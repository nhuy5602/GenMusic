"""Download a filtered Kaggle kernel output concurrently.

The official CLI downloads matching output files one at a time.  A latent
dataset contains hundreds of small tensors, so that path can take much longer
than the actual payload warrants.  This helper keeps the same authenticated
Kaggle API contract while downloading only matching files in parallel.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import (
    ApiListKernelSessionOutputRequest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.integrations.kaggle_auto import (
    kaggle_access_token,
    load_kaggle_api_tokens,
)


def _safe_destination(root: Path, file_name: str) -> Path:
    # Kaggle output names are POSIX paths.  Validate them lexically before
    # worker threads create parent directories; resolving a partially-created
    # Windows path concurrently can produce inconsistent parent identities.
    if "\\" in file_name:
        raise ValueError(f"Unsafe kernel output path: {file_name!r}")
    relative = PurePosixPath(file_name)
    if (
        not file_name
        or not relative.parts
        or relative.is_absolute()
        or ".." in relative.parts
        or "." in relative.parts
    ):
        raise ValueError(f"Unsafe kernel output path: {file_name!r}")
    resolved_root = root.resolve()
    return resolved_root.joinpath(*relative.parts)


def _list_matching_files(
    api: KaggleApi,
    kernel_ref: str,
    pattern: re.Pattern[str],
    *,
    page_size: int = 200,
) -> list[tuple[str, str, int | None]]:
    owner, slug, _version = api.parse_kernel_string(kernel_ref)
    token = None
    matches: list[tuple[str, str, int | None]] = []
    with api.build_kaggle_client() as client:
        while True:
            request = ApiListKernelSessionOutputRequest()
            request.user_name = owner
            request.kernel_slug = slug
            request.page_size = page_size
            if token:
                request.page_token = token
            response = client.kernels.kernels_api_client.list_kernel_session_output(
                request
            )
            for item in response.files or []:
                if pattern.search(item.file_name):
                    matches.append(
                        (
                            item.file_name,
                            item.url,
                            getattr(item, "file_size", None),
                        )
                    )
            token = response.next_page_token
            if not token:
                break
    return matches


def _modern_list_matching_files(
    kernel_ref: str,
    pattern: re.Pattern[str],
    access_token: str,
    *,
    page_size: int = 200,
    page_delay: float = 0.75,
    maximum_retries: int = 8,
) -> list[tuple[str, str, int | None]]:
    """List matching output through the KGAT endpoint with bounded backoff."""
    owner, slug = kernel_ref.split("/", 1)
    if "/" in slug:
        slug = slug.split("/", 1)[0]
    page_token = ""
    matches: list[tuple[str, str, int | None]] = []
    while True:
        request: dict[str, object] = {
            "userName": owner,
            "kernelSlug": slug,
            "pageSize": page_size,
        }
        if page_token:
            request["pageToken"] = page_token
        response = None
        for attempt in range(maximum_retries):
            response = requests.post(
                (
                    "https://www.kaggle.com/api/v1/"
                    "kernels.KernelsApiService/ListKernelSessionOutput"
                ),
                headers={"Authorization": f"Bearer {access_token}"},
                json=request,
                timeout=90.0,
            )
            if response.status_code != 429:
                response.raise_for_status()
                break
            if attempt + 1 == maximum_retries:
                response.raise_for_status()
            delay = min(60.0, 2.0 ** attempt)
            print(
                f"Kaggle output listing rate-limited; retrying in "
                f"{delay:.0f}s.",
                flush=True,
            )
            time.sleep(delay)
        if response is None:
            raise RuntimeError("Kaggle output listing returned no response")
        payload = response.json()
        for item in payload.get("files", []):
            file_name = str(item.get("fileName") or "")
            if pattern.search(file_name):
                matches.append(
                    (
                        file_name,
                        str(item.get("url") or ""),
                        (
                            int(item["fileSize"])
                            if item.get("fileSize") is not None
                            else None
                        ),
                    )
                )
        page_token = str(payload.get("nextPageToken") or "")
        if not page_token:
            break
        if page_delay > 0:
            time.sleep(page_delay)
    return matches


def _download_one(
    root: Path,
    item: tuple[str, str, int | None],
    *,
    force: bool,
) -> tuple[str, int, bool]:
    file_name, url, expected_size = item
    destination = _safe_destination(root, file_name)
    if (
        not force
        and destination.is_file()
        and (expected_size is None or destination.stat().st_size == expected_size)
    ):
        return file_name, destination.stat().st_size, False

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    try:
        with requests.get(url, stream=True, timeout=(20, 180)) as response:
            response.raise_for_status()
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return file_name, destination.stat().st_size, True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel_ref", help="Kaggle kernel reference owner/slug")
    parser.add_argument("--pattern", required=True, help="Regex matched against output paths")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument(
        "--page-delay",
        type=float,
        default=0.75,
        help="Delay between KGAT listing pages to avoid HTTP 429.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.workers < 1 or args.workers > 32:
        raise ValueError("--workers must be between 1 and 32")
    if not 0.0 <= args.page_delay <= 10.0:
        raise ValueError("--page-delay must be between 0 and 10 seconds")
    pattern = re.compile(args.pattern)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    access_token = kaggle_access_token(load_kaggle_api_tokens())
    if access_token:
        items = _modern_list_matching_files(
            args.kernel_ref,
            pattern,
            access_token,
            page_delay=args.page_delay,
        )
    else:
        api = KaggleApi()
        api.authenticate()
        items = _list_matching_files(api, args.kernel_ref, pattern)
    if not items:
        raise RuntimeError(
            f"No output paths from {args.kernel_ref} matched {args.pattern!r}"
        )

    print(f"Matched {len(items)} output files; downloading with {args.workers} workers.")
    downloaded = 0
    skipped = 0
    total_bytes = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(_download_one, output, item, force=args.force)
            for item in items
        ]
        for completed, future in enumerate(as_completed(futures), start=1):
            _name, size, changed = future.result()
            total_bytes += size
            downloaded += int(changed)
            skipped += int(not changed)
            if completed % 50 == 0 or completed == len(futures):
                print(
                    f"[{completed}/{len(futures)}] "
                    f"downloaded={downloaded} skipped={skipped}",
                    flush=True,
                )
    print(
        f"Complete: {len(items)} files, {total_bytes / (1024**2):.1f} MiB "
        f"under {output}"
    )


if __name__ == "__main__":
    main()
