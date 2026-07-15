"""Submit uncapped preprocessing jobs for every matching Kaggle dataset part.

This is intentionally separate from the guide/full-pipeline launchers.  Each raw
part is preprocessed in its own GPU kernel so Demucs + Whisper stay within
Kaggle's per-kernel time limit.  Completed kernel outputs can then be attached
to one downstream training kernel.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath

sys.path.append(str(Path(__file__).resolve().parents[1]))

from kaggle.api.kaggle_api_extended import KaggleApi

from scripts.run_kaggle_preprocess_all import _kernel_script_content
from src.integrations.kaggle_auto import write_source_zip
from src.integrations.kaggle_auto import load_kaggle_api_tokens, resolve_kaggle_username


DATASET_SLUG_RE = re.compile(r"vietnamese-music-dataset-version3-part(\d+)$", re.IGNORECASE)
AUDIO_SUFFIXES = {".flac", ".m4a", ".mp3", ".ogg", ".wav"}


def _discover_parts(api: KaggleApi) -> list[dict[str, object]]:
    """Find exact part-numbered datasets and count every audio file via Kaggle API."""
    matches: dict[str, tuple[int, str]] = {}
    for page in range(1, 21):
        rows = api.dataset_list(search="vietnamese-music-dataset-version3-part", page=page) or []
        if not rows:
            break
        new_matches = 0
        for dataset in rows:
            ref = str(getattr(dataset, "ref", "") or "")
            match = DATASET_SLUG_RE.fullmatch(ref.rsplit("/", 1)[-1])
            if match and ref not in matches:
                matches[ref] = (int(match.group(1)), ref.rsplit("/", 1)[-1])
                new_matches += 1
        if new_matches == 0 and page > 1:
            break

    discovered: list[dict[str, object]] = []
    for ref, (part, slug) in sorted(matches.items(), key=lambda item: item[1][0]):
        page_token = None
        audio_count = 0
        audio_bytes = 0
        while True:
            response = api.dataset_list_files(ref, page_token=page_token, page_size=200)
            for item in response.files or []:
                if PurePosixPath(str(item.name)).suffix.lower() in AUDIO_SUFFIXES:
                    audio_count += 1
                    audio_bytes += int(item.total_bytes or 0)
            page_token = response.next_page_token
            if not page_token:
                break
        discovered.append(
            {"part": part, "ref": ref, "slug": slug, "audio_count": audio_count, "audio_bytes": audio_bytes}
        )
    return discovered


def _parse_reused(values: list[str]) -> dict[int, str]:
    reused: dict[int, str] = {}
    for value in values:
        part_text, separator, kernel_ref = value.partition("=")
        if not separator or not part_text.isdigit() or "/" not in kernel_ref:
            raise ValueError("--reuse must use PART=OWNER/KERNEL-SLUG")
        reused[int(part_text)] = kernel_ref.strip()
    return reused


def _old_kaggle_cli() -> list[str]:
    """Use Kaggle 1.7 because the configured legacy API key predates Kaggle 2.x auth."""
    uvx = shutil.which("uvx")
    if not uvx:
        raise RuntimeError("uvx was not found; install uv before submitting Kaggle jobs")
    return [uvx, "--from", "kaggle==1.7.4.5", "kaggle"]


def _run_cli(cli: list[str], args: list[str], env: dict[str, str], *, timeout: int = 900) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        cli + args,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    return result


def _wait_for_dataset(cli: list[str], ref: str, env: dict[str, str]) -> None:
    for _ in range(180):
        result = _run_cli(cli, ["datasets", "status", ref], env, timeout=120)
        status = (result.stdout + result.stderr).lower()
        if result.returncode == 0 and "ready" in status:
            # Kaggle may report ready a few seconds before the dataset is mountable
            # by a newly pushed kernel, so allow its storage index to settle.
            time.sleep(15)
            return
        time.sleep(5)
    raise TimeoutError(f"Kaggle source dataset did not become ready: {ref}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reuse",
        action="append",
        default=[],
        metavar="PART=KERNEL_REF",
        help="Reuse a completed preprocess/full-pipeline kernel output for this part.",
    )
    parser.add_argument(
        "--max-new-jobs",
        type=int,
        default=2,
        help="Maximum newly submitted preprocess kernels; the current account permits two batch GPU sessions.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    tokens = load_kaggle_api_tokens()
    username = resolve_kaggle_username(tokens.get("KAGGLE_USERNAME"))
    if not username or not tokens.get("KAGGLE_KEY"):
        raise RuntimeError("Missing KAGGLE_USERNAME or KAGGLE_KEY")
    for key in ("KAGGLE_USERNAME", "KAGGLE_KEY"):
        os.environ.setdefault(key, tokens[key])
    kaggle_env = {**os.environ, **tokens, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}

    api = KaggleApi()
    api.authenticate()
    parts = _discover_parts(api)
    if not parts:
        raise RuntimeError("No exact vietnamese-music-dataset-version3-partX datasets were found")

    reused = _parse_reused(args.reuse)
    for item in parts:
        part = int(item["part"])
        item["reused_kernel_ref"] = reused.get(part, "")
    pending = [item for item in parts if not item["reused_kernel_ref"]]
    if len(pending) > args.max_new_jobs:
        selected = pending[: args.max_new_jobs]
        deferred = pending[args.max_new_jobs :]
    else:
        selected, deferred = pending, []

    print("Discovered Kaggle dataset parts:")
    for item in parts:
        reused_note = f" reuse={item['reused_kernel_ref']}" if item["reused_kernel_ref"] else ""
        print(f"  part {item['part']}: {item['audio_count']} audio files ({item['audio_bytes'] / 1024**3:.3f} GiB){reused_note}")
    print(f"Total: {sum(int(item['audio_count']) for item in parts)} audio files")

    run_id = f"allparts-{int(time.time())}"
    run_dir = project_root / "outputs" / "kaggle_all_parts" / run_id
    source_dir = run_dir / "source_dataset"
    kernels_dir = run_dir / "kernels"
    source_dir.mkdir(parents=True, exist_ok=True)
    kernels_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "state.json"
    state: dict[str, object] = {
        "run_id": run_id,
        "created_at": time.time(),
        "parts": parts,
        "deferred_parts": [int(item["part"]) for item in deferred],
        "submitted": [],
    }

    if not selected:
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        print("No new preprocessing kernels need to be submitted.")
        return

    cli = _old_kaggle_cli()
    # Keep the genmusic-source- prefix because the Kaggle-side bootstrapper uses
    # it to distinguish this code dataset from all attached raw audio datasets.
    source_slug = f"genmusic-source-allparts-{int(time.time())}"
    source_ref = f"{username}/{source_slug}"
    state["source_dataset_ref"] = source_ref
    write_source_zip(project_root, source_dir / "genmusic_vn_source.zip")
    (source_dir / "dataset-metadata.json").write_text(
        json.dumps(
            {"title": f"GenMusic all-parts source {run_id}", "id": source_ref, "licenses": [{"name": "other"}]},
            indent=2,
        ),
        encoding="utf-8",
    )
    created = _run_cli(cli, ["datasets", "create", "-p", str(source_dir), "-r", "zip"], kaggle_env)
    if created.returncode != 0:
        raise RuntimeError("Could not create the shared GenMusic source dataset on Kaggle")
    _wait_for_dataset(cli, source_ref, kaggle_env)

    submitted: list[dict[str, object]] = []
    state["submitted"] = submitted
    for item in selected:
        part = int(item["part"])
        kernel_slug = f"genmusic-prep-p{part}-{int(time.time())}"
        kernel_ref = f"{username}/{kernel_slug}"
        kernel_dir = kernels_dir / f"part{part}"
        kernel_dir.mkdir(parents=True, exist_ok=True)

        # None means the preprocessing command receives no --max-files argument.
        kernel_script = _kernel_script_content(str(item["slug"]), max_files=None)
        (kernel_dir / "run_preprocess.py").write_text(kernel_script, encoding="utf-8")
        (kernel_dir / "kernel-metadata.json").write_text(
            json.dumps(
                {
                    "id": kernel_ref,
                    "title": kernel_slug,
                    "code_file": "run_preprocess.py",
                    "language": "python",
                    "kernel_type": "script",
                    "is_private": "true",
                    "enable_gpu": "true",
                    "enable_internet": "true",
                    "machine_shape": "NvidiaTeslaT4",
                    "dataset_sources": [item["ref"], source_ref],
                    "kernel_sources": [],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        pushed = _run_cli(cli, ["kernels", "push", "-p", str(kernel_dir)], kaggle_env)
        push_output = (pushed.stdout + pushed.stderr).lower()
        push_ok = pushed.returncode == 0 and not any(
            message in push_output
            for message in ("kernel push error", "maximum batch gpu session count")
        )
        submitted_item = {
            "part": part,
            "audio_count": item["audio_count"],
            "kernel_ref": kernel_ref,
            "url": f"https://www.kaggle.com/code/{kernel_ref}",
            "status": "submitted" if push_ok else "push_failed",
        }
        submitted.append(submitted_item)
        item["kernel_ref"] = kernel_ref if push_ok else ""
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        if not push_ok:
            raise RuntimeError(f"Kaggle rejected preprocess kernel for part {part}")

    print(f"State: {state_path}")
    for item in submitted:
        print(f"part {item['part']}: {item['url']}")


if __name__ == "__main__":
    main()
