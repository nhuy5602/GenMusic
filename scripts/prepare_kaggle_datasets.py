"""Submit preprocessing jobs for a configured list of Kaggle datasets."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.run_kaggle_preprocess_raw_audio import _kernel_script_content
from src.integrations.kaggle_auto import (
    kaggle_access_token,
    kaggle_auth_available,
    kaggle_auth_environment,
    load_kaggle_api_tokens,
    resolve_kaggle_username,
    write_source_zip,
)

# Danh sách mặc định các tập dữ liệu Kaggle cần tiền xử lý
RAW_DATASETS = [
    "https://www.kaggle.com/datasets/sonlest/vietnamese-music-dataset-version3-part1",
    "https://www.kaggle.com/datasets/sonlest/vietnamese-music-dataset-version3-part2",
    "https://www.kaggle.com/datasets/sonlest/vietnamese-music-dataset-version3-part3",
    "https://www.kaggle.com/datasets/sonlest/vietnamese-music-dataset-version3-part4",
    "https://www.kaggle.com/datasets/sonlest/vietnamese-music-dataset-version3-part5",
    "https://www.kaggle.com/datasets/sonlest/vietnamese-music-dataset-version3-part6",
]

DATASET_SLUG_RE = re.compile(r"-part(\d+)$", re.IGNORECASE)


def _parse_dataset_ref(value: str) -> str:
    """Parse kaggle dataset URL or reference format into 'owner/slug'."""
    text = value.strip().rstrip("/")
    if "kaggle.com/datasets/" in text:
        text = text.split("kaggle.com/datasets/")[-1]
    elif "kaggle.com/" in text:
        text = text.split("kaggle.com/")[-1]
    parts = [p for p in text.split("/") if p]
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return text


def _old_kaggle_cli(tokens: dict[str, str] | None = None) -> list[str]:
    """Select a CLI compatible with either modern access-token or legacy auth."""
    if kaggle_access_token(tokens):
        return [sys.executable, "-m", "kaggle"]
    import shutil
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
    # Sau khi upload zip dataset source code, chờ 10s cho Kaggle index xong
    time.sleep(10)
    for _ in range(30):
        try:
            result = _run_cli(cli, ["datasets", "status", ref], env, timeout=60)
            status = (result.stdout + result.stderr).lower()
            if result.returncode == 0 and "ready" in status:
                time.sleep(5)
                return
        except Exception:
            pass
        time.sleep(3)
    # Nếu API check status bị chặn (403 Forbidden với token mới), vẫn tiếp tục vì zip file nhỏ đã upload thành công
    print("⚠️ Dataset source upload finished (continuing kernel push)...", flush=True)


def _is_kernel_finished(cli: list[str], kernel_ref: str, env: dict[str, str]) -> bool:
    """Check if a Kaggle kernel has finished running (complete, error, or cancelled)."""
    try:
        res = _run_cli(cli, ["kernels", "status", kernel_ref], env, timeout=60)
        output = (res.stdout + res.stderr).lower()
        if "complete" in output or "error" in output or "cancel" in output:
            return True
    except Exception:
        pass
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit preprocessing kernels for a list of Kaggle datasets.")
    parser.add_argument(
        "--max-new-jobs",
        type=int,
        default=2,
        help="Maximum newly submitted preprocess kernels (Kaggle permits max 2 batch GPU sessions).",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="tiny",
        help="Whisper model size for lyric transcription (tiny/base/small/...).",
    )
    parser.add_argument(
        "--wait-and-loop",
        action="store_true",
        help="Tự động chờ các job đang chạy xong rồi tự submit tiếp các part tiếp theo cho tới khi xử lý hết danh sách.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    tokens = kaggle_auth_environment(load_kaggle_api_tokens())
    username = resolve_kaggle_username(tokens.get("KAGGLE_USERNAME"))
    if not username or not kaggle_auth_available(tokens):
        raise RuntimeError("Missing KAGGLE_USERNAME or Kaggle auth")
    
    os.environ.update({key: value for key, value in tokens.items() if key.startswith("KAGGLE_")})
    kaggle_env = {**os.environ, **tokens, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}

    state_file = project_root / "outputs" / "kaggle_datasets_preparation" / "submitted_state.json"
    state_file.parent.mkdir(parents=True, exist_ok=True)
    submitted_history: dict[str, str] = {}
    if state_file.exists():
        try:
            submitted_history = json.loads(state_file.read_text(encoding="utf-8"))
        except Exception:
            submitted_history = {}

    parts = []
    for idx, raw_val in enumerate(RAW_DATASETS, start=1):
        ref = _parse_dataset_ref(raw_val)
        slug = ref.split("/")[-1]
        match = DATASET_SLUG_RE.search(slug)
        part_num = int(match.group(1)) if match else idx
        parts.append({
            "part": part_num,
            "ref": ref,
            "slug": slug,
            "already_submitted": ref in submitted_history,
            "kernel_url": submitted_history.get(ref, ""),
        })

    # Chọn các part CHƯA từng được submit
    unsubmitted = [item for item in parts if not item["already_submitted"]]
    pending = unsubmitted[: args.max_new_jobs]

    if not unsubmitted:
        print("🎉 TẤT CẢ CÁC PART ĐÃ ĐƯỢC SUBMIT LÊN KAGGLE TRƯỚC ĐÓ!")
        for item in parts:
            print(f"  part {item['part']} ({item['ref']}): {item['kernel_url']}")
        return

    print(f"Dataset parts status (submitting next {len(pending)} jobs):")
    for item in parts:
        if item["already_submitted"]:
            status_note = f" [ALREADY SUBMITTED: {item['kernel_url']}]"
        elif item in pending:
            status_note = " [PENDING SUBMIT]"
        else:
            status_note = " [DEFERRED - run again after current jobs finish]"
        print(f"  part {item['part']}: {item['ref']}{status_note}")

    run_id = f"dataprep-{int(time.time())}"
    run_dir = project_root / "outputs" / "prepare_kaggle_datasets" / run_id
    source_dir = run_dir / "source_dataset"
    kernels_dir = run_dir / "kernels"
    source_dir.mkdir(parents=True, exist_ok=True)
    kernels_dir.mkdir(parents=True, exist_ok=True)

    cli = _old_kaggle_cli(tokens)
    source_slug = f"genmusic-source-dataprep-{int(time.time())}"
    source_ref = f"{username}/{source_slug}"

    write_source_zip(project_root, source_dir / "genmusic_vn_source.zip")
    (source_dir / "dataset-metadata.json").write_text(
        json.dumps(
            {"title": f"GenMusic Source Code {run_id}", "id": source_ref, "licenses": [{"name": "other"}]},
            indent=2,
        ),
        encoding="utf-8",
    )
    
    print("\n📦 Uploading GenMusic source code...")
    created = _run_cli(cli, ["datasets", "create", "-p", str(source_dir), "-r", "zip"], kaggle_env)
    if created.returncode != 0:
        raise RuntimeError("Could not create shared GenMusic source dataset on Kaggle")
    _wait_for_dataset(cli, source_ref, kaggle_env)

    submitted = []
    for item in pending:
        part = item["part"]
        kernel_slug = f"genmusic-data-prep-p{part}-{int(time.time())}"
        kernel_ref = f"{username}/{kernel_slug}"
        kernel_dir = kernels_dir / f"part{part}"
        kernel_dir.mkdir(parents=True, exist_ok=True)

        kernel_script = _kernel_script_content(str(item["slug"]), max_files=None, whisper_model=args.whisper_model)
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
        print(f"\n🚀 Submitting Preprocess Kernel for Part {part} ({item['ref']})...")
        pushed = _run_cli(cli, ["kernels", "push", "-p", str(kernel_dir)], kaggle_env)
        push_output = (pushed.stdout + pushed.stderr).lower()
        push_ok = pushed.returncode == 0 and not any(
            message in push_output
            for message in ("kernel push error", "maximum batch gpu session count")
        )
        if not push_ok:
            raise RuntimeError(f"Kaggle rejected preprocess kernel for part {part}")
            
        url = f"https://www.kaggle.com/code/{kernel_ref}"
        submitted.append((part, url))
        
        # Lưu vào file lịch sử để lần sau không submit lại part này
        submitted_history[item["ref"]] = url
        state_file.write_text(json.dumps(submitted_history, indent=2), encoding="utf-8")

    print("\n✅ SUBMITTED PREPROCESSING KERNELS:")
    for part, url in submitted:
        print(f"➔ Part {part}: {url}")

    if getattr(args, "wait_and_loop", False):
        print("\n⏳ [--wait-and-loop enabled] Tự động theo dõi tiến độ và chờ 2 job vừa submit hoàn thành...")
        submitted_refs = [url.split("kaggle.com/code/")[-1] for _, url in submitted]
        while submitted_refs:
            time.sleep(30)
            still_running = []
            for k_ref in submitted_refs:
                if not _is_kernel_finished(cli, k_ref, kaggle_env):
                    still_running.append(k_ref)
            submitted_refs = still_running
            if submitted_refs:
                print(f"  [Waiting] {len(submitted_refs)} job(s) vẫn đang chạy trên Kaggle...", flush=True)

        print("🎉 2 job vừa xong! Tự động submit lượt tiếp theo...\n")
        main()


if __name__ == "__main__":
    main()
