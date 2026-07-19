"""Submit a bounded clean-lyrics preprocessing phase to Kaggle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.kaggle_phase_submit import (
    ensure_source_dataset,
    new_run_dir,
    submit_context,
    submit_phase_kernel,
)
from src.data.preprocess_aligned_vietnamese import DEFAULT_REPO_ID, DEFAULT_SHARD


def _kernel_code(
    *,
    repo_id: str,
    shard: str,
    max_records: int,
    max_chunks_per_song: int,
    batch_size: int,
) -> str:
    template = r'''import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

INPUT_ROOT = Path("/kaggle/input")
WORKING_ROOT = Path("/kaggle/working")

try:
    source_cli = next(
        path
        for path in INPUT_ROOT.rglob("cli.py")
        if (path.parent / "scripts/kaggle_phase_runtime.py").is_file()
    )
    source_root = WORKING_ROOT / "GenMusic"
    shutil.copytree(source_cli.parent, source_root, dirs_exist_ok=True)
    sys.path.insert(0, str(source_root))

    from scripts.kaggle_phase_runtime import gpu_preflight, run_logged

    environment = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "PYTHONPATH": str(source_root) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        "HF_HUB_DISABLE_PROGRESS_BARS": "0",
    }
    gpu_preflight()
    dependency_probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import demucs.separate, dora, librosa, pyarrow, huggingface_hub, imageio_ffmpeg",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if dependency_probe.returncode != 0:
        run_logged(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-cache-dir",
                "--prefer-binary",
                "demucs==4.0.1",
                "dora-search",
                "treetable",
                "omegaconf",
                "antlr4-python3-runtime==4.9.*",
                "retrying",
                "submitit",
                "openunmix",
                "librosa",
                "imageio-ffmpeg",
                "huggingface-hub",
                "pyarrow",
            ],
            "install_aligned_preprocess_dependencies",
            cwd=source_root,
            env=environment,
        )

    output_root = WORKING_ROOT / "processed_aligned_dataset"
    run_logged(
        [
            sys.executable,
            str(source_root / "src/data/preprocess_aligned_vietnamese.py"),
            "--repo-id",
            __REPO_ID__,
            "--shard",
            __SHARD__,
            "--output",
            str(output_root),
            "--max-records",
            str(__MAX_RECORDS__),
            "--max-chunks-per-song",
            str(__MAX_CHUNKS_PER_SONG__),
            "--batch-size",
            str(__BATCH_SIZE__),
            "--demucs-device",
            "cuda",
        ],
        "preprocess_aligned_lyrics",
        cwd=source_root,
        env=environment,
    )
    records_path = output_root / "records.jsonl"
    count = sum(1 for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip())
    print(f"aligned_dataset_records={count}", flush=True)
    if count < max(8, int(__MAX_RECORDS__ * 0.8)):
        raise RuntimeError(f"Aligned preprocessing produced only {count} records")
    shutil.rmtree(source_root, ignore_errors=True)
    (WORKING_ROOT / "success.txt").write_text("success", encoding="utf-8")
except Exception:
    traceback_text = traceback.format_exc()
    print(traceback_text, flush=True)
    (WORKING_ROOT / "error.txt").write_text(traceback_text, encoding="utf-8")
    raise
'''
    return (
        template.replace("__REPO_ID__", repr(repo_id))
        .replace("__SHARD__", repr(shard))
        .replace("__MAX_RECORDS__", str(max_records))
        .replace("__MAX_CHUNKS_PER_SONG__", str(max_chunks_per_song))
        .replace("__BATCH_SIZE__", str(batch_size))
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--shard", default=DEFAULT_SHARD)
    parser.add_argument("--max-records", type=int, default=160)
    parser.add_argument("--max-chunks-per-song", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--source-dataset-ref", default="")
    parser.add_argument("--accelerator", default="NvidiaTeslaT4")
    parser.add_argument("--session-timeout-seconds", type=int, default=3600)
    parser.add_argument("--kernel-slug", default="")
    args = parser.parse_args()
    if args.max_records < 8 or args.max_chunks_per_song < 1 or args.batch_size < 1:
        raise ValueError("max-records >= 8 and positive song/batch limits are required")

    context = submit_context()
    timestamp, run_dir = new_run_dir(context, "aligned-preprocess")
    source_ref = ensure_source_dataset(
        context,
        source_ref=args.source_dataset_ref,
        run_dir=run_dir,
        timestamp=timestamp,
        phase="aligned-preprocess",
    )
    kernel_slug = args.kernel_slug or f"genmusic-aligned-prep-{timestamp}"
    submit_phase_kernel(
        context,
        phase="aligned_preprocess",
        run_dir=run_dir,
        kernel_slug=kernel_slug,
        code=_kernel_code(
            repo_id=args.repo_id,
            shard=args.shard,
            max_records=args.max_records,
            max_chunks_per_song=args.max_chunks_per_song,
            batch_size=args.batch_size,
        ),
        dataset_sources=[source_ref],
        kernel_sources=[],
        enable_gpu=True,
        enable_internet=True,
        accelerator=args.accelerator,
        timeout_seconds=args.session_timeout_seconds,
        state={
            "source_dataset_ref": source_ref,
            "hf_repo_id": args.repo_id,
            "hf_shard": args.shard,
            "max_records": args.max_records,
            "max_chunks_per_song": args.max_chunks_per_song,
            "batch_size": args.batch_size,
            "accelerator": args.accelerator,
            "session_timeout_seconds": args.session_timeout_seconds,
        },
    )


if __name__ == "__main__":
    main()
