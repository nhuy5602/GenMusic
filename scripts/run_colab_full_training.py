"""Train GenMusic on Google Colab without replacing the Kaggle backend.

The six public Kaggle preprocessing outputs remain the source of the prepared
mel tensors. Colab supplies only the GPU runtime; Google Drive stores resumable
checkpoints and generated audio.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_KERNEL_REFS = (
    "ngochuy5602/genmusic-prep-p1-1784095999",
    "ngochuy5602/genmusic-prep-p2-1784096002",
    "ngochuy5602/genmusic-prep-p3-1784107963",
    "ngochuy5602/genmusic-prep-p4-1784108049",
    "ngochuy5602/genmusic-prep-p5-1784120546",
    "ngochuy5602/genmusic-fullexp-1784078352",
)
REQUIRED_MEL_FIELDS = ("backing_mel_path", "vocal_mel_path", "style_embed_path")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    subprocess.run(command, cwd=cwd, check=True, env=environment)


def _safe_rmtree(path: Path, allowed_root: Path) -> None:
    resolved = path.resolve()
    root = allowed_root.resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"Refusing to remove path outside the allowed root: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _read_records(records_path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def find_processed_dataset(root: Path) -> Path:
    candidates = []
    for records_path in root.rglob("records.jsonl"):
        parent = records_path.parent
        if (parent / "config.json").is_file() and (parent / "mels").is_dir():
            candidates.append(parent)
    if not candidates:
        raise FileNotFoundError(f"No processed dataset found below {root}")
    candidates.sort(key=lambda path: (0 if path.name == "processed_dataset" else 1, len(path.parts)))
    return candidates[0]


def _download_processed_output(kernel_ref: str, destination: Path, workspace: Path) -> Path:
    _safe_rmtree(destination, workspace)
    destination.mkdir(parents=True, exist_ok=True)
    filtered_command = [
        sys.executable,
        "-m",
        "kaggle",
        "kernels",
        "output",
        kernel_ref,
        "-p",
        str(destination),
        "-o",
        "--file-pattern",
        "processed_dataset/.*",
    ]
    _run(filtered_command)
    try:
        return find_processed_dataset(destination)
    except FileNotFoundError:
        # Older Kaggle output versions did not consistently expose paths to the
        # server-side regex filter, so retry the same public output unfiltered.
        _run(
            [
                sys.executable,
                "-m",
                "kaggle",
                "kernels",
                "output",
                kernel_ref,
                "-p",
                str(destination),
                "-o",
            ]
        )
        return find_processed_dataset(destination)


def _cache_dataset(source: Path, cache_dir: Path, cache_root: Path, kernel_ref: str) -> Path:
    marker = cache_dir / ".complete.json"
    if marker.is_file():
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        if payload.get("kernel_ref") == kernel_ref:
            return cache_dir
    _safe_rmtree(cache_dir, cache_root)
    shutil.copytree(source, cache_dir)
    records = _read_records(cache_dir / "records.jsonl")
    marker.write_text(
        json.dumps(
            {"kernel_ref": kernel_ref, "record_count": len(records), "cached_at": _now()},
            indent=2,
        ),
        encoding="utf-8",
    )
    return cache_dir


def prepare_processed_parts(
    kernel_refs: list[str],
    *,
    workspace: Path,
    cache_root: Path | None = None,
    on_progress=None,
) -> list[Path]:
    downloads_root = workspace / "downloads"
    downloads_root.mkdir(parents=True, exist_ok=True)
    if cache_root is not None:
        cache_root.mkdir(parents=True, exist_ok=True)

    prepared = []
    for index, kernel_ref in enumerate(kernel_refs, start=1):
        slug = kernel_ref.rsplit("/", 1)[-1]
        cache_dir = cache_root / f"part_{index:02d}_{slug}" if cache_root is not None else None
        if cache_dir is not None and (cache_dir / ".complete.json").is_file():
            try:
                marker = json.loads((cache_dir / ".complete.json").read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                marker = {}
            if marker.get("kernel_ref") == kernel_ref:
                if on_progress is not None:
                    on_progress(index, len(kernel_refs), kernel_ref, "cached")
                prepared.append(cache_dir)
                continue

        if on_progress is not None:
            on_progress(index, len(kernel_refs), kernel_ref, "downloading")
        download_dir = downloads_root / f"part_{index:02d}_{slug}"
        processed = _download_processed_output(kernel_ref, download_dir, workspace)
        if cache_dir is not None:
            if on_progress is not None:
                on_progress(index, len(kernel_refs), kernel_ref, "caching")
            _cache_dataset(processed, cache_dir, cache_root, kernel_ref)
        # Use the local download during the current session for faster training.
        prepared.append(processed)
        if on_progress is not None:
            on_progress(index, len(kernel_refs), kernel_ref, "ready")
    return prepared


def _link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.symlink(source, destination)
    except OSError:
        # Windows test environments may not permit symlinks; Colab/Linux does.
        shutil.copy2(source, destination)


def merge_processed_datasets(
    source_roots: list[Path],
    combined_root: Path,
    *,
    expected_records: int,
) -> dict[str, Any]:
    allowed_root = combined_root.parent
    _safe_rmtree(combined_root, allowed_root)
    combined_mels = combined_root / "mels"
    combined_mels.mkdir(parents=True, exist_ok=True)
    combined_records = []
    source_summaries = []
    first_config: dict[str, Any] | None = None

    for source_index, source_root in enumerate(source_roots, start=1):
        records_path = source_root / "records.jsonl"
        config_path = source_root / "config.json"
        records = _read_records(records_path)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if first_config is None:
            first_config = config
        else:
            for key in ("sample_rate", "hop_length", "n_mels"):
                if config.get(key) != first_config.get(key):
                    raise ValueError(
                        f"Dataset config mismatch for source {source_index}: {key}"
                    )

        source_summaries.append(
            {"source": str(source_root), "records": len(records)}
        )
        for record_index, record in enumerate(records, start=1):
            copied = dict(record)
            copied["id"] = "source%02d_%s" % (
                source_index,
                copied.get("id", record_index),
            )
            for field in REQUIRED_MEL_FIELDS:
                relative_path = copied.get(field)
                source_file = source_root / relative_path if relative_path else None
                if source_file is None or not source_file.is_file():
                    raise FileNotFoundError(
                        f"Missing {field} for {copied['id']}: {source_file}"
                    )
                destination_name = "source%02d_%s" % (
                    source_index,
                    source_file.name,
                )
                destination = combined_mels / destination_name
                if not destination.exists():
                    _link_or_copy(source_file.resolve(), destination)
                copied[field] = "mels/" + destination_name
            combined_records.append(copied)

    if len(combined_records) != expected_records:
        raise RuntimeError(
            f"Expected {expected_records} records, found {len(combined_records)}"
        )
    if first_config is None:
        raise RuntimeError("No processed datasets were provided")

    (combined_root / "config.json").write_text(
        json.dumps(first_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (combined_root / "records.jsonl").write_text(
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n"
            for record in combined_records
        ),
        encoding="utf-8",
    )
    summary = {
        "expected_records": expected_records,
        "combined_records": len(combined_records),
        "sources": source_summaries,
    }
    (combined_root / "combined_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def gpu_preflight() -> dict[str, Any]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        raise RuntimeError(
            "Google Colab is using a CPU runtime. Select Runtime > Change runtime type > GPU."
        )
    _run([nvidia_smi, "-L"])
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("The Colab runtime has a GPU but PyTorch cannot access CUDA.")
    smoke = torch.rand(1, device="cuda").cpu().tolist()
    report = {
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability()),
        "arches": torch.cuda.get_arch_list(),
        "cuda_smoke": smoke,
    }
    print(json.dumps(report, indent=2), flush=True)
    return report


def _write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", action="append", default=[])
    parser.add_argument("--workspace", default="/content/genmusic_colab")
    parser.add_argument("--drive-root", default="/content/drive/MyDrive/GenMusic")
    parser.add_argument("--cache-data-on-drive", action="store_true")
    parser.add_argument("--expected-records", type=int, default=1843)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--frames-per-chunk", type=int, default=384)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--checkpoint-every-steps", type=int, default=25)
    parser.add_argument("--log-every-steps", type=int, default=10)
    parser.add_argument("--skip-preflight-train", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    workspace = Path(args.workspace).resolve()
    drive_root = Path(args.drive_root).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    drive_root.mkdir(parents=True, exist_ok=True)
    state_path = drive_root / "colab_training_state.json"
    kernel_refs = args.kernel or list(DEFAULT_KERNEL_REFS)
    state: dict[str, Any] = {
        "status": "starting",
        "started_at": _now(),
        "kernel_refs": kernel_refs,
        "expected_records": args.expected_records,
        "epochs": args.epochs,
        "workspace": str(workspace),
        "drive_root": str(drive_root),
    }
    _write_state(state_path, state)

    try:
        state["gpu"] = gpu_preflight()
        state["status"] = "downloading_processed_parts"
        _write_state(state_path, state)
        cache_root = (
            drive_root / "processed_kernel_outputs"
            if args.cache_data_on_drive
            else None
        )
        def update_download_progress(
            index: int,
            total: int,
            kernel_ref: str,
            part_status: str,
        ) -> None:
            state["status"] = "preparing_processed_parts"
            state["part"] = {
                "index": index,
                "total": total,
                "kernel_ref": kernel_ref,
                "status": part_status,
            }
            state["updated_at"] = _now()
            _write_state(state_path, state)
            print(
                f"part={index}/{total} status={part_status} kernel={kernel_ref}",
                flush=True,
            )

        processed_parts = prepare_processed_parts(
            kernel_refs,
            workspace=workspace,
            cache_root=cache_root,
            on_progress=update_download_progress,
        )

        state["status"] = "merging"
        _write_state(state_path, state)
        combined_root = workspace / "combined_dataset"
        state["merge"] = merge_processed_datasets(
            processed_parts,
            combined_root,
            expected_records=args.expected_records,
        )

        checkpoint = drive_root / "checkpoints" / "baseline_all_parts.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        if not args.skip_preflight_train and not checkpoint.is_file():
            state["status"] = "preflight_training"
            _write_state(state_path, state)
            preflight_checkpoint = workspace / "preflight_all_parts.pt"
            _run(
                [
                    sys.executable,
                    str(project_root / "cli.py"),
                    "train-self",
                    "--dataset",
                    str(combined_root),
                    "--checkpoint",
                    str(preflight_checkpoint),
                    "--epochs",
                    "1",
                    "--batch-size",
                    str(args.batch_size),
                    "--max-records",
                    "4",
                    "--device",
                    "cuda",
                    "--frames-per-chunk",
                    str(args.frames_per_chunk),
                    "--dim",
                    str(args.dim),
                    "--depth",
                    str(args.depth),
                    "--heads",
                    str(args.heads),
                ],
                cwd=project_root,
            )
            preflight_checkpoint.unlink(missing_ok=True)

        state["status"] = "training"
        state["checkpoint"] = str(checkpoint)
        _write_state(state_path, state)
        train_command = [
            sys.executable,
            str(project_root / "cli.py"),
            "train-self",
            "--dataset",
            str(combined_root),
            "--checkpoint",
            str(checkpoint),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--device",
            "cuda",
            "--frames-per-chunk",
            str(args.frames_per_chunk),
            "--dim",
            str(args.dim),
            "--depth",
            str(args.depth),
            "--heads",
            str(args.heads),
            "--resume",
            "--save-every-epoch",
            "--checkpoint-every-steps",
            str(max(1, args.checkpoint_every_steps)),
            "--log-every-steps",
            str(max(1, args.log_every_steps)),
            "--progress-file",
            str(state_path),
        ]
        _run(train_command, cwd=project_root)

        reference_record = _read_records(combined_root / "records.jsonl")[0]
        lyrics = " ".join(str(reference_record.get("text", "")).split()[:20])
        if not lyrics:
            raise RuntimeError("The first combined record has no usable lyrics")
        generation_dir = drive_root / "generated_all_parts"
        state["status"] = "generating"
        _write_state(state_path, state)
        _run(
            [
                sys.executable,
                str(project_root / "cli.py"),
                "generate-local",
                "--text",
                lyrics,
                "--style",
                str(
                    reference_record.get("style")
                    or "Vietnamese music, clear vocal"
                ),
                "--duration",
                "12",
                "--checkpoint",
                str(checkpoint),
                "--steps",
                "64",
                "--guidance-scale",
                "1.5",
                "--vocoder",
                "vocos",
                "--device",
                "cuda",
                "--reference-dataset",
                str(combined_root),
                "--reference-id",
                str(reference_record["id"]),
                "--out",
                str(generation_dir),
            ],
            cwd=project_root,
        )
        state["status"] = "complete"
        state["completed_at"] = _now()
        state["generation_dir"] = str(generation_dir)
        state["generated_files"] = [
            str(path)
            for path in generation_dir.rglob("*")
            if path.is_file()
        ]
        _write_state(state_path, state)
        print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)
    except Exception:
        state["status"] = "failed"
        state["failed_at"] = _now()
        state["error"] = traceback.format_exc()
        _write_state(state_path, state)
        print(state["error"], flush=True)
        raise


if __name__ == "__main__":
    main()
