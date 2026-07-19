"""Submit native CTC pretraining as an isolated Kaggle GPU phase."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.kaggle_phase_submit import (
    ensure_source_dataset,
    new_run_dir,
    require_complete_kernels,
    submit_context,
    submit_phase_kernel,
)
from scripts.run_kaggle_iterative_self import _wait_for_dataset_visible
from scripts.run_kaggle_multi_part_training import _parse_kernel_refs


def _kernel_code(
    *,
    source_count: int,
    expected_records: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    crops_per_record: int,
    validation_max_records: int,
    patience: int,
    minimum_epochs: int,
    objective: str,
) -> str:
    template = r'''import os
import shutil
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

    from scripts.kaggle_phase_runtime import (
        build_combined_dataset,
        find_checkpoint,
        gpu_preflight,
        run_logged,
    )

    environment = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "PYTHONPATH": str(source_root) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    gpu_preflight()
    combined_root = build_combined_dataset(
        source_count=__SOURCE_COUNT__,
        expected_records=__EXPECTED_RECORDS__,
    )
    checkpoint = find_checkpoint()
    output_checkpoint = WORKING_ROOT / "self_all_parts.pt"
    report_path = WORKING_ROOT / "ctc_pretrain_report.json"
    print(f"ctc_input_checkpoint={checkpoint}", flush=True)
    run_logged(
        [
            sys.executable,
            str(source_root / "scripts/pretrain_native_ctc.py"),
            str(checkpoint),
            str(combined_root),
            str(output_checkpoint),
            "--report",
            str(report_path),
            "--epochs",
            str(__EPOCHS__),
            "--batch-size",
            str(__BATCH_SIZE__),
            "--learning-rate",
            str(__LEARNING_RATE__),
            "--crops-per-record",
            str(__CROPS_PER_RECORD__),
            "--validation-max-records",
            str(__VALIDATION_MAX_RECORDS__),
            "--patience",
            str(__PATIENCE__),
            "--minimum-epochs",
            str(__MINIMUM_EPOCHS__),
            "--objective",
            __OBJECTIVE__,
            "--device",
            "cuda",
        ],
        "native_ctc_pretrain",
        cwd=source_root,
        env=environment,
    )
    if not output_checkpoint.is_file() or not report_path.is_file():
        raise RuntimeError("Native CTC phase did not produce checkpoint/report")
    shutil.rmtree(combined_root, ignore_errors=True)
    shutil.rmtree(source_root, ignore_errors=True)
    (WORKING_ROOT / "success.txt").write_text("success", encoding="utf-8")
except Exception:
    traceback_text = traceback.format_exc()
    print(traceback_text, flush=True)
    (WORKING_ROOT / "error.txt").write_text(traceback_text, encoding="utf-8")
    raise
'''
    return (
        template.replace("__SOURCE_COUNT__", str(source_count))
        .replace("__EXPECTED_RECORDS__", str(expected_records))
        .replace("__EPOCHS__", str(epochs))
        .replace("__BATCH_SIZE__", str(batch_size))
        .replace("__LEARNING_RATE__", repr(float(learning_rate)))
        .replace("__CROPS_PER_RECORD__", str(crops_per_record))
        .replace("__VALIDATION_MAX_RECORDS__", str(validation_max_records))
        .replace("__PATIENCE__", str(patience))
        .replace("__MINIMUM_EPOCHS__", str(minimum_epochs))
        .replace("__OBJECTIVE__", repr(objective))
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", action="append", default=[], metavar="PART=KERNEL_REF")
    parser.add_argument("--checkpoint-kernel-ref", default="")
    parser.add_argument("--checkpoint-dataset-ref", default="")
    parser.add_argument("--source-dataset-ref", default="")
    parser.add_argument("--expected-records", type=int, default=1843)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--crops-per-record", type=int, default=3)
    parser.add_argument("--validation-max-records", type=int, default=128)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--minimum-epochs", type=int, default=5)
    parser.add_argument("--objective", choices=("ctc", "frame"), default="ctc")
    parser.add_argument("--accelerator", default="NvidiaTeslaT4")
    parser.add_argument("--session-timeout-seconds", type=int, default=7200)
    parser.add_argument("--kernel-slug", default="")
    args = parser.parse_args()

    if bool(args.checkpoint_kernel_ref) == bool(args.checkpoint_dataset_ref):
        raise ValueError(
            "Pass exactly one of --checkpoint-kernel-ref or --checkpoint-dataset-ref"
        )
    if args.epochs < 1 or args.minimum_epochs < 1 or args.patience < 1:
        raise ValueError("epochs, minimum-epochs and patience must be positive")

    refs_by_part = _parse_kernel_refs(args.kernel)
    context = submit_context()
    complete_refs = list(refs_by_part.values())
    if args.checkpoint_kernel_ref:
        complete_refs.append(args.checkpoint_kernel_ref)
    require_complete_kernels(context, complete_refs)
    if args.checkpoint_dataset_ref:
        _wait_for_dataset_visible(
            context.cli,
            args.checkpoint_dataset_ref,
            context.environment,
            expected_marker="self_all_parts.pt",
        )

    timestamp, run_dir = new_run_dir(context, "native-ctc")
    source_ref = ensure_source_dataset(
        context,
        source_ref=args.source_dataset_ref,
        run_dir=run_dir,
        timestamp=timestamp,
        phase="native-ctc",
    )
    kernel_slug = args.kernel_slug or f"genmusic-native-ctc-{timestamp}"
    dataset_sources = [source_ref]
    if args.checkpoint_dataset_ref:
        dataset_sources.append(args.checkpoint_dataset_ref)
    kernel_sources = [refs_by_part[part] for part in sorted(refs_by_part)]
    if args.checkpoint_kernel_ref:
        kernel_sources.append(args.checkpoint_kernel_ref)

    submit_phase_kernel(
        context,
        phase="native_ctc",
        run_dir=run_dir,
        kernel_slug=kernel_slug,
        code=_kernel_code(
            source_count=len(refs_by_part),
            expected_records=args.expected_records,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            crops_per_record=args.crops_per_record,
            validation_max_records=args.validation_max_records,
            patience=args.patience,
            minimum_epochs=args.minimum_epochs,
            objective=args.objective,
        ),
        dataset_sources=dataset_sources,
        kernel_sources=kernel_sources,
        enable_gpu=True,
        enable_internet=False,
        accelerator=args.accelerator,
        timeout_seconds=args.session_timeout_seconds,
        state={
            "source_dataset_ref": source_ref,
            "processed_kernel_refs": refs_by_part,
            "checkpoint_kernel_ref": args.checkpoint_kernel_ref or None,
            "checkpoint_dataset_ref": args.checkpoint_dataset_ref or None,
            "expected_records": args.expected_records,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "crops_per_record": args.crops_per_record,
            "validation_max_records": args.validation_max_records,
            "patience": args.patience,
            "minimum_epochs": args.minimum_epochs,
            "objective": args.objective,
            "accelerator": args.accelerator,
            "session_timeout_seconds": args.session_timeout_seconds,
        },
    )


if __name__ == "__main__":
    main()
