"""Submit report-plot generation as a separate CPU-only Kaggle phase."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.kaggle_phase_submit import (
    create_small_dataset,
    ensure_source_dataset,
    new_run_dir,
    require_complete_kernels,
    submit_context,
    submit_phase_kernel,
)
from scripts.run_kaggle_iterative_self import _wait_for_dataset_visible


def _kernel_code() -> str:
    return r'''import json
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
    from scripts.kaggle_phase_runtime import find_json_reports, run_logged

    evaluation_results = list(find_json_reports("evaluation_result.json"))
    if not evaluation_results:
        raise FileNotFoundError("Mounted evaluation kernel has no evaluation_result.json")
    evaluation_path, evaluation_result = evaluation_results[0]
    quality_report_path = evaluation_path.parent / "quality_report.json"
    if not quality_report_path.is_file():
        raise FileNotFoundError(f"Selected quality report is missing: {quality_report_path}")

    training_reports = list(find_json_reports("training_report.json"))
    if not training_reports:
        raise FileNotFoundError("No mounted training_report.json was found")
    training_report_path, training_report = max(
        training_reports,
        key=lambda item: int(item[1].get("completed_epochs") or item[1].get("epochs") or 0),
    )
    print(
        "plot_inputs="
        + json.dumps(
            {
                "training_report": str(training_report_path),
                "completed_epochs": training_report.get("completed_epochs"),
                "quality_report": str(quality_report_path),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    report_dir = WORKING_ROOT / "report_plots"
    run_logged(
        [
            sys.executable,
            str(source_root / "scripts/create_kaggle_report_plots.py"),
            str(training_report_path),
            str(quality_report_path),
            str(report_dir),
        ],
        "create_report_plots",
        cwd=source_root,
    )
    shutil.copy2(evaluation_path, WORKING_ROOT / "evaluation_result.json")
    comparison_path = evaluation_path.parent / "quality_comparison.json"
    if comparison_path.is_file():
        shutil.copy2(comparison_path, WORKING_ROOT / "quality_comparison.json")
    (WORKING_ROOT / "report_phase_result.json").write_text(
        json.dumps(
            {
                "phase": "report_plots",
                "completed_epochs": training_report.get("completed_epochs"),
                "intelligibility_pass": evaluation_result.get("intelligibility_pass"),
                "artifact_count": len(list(report_dir.iterdir())),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    shutil.rmtree(source_root, ignore_errors=True)
    (WORKING_ROOT / "success.txt").write_text("success", encoding="utf-8")
except Exception:
    traceback_text = traceback.format_exc()
    print(traceback_text, flush=True)
    (WORKING_ROOT / "error.txt").write_text(traceback_text, encoding="utf-8")
    raise
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-kernel-ref", required=True)
    parser.add_argument("--training-report", default="")
    parser.add_argument("--training-report-dataset-ref", default="")
    parser.add_argument("--source-dataset-ref", default="")
    parser.add_argument("--kernel-slug", default="")
    parser.add_argument("--session-timeout-seconds", type=int, default=1800)
    args = parser.parse_args()
    if bool(args.training_report) == bool(args.training_report_dataset_ref):
        raise ValueError(
            "Pass exactly one of --training-report or --training-report-dataset-ref"
        )

    context = submit_context()
    require_complete_kernels(context, [args.evaluation_kernel_ref])
    timestamp, run_dir = new_run_dir(context, "plots")
    source_ref = ensure_source_dataset(
        context,
        source_ref=args.source_dataset_ref,
        run_dir=run_dir,
        timestamp=timestamp,
        phase="plots",
    )

    report_dataset_ref = args.training_report_dataset_ref.strip()
    if args.training_report:
        report_path = Path(args.training_report).expanduser().resolve()
        if not report_path.is_file():
            raise FileNotFoundError(report_path)
        report_dataset_ref = f"{context.username}/genmusic-training-report-{timestamp}"
        upload_dir = run_dir / "training_report_dataset"
        upload_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(report_path, upload_dir / "training_report.json")
        create_small_dataset(
            context,
            upload_dir=upload_dir,
            dataset_ref=report_dataset_ref,
            title=f"GenMusic training report {timestamp}",
            expected_marker="training_report.json",
        )
    else:
        _wait_for_dataset_visible(
            context.cli,
            report_dataset_ref,
            context.environment,
            expected_marker="training_report.json",
        )

    kernel_slug = args.kernel_slug or f"genmusic-report-plots-{timestamp}"
    submit_phase_kernel(
        context,
        phase="plots",
        run_dir=run_dir,
        kernel_slug=kernel_slug,
        code=_kernel_code(),
        dataset_sources=[source_ref, report_dataset_ref],
        kernel_sources=[args.evaluation_kernel_ref],
        enable_gpu=False,
        enable_internet=False,
        accelerator="",
        timeout_seconds=args.session_timeout_seconds,
        state={
            "source_dataset_ref": source_ref,
            "evaluation_kernel_ref": args.evaluation_kernel_ref,
            "training_report_dataset_ref": report_dataset_ref,
            "session_timeout_seconds": args.session_timeout_seconds,
        },
    )


if __name__ == "__main__":
    main()
