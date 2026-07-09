from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

from .report_plots import generate_project_telemetry_plots


def build_project_report(
    source_root: str | Path = "outputs",
    *,
    output_root: str | Path = "outputs/project_report",
) -> dict[str, Any]:
    """Aggregate real Kaggle job telemetry from input receipt through MP3 download."""
    source_path = Path(source_root)
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)
    records = collect_project_jobs(source_path)
    summary = _summary(records)
    report = {
        "source_root": str(source_path),
        "output_root": str(output_path),
        "job_count": len(records),
        "summary": summary,
        "items": records,
    }
    report["plots"] = generate_project_telemetry_plots(report, output_path / "plots")
    report_path = output_path / "project_report.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path = output_path / "project_report.md"
    markdown_path.write_text(_markdown_report(report), encoding="utf-8")
    report["markdown_path"] = str(markdown_path)
    return report


def collect_project_jobs(source_root: str | Path) -> list[dict[str, Any]]:
    source_path = Path(source_root)
    records: list[dict[str, Any]] = []
    for state_path in sorted(source_path.rglob("job_state.json")):
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(state, dict) or not state.get("run_id"):
            continue
        records.append(_job_record(state, state_path))
    return records


def _job_record(state: dict[str, Any], state_path: Path) -> dict[str, Any]:
    input_received = state.get("input_received_at") or state.get("created_at")
    mp3_ready = state.get("mp3_ready_at") or state.get("completed_at")
    terminal_at = state.get("terminal_at") or state.get("failed_at") or state.get("checked_at")
    mp3_path = str(state.get("mp3_path") or "")
    has_mp3 = bool(mp3_path) and Path(mp3_path).exists()
    end_to_end = _elapsed_seconds(input_received, mp3_ready) if has_mp3 else None
    if end_to_end is None and state.get("input_to_mp3_seconds") is not None and has_mp3:
        end_to_end = _as_float(state.get("input_to_mp3_seconds"))
    retry_attempt = state.get("job_kind") == "tts_retry" or int(state.get("retry_count") or 0) > 0
    status = str(state.get("status") or "unknown")
    terminal_status = "complete" if status == "complete" and has_mp3 else (
        "failed_or_missing_mp3" if status == "complete" else status
    )
    return {
        "run_id": str(state.get("run_id")),
        "parent_run_id": str(state.get("parent_run_id") or ""),
        "job_kind": str(state.get("job_kind") or "music_generation"),
        "status": status,
        "terminal_status": terminal_status,
        "state_path": str(state_path),
        "input_received_at": input_received,
        "dataset_upload_started_at": state.get("dataset_upload_started_at"),
        "dataset_uploaded_at": state.get("dataset_uploaded_at"),
        "dataset_ready_at": state.get("dataset_ready_at"),
        "kernel_submit_started_at": state.get("kernel_submit_started_at"),
        "submitted_at": state.get("submitted_at"),
        "mp3_ready_at": mp3_ready,
        "terminal_at": terminal_at,
        "has_mp3": has_mp3,
        "mp3_path": mp3_path,
        "input_to_mp3_seconds": round(end_to_end, 6) if end_to_end is not None else None,
        "attempt_duration_seconds": _elapsed_seconds(input_received, terminal_at),
        "dataset_upload_seconds": _elapsed_seconds(
            state.get("dataset_upload_started_at"), state.get("dataset_uploaded_at")
        ),
        "dataset_ready_wait_seconds": _elapsed_seconds(
            state.get("dataset_uploaded_at"), state.get("dataset_ready_at")
        ),
        "kernel_to_mp3_seconds": _elapsed_seconds(state.get("submitted_at"), mp3_ready),
        "retry_count": int(state.get("retry_count") or 0),
        "retry_attempt": retry_attempt,
        "last_error": str(state.get("last_error") or ""),
        "generation_backend": str(state.get("generation_backend") or ""),
    }


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    roots = [item for item in records if not item["retry_attempt"]]
    retries = [item for item in records if item["retry_attempt"]]
    failed = [item for item in records if item["terminal_status"] != "complete"]
    retry_parent_ids = {
        item["parent_run_id"] for item in retries if item["parent_run_id"]
    }
    completed_retries = sum(1 for item in retries if item["has_mp3"])
    end_to_end = [
        float(item["input_to_mp3_seconds"])
        for item in roots
        if item["input_to_mp3_seconds"] is not None
    ]
    return {
        "total_requests": len(roots),
        "total_attempts": len(records),
        "completed_attempts": len(records) - len(failed),
        "failed_attempts": len(failed),
        "mp3_success_rate": _ratio(sum(1 for item in roots if item["has_mp3"]), len(roots)),
        "attempt_error_rate": _ratio(len(failed), len(records)),
        "retry_attempts": len(retries),
        "requests_needing_retry": len(retry_parent_ids),
        "retry_rate": _ratio(len(retry_parent_ids), len(roots)),
        "retry_success_rate": _ratio(completed_retries, len(retries)),
        "input_to_mp3_seconds_mean": _rounded_mean(end_to_end),
        "input_to_mp3_seconds_median": round(median(end_to_end), 6) if end_to_end else 0.0,
        "telemetry_complete_count": sum(
            1 for item in records if item["input_received_at"] and item["terminal_at"]
        ),
        "telemetry_missing_count": sum(
            1 for item in records if not item["input_received_at"] or not item["terminal_at"]
        ),
    }


def _markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    plots = report.get("plots", {}).get("files", {})
    lines = [
        "# GenMusic VN project report",
        "",
        f"- Jobs scanned: `{report['job_count']}`",
        f"- Requests needing Kaggle retry: `{summary['requests_needing_retry']}`",
        f"- Retry rate: `{summary['retry_rate']:.2%}`",
        f"- Attempt error rate: `{summary['attempt_error_rate']:.2%}`",
        f"- MP3 success rate: `{summary['mp3_success_rate']:.2%}`",
        f"- Mean input-to-MP3 time: `{summary['input_to_mp3_seconds_mean']:.2f}s`",
        f"- Median input-to-MP3 time: `{summary['input_to_mp3_seconds_median']:.2f}s`",
        "",
        "## Plots",
        "",
    ]
    for name, path in plots.items():
        lines.append(f"- `{name}`: `{path}`")
    return "\n".join(lines) + "\n"


def _elapsed_seconds(start: Any, end: Any) -> float | None:
    start_dt = _parse_timestamp(start)
    end_dt = _parse_timestamp(end)
    if start_dt is None or end_dt is None:
        return None
    return round(max(0.0, (end_dt - start_dt).total_seconds()), 6)


def _parse_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _rounded_mean(values: list[float]) -> float:
    return round(mean(values), 6) if values else 0.0
