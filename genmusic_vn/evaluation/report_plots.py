from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median
from typing import Any


def generate_evaluation_plots(
    report: dict[str, Any],
    output_root: str | Path,
    *,
    quality_report: dict[str, Any] | None = None,
    title_prefix: str = "Evaluation",
) -> dict[str, Any]:
    """Write Kaggle-style PNG plots and the exact plotted data used to create them."""
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)
    items = list(report.get("items") or [])
    quality_items = list((quality_report or {}).get("items") or [])
    rating_values, rating_source = _rating_values(items, quality_items)
    plot_data = {
        "duration_vs_processing_time": [
            {
                "id": item.get("id", ""),
                "duration_seconds": item.get("duration_seconds", 0),
                "processing_time_seconds": item.get("processing_time_seconds", 0.0),
                "success": bool(item.get("success", True)),
            }
            for item in items
        ],
        "emotion_vs_bpm": [
            {
                "id": item.get("id", ""),
                "emotion": item.get("predicted", {}).get("emotion") or item.get("expected_emotion", "unknown"),
                "bpm": item.get("predicted", {}).get("bpm", item.get("bpm", None)),
                "success": bool(item.get("success", True)),
            }
            for item in items
            if item.get("predicted", {}).get("bpm", item.get("bpm", None)) is not None
        ],
        "user_rating": rating_values,
        "rating_distribution": _rating_distribution(rating_values),
        "rating_summary": _rating_summary(rating_values),
        "success_error_rate": {
            "success_rate": report.get("summary", {}).get("success_rate", 0.0),
            "error_rate": report.get("summary", {}).get("error_rate", 0.0),
        },
        "rating_source": rating_source,
    }
    data_path = output_path / "plot_data.json"
    data_path.write_text(json.dumps(plot_data, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {
            "status": "plotting_unavailable",
            "error": f"{type(exc).__name__}: {exc}",
            "plot_data_path": str(data_path),
            "rating_source": rating_source,
        }

    files: dict[str, str] = {}
    files["duration_input_vs_processing_time"] = _duration_plot(
        plt, plot_data["duration_vs_processing_time"], output_path, title_prefix
    )
    files["emotion_vs_bpm"] = _emotion_bpm_plot(
        plt, plot_data["emotion_vs_bpm"], output_path, title_prefix
    )
    files["user_rating"] = _rating_plot(
        plt, rating_values, output_path, title_prefix, rating_source
    )
    files["success_error_rate"] = _success_error_plot(
        plt, plot_data["success_error_rate"], output_path, title_prefix
    )
    return {
        "status": "complete",
        "files": files,
        "plot_data_path": str(data_path),
        "rating_source": rating_source,
    }


def generate_self_improvement_plots(report: dict[str, Any], output_root: str | Path) -> dict[str, Any]:
    """Aggregate iteration reports into one four-plot self-improvement dashboard."""
    items: list[dict[str, Any]] = []
    quality_items: list[dict[str, Any]] = []
    for history_item in report.get("history", []):
        iteration = history_item.get("iteration")
        eval_path = Path(str(history_item.get("evaluation_report_path") or ""))
        quality_path = Path(str(history_item.get("quality_report_path") or ""))
        if eval_path.exists():
            try:
                eval_report = json.loads(eval_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                eval_report = {}
            for item in eval_report.get("items", []):
                copied = dict(item)
                copied["iteration"] = iteration
                items.append(copied)
        if quality_path.exists():
            try:
                quality_report = json.loads(quality_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                quality_report = {}
            for item in quality_report.get("items", []):
                copied = dict(item)
                copied["iteration"] = iteration
                quality_items.append(copied)

    aggregate = {
        "items": items,
        "summary": {
            "success_rate": _success_rate(items),
            "error_rate": round(1.0 - _success_rate(items), 4),
        },
    }
    result = generate_evaluation_plots(
        aggregate,
        output_root,
        quality_report={"items": quality_items},
        title_prefix="Self-improvement",
    )
    result["iteration_count"] = len(report.get("history", []))
    result["evaluation_item_count"] = len(items)
    result["quality_item_count"] = len(quality_items)
    return result


def _rating_values(items: list[dict[str, Any]], quality_items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    if quality_items:
        values = []
        has_supplied_rating = False
        for index, item in enumerate(quality_items, start=1):
            metrics = item.get("metrics", {})
            rating = item.get("user_rating")
            if rating is None:
                rating = _score_to_rating(metrics.get("overall_quality_score", 0.0))
            has_supplied_rating = has_supplied_rating or item.get("user_rating_source") == "user_supplied"
            values.append({"id": item.get("id", f"case_{index:03d}"), "rating": round(float(rating), 2)})
        return values, "user_supplied" if has_supplied_rating else "objective_quality_proxy"
    values = []
    for index, item in enumerate(items, start=1):
        score = item.get("metrics", {}).get("overall_score", 0.0)
        values.append({"id": item.get("id", f"case_{index:03d}"), "rating": _score_to_rating(score)})
    return values, "evaluation.overall_score_proxy"


def _score_to_rating(score: Any) -> float:
    try:
        numeric = float(score)
    except (TypeError, ValueError):
        numeric = 0.0
    # Scores below 0.35 should read as a poor result, not as an inflated four-star proxy.
    calibrated = (numeric - 0.35) / 0.65
    return round(max(1.0, min(5.0, 1.0 + 4.0 * calibrated)), 2)


def _rating_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    distribution = {str(rating): 0 for rating in range(1, 6)}
    for row in rows:
        try:
            value = float(row.get("rating", 0.0))
        except (TypeError, ValueError):
            continue
        bucket = min(5, max(1, int(value + 0.5)))
        distribution[str(bucket)] += 1
    return distribution


def _rating_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    values: list[float] = []
    for row in rows:
        try:
            values.append(float(row.get("rating", 0.0)))
        except (TypeError, ValueError):
            continue
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": round(mean(values), 2),
        "median": round(median(values), 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
    }


def _duration_plot(plt: Any, rows: list[dict[str, Any]], output_path: Path, title_prefix: str) -> str:
    figure, axis = plt.subplots(figsize=(8, 5))
    durations = [float(row.get("duration_seconds", 0)) for row in rows]
    processing = [float(row.get("processing_time_seconds", 0.0)) for row in rows]
    success = [bool(row.get("success", True)) for row in rows]
    axis.scatter(durations, processing, c=["#2e8b57" if ok else "#c0392b" for ok in success], alpha=0.8)
    axis.set_title(f"{title_prefix}: thời lượng input và thời gian xử lý")
    axis.set_xlabel("Thời lượng input (giây)")
    axis.set_ylabel("Thời gian xử lý (giây)")
    axis.grid(alpha=0.25)
    return _save_plot(plt, figure, output_path / "duration_input_vs_processing_time.png")


def _emotion_bpm_plot(plt: Any, rows: list[dict[str, Any]], output_path: Path, title_prefix: str) -> str:
    figure, axis = plt.subplots(figsize=(9, 5))
    grouped: dict[str, list[float]] = {}
    for row in rows:
        try:
            grouped.setdefault(str(row.get("emotion") or "unknown"), []).append(float(row["bpm"]))
        except (KeyError, TypeError, ValueError):
            continue
    labels = sorted(grouped)
    if not labels:
        axis.text(0.5, 0.5, "Chưa có dữ liệu BPM", ha="center", va="center", transform=axis.transAxes)
        axis.set_axis_off()
        return _save_plot(plt, figure, output_path / "emotion_vs_bpm.png")
    positions = list(range(len(labels)))
    axis.boxplot([grouped[label] for label in labels], positions=positions, widths=0.55)
    for position, label in zip(positions, labels):
        values = grouped[label]
        axis.scatter([position] * len(values), values, alpha=0.45, s=18)
    axis.set_xticks(positions, labels, rotation=30, ha="right")
    axis.set_title(f"{title_prefix}: cảm xúc và BPM")
    axis.set_xlabel("Cảm xúc")
    axis.set_ylabel("BPM")
    axis.grid(axis="y", alpha=0.25)
    return _save_plot(plt, figure, output_path / "emotion_vs_bpm.png")


def _rating_plot(plt: Any, rows: list[dict[str, Any]], output_path: Path, title_prefix: str, source: str) -> str:
    figure, axis = plt.subplots(figsize=(9, 5))
    values = [float(row.get("rating", 0.0)) for row in rows]
    if values:
        axis.hist(values, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], color="#4169e1", rwidth=0.82)
    else:
        axis.text(0.5, 0.5, "Chưa có dữ liệu rating", ha="center", va="center", transform=axis.transAxes)
    axis.set_xlim(0.5, 5.5)
    axis.set_xticks(range(1, 6))
    axis.set_title(f"{title_prefix}: phân bố rating người dùng ({source})")
    axis.set_xlabel("Mức rating (1-5)")
    axis.set_ylabel("Số case")
    axis.grid(axis="y", alpha=0.25)
    return _save_plot(plt, figure, output_path / "user_rating.png")


def _success_error_plot(plt: Any, rates: dict[str, Any], output_path: Path, title_prefix: str) -> str:
    figure, axis = plt.subplots(figsize=(6, 5))
    labels = ["thành công", "lỗi"]
    values = [float(rates.get("success_rate", 0.0)), float(rates.get("error_rate", 0.0))]
    axis.bar(labels, values, color=["#2e8b57", "#c0392b"])
    axis.set_ylim(0, 1.05)
    axis.set_title(f"{title_prefix}: tỷ lệ thành công/lỗi")
    axis.set_ylabel("Tỷ lệ")
    axis.grid(axis="y", alpha=0.25)
    return _save_plot(plt, figure, output_path / "success_error_rate.png")


def _success_rate(items: list[dict[str, Any]]) -> float:
    if not items:
        return 0.0
    return round(mean(1.0 if item.get("success", True) else 0.0 for item in items), 4)


def _save_plot(plt: Any, figure: Any, path: Path) -> str:
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return str(path)


def generate_project_telemetry_plots(report: dict[str, Any], output_root: str | Path) -> dict[str, Any]:
    """Write plots for real app/Kaggle request telemetry, including input-to-MP3 latency."""
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)
    items = list(report.get("items") or [])
    summary = dict(report.get("summary") or {})
    plot_data = {
        "input_to_mp3": [
            {
                "run_id": item.get("run_id", ""),
                "seconds": item.get("input_to_mp3_seconds"),
                "status": item.get("terminal_status", item.get("status", "unknown")),
                "retry_attempt": bool(item.get("retry_attempt", False)),
            }
            for item in items
            if item.get("input_to_mp3_seconds") is not None
        ],
        "retry_error_rates": {
            "retry_rate": summary.get("retry_rate", 0.0),
            "attempt_error_rate": summary.get("attempt_error_rate", 0.0),
            "retry_success_rate": summary.get("retry_success_rate", 0.0),
            "mp3_success_rate": summary.get("mp3_success_rate", 0.0),
        },
        "stage_timing": [
            {
                "run_id": item.get("run_id", ""),
                "dataset_upload_seconds": item.get("dataset_upload_seconds"),
                "dataset_ready_wait_seconds": item.get("dataset_ready_wait_seconds"),
                "kernel_to_mp3_seconds": item.get("kernel_to_mp3_seconds"),
                "retry_attempt": bool(item.get("retry_attempt", False)),
            }
            for item in items
        ],
        "outcomes": _project_outcomes(items),
    }
    data_path = output_path / "plot_data.json"
    data_path.write_text(json.dumps(plot_data, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {
            "status": "plotting_unavailable",
            "error": f"{type(exc).__name__}: {exc}",
            "plot_data_path": str(data_path),
        }

    files = {
        "input_to_mp3_processing_time": _input_to_mp3_plot(
            plt, plot_data["input_to_mp3"], output_path
        ),
        "kaggle_retry_error_rate": _retry_error_plot(
            plt, plot_data["retry_error_rates"], output_path
        ),
        "kaggle_stage_processing_time": _stage_timing_plot(
            plt, plot_data["stage_timing"], output_path
        ),
        "project_outcome_rate": _outcome_plot(plt, plot_data["outcomes"], output_path),
    }
    return {"status": "complete", "files": files, "plot_data_path": str(data_path)}


def _project_outcomes(items: list[dict[str, Any]]) -> dict[str, int]:
    outcomes = {"complete_with_mp3": 0, "failed_or_incomplete": 0, "retry_attempt": 0}
    for item in items:
        if item.get("retry_attempt"):
            outcomes["retry_attempt"] += 1
        if item.get("has_mp3"):
            outcomes["complete_with_mp3"] += 1
        else:
            outcomes["failed_or_incomplete"] += 1
    return outcomes


def _input_to_mp3_plot(plt: Any, rows: list[dict[str, Any]], output_path: Path) -> str:
    figure, axis = plt.subplots(figsize=(10, 5))
    values = [float(row["seconds"]) for row in rows]
    colors = ["#2e8b57" if row.get("status") == "complete" else "#c0392b" for row in rows]
    if values:
        axis.bar(range(len(values)), values, color=colors)
    else:
        axis.text(0.5, 0.5, "Chưa có telemetry MP3 hoàn tất", ha="center", va="center", transform=axis.transAxes)
    axis.set_title("Project: thời gian thực từ input tới MP3")
    axis.set_xlabel("Lần tạo nhạc")
    axis.set_ylabel("Giây từ lúc nhận input tới lúc có MP3")
    axis.grid(axis="y", alpha=0.25)
    return _save_plot(plt, figure, output_path / "input_to_mp3_processing_time.png")


def _retry_error_plot(plt: Any, rates: dict[str, Any], output_path: Path) -> str:
    figure, axis = plt.subplots(figsize=(8, 5))
    labels = ["Tỷ lệ retry", "Lỗi lần chạy", "Retry thành công", "MP3 thành công"]
    keys = ["retry_rate", "attempt_error_rate", "retry_success_rate", "mp3_success_rate"]
    values = [float(rates.get(key, 0.0)) for key in keys]
    axis.bar(labels, values, color=["#f39c12", "#c0392b", "#2e8b57", "#4169e1"])
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Tỷ lệ")
    axis.set_title("Project: tỷ lệ retry, lỗi và MP3 thành công trên Kaggle")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.25)
    return _save_plot(plt, figure, output_path / "kaggle_retry_error_rate.png")


def _stage_timing_plot(plt: Any, rows: list[dict[str, Any]], output_path: Path) -> str:
    figure, axis = plt.subplots(figsize=(9, 5))
    labels = ["Upload dataset", "Chờ dataset sẵn sàng", "Kernel tới MP3"]
    keys = ["dataset_upload_seconds", "dataset_ready_wait_seconds", "kernel_to_mp3_seconds"]
    groups: list[list[float]] = []
    for key in keys:
        groups.append([float(row[key]) for row in rows if row.get(key) is not None])
    non_empty = [(label, values) for label, values in zip(labels, groups) if values]
    if non_empty:
        box_values = [values for _, values in non_empty]
        box_labels = [label for label, _ in non_empty]
        try:
            axis.boxplot(box_values, tick_labels=box_labels)
        except TypeError:  # Matplotlib < 3.9
            axis.boxplot(box_values, labels=box_labels)
    else:
        axis.text(0.5, 0.5, "Chưa có telemetry thời gian stage", ha="center", va="center", transform=axis.transAxes)
    axis.set_title("Project: thời gian xử lý từng stage Kaggle")
    axis.set_ylabel("Giây")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.25)
    return _save_plot(plt, figure, output_path / "kaggle_stage_processing_time.png")


def _outcome_plot(plt: Any, outcomes: dict[str, int], output_path: Path) -> str:
    figure, axis = plt.subplots(figsize=(7, 5))
    labels = ["MP3 ready", "Failed/incomplete", "Retry attempts"]
    values = [
        int(outcomes.get("complete_with_mp3", 0)),
        int(outcomes.get("failed_or_incomplete", 0)),
        int(outcomes.get("retry_attempt", 0)),
    ]
    axis.bar(labels, values, color=["#2e8b57", "#c0392b", "#f39c12"])
    axis.set_title("Project: request outcome counts")
    axis.set_ylabel("Count")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.25)
    return _save_plot(plt, figure, output_path / "project_outcome_rate.png")
