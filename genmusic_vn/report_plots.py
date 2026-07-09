from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
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
    return round(max(1.0, min(5.0, 1.0 + 4.0 * numeric)), 2)


def _duration_plot(plt: Any, rows: list[dict[str, Any]], output_path: Path, title_prefix: str) -> str:
    figure, axis = plt.subplots(figsize=(8, 5))
    durations = [float(row.get("duration_seconds", 0)) for row in rows]
    processing = [float(row.get("processing_time_seconds", 0.0)) for row in rows]
    success = [bool(row.get("success", True)) for row in rows]
    axis.scatter(durations, processing, c=["#2e8b57" if ok else "#c0392b" for ok in success], alpha=0.8)
    axis.set_title(f"{title_prefix}: input duration vs processing time")
    axis.set_xlabel("Input duration (seconds)")
    axis.set_ylabel("Processing time (seconds)")
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
        axis.text(0.5, 0.5, "No BPM data", ha="center", va="center", transform=axis.transAxes)
        axis.set_axis_off()
        return _save_plot(plt, figure, output_path / "emotion_vs_bpm.png")
    positions = list(range(len(labels)))
    axis.boxplot([grouped[label] for label in labels], positions=positions, widths=0.55)
    for position, label in zip(positions, labels):
        values = grouped[label]
        axis.scatter([position] * len(values), values, alpha=0.45, s=18)
    axis.set_xticks(positions, labels, rotation=30, ha="right")
    axis.set_title(f"{title_prefix}: emotion vs BPM")
    axis.set_xlabel("Emotion")
    axis.set_ylabel("BPM")
    axis.grid(axis="y", alpha=0.25)
    return _save_plot(plt, figure, output_path / "emotion_vs_bpm.png")


def _rating_plot(plt: Any, rows: list[dict[str, Any]], output_path: Path, title_prefix: str, source: str) -> str:
    figure, axis = plt.subplots(figsize=(9, 5))
    values = [float(row.get("rating", 0.0)) for row in rows]
    labels = [str(row.get("id", index + 1))[:16] for index, row in enumerate(rows)]
    axis.bar(range(len(values)), values, color="#4169e1")
    axis.set_ylim(0, 5.2)
    axis.set_title(f"{title_prefix}: user rating ({source})")
    axis.set_xlabel("Case")
    axis.set_ylabel("Rating (1-5)")
    if len(labels) <= 24:
        axis.set_xticks(range(len(labels)), labels, rotation=70, ha="right", fontsize=8)
    else:
        axis.set_xticks([])
    axis.grid(axis="y", alpha=0.25)
    return _save_plot(plt, figure, output_path / "user_rating.png")


def _success_error_plot(plt: Any, rates: dict[str, Any], output_path: Path, title_prefix: str) -> str:
    figure, axis = plt.subplots(figsize=(6, 5))
    labels = ["success", "error"]
    values = [float(rates.get("success_rate", 0.0)), float(rates.get("error_rate", 0.0))]
    axis.bar(labels, values, color=["#2e8b57", "#c0392b"])
    axis.set_ylim(0, 1.05)
    axis.set_title(f"{title_prefix}: success/error rate")
    axis.set_ylabel("Rate")
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
