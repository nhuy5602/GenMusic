from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def audio_quality_metrics(audio: Any, sampling_rate: int) -> dict[str, float]:
    """Metric kỹ thuật cho audio sinh bởi model tự code."""
    import numpy as np

    raw_values = np.asarray(audio)
    if np.issubdtype(raw_values.dtype, np.integer):
        values = raw_values.astype("float32") / float(np.iinfo(raw_values.dtype).max)
    else:
        values = raw_values.astype("float32")
    values = values.reshape(-1)
    if values.size == 0:
        return {"duration_seconds": 0.0, "rms_db": -120.0, "peak": 0.0, "clipping_ratio": 0.0}
    peak = float(np.max(np.abs(values)))
    rms = float(np.sqrt(np.mean(np.square(values)) + 1e-12))
    clipped = float(np.mean(np.abs(values) >= 0.999))
    zero_crossings = float(np.mean(np.signbit(values[1:]) != np.signbit(values[:-1]))) if values.size > 1 else 0.0
    return {
        "duration_seconds": round(float(values.size / max(1, sampling_rate)), 4),
        "rms_db": round(float(20.0 * np.log10(max(rms, 1e-6))), 4),
        "peak": round(peak, 4),
        "clipping_ratio": round(clipped, 6),
        "zero_crossing_rate": round(zero_crossings, 6),
    }


def build_custom_music_metric_report(
    samples: list[dict[str, Any]],
    *,
    model_name: str,
    dataset_ref: str = "",
    training: bool = False,
) -> dict[str, Any]:
    metrics = [dict(item) for item in samples]
    numeric_keys = ("duration_seconds", "rms_db", "peak", "clipping_ratio", "zero_crossing_rate")
    summary: dict[str, float] = {}
    for key in numeric_keys:
        values = [float(item[key]) for item in metrics if item.get(key) is not None]
        if values:
            summary[f"mean_{key}"] = round(sum(values) / len(values), 6)
    summary["sample_count"] = float(len(metrics))
    return {
        "metric_scope": "technical_audio_proxies",
        "model_name": model_name,
        "dataset_ref": dataset_ref,
        "training": bool(training),
        "samples": metrics,
        "summary": summary,
    }


def write_custom_music_metric_plots(report: dict[str, Any], output_root: str | Path) -> dict[str, Any]:
    """Ghi plot PNG và plot_data.json để tái lập báo cáo Kaggle."""
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)
    data_path = output_path / "plot_data.json"
    data_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"status": "plotting_unavailable", "error": f"{type(exc).__name__}: {exc}", "plot_data_path": str(data_path)}

    samples = list(report.get("samples") or [])
    labels = [str(item.get("id") or f"sample_{index + 1}") for index, item in enumerate(samples)]
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.scatter(
        [float(item.get("duration_seconds", 0.0)) for item in samples],
        [float(item.get("rms_db", -120.0)) for item in samples],
        color="#2563eb",
        alpha=0.8,
    )
    axis.set_title("Model tự code: thời lượng và năng lượng mẫu sinh")
    axis.set_xlabel("Thời lượng (giây)")
    axis.set_ylabel("RMS (dB)")
    axis.grid(alpha=0.25)
    duration_energy_path = output_path / "duration_vs_energy.png"
    figure.tight_layout()
    figure.savefig(duration_energy_path, dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(labels, [float(item.get("clipping_ratio", 0.0)) * 100.0 for item in samples], color="#dc2626")
    axis.set_title("Model tự code: tỷ lệ clipping kỹ thuật")
    axis.set_xlabel("Mẫu")
    axis.set_ylabel("Clipping (%)")
    axis.tick_params(axis="x", rotation=35)
    axis.grid(axis="y", alpha=0.25)
    clipping_path = output_path / "clipping_rate.png"
    figure.tight_layout()
    figure.savefig(clipping_path, dpi=150)
    plt.close(figure)

    files = {"duration_vs_energy": str(duration_energy_path), "clipping_rate": str(clipping_path)}
    loss_history = list(report.get("loss_history") or [])
    if loss_history:
        figure, axis = plt.subplots(figsize=(10, 5))
        axis.plot(
            [int(item.get("step", index + 1)) for index, item in enumerate(loss_history)],
            [float(item.get("loss", 0.0)) for item in loss_history],
            marker="o",
            color="#7c3aed",
        )
        axis.set_title("Model tự code: loss theo bước train")
        axis.set_xlabel("Bước")
        axis.set_ylabel("Loss")
        axis.grid(alpha=0.25)
        loss_path = output_path / "loss_curve.png"
        figure.tight_layout()
        figure.savefig(loss_path, dpi=150)
        plt.close(figure)
        files["loss_curve"] = str(loss_path)
    accuracy = report.get("holdout_feature_accuracy") or {}
    if accuracy:
        figure, axis = plt.subplots(figsize=(9, 5))
        labels = list(accuracy)
        axis.bar(labels, [float(accuracy[label]) for label in labels], color="#0891b2")
        axis.set_title("Model tự code: độ chính xác trên holdout")
        axis.set_xlabel("Đặc trưng audio")
        axis.set_ylabel("Accuracy")
        axis.set_ylim(0.0, 1.05)
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
        accuracy_path = output_path / "holdout_feature_accuracy.png"
        figure.tight_layout()
        figure.savefig(accuracy_path, dpi=150)
        plt.close(figure)
        files["holdout_feature_accuracy"] = str(accuracy_path)
    return {"status": "complete", "files": files, "plot_data_path": str(data_path)}
