"""Create report-ready plots from GenMusic training and quality reports.

The script intentionally consumes only JSON artifacts, so it can run at the
end of a Kaggle training kernel without loading the model or reserving GPU
memory.  Every image is saved at presentation-friendly resolution under one
output directory, together with a CSV transcription table and Markdown index.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


COLORS = {
    "generated": "#2563eb",
    "real": "#16a34a",
    "train": "#7c3aed",
    "validation": "#ea580c",
    "threshold": "#dc2626",
    "grid": "#cbd5e1",
}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _style_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", color=COLORS["grid"], alpha=0.55, linewidth=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _save(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _sample_label(sample: dict[str, Any], index: int) -> str:
    value = str(sample.get("id") or f"sample_{index + 1}")
    return value if len(value) <= 20 else value[:17] + "..."


def _asr_values(samples: list[dict[str, Any]], key: str, metric: str) -> list[float]:
    values: list[float] = []
    for sample in samples:
        value = _number((sample.get(key) or {}).get(metric))
        values.append(value if value is not None else 0.0)
    return values


def plot_training_curves(training_report: dict[str, Any], output: Path) -> Path:
    curve = [item for item in training_report.get("loss_curve", []) if isinstance(item, dict)]
    epochs = [int(item.get("epoch", index + 1)) for index, item in enumerate(curve)]
    train = [_number(item.get("loss")) for item in curve]
    validation = [_number(item.get("validation_loss")) for item in curve]

    figure, axis = plt.subplots(figsize=(10.5, 5.5))
    if epochs and any(value is not None for value in train):
        axis.plot(
            epochs,
            [np.nan if value is None else value for value in train],
            marker="o",
            linewidth=2.2,
            color=COLORS["train"],
            label="Train loss",
        )
    if epochs and any(value is not None for value in validation):
        axis.plot(
            epochs,
            [np.nan if value is None else value for value in validation],
            marker="s",
            linewidth=2.2,
            color=COLORS["validation"],
            label="Validation loss",
        )
    best_epoch = training_report.get("best_epoch")
    best_loss = _number(training_report.get("best_validation_loss"))
    if best_epoch and best_loss is not None:
        axis.scatter([int(best_epoch)], [best_loss], s=130, marker="*", color="#facc15", edgecolor="#713f12", zorder=5)
        axis.annotate(
            f"Best: epoch {best_epoch}\nval={best_loss:.4f}",
            (int(best_epoch), best_loss),
            xytext=(8, 10),
            textcoords="offset points",
            fontsize=9,
        )
    if not curve:
        axis.text(0.5, 0.5, "No epoch loss curve recorded", ha="center", va="center", transform=axis.transAxes)
    axis.set_title("GenMusic training convergence", fontsize=15, weight="bold")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    _style_axis(axis)
    if axis.lines:
        axis.legend(frameon=False)
    path = output / "01_training_curves.png"
    _save(figure, path)
    return path


def plot_asr_metrics(quality_report: dict[str, Any], output: Path) -> Path:
    samples = [item for item in quality_report.get("samples", []) if isinstance(item, dict)]
    labels = [_sample_label(sample, index) for index, sample in enumerate(samples)]
    generated_accuracy = _asr_values(samples, "generated_asr", "word_accuracy")
    real_accuracy = _asr_values(samples, "real_vocal_asr", "word_accuracy")
    generated_wer = _asr_values(samples, "generated_asr", "wer")
    generated_cer = _asr_values(samples, "generated_asr", "cer")

    figure, axes = plt.subplots(2, 1, figsize=(11.5, 8.2), sharex=True)
    positions = np.arange(len(samples))
    width = 0.36
    axes[0].bar(positions - width / 2, generated_accuracy, width, label="Generated", color=COLORS["generated"])
    axes[0].bar(positions + width / 2, real_accuracy, width, label="Real vocal", color=COLORS["real"])
    axes[0].axhline(0.30, color=COLORS["threshold"], linestyle="--", linewidth=1.5, label="Per-sample pass (0.30)")
    axes[0].set_ylim(0, max(1.0, *(generated_accuracy + real_accuracy or [1.0])) * 1.08)
    axes[0].set_ylabel("Word accuracy")
    axes[0].set_title("Vietnamese ASR intelligibility by sample", fontsize=15, weight="bold")
    axes[0].legend(frameon=False, ncol=3)
    _style_axis(axes[0])

    axes[1].bar(positions - width / 2, generated_wer, width, label="WER", color="#9333ea")
    axes[1].bar(positions + width / 2, generated_cer, width, label="CER", color="#f97316")
    axes[1].axhline(0.70, color=COLORS["threshold"], linestyle="--", linewidth=1.5, label="Mean CER target (0.70)")
    axes[1].set_ylabel("Error rate (lower is better)")
    axes[1].set_xlabel("Evaluation sample")
    axes[1].set_xticks(positions, labels, rotation=25, ha="right")
    axes[1].legend(frameon=False, ncol=3)
    _style_axis(axes[1])
    if not samples:
        for axis in axes:
            axis.text(0.5, 0.5, "No ASR samples recorded", ha="center", va="center", transform=axis.transAxes)
    figure.tight_layout()
    path = output / "02_asr_intelligibility.png"
    _save(figure, path)
    return path


def plot_acoustic_metrics(quality_report: dict[str, Any], output: Path) -> Path:
    samples = [item for item in quality_report.get("samples", []) if isinstance(item, dict)]
    labels = [_sample_label(sample, index) for index, sample in enumerate(samples)]
    metric_specs = [
        ("voiced_ratio", "Voiced ratio", 0.35),
        ("pitch_std_semitones", "Pitch variation (semitones)", 1.0),
        ("spectral_flatness", "Spectral flatness", None),
        ("silence_ratio", "Silence ratio", None),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    positions = np.arange(len(samples))
    width = 0.36
    for axis, (metric, title, threshold) in zip(axes.flat, metric_specs):
        generated = [
            _number((sample.get("generated") or {}).get(metric)) or 0.0
            for sample in samples
        ]
        real = [
            _number((sample.get("real_vocal_same_vocoder") or {}).get(metric)) or 0.0
            for sample in samples
        ]
        axis.bar(positions - width / 2, generated, width, label="Generated", color=COLORS["generated"])
        axis.bar(positions + width / 2, real, width, label="Real vocal", color=COLORS["real"])
        if threshold is not None:
            axis.axhline(threshold, color=COLORS["threshold"], linestyle="--", linewidth=1.4, label=f"Target ({threshold:g})")
        axis.set_title(title, weight="bold")
        axis.set_xticks(positions, labels, rotation=30, ha="right", fontsize=8)
        _style_axis(axis)
        if not samples:
            axis.text(0.5, 0.5, "No samples recorded", ha="center", va="center", transform=axis.transAxes)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="upper center", ncol=max(1, len(legend_labels)), frameon=False)
    figure.suptitle("Generated audio vs. real vocal acoustic metrics", fontsize=16, weight="bold", y=1.02)
    figure.tight_layout()
    path = output / "03_acoustic_metrics.png"
    _save(figure, path)
    return path


def plot_guidance_search(quality_report: dict[str, Any], output: Path) -> Path:
    """Show the ASR effect of the CFG sweep performed inside the Kaggle run."""
    samples = [item for item in quality_report.get("samples", []) if isinstance(item, dict)]
    values_by_scale: dict[float, list[float]] = {}
    for sample in samples:
        for candidate in sample.get("guidance_candidates") or []:
            scale = _number(candidate.get("guidance_scale"))
            accuracy = _number((candidate.get("asr") or {}).get("word_accuracy"))
            if scale is not None and accuracy is not None:
                values_by_scale.setdefault(scale, []).append(accuracy)

    figure, axis = plt.subplots(figsize=(10.5, 5.5))
    scales = sorted(values_by_scale)
    means = [float(np.mean(values_by_scale[scale])) for scale in scales]
    stds = [float(np.std(values_by_scale[scale])) for scale in scales]
    if scales:
        axis.errorbar(
            scales,
            means,
            yerr=stds,
            marker="o",
            markersize=7,
            linewidth=2.2,
            capsize=5,
            color=COLORS["generated"],
            label="Mean ± std across samples",
        )
        axis.axhline(0.40, color=COLORS["threshold"], linestyle="--", label="Run pass target (0.40)")
        axis.legend(frameon=False)
    else:
        axis.text(0.5, 0.5, "No guidance sweep recorded", ha="center", va="center", transform=axis.transAxes)
    axis.set_title("Classifier-free lyric guidance search", fontsize=15, weight="bold")
    axis.set_xlabel("Guidance scale")
    axis.set_ylabel("Whisper word accuracy")
    axis.set_ylim(bottom=0)
    _style_axis(axis)
    path = output / "04_guidance_search.png"
    _save(figure, path)
    return path


def plot_text_conditioning(training_report: dict[str, Any], output: Path) -> Path:
    """Visualize whether changing/removing lyrics materially changes predictions."""
    curve = [item for item in training_report.get("loss_curve", []) if isinstance(item, dict)]
    points = [
        (int(item.get("epoch", index + 1)), _number(item.get("text_conditioning_sensitivity")))
        for index, item in enumerate(curve)
    ]
    points = [(epoch, value) for epoch, value in points if value is not None]
    figure, axis = plt.subplots(figsize=(10.5, 5.5))
    if points:
        axis.plot(
            [item[0] for item in points],
            [item[1] for item in points],
            marker="o",
            linewidth=2.2,
            color="#0f766e",
            label="Relative conditioned/unconditioned difference",
        )
        axis.axhline(0.10, color=COLORS["threshold"], linestyle="--", label="Collapse warning boundary")
        axis.axhline(0.20, color="#2563eb", linestyle=":", label="Training response target")
        axis.legend(frameon=False)
    else:
        axis.text(0.5, 0.5, "No conditioning-sensitivity series recorded", ha="center", va="center", transform=axis.transAxes)
    axis.set_title("Lyric-conditioning sensitivity", fontsize=15, weight="bold")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Relative prediction change")
    axis.set_ylim(bottom=0)
    _style_axis(axis)
    path = output / "05_text_conditioning.png"
    _save(figure, path)
    return path


def plot_sample_quality_heatmap(quality_report: dict[str, Any], output: Path) -> Path:
    """Summarize per-sample intelligibility and acoustic health in one figure."""
    samples = [item for item in quality_report.get("samples", []) if isinstance(item, dict)]
    labels = [_sample_label(sample, index) for index, sample in enumerate(samples)]
    columns = [
        "Word accuracy\n/ 0.30",
        "1 - CER\n/ 0.30",
        "Voiced\nratio / 0.35",
        "Pitch std\n/ 1 semitone",
        "Real-vocal WA\n/ 0.30",
    ]
    rows: list[list[float]] = []
    annotations: list[list[str]] = []
    for sample in samples:
        generated_asr = sample.get("generated_asr") or {}
        real_asr = sample.get("real_vocal_asr") or {}
        generated = sample.get("generated") or {}
        word_accuracy = _number(generated_asr.get("word_accuracy")) or 0.0
        cer = _number(generated_asr.get("cer"))
        voiced_ratio = _number(generated.get("voiced_ratio")) or 0.0
        pitch_std = _number(generated.get("pitch_std_semitones")) or 0.0
        real_word_accuracy = _number(real_asr.get("word_accuracy")) or 0.0
        raw = [
            word_accuracy,
            max(0.0, 1.0 - (cer if cer is not None else 1.0)),
            voiced_ratio,
            pitch_std,
            real_word_accuracy,
        ]
        normalized = [raw[0] / 0.30, raw[1] / 0.30, raw[2] / 0.35, raw[3] / 1.0, raw[4] / 0.30]
        # Cap the colors at the pass target while retaining exact values as labels.
        rows.append([min(1.0, value) for value in normalized])
        annotations.append([f"{value:.2f}" for value in raw])

    height = max(4.8, 0.62 * len(samples) + 2.2)
    figure, axis = plt.subplots(figsize=(11.5, height))
    if rows:
        image = axis.imshow(np.asarray(rows), aspect="auto", vmin=0.0, vmax=1.0, cmap="RdYlGn")
        for row_index, row in enumerate(annotations):
            for column_index, label in enumerate(row):
                color = "white" if rows[row_index][column_index] < 0.28 else "black"
                axis.text(column_index, row_index, label, ha="center", va="center", color=color, fontsize=9)
        figure.colorbar(image, ax=axis, pad=0.02, label="Normalized quality (1.0 = target reached)")
        axis.set_xticks(np.arange(len(columns)), columns)
        axis.set_yticks(np.arange(len(labels)), labels)
    else:
        axis.text(0.5, 0.5, "No evaluation samples recorded", ha="center", va="center", transform=axis.transAxes)
        axis.set_xticks([])
        axis.set_yticks([])
    axis.set_title("Per-sample Vietnamese generation quality", fontsize=15, weight="bold", pad=14)
    axis.tick_params(axis="x", labelsize=9)
    figure.tight_layout()
    path = output / "06_sample_quality_heatmap.png"
    _save(figure, path)
    return path


def _mono_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    return np.mean(audio, axis=1), int(sample_rate)


def plot_audio_evidence(quality_report: dict[str, Any], audio_dir: Path, output: Path) -> Path:
    """Compare real and generated waveforms/spectrograms for the best ASR sample."""
    samples = [item for item in quality_report.get("samples", []) if isinstance(item, dict)]
    samples.sort(
        key=lambda item: _number((item.get("generated_asr") or {}).get("word_accuracy")) or 0.0,
        reverse=True,
    )
    selected: dict[str, Any] | None = None
    generated_path: Path | None = None
    real_path: Path | None = None
    for sample in samples:
        record_id = str(sample.get("id") or "")
        candidate_generated = audio_dir / f"{record_id}_generated.wav"
        candidate_real = audio_dir / f"{record_id}_real.wav"
        if candidate_generated.is_file() and candidate_real.is_file():
            selected = sample
            generated_path = candidate_generated
            real_path = candidate_real
            break

    figure, axes = plt.subplots(2, 2, figsize=(14, 8.5))
    if selected is not None and generated_path is not None and real_path is not None:
        real_audio, real_rate = _mono_audio(real_path)
        generated_audio, generated_rate = _mono_audio(generated_path)
        pairs = [
            (real_audio, real_rate, "Real vocal (same vocoder)", COLORS["real"]),
            (generated_audio, generated_rate, "Generated vocal", COLORS["generated"]),
        ]
        for column, (audio, sample_rate, label, color) in enumerate(pairs):
            times = np.arange(len(audio), dtype=np.float32) / max(sample_rate, 1)
            axes[0, column].plot(times, audio, color=color, linewidth=0.65)
            axes[0, column].set_title(f"{label} waveform", weight="bold")
            axes[0, column].set_xlabel("Time (s)")
            axes[0, column].set_ylabel("Amplitude")
            axes[0, column].set_ylim(-1.0, 1.0)
            _style_axis(axes[0, column])
            axes[1, column].specgram(
                audio,
                NFFT=1024,
                Fs=sample_rate,
                noverlap=768,
                cmap="magma",
                scale="dB",
            )
            axes[1, column].set_title(f"{label} spectrogram", weight="bold")
            axes[1, column].set_xlabel("Time (s)")
            axes[1, column].set_ylabel("Frequency (Hz)")
            axes[1, column].set_ylim(0, min(12_000, sample_rate / 2))
        accuracy = _number((selected.get("generated_asr") or {}).get("word_accuracy")) or 0.0
        figure.suptitle(
            f"Best generated sample evidence: {_sample_label(selected, 0)} (word accuracy={accuracy:.3f})",
            fontsize=16,
            weight="bold",
        )
    else:
        for axis in axes.flat:
            axis.axis("off")
            axis.text(0.5, 0.5, "Paired evaluation WAV files not found", ha="center", va="center")
        figure.suptitle("Real vs. generated audio evidence", fontsize=16, weight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    path = output / "07_audio_evidence.png"
    _save(figure, path)
    return path


def plot_summary_dashboard(
    training_report: dict[str, Any], quality_report: dict[str, Any], output: Path
) -> Path:
    summary = quality_report.get("summary") or {}
    samples = [item for item in quality_report.get("samples", []) if isinstance(item, dict)]
    generated_accuracy = _asr_values(samples, "generated_asr", "word_accuracy")
    real_accuracy = _asr_values(samples, "real_vocal_asr", "word_accuracy")
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))

    curve = [item for item in training_report.get("loss_curve", []) if isinstance(item, dict)]
    epochs = [int(item.get("epoch", index + 1)) for index, item in enumerate(curve)]
    axes[0, 0].plot(epochs, [_number(item.get("loss")) or np.nan for item in curve], marker="o", color=COLORS["train"], label="Train")
    axes[0, 0].plot(epochs, [_number(item.get("validation_loss")) or np.nan for item in curve], marker="s", color=COLORS["validation"], label="Validation")
    axes[0, 0].set_title("Convergence", weight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    _style_axis(axes[0, 0])
    if curve:
        axes[0, 0].legend(frameon=False)

    positions = np.arange(len(samples))
    width = 0.36
    axes[0, 1].bar(positions - width / 2, generated_accuracy, width, color=COLORS["generated"], label="Generated")
    axes[0, 1].bar(positions + width / 2, real_accuracy, width, color=COLORS["real"], label="Real vocal")
    axes[0, 1].axhline(0.30, color=COLORS["threshold"], linestyle="--", linewidth=1.3)
    axes[0, 1].set_title("Word accuracy", weight="bold")
    axes[0, 1].set_xticks(positions, [str(index + 1) for index in range(len(samples))])
    axes[0, 1].set_ylim(0, max(1.0, *(generated_accuracy + real_accuracy or [1.0])) * 1.08)
    _style_axis(axes[0, 1])
    axes[0, 1].legend(frameon=False)

    metric_names = ["Word accuracy", "Voiced ratio", "Pitch std"]
    values = [
        _number(summary.get("mean_word_accuracy_generated")) or 0.0,
        _number(summary.get("mean_voiced_ratio_generated")) or 0.0,
        _number(summary.get("mean_pitch_std_semitones_generated")) or 0.0,
    ]
    targets = [0.40, 0.35, 1.0]
    normalized = [value / target for value, target in zip(values, targets)]
    bars = axes[1, 0].bar(metric_names, normalized, color=[COLORS["generated"], "#0891b2", "#ca8a04"])
    axes[1, 0].axhline(1.0, color=COLORS["threshold"], linestyle="--", linewidth=1.5, label="Pass target")
    axes[1, 0].set_ylabel("Ratio to target (1.0 = pass)")
    axes[1, 0].set_title("Key quality gates", weight="bold")
    axes[1, 0].tick_params(axis="x", rotation=15)
    axes[1, 0].bar_label(bars, labels=[f"{value:.2f}x" for value in normalized], padding=3)
    axes[1, 0].legend(frameon=False)
    _style_axis(axes[1, 0])

    passed = bool(summary.get("intelligibility_pass"))
    status = "PASS" if passed else "NEEDS IMPROVEMENT"
    status_color = "#15803d" if passed else "#b91c1c"
    best_epoch = training_report.get("best_epoch", "n/a")
    best_val = _number(training_report.get("best_validation_loss"))
    lines = [
        f"Intelligibility: {status}",
        f"Best epoch: {best_epoch}",
        f"Best validation loss: {best_val:.4f}" if best_val is not None else "Best validation loss: n/a",
        f"Mean word accuracy: {(_number(summary.get('mean_word_accuracy_generated')) or 0.0):.3f}",
        f"Mean CER: {(_number(summary.get('mean_cer_generated')) or 0.0):.3f}",
        f"ASR passing samples: {int(summary.get('asr_passing_samples') or 0)}/{len(samples)}",
        f"Mean voiced ratio: {(_number(summary.get('mean_voiced_ratio_generated')) or 0.0):.3f}",
        f"Pitch std: {(_number(summary.get('mean_pitch_std_semitones_generated')) or 0.0):.3f} semitones",
        f"Text sensitivity: {(_number(training_report.get('final_text_conditioning_sensitivity')) or 0.0):.3f}",
    ]
    axes[1, 1].axis("off")
    axes[1, 1].text(0.02, 0.93, lines[0], color=status_color, fontsize=18, weight="bold", va="top")
    axes[1, 1].text(0.02, 0.79, "\n".join(lines[1:]), fontsize=12, linespacing=1.65, va="top")
    axes[1, 1].set_title("Run summary", weight="bold", loc="left")

    figure.suptitle("GenMusic Vietnamese vocal generation report", fontsize=18, weight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    path = output / "00_report_dashboard.png"
    _save(figure, path)
    return path


def write_transcription_table(quality_report: dict[str, Any], output: Path) -> Path:
    path = output / "transcription_table.csv"
    fields = [
        "id",
        "selected_guidance_scale",
        "reference",
        "generated_transcript",
        "real_vocal_transcript",
        "generated_word_accuracy",
        "generated_wer",
        "generated_cer",
        "real_word_accuracy",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample in quality_report.get("samples", []):
            if not isinstance(sample, dict):
                continue
            generated = sample.get("generated_asr") or {}
            real = sample.get("real_vocal_asr") or {}
            writer.writerow({
                "id": sample.get("id", ""),
                "selected_guidance_scale": sample.get("selected_guidance_scale", ""),
                "reference": sample.get("reference_text") or generated.get("reference", ""),
                "generated_transcript": generated.get("hypothesis", ""),
                "real_vocal_transcript": real.get("hypothesis", ""),
                "generated_word_accuracy": generated.get("word_accuracy", ""),
                "generated_wer": generated.get("wer", ""),
                "generated_cer": generated.get("cer", ""),
                "real_word_accuracy": real.get("word_accuracy", ""),
            })
    return path


def write_markdown_index(
    training_report: dict[str, Any], quality_report: dict[str, Any], artifacts: list[Path], output: Path
) -> Path:
    summary = quality_report.get("summary") or {}
    status = "PASS" if summary.get("intelligibility_pass") else "NEEDS IMPROVEMENT"
    lines = [
        "# GenMusic Kaggle evaluation report",
        "",
        f"- Intelligibility gate: **{status}**",
        f"- Best epoch: `{training_report.get('best_epoch', 'n/a')}`",
        f"- Best validation loss: `{training_report.get('best_validation_loss', 'n/a')}`",
        f"- Mean generated word accuracy: `{summary.get('mean_word_accuracy_generated', 'n/a')}`",
        f"- Mean generated CER: `{summary.get('mean_cer_generated', 'n/a')}`",
        f"- Mean generated voiced ratio: `{summary.get('mean_voiced_ratio_generated', 'n/a')}`",
        f"- Mean generated pitch std: `{summary.get('mean_pitch_std_semitones_generated', 'n/a')}`",
        "",
        "## Report artifacts",
        "",
    ]
    lines.extend(f"- [{path.name}]({path.name})" for path in artifacts)
    path = output / "REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def create_report_artifacts(training_report_path: Path, quality_report_path: Path, output: Path) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    training_report = _load_json(training_report_path)
    quality_report = _load_json(quality_report_path)
    artifacts = [
        plot_summary_dashboard(training_report, quality_report, output),
        plot_training_curves(training_report, output),
        plot_asr_metrics(quality_report, output),
        plot_acoustic_metrics(quality_report, output),
        plot_guidance_search(quality_report, output),
        plot_text_conditioning(training_report, output),
        plot_sample_quality_heatmap(quality_report, output),
        plot_audio_evidence(quality_report, quality_report_path.parent, output),
        write_transcription_table(quality_report, output),
    ]
    artifacts.append(write_markdown_index(training_report, quality_report, artifacts, output))
    manifest_path = output / "report_artifacts.json"
    manifest_path.write_text(
        json.dumps({"artifacts": [path.name for path in artifacts]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    artifacts.append(manifest_path)
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("training_report")
    parser.add_argument("quality_report")
    parser.add_argument("output_dir")
    args = parser.parse_args()
    artifacts = create_report_artifacts(
        Path(args.training_report), Path(args.quality_report), Path(args.output_dir)
    )
    print("Created report artifacts:")
    for artifact in artifacts:
        print(f"- {artifact}")


if __name__ == "__main__":
    main()
