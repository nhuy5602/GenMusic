"""Academic evaluation plots for the self-authored music model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_jam_plots(report: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        return {"status": "unavailable", "reason": "matplotlib chưa được cài", "plots": []}
    objective = report.get("objective", report)
    labels = ["FAD", "MCD", "FFE", "WER"]
    values = [objective.get(label) for label in labels]
    numeric = [float(value) if value is not None else 0.0 for value in values]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(labels, numeric, color=["#2563eb", "#0f766e", "#b45309", "#be123c"])
    axis.set_title("Chỉ số khách quan của mô hình")
    axis.set_ylabel("Giá trị (giá trị 0 thể hiện chưa đo)")
    figure.tight_layout()
    objective_path = destination / "objective_metrics.png"
    figure.savefig(objective_path, dpi=160)
    plt.close(figure)

    variants = report.get("subjective", {}).get("variants", {})
    figure, axis = plt.subplots(figsize=(8, 4.5))
    names = list(variants)
    musicality = [variants[name].get("MOS_musicality") or 0 for name in names]
    intelligibility = [variants[name].get("MOS_intelligibility") or 0 for name in names]
    positions = list(range(len(names)))
    axis.bar([position - 0.2 for position in positions], musicality, width=0.4, label="MOS musicality")
    axis.bar([position + 0.2 for position in positions], intelligibility, width=0.4, label="MOS intelligibility")
    axis.set_xticks(positions, names)
    axis.set_ylim(0, 5)
    axis.set_title("MOS từ khảo sát người nghe")
    axis.legend()
    figure.tight_layout()
    subjective_path = destination / "mos_intelligibility_musicality.png"
    figure.savefig(subjective_path, dpi=160)
    plt.close(figure)

    plot_report = {"status": "created", "plots": [str(objective_path.resolve()), str(subjective_path.resolve())]}
    (destination / "plots.json").write_text(json.dumps(plot_report, ensure_ascii=False, indent=2), encoding="utf-8")
    return plot_report
