"""Post-training INT8 export with explicit coverage metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def quantize_linear_layers(checkpoint: str | Path, output_path: str | Path) -> dict[str, Any]:
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Cần torch để lượng hóa INT8.") from exc
    source = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_config = source.get("model_config")
    if not model_config:
        raise ValueError("Checkpoint thiếu model_config, không thể khởi tạo model để PTQ.")
    from ..models.jam_diffrhythm import ConditionalDiT, DiTConfig

    model = ConditionalDiT(DiTConfig(**model_config))
    model.load_state_dict(source["model_state"])
    quantized = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": quantized.state_dict(), "model_config": model_config, "quantization": "dynamic-int8-linear", "source": str(Path(checkpoint).resolve())}, destination)
    report = {"input": str(Path(checkpoint).resolve()), "output": str(destination.resolve()), "method": "dynamic-int8-linear", "status": "exported", "coverage": "Linear layers only; attention and embeddings remain non-quantized unless a backend-specific exporter is used."}
    destination.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
