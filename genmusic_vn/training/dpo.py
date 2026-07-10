"""DPO fine-tuning over preferred/dispreferred latent pairs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .preference import dpo_logistic_loss
from ..models.jam_diffrhythm import ConditionalDiT, ConditionalFlowMatching, DiTConfig, count_parameters, require_torch


@dataclass(frozen=True)
class DPOConfig:
    steps: int = 200
    learning_rate: float = 5e-5
    beta: float = 0.1
    device: str = "auto"
    seed: int = 5602


def _load(path: str | Path) -> Any:
    import torch

    value = torch.load(path, map_location="cpu", weights_only=False)
    return value["tensor"] if isinstance(value, dict) and "tensor" in value else value


def train_dpo_from_pairs(checkpoint: str | Path, pairs_path: str | Path, output_dir: str | Path, *, config: DPOConfig | None = None) -> dict[str, Any]:
    require_torch()
    import torch

    config = config or DPOConfig()
    source = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_config = DiTConfig(**source["model_config"])
    model = ConditionalDiT(model_config)
    model.load_state_dict(source["model_state"])
    device = torch.device("cuda" if config.device == "auto" and torch.cuda.is_available() else ("cpu" if config.device == "auto" else config.device))
    model.to(device)
    flow = ConditionalFlowMatching(model)
    pairs = [json.loads(line) for line in Path(pairs_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    usable = [pair for pair in pairs if Path(str(pair.get("preferred", ""))).exists() and Path(str(pair.get("dispreferred", ""))).exists()]
    if not usable:
        raise ValueError("Không có preference pair nào trỏ tới latent tồn tại.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    history: list[float] = []
    model.train()
    for step in range(config.steps):
        pair = usable[step % len(usable)]
        preferred = _load(pair["preferred"]).float().unsqueeze(0).to(device)
        dispreferred = _load(pair["dispreferred"]).float().unsqueeze(0).to(device)
        frames = min(preferred.shape[1], dispreferred.shape[1])
        preferred = preferred[:, :frames]
        dispreferred = dispreferred[:, :frames]
        cond = torch.zeros_like(preferred)
        text_ids = torch.tensor([pair.get("text_ids") or [0]], dtype=torch.long, device=device)
        style = _load(pair["style_path"]).float().reshape(1, -1).to(device) if pair.get("style_path") else torch.zeros(1, model_config.style_dim, device=device)
        preferred_loss = flow.loss(preferred, cond, text_ids, style)
        dispreferred_loss = flow.loss(dispreferred, cond, text_ids, style)
        loss = dpo_logistic_loss(preferred_loss, dispreferred_loss, beta=config.beta)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().cpu()))
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint_path = destination / "dpo_checkpoint.pt"
    torch.save({"model_state": model.state_dict(), "model_config": model_config.as_dict(), "training_config": asdict(config), "history": history, "parameter_count": count_parameters(model)}, checkpoint_path)
    report = {"stage": "dpo-tone-alignment", "pair_count": len(usable), "steps": config.steps, "initial_loss": history[0], "final_loss": history[-1], "checkpoint": str(checkpoint_path.resolve()), "training_config": asdict(config), "status": "trained"}
    (destination / "dpo_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
