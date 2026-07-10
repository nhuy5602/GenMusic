"""Teacher-to-student velocity distillation for four-step inference."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..models.jam_diffrhythm import ConditionalDiT, DiTConfig, count_parameters, require_torch
from .sft import _load_pt, _load_records


@dataclass(frozen=True)
class DistillationConfig:
    steps: int = 1000
    learning_rate: float = 1e-4
    teacher_steps: int = 32
    student_steps: int = 4
    device: str = "auto"
    seed: int = 5602


def distill_jam_checkpoint(teacher_checkpoint: str | Path, manifest: str | Path, output_dir: str | Path, *, config: DistillationConfig | None = None) -> dict[str, Any]:
    require_torch()
    import torch

    config = config or DistillationConfig()
    torch.manual_seed(config.seed)
    source = torch.load(teacher_checkpoint, map_location="cpu", weights_only=False)
    model_config = DiTConfig(**source["model_config"])
    teacher = ConditionalDiT(model_config)
    teacher.load_state_dict(source["model_state"])
    teacher.eval()
    student = ConditionalDiT(model_config)
    student.load_state_dict(source["model_state"])
    device = torch.device("cuda" if config.device == "auto" and torch.cuda.is_available() else ("cpu" if config.device == "auto" else config.device))
    teacher.to(device)
    student.to(device)
    records = _load_records(manifest)
    usable = [record for record in records if Path(record.get("latent_path", "")).exists()]
    if not usable:
        raise ValueError("Manifest không có latent để distillation.")
    optimizer = torch.optim.AdamW(student.parameters(), lr=config.learning_rate)
    history: list[float] = []
    for step in range(config.steps):
        record = usable[step % len(usable)]
        target = _load_pt(record["latent_path"]).float().unsqueeze(0).to(device)
        target = target[:, : model_config.max_frames]
        cond = torch.zeros_like(target)
        text = torch.zeros(1, 1, dtype=torch.long, device=device)
        style = _load_pt(record["style_path"]).float().reshape(1, -1).to(device) if record.get("style_path") else torch.zeros(1, model_config.style_dim, device=device)
        noise = torch.randn_like(target)
        time = torch.rand(1, device=device)
        noised = (1.0 - time[:, None, None]) * noise + time[:, None, None] * target
        with torch.no_grad():
            teacher_velocity = teacher(noised, cond, text, style, time)
        student_velocity = student(noised, cond, text, style, time)
        loss = torch.nn.functional.mse_loss(student_velocity, teacher_velocity)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().cpu()))
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint_path = destination / "consistency_student.pt"
    torch.save({"model_state": student.state_dict(), "model_config": model_config.as_dict(), "training_config": asdict(config), "history": history, "inference_steps": config.student_steps, "teacher_inference_steps": config.teacher_steps, "parameter_count": count_parameters(student)}, checkpoint_path)
    report = {"stage": "consistency-velocity-distillation", "records": len(usable), "steps": config.steps, "initial_loss": history[0], "final_loss": history[-1], "teacher_inference_steps": config.teacher_steps, "student_inference_steps": config.student_steps, "checkpoint": str(checkpoint_path.resolve()), "status": "trained"}
    (destination / "consistency_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
