"""Consistency distillation helpers for reducing CFM sampling steps."""

from __future__ import annotations

from typing import Any, Callable


def consistency_distillation_loss(student_prediction: Any, teacher_prediction: Any) -> Any:
    try:
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Cần torch để distill.") from exc
    return functional.mse_loss(student_prediction, teacher_prediction.detach())


def teacher_euler_target(teacher: Callable[..., Any], state: Any, condition: Any, *, steps: int = 32, **kwargs: Any) -> Any:
    """Integrate a teacher velocity field to a target with no student gradient."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Cần torch để distill.") from exc
    with torch.no_grad():
        value = state
        delta = 1.0 / max(1, steps)
        for index in range(max(1, steps)):
            time = torch.full((state.shape[0],), index / max(1, steps), device=state.device, dtype=state.dtype)
            value = value + delta * teacher(value, condition, time, **kwargs)
        return value
