"""Configurable conditional flow-matching DiT for the JAM/DiffRhythm recipe.

This module owns the training architecture used by this project. It does not
silently download or bundle a third-party checkpoint.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

try:
    import torch
    from torch import Tensor, nn
except ImportError:  # pragma: no cover - exercised only on lightweight installs
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]
    nn = None  # type: ignore[assignment]


@dataclass(frozen=True)
class DiTConfig:
    name: str = "genmusic-jam-demo"
    latent_dim: int = 64
    style_dim: int = 512
    vocab_size: int = 4096
    hidden_size: int = 256
    depth: int = 4
    heads: int = 8
    ff_mult: int = 4
    dropout: float = 0.1
    max_frames: int = 6144
    declared_parameter_count: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def model_config(preset: str = "demo") -> DiTConfig:
    presets = {
        "demo": DiTConfig(),
        "jam": DiTConfig(name="genmusic-jam-compatible", hidden_size=1024, depth=16, heads=16),
        "diffrhythm": DiTConfig(name="genmusic-diffrhythm-compatible", hidden_size=2048, depth=16, heads=32),
    }
    try:
        return presets[preset]
    except KeyError as exc:
        raise ValueError(f"Preset không tồn tại: {preset}. Chọn demo, jam hoặc diffrhythm.") from exc


def require_torch() -> None:
    if torch is None:
        raise RuntimeError("Cần cài torch để khởi tạo DiT/CFM; có thể chạy preprocessing text không cần torch.")


def _time_embedding(time: Tensor, dimension: int) -> Tensor:
    half = dimension // 2
    frequencies = torch.exp(torch.arange(half, device=time.device, dtype=time.dtype) * (-math.log(10_000.0) / max(1, half - 1)))
    values = time[:, None] * frequencies[None, :]
    embedding = torch.cat([values.sin(), values.cos()], dim=-1)
    if embedding.shape[-1] < dimension:
        embedding = torch.nn.functional.pad(embedding, (0, dimension - embedding.shape[-1]))
    return embedding


if torch is not None:

    class ConditionalDiT(nn.Module):
        """Text + style conditioned latent velocity predictor."""

        def __init__(self, config: DiTConfig):
            super().__init__()
            self.config = config
            self.text_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
            self.style_projection = nn.Sequential(nn.Linear(config.style_dim, config.hidden_size), nn.SiLU(), nn.Linear(config.hidden_size, config.hidden_size))
            self.time_projection = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size), nn.SiLU(), nn.Linear(config.hidden_size, config.hidden_size))
            self.input_projection = nn.Linear(config.latent_dim * 2 + config.hidden_size * 3, config.hidden_size)
            layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.heads,
                dim_feedforward=config.hidden_size * config.ff_mult,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.blocks = nn.TransformerEncoder(layer, num_layers=config.depth)
            self.output_norm = nn.LayerNorm(config.hidden_size)
            self.output_projection = nn.Linear(config.hidden_size, config.latent_dim)

        @staticmethod
        def _maybe_drop(value: Tensor, drop: bool | Tensor) -> Tensor:
            if isinstance(drop, bool):
                return torch.zeros_like(value) if drop else value
            mask = drop.to(device=value.device, dtype=torch.bool).reshape(-1, 1, 1)
            return torch.where(mask, torch.zeros_like(value), value)

        def forward(
            self,
            x: Tensor,
            cond: Tensor,
            text_ids: Tensor,
            style: Tensor,
            time: Tensor,
            *,
            drop_audio: bool | Tensor = False,
            drop_text: bool | Tensor = False,
            drop_style: bool | Tensor = False,
        ) -> Tensor:
            if x.ndim != 3 or cond.shape != x.shape:
                raise ValueError("x và cond phải có cùng shape [batch, frames, latent_dim].")
            text_context = self.text_embedding(text_ids).mean(dim=1)
            text_context = self._maybe_drop(text_context[:, None, :], drop_text)
            style_context = self._maybe_drop(self.style_projection(style)[:, None, :], drop_style)
            time_context = self.time_projection(_time_embedding(time, self.config.hidden_size))[:, None, :]
            audio_condition = self._maybe_drop(cond, drop_audio)
            time_context = time_context.expand(-1, x.shape[1], -1)
            text_context = text_context.expand(-1, x.shape[1], -1)
            style_context = style_context.expand(-1, x.shape[1], -1)
            hidden = self.input_projection(torch.cat([x, audio_condition, text_context, style_context, time_context], dim=-1))
            hidden = self.blocks(hidden)
            return self.output_projection(self.output_norm(hidden))


    class ConditionalFlowMatching(nn.Module):
        """CFM objective and Euler sampler used by SFT and distillation."""

        def __init__(self, model: ConditionalDiT, *, sigma: float = 1.0):
            super().__init__()
            self.model = model
            self.sigma = sigma

        def loss(self, target: Tensor, cond: Tensor, text_ids: Tensor, style: Tensor) -> Tensor:
            noise = torch.randn_like(target) * self.sigma
            time = torch.rand(target.shape[0], device=target.device, dtype=target.dtype)
            path = (1.0 - time[:, None, None]) * noise + time[:, None, None] * target
            flow = target - noise
            prediction = self.model(path, cond, text_ids, style, time, drop_audio=False, drop_text=False, drop_style=False)
            return torch.nn.functional.mse_loss(prediction, flow)

        @torch.no_grad()
        def sample(self, cond: Tensor, text_ids: Tensor, style: Tensor, *, frames: int, steps: int = 32, seed: int | None = None) -> Tensor:
            if seed is not None:
                torch.manual_seed(seed)
            state = torch.randn(cond.shape[0], frames, self.model.config.latent_dim, device=cond.device, dtype=cond.dtype)
            step = 1.0 / max(1, steps)
            for index in range(max(1, steps)):
                time = torch.full((cond.shape[0],), index / max(1, steps), device=cond.device, dtype=cond.dtype)
                prediction = self.model(state, cond[:, :frames], text_ids, style, time)
                state = state + step * prediction
            return state

else:

    class ConditionalDiT:  # type: ignore[no-redef]
        def __init__(self, config: DiTConfig):
            require_torch()

    class ConditionalFlowMatching:  # type: ignore[no-redef]
        def __init__(self, model: ConditionalDiT, *, sigma: float = 1.0):
            require_torch()


def count_parameters(model: Any) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
