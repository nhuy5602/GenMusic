"""Native-duration waveform extraction primitives."""

from __future__ import annotations

import torch

from src.audio.waveform_units import SAMPLE_RATE, Unit


def extract_source_unit(
    waveform: torch.Tensor,
    unit: Unit,
    *,
    zero_crossing_search_ms: float = 12.0,
    fade_ms: float = 8.0,
) -> torch.Tensor:
    """Extract a donor unit without pitch shifting or time stretching."""
    source = waveform.detach().float().flatten()
    start = max(0, round(unit.start * SAMPLE_RATE))
    end = min(source.numel(), round(unit.end * SAMPLE_RATE))
    search = max(
        1,
        round(zero_crossing_search_ms * SAMPLE_RATE / 1000.0),
    )

    def quiet_index(center: int, lower: int, upper: int) -> int:
        left = max(lower, center - search)
        right = min(upper, center + search + 1)
        if right <= left:
            return center
        return left + int(source[left:right].abs().argmin())

    start = quiet_index(start, 0, max(start + 1, end - 64))
    end = quiet_index(
        end,
        min(end - 1, start + 64),
        source.numel(),
    )
    rendered = source[start:end].clone()
    if rendered.numel() < 64:
        raise RuntimeError(f"Waveform unit is too short: {unit}")
    rendered -= rendered.mean()
    fade = min(
        round(fade_ms * SAMPLE_RATE / 1000.0),
        max(1, rendered.numel() // 6),
    )
    if fade > 1:
        rendered[:fade] *= torch.linspace(0.0, 1.0, fade)
        rendered[-fade:] *= torch.linspace(1.0, 0.0, fade)
    return rendered
