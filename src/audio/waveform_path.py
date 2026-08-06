"""Native-duration waveform extraction primitives."""

from __future__ import annotations

import torch

from src.audio.waveform_units import SAMPLE_RATE, Unit


def resolve_source_unit_bounds(
    waveform: torch.Tensor,
    unit: Unit,
    *,
    zero_crossing_search_ms: float = 12.0,
) -> tuple[int, int]:
    """Resolve V80's quiet-trim bounds without changing the source waveform.

    The returned indices are deliberately pre-fade and pre-normalisation.  An
    audit can therefore establish that an edge was intrinsically quiet instead
    of mistaking the renderer's cosmetic fade for acoustic silence.
    """

    source = waveform.detach().float().flatten()
    start = max(0, round(unit.start * SAMPLE_RATE))
    end = min(source.numel(), round(unit.end * SAMPLE_RATE))
    if start >= source.numel() or end - start < 64:
        raise RuntimeError(
            "Waveform unit timestamp is outside the available audio: "
            f"start={unit.start:.6f}, end={unit.end:.6f}, "
            f"samples={source.numel()}"
        )
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
    if end - start < 64:
        raise RuntimeError(f"Waveform unit is too short: {unit}")
    return start, end


def extract_source_unit(
    waveform: torch.Tensor,
    unit: Unit,
    *,
    zero_crossing_search_ms: float = 12.0,
    fade_ms: float = 8.0,
) -> torch.Tensor:
    """Extract a donor unit without pitch shifting or time stretching."""
    source = waveform.detach().float().flatten()
    start, end = resolve_source_unit_bounds(
        source,
        unit,
        zero_crossing_search_ms=zero_crossing_search_ms,
    )
    rendered = source[start:end].clone()
    rendered -= rendered.mean()
    fade = min(
        round(fade_ms * SAMPLE_RATE / 1000.0),
        max(1, rendered.numel() // 6),
    )
    if fade > 1:
        rendered[:fade] *= torch.linspace(0.0, 1.0, fade)
        rendered[-fade:] *= torch.linspace(1.0, 0.0, fade)
    return rendered
