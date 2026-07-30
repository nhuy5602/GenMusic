"""Validation and windowing for exactly aligned waveform stems."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def validate_raw_records(
    dataset: Path,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate waveform shape, finiteness, alignment, and activity."""
    unique_songs = {
        str(record.get("song_id") or record["id"])
        for record in records
    }
    exact = 0
    finite = 0
    valid_channels = 0
    duration_valid = 0
    non_silent_vocal = 0
    non_silent_backing = 0
    for record in records:
        if bool(record.get("exact_word_timestamps")):
            exact += 1
        vocal_path = dataset / str(record["vocal_wav_path"])
        backing_path = dataset / str(record["backing_wav_path"])
        if not vocal_path.is_file() or not backing_path.is_file():
            raise FileNotFoundError(
                f"Missing stems for {record['id']}: "
                f"{vocal_path}, {backing_path}"
            )
        vocal = torch.load(
            vocal_path,
            map_location="cpu",
            weights_only=True,
        )
        backing = torch.load(
            backing_path,
            map_location="cpu",
            weights_only=True,
        )
        if vocal.dim() == 1 and backing.dim() == 1:
            valid_channels += 1
        if bool(torch.isfinite(vocal).all() and torch.isfinite(backing).all()):
            finite += 1
        seconds = min(vocal.numel(), backing.numel()) / 24_000.0
        if 8.0 <= seconds <= 22.5:
            duration_valid += 1
        if float(vocal.float().square().mean().sqrt()) >= 1e-4:
            non_silent_vocal += 1
        if float(backing.float().square().mean().sqrt()) >= 1e-4:
            non_silent_backing += 1
    count = len(records)
    return {
        "records": count,
        "unique_songs": len(unique_songs),
        "exact_timestamp_fraction": exact / max(1, count),
        "finite_fraction": finite / max(1, count),
        "mono_waveform_fraction": valid_channels / max(1, count),
        "duration_valid_fraction": duration_valid / max(1, count),
        "non_silent_vocal_fraction": non_silent_vocal / max(1, count),
        "non_silent_backing_fraction": non_silent_backing / max(1, count),
    }
