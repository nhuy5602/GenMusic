"""Frequency-aware vocal and backing mixing."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf


def _load_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(
        str(path),
        dtype="float32",
        always_2d=True,
    )
    return audio.mean(axis=1).astype(np.float32), int(sample_rate)


def _fit_length(audio: np.ndarray, length: int) -> np.ndarray:
    if audio.size < 1:
        return np.zeros(length, dtype=np.float32)
    if audio.size < length:
        audio = np.tile(audio, math.ceil(length / audio.size))
    return np.asarray(audio[:length], dtype=np.float32)


def presence_band_weight(
    frequencies: np.ndarray,
    *,
    lower_start_hz: float = 500.0,
    lower_full_hz: float = 900.0,
    upper_full_hz: float = 4_500.0,
    upper_end_hz: float = 7_000.0,
) -> np.ndarray:
    """Return a smooth unit-gain mask for the consonant-presence band."""
    frequencies = np.asarray(frequencies, dtype=np.float64)
    rising = np.clip(
        (frequencies - lower_start_hz)
        / max(lower_full_hz - lower_start_hz, 1e-8),
        0.0,
        1.0,
    )
    falling = np.clip(
        (upper_end_hz - frequencies)
        / max(upper_end_hz - upper_full_hz, 1e-8),
        0.0,
        1.0,
    )
    return np.minimum(rising, falling).astype(np.float32)


def _vocal_activity(vocal_stft: np.ndarray) -> np.ndarray:
    energy = np.sqrt(
        np.mean(
            np.square(np.abs(vocal_stft), dtype=np.float64),
            axis=0,
        )
        + 1e-12
    )
    low = float(np.quantile(energy, 0.20))
    high = float(np.quantile(energy, 0.80))
    activity = np.clip(
        (energy - low) / max(high - low, 1e-8),
        0.0,
        1.0,
    )
    if activity.size >= 5:
        activity = np.convolve(
            activity,
            np.ones(5, dtype=np.float64) / 5.0,
            mode="same",
        )
    return activity.astype(np.float32)


def attenuate_backing_presence(
    backing: np.ndarray,
    vocal: np.ndarray,
    sample_rate: int,
    *,
    attenuation_db: float,
    dynamic: bool,
) -> np.ndarray:
    """Duck only the backing frequencies that mask Vietnamese consonants."""
    if attenuation_db <= 0.0:
        return np.asarray(backing, dtype=np.float32).copy()
    n_fft = 2_048
    hop_length = 512
    backing_stft = librosa.stft(
        np.asarray(backing, dtype=np.float32),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
    )
    vocal_stft = librosa.stft(
        np.asarray(vocal, dtype=np.float32),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
    )
    frequencies = librosa.fft_frequencies(
        sr=sample_rate,
        n_fft=n_fft,
    )
    band = presence_band_weight(frequencies)
    activity = (
        _vocal_activity(vocal_stft)
        if dynamic
        else np.ones(backing_stft.shape[1], dtype=np.float32)
    )
    gain_db = (
        -float(attenuation_db)
        * band[:, None]
        * activity[None, :]
    )
    gain = np.power(10.0, gain_db / 20.0)
    processed = librosa.istft(
        backing_stft * gain,
        hop_length=hop_length,
        win_length=n_fft,
        length=backing.size,
    )
    return np.asarray(processed, dtype=np.float32)


def span_duck_envelope(
    length: int,
    sample_rate: int,
    spans_seconds: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    *,
    attenuation_db: float,
    ramp_ms: float = 80.0,
) -> np.ndarray:
    """Build a smooth full-band duck only around designated novel-word spans."""
    envelope = np.ones(max(0, int(length)), dtype=np.float32)
    if envelope.size == 0 or attenuation_db <= 0.0 or not spans_seconds:
        return envelope
    floor = float(np.power(10.0, -float(attenuation_db) / 20.0))
    ramp = max(1, round(float(ramp_ms) * sample_rate / 1_000.0))
    for start_seconds, end_seconds in spans_seconds:
        start = max(0, min(envelope.size, round(float(start_seconds) * sample_rate)))
        end = max(start, min(envelope.size, round(float(end_seconds) * sample_rate)))
        if end <= start:
            continue
        local = np.ones_like(envelope)
        local[start:end] = floor
        left_start = max(0, start - ramp)
        if start > left_start:
            local[left_start:start] = np.linspace(
                1.0,
                floor,
                start - left_start,
                endpoint=False,
                dtype=np.float32,
            )
        right_end = min(envelope.size, end + ramp)
        if right_end > end:
            local[end:right_end] = np.linspace(
                floor,
                1.0,
                right_end - end,
                endpoint=False,
                dtype=np.float32,
            )
        envelope = np.minimum(envelope, local)
    return envelope


def mix_clarity_candidate(
    vocal_path: Path,
    backing_path: Path,
    destination: Path,
    *,
    backing_ratio: float,
    presence_attenuation_db: float,
    dynamic: bool,
    focus_spans_seconds: list[tuple[float, float]]
    | tuple[tuple[float, float], ...] = (),
    focus_duck_db: float = 0.0,
    focus_ramp_ms: float = 80.0,
) -> dict[str, Any]:
    """Mix a vocal-forward full track and persist it as PCM WAV."""
    vocal, sample_rate = _load_mono(vocal_path)
    backing, backing_rate = _load_mono(backing_path)
    if backing_rate != sample_rate:
        backing = librosa.resample(
            backing,
            orig_sr=backing_rate,
            target_sr=sample_rate,
        ).astype(np.float32)
    backing = _fit_length(backing, vocal.size)
    processed = attenuate_backing_presence(
        backing,
        vocal,
        sample_rate,
        attenuation_db=presence_attenuation_db,
        dynamic=dynamic,
    )
    focus_envelope = span_duck_envelope(
        processed.size,
        sample_rate,
        focus_spans_seconds,
        attenuation_db=float(focus_duck_db),
        ramp_ms=float(focus_ramp_ms),
    )
    processed = processed * focus_envelope

    epsilon = 1e-8
    vocal_rms = float(
        np.sqrt(np.mean(np.square(vocal, dtype=np.float64)) + epsilon)
    )
    processed_rms = float(
        np.sqrt(
            np.mean(np.square(processed, dtype=np.float64)) + epsilon
        )
    )
    scale = (
        vocal_rms * float(backing_ratio) / processed_rms
        if processed_rms > epsilon
        else 0.0
    )
    contribution = processed * scale
    mixed = vocal + contribution
    peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    limiter_scale = min(1.0, 0.98 / peak) if peak > 0.0 else 1.0
    mixed = np.asarray(mixed * limiter_scale, dtype=np.float32)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(destination), mixed, sample_rate, subtype="PCM_16")

    contribution_rms = float(
        np.sqrt(
            np.mean(np.square(contribution, dtype=np.float64)) + epsilon
        )
    )
    return {
        "audio_path": str(destination),
        "sample_rate": sample_rate,
        "duration_seconds": float(mixed.size / sample_rate),
        "vocal_rms": vocal_rms,
        "backing_rms_after_presence_filter": processed_rms,
        "backing_scale": float(scale),
        "effective_backing_to_vocal_rms": (
            contribution_rms / max(vocal_rms, epsilon)
        ),
        "configured_backing_ratio": float(backing_ratio),
        "presence_attenuation_db": float(presence_attenuation_db),
        "dynamic_presence_ducking": bool(dynamic),
        "focus_spans_seconds": [
            [float(start), float(end)] for start, end in focus_spans_seconds
        ],
        "focus_duck_db": float(focus_duck_db),
        "focus_ramp_ms": float(focus_ramp_ms),
        "limiter_scale": float(limiter_scale),
        "clip_ratio": float(np.mean(np.abs(mixed) >= 0.999)),
    }
