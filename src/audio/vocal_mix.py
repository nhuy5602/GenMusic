"""Cross-song vocal/backing mix-clarity utilities.

The generated vocals, prompts, and cross-song backing tracks are copied
unchanged from V58.  V67 only tests whether a target-free, frequency-aware
mix can stop the backing from masking Vietnamese consonants.  PhoWhisper is
used for evaluation only.  The selected configuration must later pass on
fresh held-out renders before it can be treated as final goal evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_generation_quality import (
    lyric_wer,
    transcription_metrics,
    wav_metrics,
)


STATE_NAME = "master_vocal_mix_clarity_v67_state.json"
CANDIDATES = (
    {
        "name": "linear_0p45",
        "backing_ratio": 0.45,
        "presence_attenuation_db": 0.0,
        "dynamic": False,
    },
    {
        "name": "linear_0p30",
        "backing_ratio": 0.30,
        "presence_attenuation_db": 0.0,
        "dynamic": False,
    },
    {
        "name": "static_presence_0p42",
        "backing_ratio": 0.42,
        "presence_attenuation_db": 7.0,
        "dynamic": False,
    },
    {
        "name": "sidechain_presence_0p45",
        "backing_ratio": 0.45,
        "presence_attenuation_db": 10.0,
        "dynamic": True,
    },
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


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
    """Return a smooth unit gain mask for the consonant-presence band."""
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


def _vocal_activity(
    vocal_stft: np.ndarray,
) -> np.ndarray:
    energy = np.sqrt(
        np.mean(np.square(np.abs(vocal_stft), dtype=np.float64), axis=0)
        + 1e-12
    )
    low = float(np.quantile(energy, 0.20))
    high = float(np.quantile(energy, 0.80))
    activity = np.clip((energy - low) / max(high - low, 1e-8), 0.0, 1.0)
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
    """Attenuate only the backing presence band, optionally by vocal activity."""
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


def mix_clarity_candidate(
    vocal_path: Path,
    backing_path: Path,
    destination: Path,
    *,
    backing_ratio: float,
    presence_attenuation_db: float,
    dynamic: bool,
) -> dict[str, Any]:
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

    epsilon = 1e-8
    vocal_rms = float(
        np.sqrt(np.mean(np.square(vocal, dtype=np.float64)) + epsilon)
    )
    processed_rms = float(
        np.sqrt(np.mean(np.square(processed, dtype=np.float64)) + epsilon)
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
        "limiter_scale": float(limiter_scale),
        "clip_ratio": float(np.mean(np.abs(mixed) >= 0.999)),
    }


def candidate_gate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    word_accuracy = [
        float(sample["asr"]["word_accuracy"]) for sample in samples
    ]
    hypotheses = [
        str(sample["asr"]["hypothesis"]).strip().casefold()
        for sample in samples
    ]
    effective_ratios = [
        float(sample["mix"]["effective_backing_to_vocal_rms"])
        for sample in samples
    ]
    mean_wa = float(np.mean(word_accuracy))
    gate = {
        "mean_word_accuracy": mean_wa,
        "samples_word_accuracy_at_least_0p25": sum(
            value >= 0.25 for value in word_accuracy
        ),
        "distinct_nonempty_hypotheses": len(
            {value for value in hypotheses if value}
        ),
        "minimum_effective_backing_to_vocal_rms": min(effective_ratios),
        "minimum_duration_seconds": min(
            float(sample["acoustics"]["duration_seconds"])
            for sample in samples
        ),
        "maximum_clip_ratio": max(
            float(sample["acoustics"]["clip_ratio"])
            for sample in samples
        ),
        "mean_voiced_ratio": float(np.mean([
            float(sample["acoustics"].get("voiced_ratio") or 0.0)
            for sample in samples
        ])),
    }
    gate["pass"] = bool(
        gate["mean_word_accuracy"] >= 0.35
        and gate["samples_word_accuracy_at_least_0p25"] >= 2
        and gate["distinct_nonempty_hypotheses"] >= 2
        and gate["minimum_effective_backing_to_vocal_rms"] >= 0.25
        and gate["minimum_duration_seconds"] >= 15.5
        and gate["maximum_clip_ratio"] <= 0.01
        and gate["mean_voiced_ratio"] >= 0.50
    )
    return gate


def _local_wav(source_root: Path, remote_path: str) -> Path:
    return source_root / "held_out_16s" / Path(remote_path).name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--source-state-name",
        default="master_vocal_phrase_arbitrary_v58_state.json",
    )
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    state_path = output_root / STATE_NAME
    source_state_path = source_root / args.source_state_name
    source_state = json.loads(
        source_state_path.read_text(encoding="utf-8")
    )
    if source_state.get("status") != "pilot_passed":
        raise RuntimeError(
            f"V67 requires completed V58 pilot, got {source_state.get('status')}"
        )
    samples = list(source_state.get("samples") or [])
    if len(samples) != 3:
        raise RuntimeError(f"V67 requires exactly three V58 samples, got {len(samples)}")
    if not all(
        str(sample["prompt_song_id"]) != str(sample["backing_song_id"])
        for sample in samples
    ):
        raise RuntimeError("V67 requires cross-song V58 backing")

    state: dict[str, Any] = {
        "status": "running",
        "training": False,
        "goal_eligible_vocal_source": True,
        "fresh_heldout_confirmation_required": True,
        "selection_set_only": True,
        "source_state": str(source_state_path),
        "source_gate": source_state.get("gate"),
        "pretrained_tts_used": False,
        "pretrained_asr_used": True,
        "asr_evaluation_only": True,
        "vocal_audio_changed": False,
        "prompt_or_timing_changed": False,
        "candidates": {},
    }
    _write_json(state_path, state)

    for specification in CANDIDATES:
        name = str(specification["name"])
        candidate_root = output_root / "mixes" / name
        candidate_samples: list[dict[str, Any]] = []
        for source_sample in samples:
            stem = str(source_sample["id"])
            vocal_path = _local_wav(
                source_root,
                str(source_sample["generated_vocal_wav"]),
            )
            backing_path = _local_wav(
                source_root,
                str(source_sample["backing_wav"]),
            )
            mix_path = candidate_root / f"{stem}_full_mix.wav"
            mix_report = mix_clarity_candidate(
                vocal_path,
                backing_path,
                mix_path,
                backing_ratio=float(specification["backing_ratio"]),
                presence_attenuation_db=float(
                    specification["presence_attenuation_db"]
                ),
                dynamic=bool(specification["dynamic"]),
            )
            prompt = str(source_sample["reference_text"])
            hypothesis = lyric_wer(mix_path, prompt)["predicted_text"]
            asr = transcription_metrics(prompt, hypothesis)
            candidate_samples.append({
                "id": stem,
                "prompt_song_id": source_sample["prompt_song_id"],
                "backing_song_id": source_sample["backing_song_id"],
                "reference_text": prompt,
                "asr": asr,
                "acoustics": wav_metrics(mix_path),
                "mix": mix_report,
                "audio_path": str(mix_path),
            })
            print(
                "V67_SAMPLE",
                name,
                stem,
                f"WA={asr['word_accuracy']:.6f}",
                flush=True,
            )
        state["candidates"][name] = {
            "specification": specification,
            "samples": candidate_samples,
            "gate": candidate_gate(candidate_samples),
        }
        _write_json(state_path, state)

    ranked = sorted(
        state["candidates"].items(),
        key=lambda item: (
            bool(item[1]["gate"]["pass"]),
            float(item[1]["gate"]["mean_word_accuracy"]),
            float(
                item[1]["gate"][
                    "minimum_effective_backing_to_vocal_rms"
                ]
            ),
        ),
        reverse=True,
    )
    state["selected_candidate"] = ranked[0][0]
    state["selected_gate"] = ranked[0][1]["gate"]
    state["status"] = (
        "mix_diagnostic_passed"
        if ranked[0][1]["gate"]["pass"]
        else "mix_diagnostic_failed"
    )
    state["next_if_pass"] = (
        "freeze mixer and validate on fresh held-out cross-song renders"
    )
    state["next_if_fail"] = (
        "mix is not the main bottleneck; train a separately supervised "
        "acoustic/phone model instead of more DSP or waveform-unit retrieval"
    )
    _write_json(state_path, state)
    print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
