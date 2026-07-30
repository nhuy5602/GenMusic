"""Validate a raw, exactly aligned Vietnamese stem dataset.

V63 is a data/representation gate, not a product model.  It verifies that the
same clean aligned corpus used by the successful mel experiments can retain
the separated 24 kHz vocal and backing waveforms without a Vocos round-trip.
PhoWhisper is used only to score three fixed held-out target stems.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

MINIMUM_RECORDS = 64


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def collect_exact_words(record: dict[str, Any]) -> list[dict[str, Any]]:
    words = []
    for segment in record.get("segments") or []:
        for item in segment.get("words") or []:
            word = str(item.get("word") or "").strip()
            start = float(item.get("start") or 0.0)
            end = float(item.get("end") or start)
            if word and end > start:
                words.append({"word": word, "start": start, "end": end})
    return sorted(words, key=lambda item: (item["start"], item["end"]))


def select_word_window(
    record: dict[str, Any],
    *,
    duration_seconds: float = 16.0,
) -> dict[str, Any]:
    """Select the deterministic 16-second window containing most exact words."""
    words = collect_exact_words(record)
    if not words:
        raise ValueError("V63 record has no exact words")
    total_duration = float(record["frames"]) / 24_000.0
    duration = min(float(duration_seconds), total_duration)
    maximum_start = max(0.0, total_duration - duration)
    candidates = {0.0, maximum_start}
    candidates.update(
        min(maximum_start, max(0.0, float(word["start"]) - 0.25))
        for word in words
    )
    best = None
    for start in sorted(candidates):
        end = start + duration
        selected = [
            word
            for word in words
            if (float(word["start"]) + float(word["end"])) * 0.5 >= start
            and (float(word["start"]) + float(word["end"])) * 0.5 < end
        ]
        score = (len(selected), -start)
        if best is None or score > best[0]:
            best = (score, start, end, selected)
    assert best is not None
    _, start, end, selected = best
    return {
        "start_seconds": start,
        "end_seconds": end,
        "duration_seconds": end - start,
        "word_count": len(selected),
        "prompt": " ".join(str(word["word"]) for word in selected),
    }


def mix_raw_stems(
    vocal: torch.Tensor,
    backing: torch.Tensor,
    *,
    backing_to_vocal_rms: float = 0.45,
) -> torch.Tensor:
    """Mix mono stems at a bounded RMS ratio without clipping."""
    vocal = vocal.detach().float().flatten()
    backing = backing.detach().float().flatten()
    samples = min(int(vocal.numel()), int(backing.numel()))
    if samples < 1:
        raise ValueError("V63 stems are empty")
    vocal = vocal[:samples]
    backing = backing[:samples]
    vocal_rms = vocal.square().mean().sqrt().clamp_min(1e-6)
    backing_rms = backing.square().mean().sqrt().clamp_min(1e-6)
    backing = backing * (
        vocal_rms * max(0.0, float(backing_to_vocal_rms)) / backing_rms
    )
    mixed = vocal + backing
    peak = mixed.abs().max()
    if float(peak) > 0.995:
        mixed = mixed * (0.995 / peak)
    return mixed


def validate_raw_records(
    dataset: Path,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    unique_songs = {
        str(record.get("song_id") or record["id"]) for record in records
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
                f"Missing V63 stems for {record['id']}: "
                f"{vocal_path}, {backing_path}"
            )
        vocal = torch.load(vocal_path, map_location="cpu", weights_only=True)
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


def materialization_gate(
    validation: dict[str, Any],
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    mean_vocal_wa = sum(
        float(sample["target_vocal_asr"]["word_accuracy"])
        for sample in samples
    ) / max(1, len(samples))
    mean_mix_wa = sum(
        float(sample["target_full_mix_asr"]["word_accuracy"])
        for sample in samples
    ) / max(1, len(samples))
    pass_value = bool(
        int(validation["records"]) >= MINIMUM_RECORDS
        and int(validation["unique_songs"]) == int(validation["records"])
        and float(validation["exact_timestamp_fraction"]) >= 0.99
        and float(validation["finite_fraction"]) == 1.0
        and float(validation["mono_waveform_fraction"]) == 1.0
        and float(validation["duration_valid_fraction"]) >= 0.99
        and float(validation["non_silent_vocal_fraction"]) >= 0.99
        and float(validation["non_silent_backing_fraction"]) >= 0.99
        and len(samples) == 3
        and mean_vocal_wa >= 0.50
        and mean_mix_wa >= 0.40
        and all(
            float(sample["target_vocal_acoustics"]["voiced_ratio"]) >= 0.50
            and float(sample["target_full_mix_acoustics"]["duration_seconds"])
            >= 15.5
            and float(sample["target_full_mix_acoustics"]["clip_ratio"]) <= 0.01
            for sample in samples
        )
    )
    return {
        "pass": pass_value,
        **validation,
        "mean_target_vocal_word_accuracy": mean_vocal_wa,
        "mean_target_full_mix_word_accuracy": mean_mix_wa,
        "samples_at_least_vocal_wa_0p25": sum(
            float(sample["target_vocal_asr"]["word_accuracy"]) >= 0.25
            for sample in samples
        ),
        "thresholds": {
            "minimum_records": MINIMUM_RECORDS,
            "unique_song_fraction": 1.0,
            "exact_timestamp_fraction": 0.99,
            "finite_fraction": 1.0,
            "mono_waveform_fraction": 1.0,
            "duration_valid_fraction": 0.99,
            "non_silent_fraction": 0.99,
            "target_vocal_word_accuracy": 0.50,
            "target_full_mix_word_accuracy": 0.40,
            "target_vocal_voiced_ratio": 0.50,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=16.0)
    parser.add_argument("--backing-to-vocal-rms", type=float, default=0.45)
    args = parser.parse_args()

    import soundfile as sf

    from scripts.evaluate_generation_quality import (
        lyric_wer,
        transcription_metrics,
        wav_metrics,
    )

    dataset = args.dataset.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "master_aligned_raw_v63_state.json"
    records = [
        json.loads(line)
        for line in (dataset / "records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    config = json.loads((dataset / "config.json").read_text(encoding="utf-8"))
    state: dict[str, Any] = {
        "status": "validating",
        "training": False,
        "goal_eligible_prediction": False,
        "oracle_data_gate_only": True,
        "design": (
            "exactly aligned raw 24kHz Demucs vocal/backing stems with no "
            "mel or vocoder round-trip"
        ),
        "records": len(records),
        "raw_audio_mode": bool(config.get("raw_audio_mode")),
        "pretrained_tts_used": False,
        "pretrained_asr_used": True,
        "asr_evaluation_only": True,
        "pretrained_source_separator_used": True,
    }
    _write_json(state_path, state)
    if not state["raw_audio_mode"]:
        raise RuntimeError("V63 dataset is not raw-audio mode")
    validation = validate_raw_records(dataset, records)
    state["validation"] = validation
    state["status"] = "rendering_oracle_gate"
    _write_json(state_path, state)

    candidates = []
    for record in records:
        try:
            window = select_word_window(
                record,
                duration_seconds=float(args.duration),
            )
        except ValueError:
            continue
        if window["duration_seconds"] >= 15.5 and window["word_count"] >= 12:
            candidates.append((record, window))
    candidates.sort(
        key=lambda item: (
            str(item[0].get("song_id") or item[0]["id"]),
            str(item[0]["id"]),
        )
    )
    if len(candidates) < 3:
        raise RuntimeError("V63 lacks three 16-second exact-word oracle windows")
    selected = [candidates[0], candidates[len(candidates) // 2], candidates[-1]]

    sample_root = output_root / "oracle_16s"
    sample_root.mkdir(parents=True, exist_ok=True)
    samples = []
    for record, window in selected:
        start = round(float(window["start_seconds"]) * 24_000)
        length = round(float(args.duration) * 24_000)
        vocal = torch.load(
            dataset / str(record["vocal_wav_path"]),
            map_location="cpu",
            weights_only=True,
        ).float()[start : start + length]
        backing = torch.load(
            dataset / str(record["backing_wav_path"]),
            map_location="cpu",
            weights_only=True,
        ).float()[start : start + length]
        if vocal.numel() < length:
            vocal = torch.nn.functional.pad(vocal, (0, length - vocal.numel()))
        if backing.numel() < length:
            backing = torch.nn.functional.pad(
                backing,
                (0, length - backing.numel()),
            )
        mixed = mix_raw_stems(
            vocal,
            backing,
            backing_to_vocal_rms=float(args.backing_to_vocal_rms),
        )
        stem = str(record["id"])
        vocal_path = sample_root / f"{stem}_target_vocal.wav"
        mix_path = sample_root / f"{stem}_target_full_mix.wav"
        sf.write(vocal_path, vocal.numpy(), 24_000, subtype="PCM_16")
        sf.write(mix_path, mixed.numpy(), 24_000, subtype="PCM_16")
        prompt = str(window["prompt"])

        def asr(path: Path) -> dict[str, Any]:
            hypothesis = lyric_wer(path, prompt)["predicted_text"]
            return transcription_metrics(prompt, hypothesis)

        samples.append({
            "id": stem,
            "song_id": str(record.get("song_id") or stem),
            "reference_text": prompt,
            "window": window,
            "target_vocal_asr": asr(vocal_path),
            "target_full_mix_asr": asr(mix_path),
            "target_vocal_acoustics": wav_metrics(vocal_path),
            "target_full_mix_acoustics": wav_metrics(mix_path),
            "target_vocal_wav": str(vocal_path),
            "target_full_mix_wav": str(mix_path),
        })
        state["samples"] = samples
        _write_json(state_path, state)

    gate = materialization_gate(validation, samples)
    state.update({
        "status": "materialization_passed" if gate["pass"] else "gate_failed",
        "samples": samples,
        "gate": gate,
    })
    _write_json(state_path, state)
    print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
