"""Target-free raw-waveform singing-unit pilot on exactly aligned stems.

V64 deliberately leaves the failed mel-retrieval family behind.  It retrieves
only from training-song *waveform* units, keeps held-out vocals unavailable to
generation, derives pacing from donor statistics, and uses the held-out vocal
only for evaluation.  Matching held-out backing is allowed for this bounded
pilot; a cross-song backing audit is still required before the project goal can
be declared complete.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import torch

from scripts.run_colab_master_aligned_raw_v63 import (
    mix_raw_stems,
    select_word_window,
)


SAMPLE_RATE = 24_000


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def normalize_word(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    return "".join(
        character
        for character in normalized
        if character.isalnum() or unicodedata.category(character).startswith("L")
    )


def accentless(text: str) -> str:
    text = normalize_word(text).replace("đ", "d")
    return "".join(
        character
        for character in unicodedata.normalize("NFD", text)
        if unicodedata.category(character) != "Mn"
    )


@dataclass(frozen=True)
class Unit:
    song_id: str
    record_id: str
    waveform_path: str
    words: tuple[str, ...]
    normalized_words: tuple[str, ...]
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(1e-3, self.end - self.start)


def words_in_window(
    record: dict[str, Any],
    window: dict[str, Any],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    start = float(window["start_seconds"])
    end = float(window["end_seconds"])
    for segment_index, segment in enumerate(record.get("segments") or []):
        for word in segment.get("words") or []:
            text = str(word.get("word") or "").strip()
            word_start = float(word.get("start") or 0.0)
            word_end = float(word.get("end") or word_start)
            midpoint = (word_start + word_end) * 0.5
            if text and word_end > word_start and start <= midpoint < end:
                selected.append({
                    "word": text,
                    "normalized": normalize_word(text),
                    "start": word_start,
                    "end": word_end,
                    "segment_index": segment_index,
                })
    return selected


def build_inventory(
    records: list[dict[str, Any]],
    *,
    heldout_song_ids: set[str],
) -> tuple[list[Unit], dict[tuple[str, ...], list[Unit]]]:
    units: list[Unit] = []
    index: dict[tuple[str, ...], list[Unit]] = defaultdict(list)
    for record in records:
        song_id = str(record.get("song_id") or record["id"])
        if song_id in heldout_song_ids:
            continue
        waveform_path = str(record["vocal_wav_path"])
        for segment in record.get("segments") or []:
            words = [
                {
                    "word": str(item.get("word") or "").strip(),
                    "start": float(item.get("start") or 0.0),
                    "end": float(item.get("end") or item.get("start") or 0.0),
                }
                for item in segment.get("words") or []
            ]
            words = [
                item
                for item in words
                if item["word"] and item["end"] > item["start"]
            ]
            for offset in range(len(words)):
                for size in (1, 2, 3):
                    group = words[offset : offset + size]
                    if len(group) != size:
                        continue
                    normalized = tuple(
                        normalize_word(item["word"]) for item in group
                    )
                    if not all(normalized):
                        continue
                    unit = Unit(
                        song_id=song_id,
                        record_id=str(record["id"]),
                        waveform_path=waveform_path,
                        words=tuple(item["word"] for item in group),
                        normalized_words=normalized,
                        start=float(group[0]["start"]),
                        end=float(group[-1]["end"]),
                    )
                    units.append(unit)
                    index[normalized].append(unit)
    return units, index


def duration_statistics(
    units: list[Unit],
) -> dict[str, Any]:
    word_units = [unit for unit in units if len(unit.words) == 1]
    by_word: dict[str, list[float]] = defaultdict(list)
    by_length: dict[int, list[float]] = defaultdict(list)
    for unit in word_units:
        duration = min(1.25, max(0.12, unit.duration))
        by_word[unit.normalized_words[0]].append(duration)
        by_length[min(8, len(accentless(unit.normalized_words[0])))].append(
            duration
        )
    global_duration = median(
        [value for values in by_word.values() for value in values]
    )
    return {
        "global": float(global_duration),
        "by_word": {
            key: float(median(values)) for key, values in by_word.items()
        },
        "by_length": {
            int(key): float(median(values)) for key, values in by_length.items()
        },
    }


def schedule_words(
    words: list[dict[str, Any]],
    statistics: dict[str, Any],
    *,
    duration_seconds: float = 16.0,
) -> list[dict[str, Any]]:
    """Create a lyric-only schedule from donor medians, never target timings."""
    if not words:
        raise ValueError("V64 requires at least one lyric word")
    by_word = statistics["by_word"]
    by_length = statistics["by_length"]
    global_duration = float(statistics["global"])
    durations = []
    for item in words:
        normalized = str(item["normalized"])
        character_count = min(8, len(accentless(normalized)))
        estimate = float(
            by_word.get(
                normalized,
                by_length.get(character_count, global_duration),
            )
        )
        durations.append(min(0.82, max(0.26, estimate)))
    gaps = []
    for index in range(len(words) - 1):
        line_break = (
            int(words[index]["segment_index"])
            != int(words[index + 1]["segment_index"])
        )
        gaps.append(0.30 if line_break else 0.065)

    target_span = max(1.0, float(duration_seconds) - 0.60)
    gap_total = sum(gaps)
    word_total = sum(durations)
    scale = (target_span - gap_total) / max(word_total, 1e-6)
    scale = min(1.35, max(0.78, scale))
    durations = [duration * scale for duration in durations]
    total = sum(durations) + gap_total
    remaining = max(0.0, target_span - total)
    if gaps and remaining:
        # Preserve explicit phrase pauses while distributing modest breathing
        # room; do not stretch phones until words sound compressed or dragged.
        weights = [
            3.0
            if int(words[index]["segment_index"])
            != int(words[index + 1]["segment_index"])
            else 1.0
            for index in range(len(gaps))
        ]
        weight_total = sum(weights)
        for index, weight in enumerate(weights):
            cap = 0.55 if weight > 1.0 else 0.18
            gaps[index] = min(
                cap,
                gaps[index] + remaining * weight / weight_total,
            )

    cursor = 0.30
    schedule = []
    for index, (item, word_duration) in enumerate(zip(words, durations)):
        end = min(float(duration_seconds) - 0.10, cursor + word_duration)
        schedule.append({
            **item,
            "scheduled_start": cursor,
            "scheduled_end": end,
            "scheduled_duration": end - cursor,
        })
        cursor = end
        if index < len(gaps):
            cursor += gaps[index]
    return schedule


def _similarity(left: str, right: str) -> float:
    if left == right:
        return 1.0
    plain_left = accentless(left)
    plain_right = accentless(right)
    return max(
        SequenceMatcher(None, left, right).ratio(),
        0.92 * SequenceMatcher(None, plain_left, plain_right).ratio(),
    )


def select_unit(
    normalized_words: tuple[str, ...],
    target_duration: float,
    *,
    units: list[Unit],
    exact_index: dict[tuple[str, ...], list[Unit]],
    usage: Counter[tuple[str, float, float]],
    preferred_song: str | None,
) -> tuple[Unit, bool, float]:
    exact = list(exact_index.get(normalized_words) or [])
    candidates = exact
    if not candidates:
        candidates = [
            unit
            for unit in units
            if len(unit.normalized_words) == len(normalized_words)
        ]
    if not candidates:
        raise RuntimeError("V64 donor inventory is empty")

    def score(unit: Unit) -> tuple[float, str, float]:
        similarity = sum(
            _similarity(left, right)
            for left, right in zip(normalized_words, unit.normalized_words)
        ) / len(normalized_words)
        duration_error = abs(math.log(max(1e-3, unit.duration / target_duration)))
        key = (unit.record_id, unit.start, unit.end)
        value = (
            4.0 * float(unit.normalized_words == normalized_words)
            + 1.8 * similarity
            - 0.45 * duration_error
            - 0.30 * usage.get(key, 0)
            + 0.18 * float(preferred_song == unit.song_id)
        )
        return value, unit.record_id, -unit.start

    selected = max(candidates, key=score)
    key = (selected.record_id, selected.start, selected.end)
    usage[key] = usage.get(key, 0) + 1
    mean_similarity = sum(
        _similarity(left, right)
        for left, right in zip(normalized_words, selected.normalized_words)
    ) / len(normalized_words)
    return selected, bool(exact), float(mean_similarity)


def _fit_length(waveform: np.ndarray, samples: int) -> np.ndarray:
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if waveform.size < samples:
        waveform = np.pad(waveform, (0, samples - waveform.size))
    return waveform[:samples]


def stretch_unit(
    waveform: torch.Tensor,
    unit: Unit,
    target_duration: float,
    *,
    context_seconds: float = 0.08,
) -> torch.Tensor:
    """Pitch-preserving phase-vocoder stretch with context then safe crop."""
    import librosa

    source = waveform.detach().float().flatten().cpu().numpy()
    context = max(0, round(context_seconds * SAMPLE_RATE))
    start = max(0, round(unit.start * SAMPLE_RATE) - context)
    end = min(source.size, round(unit.end * SAMPLE_RATE) + context)
    excerpt = source[start:end]
    target_samples = max(1, round(float(target_duration) * SAMPLE_RATE))
    padded_target = target_samples + 2 * context
    if excerpt.size < 256:
        stretched = np.interp(
            np.linspace(0, max(0, excerpt.size - 1), padded_target),
            np.arange(max(1, excerpt.size)),
            excerpt if excerpt.size else np.zeros(1, dtype=np.float32),
        ).astype(np.float32)
    else:
        rate = excerpt.size / max(1, padded_target)
        stretched = librosa.effects.time_stretch(
            excerpt.astype(np.float32),
            rate=float(rate),
            n_fft=512,
            hop_length=128,
        )
        stretched = _fit_length(stretched, padded_target)
    cropped = _fit_length(
        stretched[context : context + target_samples],
        target_samples,
    )
    cropped = np.nan_to_num(cropped, copy=False)
    cropped -= float(cropped.mean())
    return torch.from_numpy(cropped.copy()).float()


def synthesize_vocal(
    dataset: Path,
    schedule: list[dict[str, Any]],
    *,
    units: list[Unit],
    exact_index: dict[tuple[str, ...], list[Unit]],
    duration_seconds: float = 16.0,
    crossfade_ms: float = 20.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    output = torch.zeros(round(duration_seconds * SAMPLE_RATE))
    cache: dict[str, torch.Tensor] = {}
    usage: Counter[tuple[str, float, float]] = Counter()
    donor_song_counts: Counter[str] = Counter()
    preferred_song = None
    selected_groups = []
    cursor = 0
    while cursor < len(schedule):
        selected = None
        for size in (3, 2, 1):
            group = schedule[cursor : cursor + size]
            if len(group) != size:
                continue
            normalized = tuple(str(item["normalized"]) for item in group)
            target_duration = (
                float(group[-1]["scheduled_end"])
                - float(group[0]["scheduled_start"])
            )
            if size > 1 and normalized not in exact_index:
                continue
            selected = (
                size,
                group,
                *select_unit(
                    normalized,
                    target_duration,
                    units=units,
                    exact_index=exact_index,
                    usage=usage,
                    preferred_song=preferred_song,
                ),
            )
            break
        assert selected is not None
        size, group, unit, exact, similarity = selected
        preferred_song = preferred_song or unit.song_id
        donor_song_counts[unit.song_id] += size
        path = dataset / unit.waveform_path
        if unit.waveform_path not in cache:
            cache[unit.waveform_path] = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            ).float()
        target_duration = (
            float(group[-1]["scheduled_end"])
            - float(group[0]["scheduled_start"])
        )
        rendered = stretch_unit(
            cache[unit.waveform_path],
            unit,
            target_duration,
        )
        rms = rendered.square().mean().sqrt().clamp_min(1e-5)
        rendered = rendered * torch.clamp(torch.tensor(0.085) / rms, 0.55, 1.8)
        fade = min(
            round(float(crossfade_ms) * SAMPLE_RATE / 1000.0),
            max(1, rendered.numel() // 5),
        )
        if fade > 1:
            rendered[:fade] *= torch.linspace(0.0, 1.0, fade)
            rendered[-fade:] *= torch.linspace(1.0, 0.0, fade)
        start = round(float(group[0]["scheduled_start"]) * SAMPLE_RATE)
        end = min(output.numel(), start + rendered.numel())
        output[start:end] += rendered[: end - start]
        selected_groups.append({
            "target_words": [str(item["word"]) for item in group],
            "donor_words": list(unit.words),
            "donor_song_id": unit.song_id,
            "donor_record_id": unit.record_id,
            "source_start": unit.start,
            "source_end": unit.end,
            "target_start": float(group[0]["scheduled_start"]),
            "target_end": float(group[-1]["scheduled_end"]),
            "exact": exact,
            "similarity": similarity,
        })
        cursor += size

    peak = output.abs().max()
    if float(peak) > 0.97:
        output *= 0.97 / peak
    exact_words = sum(
        len(group["target_words"]) for group in selected_groups if group["exact"]
    )
    phrase_words = sum(
        len(group["target_words"])
        for group in selected_groups
        if group["exact"] and len(group["target_words"]) > 1
    )
    return output, {
        "groups": selected_groups,
        "exact_word_fraction": exact_words / max(1, len(schedule)),
        "exact_phrase_word_fraction": phrase_words / max(1, len(schedule)),
        "mean_similarity": float(
            np.mean([group["similarity"] for group in selected_groups])
        ),
        "donor_song_count": len(donor_song_counts),
        "dominant_donor_song": donor_song_counts.most_common(1)[0][0],
    }


def pilot_gate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    mean_vocal_wa = float(np.mean([
        sample["generated_vocal_asr"]["word_accuracy"] for sample in samples
    ]))
    mean_mix_wa = float(np.mean([
        sample["generated_full_mix_asr"]["word_accuracy"] for sample in samples
    ]))
    mean_voiced_relative = float(np.mean([
        sample["generated_vocal_acoustics"]["voiced_ratio"]
        / max(1e-6, sample["target_vocal_acoustics"]["voiced_ratio"])
        for sample in samples
    ]))
    hypotheses = {
        sample["generated_full_mix_asr"]["hypothesis"].strip().casefold()
        for sample in samples
        if sample["generated_full_mix_asr"]["hypothesis"].strip()
    }
    pass_value = bool(
        len(samples) == 3
        and mean_vocal_wa >= 0.35
        and mean_mix_wa >= 0.30
        and sum(
            sample["generated_full_mix_asr"]["word_accuracy"] >= 0.25
            for sample in samples
        ) >= 2
        and mean_voiced_relative >= 0.70
        and len(hypotheses) >= 2
        and all(
            sample["generated_full_mix_acoustics"]["duration_seconds"] >= 15.5
            and sample["generated_full_mix_acoustics"]["clip_ratio"] <= 0.01
            and sample["retrieval"]["mean_similarity"] >= 0.60
            for sample in samples
        )
    )
    return {
        "pass": pass_value,
        "mean_generated_vocal_word_accuracy": mean_vocal_wa,
        "mean_generated_full_mix_word_accuracy": mean_mix_wa,
        "samples_full_mix_wa_at_least_0p25": sum(
            sample["generated_full_mix_asr"]["word_accuracy"] >= 0.25
            for sample in samples
        ),
        "mean_voiced_relative_to_target": mean_voiced_relative,
        "distinct_full_mix_hypotheses": len(hypotheses),
        "thresholds": {
            "mean_generated_vocal_word_accuracy": 0.35,
            "mean_generated_full_mix_word_accuracy": 0.30,
            "samples_full_mix_wa_at_least_0p25": 2,
            "mean_voiced_relative_to_target": 0.70,
            "distinct_full_mix_hypotheses": 2,
            "mean_retrieval_similarity": 0.60,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=16.0)
    parser.add_argument("--backing-to-vocal-rms", type=float, default=0.38)
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
    state_path = output_root / "master_waveform_unit_v64_state.json"
    records = [
        json.loads(line)
        for line in (dataset / "records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    candidates = []
    for record in records:
        try:
            window = select_word_window(
                record,
                duration_seconds=float(args.duration),
            )
        except ValueError:
            continue
        if window["duration_seconds"] >= 15.5 and window["word_count"] >= 16:
            candidates.append((record, window))
    candidates.sort(
        key=lambda item: (
            str(item[0].get("song_id") or item[0]["id"]),
            str(item[0]["id"]),
        )
    )
    if len(candidates) < 3:
        raise RuntimeError("V64 lacks three held-out 16-second lyric windows")
    heldout = [
        candidates[0],
        candidates[len(candidates) // 2],
        candidates[-1],
    ]
    heldout_song_ids = {
        str(record.get("song_id") or record["id"]) for record, _ in heldout
    }
    units, exact_index = build_inventory(
        records,
        heldout_song_ids=heldout_song_ids,
    )
    statistics = duration_statistics(units)
    state: dict[str, Any] = {
        "status": "rendering",
        "training": False,
        "goal_eligible_matching_backing": True,
        "arbitrary_backing_audit_required": True,
        "design": (
            "training-song-only exact/fuzzy raw waveform units; longest exact "
            "trigram/bigram first; pitch-preserving contextual stretch; donor "
            "median pacing; held-out vocal used only for scoring"
        ),
        "records": len(records),
        "heldout_song_ids": sorted(heldout_song_ids),
        "donor_songs": len({
            str(record.get("song_id") or record["id"])
            for record in records
            if str(record.get("song_id") or record["id"])
            not in heldout_song_ids
        }),
        "unit_count": len(units),
        "exact_ngram_keys": len(exact_index),
        "duration_statistics": statistics,
        "pretrained_tts_used": False,
        "pretrained_asr_used": True,
        "asr_evaluation_only": True,
    }
    _write_json(state_path, state)

    sample_root = output_root / "heldout_16s"
    sample_root.mkdir(parents=True, exist_ok=True)
    samples = []
    for record, window in heldout:
        words = words_in_window(record, window)
        schedule = schedule_words(
            words,
            statistics,
            duration_seconds=float(args.duration),
        )
        generated_vocal, retrieval = synthesize_vocal(
            dataset,
            schedule,
            units=units,
            exact_index=exact_index,
            duration_seconds=float(args.duration),
        )
        reference_start = round(
            float(window["start_seconds"]) * SAMPLE_RATE
        )
        length = round(float(args.duration) * SAMPLE_RATE)
        target_vocal_full = torch.load(
            dataset / str(record["vocal_wav_path"]),
            map_location="cpu",
            weights_only=True,
        ).float()
        target_backing_full = torch.load(
            dataset / str(record["backing_wav_path"]),
            map_location="cpu",
            weights_only=True,
        ).float()
        target_vocal = target_vocal_full[
            reference_start : reference_start + length
        ]
        target_backing = target_backing_full[
            reference_start : reference_start + length
        ]
        if target_vocal.numel() < length:
            target_vocal = torch.nn.functional.pad(
                target_vocal,
                (0, length - target_vocal.numel()),
            )
        if target_backing.numel() < length:
            target_backing = torch.nn.functional.pad(
                target_backing,
                (0, length - target_backing.numel()),
            )
        generated_mix = mix_raw_stems(
            generated_vocal,
            target_backing,
            backing_to_vocal_rms=float(args.backing_to_vocal_rms),
        )
        target_mix = mix_raw_stems(
            target_vocal,
            target_backing,
            backing_to_vocal_rms=float(args.backing_to_vocal_rms),
        )
        stem = str(record["id"])
        paths = {
            "generated_vocal": sample_root / f"{stem}_generated_vocal.wav",
            "generated_full_mix": sample_root / f"{stem}_generated_full_mix.wav",
            "target_vocal": sample_root / f"{stem}_target_vocal.wav",
            "target_full_mix": sample_root / f"{stem}_target_full_mix.wav",
        }
        tensors = {
            "generated_vocal": generated_vocal,
            "generated_full_mix": generated_mix,
            "target_vocal": target_vocal,
            "target_full_mix": target_mix,
        }
        for name, path in paths.items():
            sf.write(
                path,
                tensors[name].numpy(),
                SAMPLE_RATE,
                subtype="PCM_16",
            )
        prompt = " ".join(str(item["word"]) for item in words)

        def asr(path: Path) -> dict[str, Any]:
            hypothesis = lyric_wer(path, prompt)["predicted_text"]
            return transcription_metrics(prompt, hypothesis)

        sample = {
            "id": stem,
            "song_id": str(record.get("song_id") or stem),
            "reference_text": prompt,
            "reference_window": window,
            "schedule": schedule,
            "retrieval": retrieval,
            "generated_vocal_asr": asr(paths["generated_vocal"]),
            "generated_full_mix_asr": asr(paths["generated_full_mix"]),
            "target_vocal_asr": asr(paths["target_vocal"]),
            "target_full_mix_asr": asr(paths["target_full_mix"]),
            "generated_vocal_acoustics": wav_metrics(
                paths["generated_vocal"]
            ),
            "generated_full_mix_acoustics": wav_metrics(
                paths["generated_full_mix"]
            ),
            "target_vocal_acoustics": wav_metrics(paths["target_vocal"]),
            "target_full_mix_acoustics": wav_metrics(
                paths["target_full_mix"]
            ),
            **{f"{name}_wav": str(path) for name, path in paths.items()},
        }
        samples.append(sample)
        state["samples"] = samples
        _write_json(state_path, state)

    gate = pilot_gate(samples)
    state.update({
        "status": "pilot_passed" if gate["pass"] else "pilot_failed",
        "samples": samples,
        "gate": gate,
    })
    _write_json(state_path, state)
    print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
