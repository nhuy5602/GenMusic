"""Native long-phrase singing synthesis on an aligned raw-song corpus.

The raw-data oracle (V74) showed a sharp context threshold: isolated words
were unintelligible while eight-word excerpts reached mean ASR word accuracy
0.5625.  V64/V65 used only 80 songs and mostly one-to-three-word units, so
their many joins destroyed that context.  V78 scales the donor bank to the
validated 2048-song raw corpus and searches a complete native-duration path
with exact phrases up to eight words.

No target vocal or target timing enters synthesis.  Donor phrases preserve
their original pitch and duration; the renderer only trims to nearby quiet
zero crossings and adds short fades.  The supplied backing is deliberately
from another song and receives deterministic sidechain presence attenuation.
PhoWhisper is used only after all audio is rendered for final evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import torch

from src.data.aligned_raw import (
    mix_raw_stems,
    select_word_window,
)
from src.audio.vocal_mix import (
    mix_clarity_candidate,
)
from src.audio.waveform_path import (
    extract_source_unit,
)
from src.audio.waveform_units import (
    SAMPLE_RATE,
    Unit,
    _similarity,
    duration_statistics,
    normalize_word,
    words_in_window,
)


STATE_NAME = "master_raw_long_phrase_v78_state.json"
MAX_PHRASE_WORDS = 8


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def build_long_inventory(
    records: list[dict[str, Any]],
    *,
    heldout_song_ids: set[str],
    maximum_words: int = MAX_PHRASE_WORDS,
) -> tuple[list[Unit], dict[tuple[str, ...], list[Unit]]]:
    """Index exact contiguous raw-vocal units up to eight words."""
    if maximum_words < 2:
        raise ValueError("V78 maximum phrase size must be at least two")
    units: list[Unit] = []
    index: dict[tuple[str, ...], list[Unit]] = defaultdict(list)
    for record in records:
        song_id = str(record.get("song_id") or record["id"])
        if song_id in heldout_song_ids:
            continue
        waveform_path = str(record["vocal_wav_path"])
        for segment in record.get("segments") or []:
            words = []
            for item in segment.get("words") or []:
                text = str(item.get("word") or "").strip()
                try:
                    start = float(item["start"])
                    end = float(item["end"])
                except (KeyError, TypeError, ValueError):
                    continue
                normalized = normalize_word(text)
                if text and normalized and end > start:
                    words.append({
                        "word": text,
                        "normalized": normalized,
                        "start": start,
                        "end": end,
                    })
            for offset in range(len(words)):
                for size in range(
                    1,
                    min(maximum_words, len(words) - offset) + 1,
                ):
                    group = words[offset : offset + size]
                    unit = Unit(
                        song_id=song_id,
                        record_id=str(record["id"]),
                        waveform_path=waveform_path,
                        words=tuple(item["word"] for item in group),
                        normalized_words=tuple(
                            item["normalized"] for item in group
                        ),
                        start=float(group[0]["start"]),
                        end=float(group[-1]["end"]),
                    )
                    units.append(unit)
                    index[unit.normalized_words].append(unit)
    return units, index


def lexical_set_cover(
    target_words: list[dict[str, Any]],
    units: list[Unit],
    *,
    maximum_songs: int = 8,
) -> list[str]:
    required = {
        (index, str(item["normalized"]))
        for index, item in enumerate(target_words)
    }
    coverage: dict[str, set[tuple[int, str]]] = defaultdict(set)
    for unit in units:
        if len(unit.words) != 1:
            continue
        token = unit.normalized_words[0]
        for indexed in required:
            if indexed[1] == token:
                coverage[unit.song_id].add(indexed)
    uncovered = set(required)
    selected: list[str] = []
    while uncovered and len(selected) < maximum_songs:
        ranked = sorted(
            (
                (len(values & uncovered), len(values), song_id)
                for song_id, values in coverage.items()
                if song_id not in selected
            ),
            reverse=True,
        )
        if not ranked or ranked[0][0] <= 0:
            break
        song_id = ranked[0][2]
        selected.append(song_id)
        uncovered -= coverage[song_id]
    return selected


def _target_group_duration(
    normalized: tuple[str, ...],
    statistics: dict[str, Any],
) -> float:
    by_word = statistics["by_word"]
    global_duration = float(statistics["global"])
    return sum(float(by_word.get(word, global_duration)) for word in normalized)


def _rank_candidates(
    normalized: tuple[str, ...],
    *,
    exact_index: dict[tuple[str, ...], list[Unit]],
    word_units: list[Unit],
    statistics: dict[str, Any],
    preferred_songs: set[str],
    current_song: str | None,
    maximum_candidates: int = 6,
) -> list[tuple[Unit, bool, float]]:
    exact = list(exact_index.get(normalized) or [])
    target_duration = _target_group_duration(normalized, statistics)
    if exact:
        candidates = exact
    elif len(normalized) == 1:
        candidates = sorted(
            word_units,
            key=lambda unit: _similarity(
                normalized[0],
                unit.normalized_words[0],
            ),
            reverse=True,
        )[:256]
    else:
        return []

    def score(unit: Unit) -> tuple[float, str, float]:
        similarity = float(np.mean([
            _similarity(left, right)
            for left, right in zip(normalized, unit.normalized_words)
        ]))
        duration_error = abs(
            math.log(max(1e-3, unit.duration / max(1e-3, target_duration)))
        )
        value = (
            4.0 * float(unit.normalized_words == normalized)
            + 1.4 * similarity
            + 0.55 * float(unit.song_id in preferred_songs)
            + 0.75 * float(unit.song_id == current_song)
            - 0.70 * duration_error
        )
        return value, unit.record_id, -unit.start

    ranked = sorted(candidates, key=score, reverse=True)
    result = []
    seen_songs: Counter[str] = Counter()
    for unit in ranked:
        # Keep candidate diversity without allowing one prolific song to
        # consume the whole bounded beam.
        if seen_songs[unit.song_id] >= 2:
            continue
        seen_songs[unit.song_id] += 1
        similarity = float(np.mean([
            _similarity(left, right)
            for left, right in zip(normalized, unit.normalized_words)
        ]))
        result.append((
            unit,
            unit.normalized_words == normalized,
            similarity,
        ))
        if len(result) >= maximum_candidates:
            break
    return result


@dataclass
class PathState:
    position: int
    duration: float
    score: float
    last_song: str | None
    last_segment: int | None
    choices: list[dict[str, Any]]


def select_long_phrase_path(
    target_words: list[dict[str, Any]],
    units: list[Unit],
    exact_index: dict[tuple[str, ...], list[Unit]],
    statistics: dict[str, Any],
    *,
    duration_seconds: float = 16.0,
    maximum_phrase_words: int = MAX_PHRASE_WORDS,
    beam_size: int = 48,
    respect_segment_boundaries: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Beam-search a complete native-duration path with long exact phrases."""
    if len(target_words) < 8:
        raise ValueError("V78 requires at least eight lyric words")
    preferred = lexical_set_cover(target_words, units)
    preferred_set = set(preferred)
    word_units = [unit for unit in units if len(unit.words) == 1]
    beams: dict[int, list[PathState]] = {
        0: [PathState(0, 0.20, 0.0, None, None, [])]
    }
    maximum_duration = float(duration_seconds) - 0.08
    word_count = len(target_words)
    for position in range(word_count):
        states = beams.get(position) or []
        if not states:
            continue
        for state in states:
            for size in range(
                min(maximum_phrase_words, word_count - position),
                0,
                -1,
            ):
                group = target_words[position : position + size]
                if (
                    respect_segment_boundaries
                    and len({
                        int(item["segment_index"])
                        for item in group
                    }) > 1
                ):
                    continue
                normalized = tuple(
                    str(item["normalized"]) for item in group
                )
                candidates = _rank_candidates(
                    normalized,
                    exact_index=exact_index,
                    word_units=word_units,
                    statistics=statistics,
                    preferred_songs=preferred_set,
                    current_song=state.last_song,
                )
                for unit, exact, similarity in candidates:
                    segment = int(group[0]["segment_index"])
                    gap = 0.0
                    if state.last_segment is not None:
                        gap = (
                            0.24
                            if segment != state.last_segment
                            else 0.045
                        )
                    new_duration = state.duration + gap + unit.duration
                    if new_duration > maximum_duration:
                        continue
                    switch = (
                        state.last_song is not None
                        and state.last_song != unit.song_id
                    )
                    target_duration = _target_group_duration(
                        normalized,
                        statistics,
                    )
                    duration_error = abs(math.log(
                        max(1e-3, unit.duration / max(1e-3, target_duration))
                    ))
                    increment = (
                        2.4 * size * float(exact)
                        + 1.5 * max(0, size - 1)
                        + 0.2 * size * similarity
                        + 0.15 * size * float(
                            unit.song_id in preferred_set
                        )
                        + 0.55 * float(unit.song_id == state.last_song)
                        - 0.65 * float(switch)
                        - 0.60 * float(bool(state.choices))
                        - 0.55 * duration_error
                    )
                    choice = {
                        "target": group,
                        "unit": unit,
                        "exact": exact,
                        "similarity": similarity,
                        "gap_before": gap,
                    }
                    destination = position + size
                    beams.setdefault(destination, []).append(PathState(
                        position=destination,
                        duration=new_duration,
                        score=state.score + increment,
                        last_song=unit.song_id,
                        last_segment=int(group[-1]["segment_index"]),
                        choices=state.choices + [choice],
                    ))
        for destination in range(position + 1, min(
            word_count,
            position + maximum_phrase_words,
        ) + 1):
            candidates = beams.get(destination)
            if not candidates or len(candidates) <= beam_size:
                continue
            expected = maximum_duration * destination / word_count
            beams[destination] = sorted(
                candidates,
                key=lambda item: (
                    item.score
                    - 0.30 * abs(item.duration - expected),
                    -len(item.choices),
                ),
                reverse=True,
            )[:beam_size]
    complete = beams.get(word_count) or []
    if not complete:
        raise RuntimeError("V78 could not find a complete native-duration path")
    selected = max(
        complete,
        key=lambda item: (
            item.score - 0.45 * abs(item.duration - 15.2),
            -len(item.choices),
        ),
    )
    choices = selected.choices
    exact_words = sum(
        len(item["target"]) for item in choices if item["exact"]
    )
    phrase_words = sum(
        len(item["target"])
        for item in choices
        if item["exact"] and len(item["target"]) > 1
    )
    long_phrase_words = sum(
        len(item["target"])
        for item in choices
        if item["exact"] and len(item["target"]) >= 4
    )
    donor_switches = sum(
        choices[index - 1]["unit"].song_id
        != choices[index]["unit"].song_id
        for index in range(1, len(choices))
    )
    segment_spanning_groups = sum(
        len({
            int(item["segment_index"])
            for item in choice["target"]
        }) > 1
        for choice in choices
    )
    return choices, {
        "preferred_donor_songs": preferred,
        "selected_groups": len(choices),
        "selected_donor_songs": len({
            item["unit"].song_id for item in choices
        }),
        "donor_switches": donor_switches,
        "estimated_span_seconds": selected.duration,
        "exact_word_fraction": exact_words / word_count,
        "exact_phrase_word_fraction": phrase_words / word_count,
        "exact_long_phrase_word_fraction": long_phrase_words / word_count,
        "maximum_selected_phrase_words": max(
            len(item["target"]) for item in choices
        ),
        "mean_similarity": float(np.mean([
            item["similarity"] for item in choices
        ])),
        "path_score": selected.score,
        "respect_segment_boundaries": bool(
            respect_segment_boundaries
        ),
        "target_segment_spanning_groups": segment_spanning_groups,
        "source_duration_preserved": True,
        "per_unit_time_stretch_used": False,
    }


def render_long_phrase_path(
    dataset: Path,
    choices: list[dict[str, Any]],
    *,
    duration_seconds: float = 16.0,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    output = torch.zeros(round(duration_seconds * SAMPLE_RATE))
    cache: dict[str, torch.Tensor] = {}
    cursor = 0.20
    running_rms: float | None = None
    rendered_groups: list[dict[str, Any]] = []
    for choice in choices:
        cursor += float(choice["gap_before"])
        unit: Unit = choice["unit"]
        if unit.waveform_path not in cache:
            cache[unit.waveform_path] = torch.load(
                dataset / unit.waveform_path,
                map_location="cpu",
                weights_only=True,
            ).float()
        rendered = extract_source_unit(
            cache[unit.waveform_path],
            unit,
            zero_crossing_search_ms=10.0,
            fade_ms=7.0,
        )
        rms = float(rendered.square().mean().sqrt().clamp_min(1e-5))
        if running_rms is None:
            running_rms = min(0.11, max(0.060, rms))
        gain = min(1.35, max(0.72, running_rms / rms))
        rendered *= gain
        running_rms = (
            0.90 * running_rms
            + 0.10 * float(rendered.square().mean().sqrt())
        )
        start = round(cursor * SAMPLE_RATE)
        end = min(output.numel(), start + rendered.numel())
        if end <= start:
            break
        output[start:end] += rendered[: end - start]
        rendered_groups.append({
            "target_words": [
                str(item["word"]) for item in choice["target"]
            ],
            "donor_words": list(unit.words),
            "donor_song_id": unit.song_id,
            "donor_record_id": unit.record_id,
            "source_start": unit.start,
            "source_end": unit.end,
            "source_duration": unit.duration,
            "rendered_start": cursor,
            "rendered_end": end / SAMPLE_RATE,
            "exact": bool(choice["exact"]),
            "similarity": float(choice["similarity"]),
        })
        cursor = end / SAMPLE_RATE
    peak = output.abs().max()
    if float(peak) > 0.97:
        output *= 0.97 / peak
    return output, rendered_groups


def fill_natural_path_gaps(
    choices: list[dict[str, Any]],
    *,
    duration_seconds: float = 16.0,
    same_line_pause_cap: float = 0.48,
    line_break_pause_cap: float = 0.90,
) -> dict[str, Any]:
    """Use pauses, never phone compression, to occupy the requested duration."""
    if not choices:
        raise ValueError("V78 cannot pace an empty path")
    target_span = float(duration_seconds) - 0.45
    current_span = 0.20 + sum(
        float(choice["gap_before"]) + float(choice["unit"].duration)
        for choice in choices
    )
    remaining = max(0.0, target_span - current_span)
    boundaries = list(range(1, len(choices)))
    for _ in range(4):
        active = []
        for index in boundaries:
            previous_segment = int(
                choices[index - 1]["target"][-1]["segment_index"]
            )
            current_segment = int(
                choices[index]["target"][0]["segment_index"]
            )
            line_break = previous_segment != current_segment
            cap = (
                float(line_break_pause_cap)
                if line_break
                else float(same_line_pause_cap)
            )
            available = cap - float(choices[index]["gap_before"])
            if available > 1e-6:
                active.append((index, 3.0 if line_break else 1.0, available))
        if not active or remaining <= 1e-6:
            break
        weight_total = sum(weight for _, weight, _ in active)
        distributed = 0.0
        for index, weight, available in active:
            addition = min(available, remaining * weight / weight_total)
            choices[index]["gap_before"] = (
                float(choices[index]["gap_before"]) + addition
            )
            distributed += addition
        remaining = max(0.0, remaining - distributed)
    final_span = 0.20 + sum(
        float(choice["gap_before"]) + float(choice["unit"].duration)
        for choice in choices
    )
    word_count = sum(len(choice["target"]) for choice in choices)
    same_line_gaps = []
    line_break_gaps = []
    for index in range(1, len(choices)):
        previous_segment = int(
            choices[index - 1]["target"][-1]["segment_index"]
        )
        current_segment = int(
            choices[index]["target"][0]["segment_index"]
        )
        destination = (
            line_break_gaps
            if previous_segment != current_segment
            else same_line_gaps
        )
        destination.append(float(choices[index]["gap_before"]))
    return {
        "pacing_source": (
            "native donor phrase durations plus lyric-line-aware fixed "
            "pause allocation; no target timestamps"
        ),
        "span_before_gap_fill_seconds": current_span,
        "span_after_gap_fill_seconds": final_span,
        "unallocated_trailing_seconds": max(
            0.0,
            float(duration_seconds) - final_span,
        ),
        "words_per_rendered_span_second": (
            word_count / max(1e-6, final_span)
        ),
        "same_line_pause_cap_seconds": float(same_line_pause_cap),
        "line_break_pause_cap_seconds": float(line_break_pause_cap),
        "maximum_same_line_gap_seconds": max(
            same_line_gaps,
            default=0.0,
        ),
        "maximum_line_break_gap_seconds": max(
            line_break_gaps,
            default=0.0,
        ),
        "same_line_boundary_count": len(same_line_gaps),
        "line_break_boundary_count": len(line_break_gaps),
        "per_unit_time_stretch_used": False,
    }


def goal_gate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    vocal_wa = float(np.mean([
        sample["generated_vocal_asr"]["word_accuracy"]
        for sample in samples
    ]))
    mix_wa = float(np.mean([
        sample["generated_full_mix_asr"]["word_accuracy"]
        for sample in samples
    ]))
    voiced_relative = float(np.mean([
        sample["generated_vocal_acoustics"]["voiced_ratio"]
        / max(1e-6, sample["target_vocal_acoustics"]["voiced_ratio"])
        for sample in samples
    ]))
    hypotheses = {
        sample["generated_full_mix_asr"]["hypothesis"].strip().casefold()
        for sample in samples
        if sample["generated_full_mix_asr"]["hypothesis"].strip()
    }
    objective_pass = bool(
        len(samples) == 3
        and vocal_wa >= 0.35
        and mix_wa >= 0.35
        and sum(
            sample["generated_full_mix_asr"]["word_accuracy"] >= 0.25
            for sample in samples
        ) >= 2
        and voiced_relative >= 0.70
        and len(hypotheses) >= 2
        and all(
            sample["generated_full_mix_acoustics"]["duration_seconds"] >= 15.5
            and sample["generated_full_mix_acoustics"]["clip_ratio"] <= 0.01
            and sample["retrieval"]["exact_word_fraction"] >= 0.95
            and sample["retrieval"]["mean_similarity"] >= 0.95
            for sample in samples
        )
    )
    # V65 on 80 songs had vocal .21288/mix .12682.  The scaled corpus must
    # improve both by a substantive absolute amount even if the goal misses.
    pilot_pass = bool(
        vocal_wa >= 0.30
        and mix_wa >= 0.25
        and vocal_wa - 0.21287696365095746 >= 0.08
        and mix_wa - 0.12682031877078317 >= 0.10
        and voiced_relative >= 0.70
    )
    return {
        "pilot_pass": pilot_pass,
        "objective_pass": objective_pass,
        "human_listening_required": True,
        "mean_generated_vocal_word_accuracy": vocal_wa,
        "mean_generated_full_mix_word_accuracy": mix_wa,
        "vocal_word_accuracy_gain_over_v65": (
            vocal_wa - 0.21287696365095746
        ),
        "full_mix_word_accuracy_gain_over_v65": (
            mix_wa - 0.12682031877078317
        ),
        "generated_full_mix_samples_at_least_0p25": sum(
            sample["generated_full_mix_asr"]["word_accuracy"] >= 0.25
            for sample in samples
        ),
        "mean_generated_vocal_voiced_relative_to_target": voiced_relative,
        "distinct_generated_full_mix_hypotheses": len(hypotheses),
        "duration_pass": all(
            sample["generated_full_mix_acoustics"]["duration_seconds"] >= 15.5
            for sample in samples
        ),
        "clipping_pass": all(
            sample["generated_full_mix_acoustics"]["clip_ratio"] <= 0.01
            for sample in samples
        ),
        "cross_song_backing_pass": all(
            sample["song_id"] != sample["backing_song_id"]
            for sample in samples
        ),
    }


def _fit_audio(value: torch.Tensor, samples: int) -> torch.Tensor:
    value = value.detach().float().flatten()
    if value.numel() < samples:
        value = torch.nn.functional.pad(
            value,
            (0, samples - value.numel()),
        )
    return value[:samples]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=16.0)
    parser.add_argument("--backing-ratio", type=float, default=0.45)
    parser.add_argument("--presence-attenuation-db", type=float, default=10.0)
    parser.add_argument("--expected-records", type=int, default=2_048)
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
    state_path = output_root / STATE_NAME
    records = [
        json.loads(line)
        for line in (dataset / "records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    if len(records) != args.expected_records:
        raise RuntimeError(
            f"V78 expected {args.expected_records} records, got {len(records)}"
        )
    candidates = []
    for record in records:
        try:
            window = select_word_window(
                record,
                duration_seconds=float(args.duration),
            )
        except ValueError:
            continue
        if window["duration_seconds"] >= 15.5 and window["word_count"] >= 18:
            candidates.append((record, window))
    candidates.sort(
        key=lambda item: (
            str(item[0].get("song_id") or item[0]["id"]),
            str(item[0]["id"]),
        )
    )
    if len(candidates) < 96:
        raise RuntimeError(
            f"V78 held-out candidate shortfall: {len(candidates)}"
        )
    heldout = [
        candidates[0],
        candidates[len(candidates) // 2],
        candidates[-1],
    ]
    heldout_song_ids = {
        str(record.get("song_id") or record["id"])
        for record, _ in heldout
    }
    units, exact_index = build_long_inventory(
        records,
        heldout_song_ids=heldout_song_ids,
    )
    statistics = duration_statistics(units)
    state: dict[str, Any] = {
        "status": "rendering",
        "training": False,
        "goal_eligible_prediction": True,
        "design": (
            "2048-song training-only raw exact phrases up to eight words; "
            "beam-searched native-duration path; no pitch/time stretch; "
            "cross-song sidechain backing"
        ),
        "records": len(records),
        "heldout_candidates": len(candidates),
        "heldout_song_ids": sorted(heldout_song_ids),
        "donor_songs": len({
            str(record.get("song_id") or record["id"])
            for record in records
        } - heldout_song_ids),
        "unit_count": len(units),
        "exact_ngram_keys": len(exact_index),
        "maximum_phrase_words": MAX_PHRASE_WORDS,
        "target_audio_used_at_product_inference": False,
        "target_timing_used_at_product_inference": False,
        "matching_backing_used": False,
        "cross_song_backing_used": True,
        "pretrained_tts_used": False,
        "pretrained_asr_used": True,
        "asr_evaluation_only": True,
        "pretrained_source_separator_used": True,
    }
    _write_json(state_path, state)

    sample_root = output_root / "heldout_16s"
    sample_root.mkdir(parents=True, exist_ok=True)
    samples = []
    length = round(float(args.duration) * SAMPLE_RATE)
    for sample_index, (record, window) in enumerate(heldout):
        words = words_in_window(record, window)
        choices, retrieval = select_long_phrase_path(
            words,
            units,
            exact_index,
            statistics,
            duration_seconds=float(args.duration),
        )
        retrieval["pacing"] = fill_natural_path_gaps(
            choices,
            duration_seconds=float(args.duration),
        )
        generated_vocal, rendered_groups = render_long_phrase_path(
            dataset,
            choices,
            duration_seconds=float(args.duration),
        )
        retrieval["groups"] = rendered_groups
        reference_start = round(
            float(window["start_seconds"]) * SAMPLE_RATE
        )
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
        target_vocal = _fit_audio(
            target_vocal_full[reference_start : reference_start + length],
            length,
        )
        target_backing = _fit_audio(
            target_backing_full[reference_start : reference_start + length],
            length,
        )
        backing_record, backing_window = heldout[
            (sample_index + 1) % len(heldout)
        ]
        backing_start = round(
            float(backing_window["start_seconds"]) * SAMPLE_RATE
        )
        cross_backing_full = torch.load(
            dataset / str(backing_record["backing_wav_path"]),
            map_location="cpu",
            weights_only=True,
        ).float()
        cross_backing = _fit_audio(
            cross_backing_full[backing_start : backing_start + length],
            length,
        )
        stem = str(record["id"])
        paths = {
            "generated_vocal": sample_root / f"{stem}_generated_vocal.wav",
            "cross_song_backing": sample_root / f"{stem}_cross_song_backing.wav",
            "generated_full_mix": sample_root / f"{stem}_generated_full_mix.wav",
            "target_vocal": sample_root / f"{stem}_target_vocal.wav",
            "target_full_mix": sample_root / f"{stem}_target_full_mix.wav",
        }
        sf.write(
            paths["generated_vocal"],
            generated_vocal.numpy(),
            SAMPLE_RATE,
            subtype="PCM_16",
        )
        sf.write(
            paths["cross_song_backing"],
            cross_backing.numpy(),
            SAMPLE_RATE,
            subtype="PCM_16",
        )
        mix_metadata = mix_clarity_candidate(
            paths["generated_vocal"],
            paths["cross_song_backing"],
            paths["generated_full_mix"],
            backing_ratio=float(args.backing_ratio),
            presence_attenuation_db=float(
                args.presence_attenuation_db
            ),
            dynamic=True,
        )
        target_mix = mix_raw_stems(
            target_vocal,
            target_backing,
            backing_to_vocal_rms=0.45,
        )
        sf.write(
            paths["target_vocal"],
            target_vocal.numpy(),
            SAMPLE_RATE,
            subtype="PCM_16",
        )
        sf.write(
            paths["target_full_mix"],
            target_mix.numpy(),
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
            "backing_song_id": str(
                backing_record.get("song_id") or backing_record["id"]
            ),
            "reference_text": prompt,
            "reference_window": window,
            "retrieval": retrieval,
            "mix": mix_metadata,
            "generated_vocal_asr": asr(paths["generated_vocal"]),
            "generated_full_mix_asr": asr(
                paths["generated_full_mix"]
            ),
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
        print(
            "V78_SAMPLE "
            f"id={stem} "
            f"groups={retrieval['selected_groups']} "
            f"max_phrase={retrieval['maximum_selected_phrase_words']} "
            f"exact_phrase={retrieval['exact_phrase_word_fraction']:.6f} "
            f"vocal_wa={sample['generated_vocal_asr']['word_accuracy']:.6f} "
            f"mix_wa={sample['generated_full_mix_asr']['word_accuracy']:.6f}",
            flush=True,
        )

    gate = goal_gate(samples)
    state.update({
        "status": (
            "objective_candidate_passed"
            if gate["objective_pass"]
            else "pilot_passed"
            if gate["pilot_pass"]
            else "pilot_failed"
        ),
        "samples": samples,
        "gate": gate,
    })
    _write_json(state_path, state)
    print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
