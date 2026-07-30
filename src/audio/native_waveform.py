"""Target-free native Vietnamese waveform retrieval and synthesis.

Donor phrases preserve their pitch and duration. The renderer performs only
quiet-boundary trimming, short fades, line-aware pauses, and vocal-forward
mixing with a backing track from another song.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

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
)

MAX_PHRASE_WORDS = 8


def write_json(path: Path, payload: dict[str, Any]) -> None:
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
        raise ValueError("Maximum phrase size must be at least two")
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
        raise ValueError("Native waveform synthesis requires eight words")
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
        raise RuntimeError("Could not find a complete native-duration path")
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
        raise ValueError("Cannot pace an empty waveform path")
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


def _fit_audio(value: torch.Tensor, samples: int) -> torch.Tensor:
    value = value.detach().float().flatten()
    if value.numel() < samples:
        value = torch.nn.functional.pad(
            value,
            (0, samples - value.numel()),
        )
    return value[:samples]


def target_words_from_text(text: str) -> list[dict[str, Any]]:
    """Convert user lyrics into line-aware words without target timestamps."""
    words: list[dict[str, Any]] = []
    for segment_index, line in enumerate(text.splitlines() or [text]):
        for token in line.split():
            normalized = normalize_word(token)
            if normalized:
                words.append({
                    "word": token.strip(),
                    "normalized": normalized,
                    "segment_index": segment_index,
                })
    if len(words) < 8:
        raise ValueError(
            "Native waveform generation requires at least eight lyric words"
        )
    return words


def _select_cross_song_backing(
    records: list[dict[str, Any]],
    dataset: Path,
    *,
    excluded_song_ids: set[str],
    duration_seconds: float,
    selection_key: str,
) -> tuple[dict[str, Any], torch.Tensor, float]:
    """Choose a deterministic, non-silent backing outside the donor set."""
    length = round(float(duration_seconds) * SAMPLE_RATE)
    ranked = sorted(
        records,
        key=lambda record: hashlib.sha256(
            (
                selection_key
                + "\0"
                + str(record.get("song_id") or record["id"])
            ).encode("utf-8")
        ).digest(),
    )
    for record in ranked:
        song_id = str(record.get("song_id") or record["id"])
        if song_id in excluded_song_ids:
            continue
        path = dataset / str(record["backing_wav_path"])
        if not path.is_file():
            continue
        waveform = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        ).float().flatten()
        if not bool(torch.isfinite(waveform).all()) or waveform.numel() < 1:
            continue
        maximum_start = max(0, waveform.numel() - length)
        digest = hashlib.sha256(
            (selection_key + "\0" + song_id).encode("utf-8")
        ).digest()
        start = (
            int.from_bytes(digest[:8], "big") % (maximum_start + 1)
            if maximum_start
            else 0
        )
        backing = _fit_audio(waveform[start : start + length], length)
        if float(backing.square().mean().sqrt()) >= 1e-4:
            return record, backing, start / SAMPLE_RATE
    raise RuntimeError("Could not find a non-silent cross-song backing")


def generate_text_candidate(
    records: list[dict[str, Any]],
    dataset: Path,
    output_root: Path,
    *,
    text: str,
    duration_seconds: float,
    backing_ratio: float,
    presence_attenuation_db: float,
    genre: str = "",
    path_selector: Callable[..., tuple[list[dict[str, Any]], dict[str, Any]]] = (
        select_long_phrase_path
    ),
    gap_filler: Callable[..., dict[str, Any]] = fill_natural_path_gaps,
) -> dict[str, Any]:
    """Render one target-free full mix for user-provided Vietnamese lyrics."""
    import soundfile as sf

    target_words = target_words_from_text(text)
    units, exact_index = build_long_inventory(
        records,
        heldout_song_ids=set(),
    )
    statistics = duration_statistics(units)
    choices, retrieval = path_selector(
        target_words,
        units,
        exact_index,
        statistics,
        duration_seconds=float(duration_seconds),
        respect_segment_boundaries=True,
    )
    retrieval["pacing"] = gap_filler(
        choices,
        duration_seconds=float(duration_seconds),
    )
    generated_vocal, rendered_groups = render_long_phrase_path(
        dataset,
        choices,
        duration_seconds=float(duration_seconds),
    )
    retrieval["groups"] = rendered_groups
    donor_song_ids = {
        str(choice["unit"].song_id)
        for choice in choices
    }
    backing_record, backing, backing_start = _select_cross_song_backing(
        records,
        dataset,
        excluded_song_ids=donor_song_ids,
        duration_seconds=float(duration_seconds),
        selection_key=f"{genre}\0{text}",
    )

    vocal_path = output_root / "final_vocal.wav"
    backing_path = output_root / "final_backing.wav"
    mix_path = output_root / "final.wav"
    sf.write(
        vocal_path,
        generated_vocal.numpy(),
        SAMPLE_RATE,
        subtype="PCM_16",
    )
    sf.write(
        backing_path,
        backing.numpy(),
        SAMPLE_RATE,
        subtype="PCM_16",
    )
    mix_metadata = mix_clarity_candidate(
        vocal_path,
        backing_path,
        mix_path,
        backing_ratio=float(backing_ratio),
        presence_attenuation_db=float(presence_attenuation_db),
        dynamic=True,
    )
    mp3_path = output_root / "final.mp3"
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        conversion = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(mix_path),
                "-codec:a",
                "libmp3lame",
                "-q:a",
                "2",
                str(mp3_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if conversion.returncode != 0:
            mp3_path.unlink(missing_ok=True)

    return {
        "status": "complete",
        "mode": "text",
        "training": False,
        "goal_eligible_prediction": True,
        "backend": "genmusic-native-waveform-v80",
        "model": "native-waveform-v80",
        "text": text,
        "genre": genre,
        "duration_seconds": float(duration_seconds),
        "records": len(records),
        "unit_count": len(units),
        "retrieval": retrieval,
        "donor_song_ids": sorted(donor_song_ids),
        "backing_song_id": str(
            backing_record.get("song_id") or backing_record["id"]
        ),
        "backing_start_seconds": backing_start,
        "mix": mix_metadata,
        "final_vocal_wav": str(vocal_path),
        "final_backing_wav": str(backing_path),
        "final_wav": str(mix_path),
        "final_mp3": str(mp3_path) if mp3_path.is_file() else None,
        "target_audio_used_at_product_inference": False,
        "target_timing_used_at_product_inference": False,
        "per_unit_time_stretch_used": False,
        "cross_song_backing_used": True,
        "pretrained_tts_used": False,
        "pretrained_asr_used": False,
    }
