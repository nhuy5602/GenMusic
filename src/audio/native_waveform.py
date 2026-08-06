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
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    target_context: tuple[str, str, str] | None,
    exact_index: dict[tuple[str, ...], list[Unit]],
    word_units: list[Unit],
    statistics: dict[str, Any],
    preferred_songs: set[str],
    current_song: str | None,
    candidate_scorer: Callable[
        [tuple[str, ...], list[Unit]], list[float]
    ]
    | None = None,
    candidate_score_weight: float = 0.0,
    candidate_reranker: Callable[
        [tuple[str, str, str], float, list[Unit]], dict[str, Any]
    ]
    | None = None,
    maximum_candidates: int = 6,
) -> list[tuple[Unit, bool, float, float | None, dict[str, Any] | None]]:
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

    content_scores: dict[Unit, float] = {}
    if candidate_scorer is not None and not exact:
        raw_scores = [float(value) for value in candidate_scorer(normalized, candidates)]
        if len(raw_scores) != len(candidates):
            raise ValueError("candidate_scorer must return one score per unit")
        if not all(math.isfinite(value) for value in raw_scores):
            raise ValueError("candidate_scorer returned a non-finite score")
        mean = float(np.mean(raw_scores))
        std = float(np.std(raw_scores))
        normalized_scores = (
            [(value - mean) / std for value in raw_scores]
            if std > 1e-8
            else [0.0] * len(raw_scores)
        )
        content_scores = dict(zip(candidates, normalized_scores))

    hierarchical_scores: dict[Unit, dict[str, Any]] = {}
    hierarchical_applied = False
    hierarchical_reason = "not_requested"
    if (
        candidate_reranker is not None
        and not exact
        and len(normalized) == 1
        and target_context is not None
    ):
        payload = candidate_reranker(
            target_context,
            float(target_duration),
            candidates,
        )
        if not isinstance(payload, dict):
            raise ValueError("candidate_reranker must return a dictionary")
        rows = list(payload.get("scores") or [])
        if len(rows) != len(candidates):
            raise ValueError("candidate_reranker must return one score per unit")
        hierarchical_reason = str(payload.get("reason") or "unspecified")
        hierarchical_applied = bool(payload.get("apply"))
        for unit, row in zip(candidates, rows):
            if row is None:
                continue
            if not isinstance(row, dict):
                raise TypeError("candidate_reranker score rows must be dictionaries")
            values = {
                name: float(row[name])
                for name in ("semantic", "context", "duration", "joint")
            }
            if not all(
                math.isfinite(value) and 0.0 <= value <= 1.0
                for value in values.values()
            ):
                raise ValueError("candidate_reranker returned an invalid bounded score")
            hierarchical_scores[unit] = {
                **values,
                "eligible": bool(row.get("eligible", True)),
                "applied": hierarchical_applied,
                "reason": hierarchical_reason,
            }
        if hierarchical_applied and not any(
            row["eligible"] for row in hierarchical_scores.values()
        ):
            raise ValueError("candidate_reranker applied without an eligible donor")

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
            + float(candidate_score_weight) * content_scores.get(unit, 0.0)
        )
        return value, unit.record_id, -unit.start

    if hierarchical_applied:
        # Semantic compatibility is a hard gate.  The bounded V216 joint score
        # (semantic/context/duration = 0.70/0.20/0.10) is the primary reranker;
        # V80's native lexical/duration score remains only a deterministic
        # tiebreaker.  If the scorer cannot establish confidence, callers set
        # ``apply=False`` and this path is byte-for-byte the V80 ordering.
        ranked = sorted(
            (
                unit
                for unit in candidates
                if hierarchical_scores.get(unit, {}).get("eligible", False)
            ),
            key=lambda unit: (
                hierarchical_scores[unit]["joint"],
                hierarchical_scores[unit]["semantic"],
                score(unit),
            ),
            reverse=True,
        )
    else:
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
            content_scores.get(unit) if content_scores else None,
            hierarchical_scores.get(unit),
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
    path_scorer: Callable[[list[dict[str, Any]]], dict[str, Any]]
    | None = None,
    path_score_weight: float = 0.35,
    candidate_scorer: Callable[
        [tuple[str, ...], list[Unit]], list[float]
    ]
    | None = None,
    candidate_score_weight: float = 0.0,
    candidate_reranker: Callable[
        [tuple[str, str, str], float, list[Unit]], dict[str, Any]
    ]
    | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Beam-search a complete native-duration path with long exact phrases."""
    if len(target_words) < 8:
        raise ValueError("Native waveform synthesis requires eight words")
    if candidate_score_weight < 0.0:
        raise ValueError("candidate_score_weight must be non-negative")
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
                target_context = None
                if size == 1:
                    centre = str(group[0]["normalized"])
                    left = (
                        str(target_words[position - 1]["normalized"])
                        if position > 0
                        else centre
                    )
                    right = (
                        str(target_words[position + 1]["normalized"])
                        if position + 1 < word_count
                        else centre
                    )
                    target_context = (left, centre, right)
                candidates = _rank_candidates(
                    normalized,
                    target_context=target_context,
                    exact_index=exact_index,
                    word_units=word_units,
                    statistics=statistics,
                    preferred_songs=preferred_set,
                    current_song=state.last_song,
                    candidate_scorer=candidate_scorer,
                    candidate_score_weight=candidate_score_weight,
                    candidate_reranker=candidate_reranker,
                )
                for (
                    unit,
                    exact,
                    similarity,
                    content_score,
                    hierarchical_score,
                ) in candidates:
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
                        "content_candidate_score": content_score,
                        "hierarchical_candidate_score": hierarchical_score,
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
    base_scores = [
        item.score - 0.45 * abs(item.duration - 15.2)
        for item in complete
    ]
    score_payloads: list[dict[str, Any]] = []
    adjusted_scores = list(base_scores)
    compatibility_values: list[float] = []
    compatibility_mean = 0.0
    compatibility_std = 0.0
    if path_scorer is not None:
        if path_score_weight < 0.0:
            raise ValueError("path_score_weight must be non-negative")
        score_payloads = [path_scorer(item.choices) for item in complete]
        compatibility_values = [
            float(payload["score"]) for payload in score_payloads
        ]
        if not all(math.isfinite(value) for value in compatibility_values):
            raise ValueError("path_scorer returned a non-finite score")
        compatibility_mean = float(np.mean(compatibility_values))
        compatibility_std = float(np.std(compatibility_values))
        if compatibility_std > 1e-8:
            adjusted_scores = [
                base
                + float(path_score_weight)
                * float(np.clip(
                    (compatibility - compatibility_mean)
                    / compatibility_std,
                    -3.0,
                    3.0,
                ))
                for base, compatibility in zip(
                    base_scores,
                    compatibility_values,
                )
            ]
    selected_index = max(
        range(len(complete)),
        key=lambda index: (
            adjusted_scores[index],
            -len(complete[index].choices),
        ),
    )
    selected = complete[selected_index]
    choices = selected.choices
    selected_score_payload: dict[str, Any] = {}
    if score_payloads:
        selected_score_payload = score_payloads[selected_index]
        transition_scores = list(
            selected_score_payload.get("transition_scores") or []
        )
        if transition_scores and len(transition_scores) != len(choices):
            raise ValueError(
                "path_scorer transition_scores must align with choices"
            )
        for index, transition_score in enumerate(transition_scores):
            choices[index]["transition_compatibility_score"] = (
                None
                if transition_score is None
                else float(transition_score)
            )
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
        "path_score_after_compatibility": adjusted_scores[selected_index],
        "compatibility_scorer_used": path_scorer is not None,
        "content_candidate_scorer_used": candidate_scorer is not None,
        "content_candidate_score_weight": (
            float(candidate_score_weight)
            if candidate_scorer is not None
            else 0.0
        ),
        "content_scored_selected_groups": sum(
            item.get("content_candidate_score") is not None
            for item in choices
        ),
        "hierarchical_candidate_reranker_used": candidate_reranker is not None,
        "hierarchical_reranked_selected_groups": sum(
            bool((item.get("hierarchical_candidate_score") or {}).get("applied"))
            for item in choices
        ),
        "hierarchical_fallback_selected_groups": sum(
            item.get("hierarchical_candidate_score") is not None
            and not bool(item["hierarchical_candidate_score"].get("applied"))
            for item in choices
        ),
        "compatibility_score_weight": (
            float(path_score_weight) if path_scorer is not None else 0.0
        ),
        "compatibility_score": (
            float(compatibility_values[selected_index])
            if compatibility_values
            else None
        ),
        "compatibility_candidate_mean": (
            compatibility_mean if compatibility_values else None
        ),
        "compatibility_candidate_std": (
            compatibility_std if compatibility_values else None
        ),
        "compatibility_evaluated_transitions": int(
            selected_score_payload.get("evaluated_transition_count") or 0
        ),
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
    unit_transform: Callable[
        [dict[str, Any], torch.Tensor], torch.Tensor
    ]
    | None = None,
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
        if unit_transform is not None:
            transformed = torch.as_tensor(
                unit_transform(choice, rendered.clone())
            ).detach().float().flatten()
            if transformed.shape != rendered.shape:
                raise ValueError(
                    "unit_transform must preserve the donor duration."
                )
            if not bool(torch.isfinite(transformed).all()):
                raise ValueError("unit_transform returned non-finite audio.")
            rendered = transformed
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
                "content_candidate_score": choice.get(
                    "content_candidate_score"
                ),
                "transition_compatibility_score": choice.get(
                "transition_compatibility_score"
            ),
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
