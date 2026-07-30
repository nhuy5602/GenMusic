"""Raw source-duration waveform path synthesis.

V64 proved that the raw corpus has high lexical coverage and healthy voicing,
but per-unit phase-vocoder stretching plus 14--20 donor switches destroyed
spectral structure and intelligibility.  V65 changes both causal factors:

* choose a small donor-song set with greedy lexical set cover;
* prefer exact contiguous phrases from that set;
* keep every selected waveform unit at its original duration and pitch;
* trim only to nearby zero crossings and use short boundary fades.

Held-out vocal waveforms remain evaluation-only.  Matching backing is used for
this pilot and still requires a later arbitrary-backing audit.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data.aligned_raw import (
    mix_raw_stems,
    select_word_window,
)
from src.audio.waveform_units import (
    SAMPLE_RATE,
    Unit,
    _similarity,
    build_inventory,
    duration_statistics,
    words_in_window,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def donor_set_cover(
    target_words: list[dict[str, Any]],
    units: list[Unit],
    *,
    maximum_songs: int = 5,
) -> list[str]:
    """Choose a small donor set maximizing exact target-token coverage."""
    required = {
        (index, str(item["normalized"]))
        for index, item in enumerate(target_words)
    }
    coverage: dict[str, set[tuple[int, str]]] = defaultdict(set)
    for unit in units:
        if len(unit.normalized_words) != 1:
            continue
        token = unit.normalized_words[0]
        for indexed in required:
            if indexed[1] == token:
                coverage[unit.song_id].add(indexed)
    uncovered = set(required)
    selected = []
    while uncovered and len(selected) < maximum_songs:
        candidates = [
            (
                len(values & uncovered),
                len(values),
                song_id,
            )
            for song_id, values in coverage.items()
            if song_id not in selected
        ]
        if not candidates:
            break
        gain, _, song_id = max(candidates)
        if gain <= 0:
            break
        selected.append(song_id)
        uncovered -= coverage[song_id]
    return selected


def _unit_score(
    unit: Unit,
    target_words: tuple[str, ...],
    target_duration: float,
    *,
    preferred_songs: set[str],
    current_song: str | None,
    usage: Counter[tuple[str, float, float]],
) -> float:
    similarity = float(np.mean([
        _similarity(left, right)
        for left, right in zip(target_words, unit.normalized_words)
    ]))
    duration_ratio = max(1e-3, unit.duration / max(1e-3, target_duration))
    duration_error = abs(float(np.log(duration_ratio)))
    key = (unit.record_id, unit.start, unit.end)
    return (
        4.5 * float(unit.normalized_words == target_words)
        + 1.8 * similarity
        + 1.25 * float(unit.song_id in preferred_songs)
        + 1.65 * float(unit.song_id == current_song)
        + 0.35 * (len(unit.words) - 1)
        - 1.35 * duration_error
        - 0.35 * usage[key]
    )


def select_source_path(
    target_words: list[dict[str, Any]],
    units: list[Unit],
    exact_index: dict[tuple[str, ...], list[Unit]],
    statistics: dict[str, Any],
    *,
    maximum_preferred_songs: int = 5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select unmodified source units with few donor-song switches."""
    preferred = donor_set_cover(
        target_words,
        units,
        maximum_songs=maximum_preferred_songs,
    )
    preferred_set = set(preferred)
    word_units = [unit for unit in units if len(unit.words) == 1]
    by_word = statistics["by_word"]
    global_duration = float(statistics["global"])
    line_breaks = sum(
        int(target_words[index]["segment_index"])
        != int(target_words[index + 1]["segment_index"])
        for index in range(len(target_words) - 1)
    )
    estimated_gap_seconds = (
        0.055 * max(0, len(target_words) - 1 - line_breaks)
        + 0.28 * line_breaks
    )
    budget_per_word = max(
        0.28,
        (15.35 - 0.28 - estimated_gap_seconds)
        / max(1, len(target_words)),
    )
    usage: Counter[tuple[str, float, float]] = Counter()
    selected = []
    current_song = preferred[0] if preferred else None
    cursor = 0
    while cursor < len(target_words):
        choice = None
        for size in (3, 2, 1):
            group = target_words[cursor : cursor + size]
            if len(group) != size:
                continue
            normalized = tuple(str(item["normalized"]) for item in group)
            exact = list(exact_index.get(normalized) or [])
            if size > 1 and not exact:
                continue
            if exact:
                preferred_exact = [
                    unit for unit in exact if unit.song_id in preferred_set
                ]
                current_exact = [
                    unit for unit in exact if unit.song_id == current_song
                ]
                candidates = current_exact or preferred_exact or exact
            else:
                preferred_fuzzy = [
                    unit
                    for unit in word_units
                    if unit.song_id in preferred_set
                ]
                candidates = preferred_fuzzy or word_units
            donor_median_duration = sum(
                float(by_word.get(word, global_duration))
                for word in normalized
            )
            target_duration = min(
                donor_median_duration,
                budget_per_word * size,
            )
            candidates = [
                unit
                for unit in candidates
                if len(unit.words) == size
                and (
                    size > 1
                    or unit.duration <= max(1.05, target_duration * 1.9)
                )
            ] or candidates
            if size == 1 and not exact:
                # Keep the fuzzy search bounded to plausible phones and donor
                # set; lexical similarity still dominates the final score.
                ranked = sorted(
                    candidates,
                    key=lambda unit: _similarity(
                        normalized[0],
                        unit.normalized_words[0],
                    ),
                    reverse=True,
                )
                candidates = ranked[:96]
            if not candidates:
                continue
            unit = max(
                candidates,
                key=lambda candidate: _unit_score(
                    candidate,
                    normalized,
                    target_duration,
                    preferred_songs=preferred_set,
                    current_song=current_song,
                    usage=usage,
                ),
            )
            similarity = float(np.mean([
                _similarity(left, right)
                for left, right in zip(normalized, unit.normalized_words)
            ]))
            choice = {
                "size": size,
                "target": group,
                "unit": unit,
                "exact": unit.normalized_words == normalized,
                "similarity": similarity,
            }
            break
        if choice is None:
            raise RuntimeError("V65 could not select a waveform unit")
        unit = choice["unit"]
        key = (unit.record_id, unit.start, unit.end)
        usage[key] += 1
        current_song = unit.song_id
        selected.append(choice)
        cursor += int(choice["size"])

    switches = sum(
        selected[index - 1]["unit"].song_id
        != selected[index]["unit"].song_id
        for index in range(1, len(selected))
    )
    return selected, {
        "preferred_donor_songs": preferred,
        "preferred_donor_song_count": len(preferred),
        "selected_donor_song_count": len({
            item["unit"].song_id for item in selected
        }),
        "donor_switches": switches,
        "budget_per_word_seconds": budget_per_word,
    }


def extract_source_unit(
    waveform: torch.Tensor,
    unit: Unit,
    *,
    zero_crossing_search_ms: float = 12.0,
    fade_ms: float = 8.0,
) -> torch.Tensor:
    """Extract at native duration; adjust only to nearby quiet zero crossings."""
    source = waveform.detach().float().flatten()
    start = max(0, round(unit.start * SAMPLE_RATE))
    end = min(source.numel(), round(unit.end * SAMPLE_RATE))
    search = max(1, round(zero_crossing_search_ms * SAMPLE_RATE / 1000.0))

    def quiet_index(center: int, lower: int, upper: int) -> int:
        left = max(lower, center - search)
        right = min(upper, center + search + 1)
        if right <= left:
            return center
        return left + int(source[left:right].abs().argmin())

    start = quiet_index(start, 0, max(start + 1, end - 64))
    end = quiet_index(end, min(end - 1, start + 64), source.numel())
    rendered = source[start:end].clone()
    if rendered.numel() < 64:
        raise RuntimeError(f"V65 unit too short: {unit}")
    rendered -= rendered.mean()
    fade = min(
        round(fade_ms * SAMPLE_RATE / 1000.0),
        max(1, rendered.numel() // 6),
    )
    if fade > 1:
        rendered[:fade] *= torch.linspace(0.0, 1.0, fade)
        rendered[-fade:] *= torch.linspace(1.0, 0.0, fade)
    return rendered


def estimate_source_path_duration(selected: list[dict[str, Any]]) -> float:
    if not selected:
        return 0.0
    duration = 0.28 + sum(
        float(item["unit"].duration) for item in selected
    )
    for index in range(1, len(selected)):
        previous_segment = int(
            selected[index - 1]["target"][-1]["segment_index"]
        )
        current_segment = int(
            selected[index]["target"][0]["segment_index"]
        )
        duration += 0.28 if current_segment != previous_segment else 0.055
    return duration


def render_source_path(
    dataset: Path,
    selected: list[dict[str, Any]],
    *,
    duration_seconds: float = 16.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    output = torch.zeros(round(duration_seconds * SAMPLE_RATE))
    cache: dict[str, torch.Tensor] = {}
    groups = []
    cursor_seconds = 0.28
    previous_segment = None
    running_rms = None
    for item in selected:
        group = item["target"]
        unit: Unit = item["unit"]
        segment = int(group[0]["segment_index"])
        if previous_segment is not None:
            cursor_seconds += 0.28 if segment != previous_segment else 0.055
        previous_segment = int(group[-1]["segment_index"])
        if unit.waveform_path not in cache:
            cache[unit.waveform_path] = torch.load(
                dataset / unit.waveform_path,
                map_location="cpu",
                weights_only=True,
            ).float()
        rendered = extract_source_unit(cache[unit.waveform_path], unit)
        rms = float(rendered.square().mean().sqrt().clamp_min(1e-5))
        if running_rms is None:
            running_rms = min(0.11, max(0.055, rms))
        gain = min(1.45, max(0.70, running_rms / rms))
        rendered *= gain
        running_rms = 0.85 * running_rms + 0.15 * float(
            rendered.square().mean().sqrt()
        )
        start = round(cursor_seconds * SAMPLE_RATE)
        end = min(output.numel(), start + rendered.numel())
        if end <= start:
            break
        output[start:end] += rendered[: end - start]
        groups.append({
            "target_words": [str(word["word"]) for word in group],
            "donor_words": list(unit.words),
            "donor_song_id": unit.song_id,
            "donor_record_id": unit.record_id,
            "source_duration": unit.duration,
            "rendered_duration": rendered.numel() / SAMPLE_RATE,
            "target_start": cursor_seconds,
            "target_end": end / SAMPLE_RATE,
            "exact": bool(item["exact"]),
            "similarity": float(item["similarity"]),
        })
        cursor_seconds = end / SAMPLE_RATE
    peak = output.abs().max()
    if float(peak) > 0.97:
        output *= 0.97 / peak
    target_word_count = sum(len(item["target"]) for item in selected)
    exact_words = sum(
        len(item["target"]) for item in selected if item["exact"]
    )
    phrase_words = sum(
        len(item["target"])
        for item in selected
        if item["exact"] and len(item["target"]) > 1
    )
    return output, {
        "groups": groups,
        "rendered_span_seconds": min(duration_seconds, cursor_seconds),
        "exact_word_fraction": exact_words / max(1, target_word_count),
        "exact_phrase_word_fraction": phrase_words / max(1, target_word_count),
        "mean_similarity": float(np.mean([
            item["similarity"] for item in selected
        ])),
        "selected_donor_song_count": len({
            item["unit"].song_id for item in selected
        }),
        "donor_switches": sum(
            selected[index - 1]["unit"].song_id
            != selected[index]["unit"].song_id
            for index in range(1, len(selected))
        ),
        "source_duration_preserved": True,
        "per_unit_time_stretch_used": False,
    }


def pilot_gate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    vocal_wa = float(np.mean([
        sample["generated_vocal_asr"]["word_accuracy"] for sample in samples
    ]))
    mix_wa = float(np.mean([
        sample["generated_full_mix_asr"]["word_accuracy"] for sample in samples
    ]))
    voiced_relative = float(np.mean([
        sample["generated_vocal_acoustics"]["voiced_ratio"]
        / max(1e-6, sample["target_vocal_acoustics"]["voiced_ratio"])
        for sample in samples
    ]))
    mean_flatness = float(np.mean([
        sample["generated_vocal_acoustics"]["spectral_flatness"]
        for sample in samples
    ]))
    hypotheses = {
        sample["generated_full_mix_asr"]["hypothesis"].strip().casefold()
        for sample in samples
        if sample["generated_full_mix_asr"]["hypothesis"].strip()
    }
    # V64 baseline: vocal .12309, mix .14524, flatness about .1304.
    substantive_gain = vocal_wa - 0.12309368191721133
    mix_gain = mix_wa - 0.1452432824981845
    pass_value = bool(
        len(samples) == 3
        and substantive_gain >= 0.10
        and mix_gain >= 0.10
        and vocal_wa >= 0.25
        and mix_wa >= 0.25
        and sum(
            sample["generated_full_mix_asr"]["word_accuracy"] >= 0.20
            for sample in samples
        ) >= 2
        and voiced_relative >= 0.70
        and mean_flatness <= 0.065
        and len(hypotheses) >= 2
        and all(
            sample["generated_full_mix_acoustics"]["duration_seconds"] >= 15.5
            and sample["generated_full_mix_acoustics"]["clip_ratio"] <= 0.01
            for sample in samples
        )
    )
    return {
        "pass": pass_value,
        "mean_generated_vocal_word_accuracy": vocal_wa,
        "mean_generated_full_mix_word_accuracy": mix_wa,
        "vocal_word_accuracy_gain_over_v64": substantive_gain,
        "full_mix_word_accuracy_gain_over_v64": mix_gain,
        "mean_voiced_relative_to_target": voiced_relative,
        "mean_generated_vocal_flatness": mean_flatness,
        "distinct_full_mix_hypotheses": len(hypotheses),
        "samples_full_mix_wa_at_least_0p20": sum(
            sample["generated_full_mix_asr"]["word_accuracy"] >= 0.20
            for sample in samples
        ),
        "actual_goal_matching_backing_pass": bool(
            mix_wa >= 0.35
            and sum(
                sample["generated_full_mix_asr"]["word_accuracy"] >= 0.25
                for sample in samples
            ) >= 2
            and voiced_relative >= 0.70
            and len(hypotheses) >= 2
        ),
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
    state_path = output_root / "master_waveform_path_v65_state.json"
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
        raise RuntimeError("V65 lacks held-out 16-second lyric windows")
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
            "greedy donor-song set cover + exact longest phrase path + "
            "unmodified source-duration waveform units; no unit time stretch"
        ),
        "records": len(records),
        "heldout_song_ids": sorted(heldout_song_ids),
        "unit_count": len(units),
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
        selected, path_summary = select_source_path(
            words,
            units,
            exact_index,
            statistics,
        )
        planned_span = estimate_source_path_duration(selected)
        render_duration = min(
            20.0,
            max(float(args.duration), planned_span + 0.20),
        )
        generated_vocal, retrieval = render_source_path(
            dataset,
            selected,
            duration_seconds=render_duration,
        )
        retrieval.update(path_summary)
        retrieval["planned_span_seconds"] = planned_span
        retrieval["render_duration_seconds"] = render_duration
        reference_start = round(
            float(window["start_seconds"]) * SAMPLE_RATE
        )
        length = round(render_duration * SAMPLE_RATE)
        target_vocal = torch.load(
            dataset / str(record["vocal_wav_path"]),
            map_location="cpu",
            weights_only=True,
        ).float()[reference_start : reference_start + length]
        target_backing = torch.load(
            dataset / str(record["backing_wav_path"]),
            map_location="cpu",
            weights_only=True,
        ).float()[reference_start : reference_start + length]
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
