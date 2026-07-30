"""Line-locked native-duration pacing refinement for the V78 candidate.

V78 was the first goal-eligible path to pass the objective ASR/acoustic gate,
but its first held-out sample exposed a structural pacing defect: an exact
donor phrase could span two target lyric lines and the remaining duration was
then spread across same-line word boundaries.  V79 keeps the same held-out
songs, donor corpus, native source durations, and cross-song backing while:

* forbidding a selected phrase from crossing a target lyric-line boundary;
* capping pauses inside a lyric line at 140 ms;
* allocating remaining duration at lyric-line boundaries, capped at 1.8 s.

No target waveform or target timestamps enter synthesis.  V79 is accepted
only if the V78 objective gate still passes, ASR does not materially regress,
and all structural continuity checks pass.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from scripts import run_colab_master_raw_long_phrase_v78 as v78


STATE_NAME = "master_raw_line_pacing_v79_state.json"
BASELINE_V78_VOCAL_WA = 0.5202702702702703
BASELINE_V78_MIX_WA = 0.45990990990990993
BASELINE_V78_MAX_SAME_LINE_GAP = 0.476357142857143
SAME_LINE_PAUSE_CAP = 0.14
LINE_BREAK_PAUSE_CAP = 1.80

_select_long_phrase_path = v78.select_long_phrase_path
_fill_natural_path_gaps = v78.fill_natural_path_gaps


def select_line_locked_path(
    target_words: list[dict[str, Any]],
    units: list[Any],
    exact_index: dict[tuple[str, ...], list[Any]],
    statistics: dict[str, Any],
    **kwargs: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Use the V78 beam while preserving target lyric-line boundaries."""
    return _select_long_phrase_path(
        target_words,
        units,
        exact_index,
        statistics,
        respect_segment_boundaries=True,
        **kwargs,
    )


def fill_line_locked_gaps(
    choices: list[dict[str, Any]],
    **kwargs: Any,
) -> dict[str, Any]:
    """Keep phrases close inside a line and place rests between lines."""
    return _fill_natural_path_gaps(
        choices,
        same_line_pause_cap=SAME_LINE_PAUSE_CAP,
        line_break_pause_cap=LINE_BREAK_PAUSE_CAP,
        **kwargs,
    )


def continuity_gate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Require structural improvement without sacrificing V78 semantics."""
    base_gate = v78.goal_gate(samples)
    vocal_wa = float(np.mean([
        sample["generated_vocal_asr"]["word_accuracy"]
        for sample in samples
    ]))
    mix_wa = float(np.mean([
        sample["generated_full_mix_asr"]["word_accuracy"]
        for sample in samples
    ]))
    maximum_same_line_gap = max(
        float(sample["retrieval"]["pacing"][
            "maximum_same_line_gap_seconds"
        ])
        for sample in samples
    )
    maximum_spanning_groups = max(
        int(sample["retrieval"]["target_segment_spanning_groups"])
        for sample in samples
    )
    line_breaks = sum(
        int(sample["retrieval"]["pacing"]["line_break_boundary_count"])
        for sample in samples
    )
    continuity_pass = bool(
        base_gate["objective_pass"]
        and maximum_spanning_groups == 0
        and maximum_same_line_gap <= SAME_LINE_PAUSE_CAP + 1e-6
        and maximum_same_line_gap
        <= BASELINE_V78_MAX_SAME_LINE_GAP - 0.20
        and line_breaks >= 3
        and vocal_wa >= BASELINE_V78_VOCAL_WA - 0.01
        and mix_wa >= BASELINE_V78_MIX_WA - 0.01
        and all(
            sample["generated_full_mix_asr"]["word_accuracy"] >= 0.25
            for sample in samples
        )
    )
    return {
        **base_gate,
        "v79_continuity_pass": continuity_pass,
        "baseline_v78_mean_generated_vocal_word_accuracy": (
            BASELINE_V78_VOCAL_WA
        ),
        "baseline_v78_mean_generated_full_mix_word_accuracy": (
            BASELINE_V78_MIX_WA
        ),
        "vocal_word_accuracy_delta_vs_v78": (
            vocal_wa - BASELINE_V78_VOCAL_WA
        ),
        "full_mix_word_accuracy_delta_vs_v78": (
            mix_wa - BASELINE_V78_MIX_WA
        ),
        "baseline_v78_maximum_same_line_gap_seconds": (
            BASELINE_V78_MAX_SAME_LINE_GAP
        ),
        "maximum_same_line_gap_seconds": maximum_same_line_gap,
        "maximum_target_segment_spanning_groups": maximum_spanning_groups,
        "line_break_boundary_count": line_breaks,
    }


def _argument_path(flag: str) -> Path:
    try:
        value = sys.argv[sys.argv.index(flag) + 1]
    except (ValueError, IndexError) as error:
        raise RuntimeError(f"V79 missing required argument {flag}") from error
    return Path(value).resolve()


def main() -> None:
    v78.STATE_NAME = STATE_NAME
    v78.select_long_phrase_path = select_line_locked_path
    v78.fill_natural_path_gaps = fill_line_locked_gaps
    v78.main()

    state_path = _argument_path("--output-root") / STATE_NAME
    state = json.loads(state_path.read_text(encoding="utf-8"))
    samples = list(state.get("samples") or [])
    gate = continuity_gate(samples)
    state.update({
        "version": "V79",
        "status": (
            "continuity_candidate_passed"
            if gate["v79_continuity_pass"]
            else "continuity_candidate_failed"
        ),
        "design": (
            "V78 2048-song exact raw phrases with phrases locked to target "
            "lyric lines; native donor durations; <=140ms same-line gaps; "
            "<=1.8s line rests; cross-song sidechain backing"
        ),
        "target_audio_used_at_product_inference": False,
        "target_timing_used_at_product_inference": False,
        "per_unit_time_stretch_used": False,
        "same_line_pause_cap_seconds": SAME_LINE_PAUSE_CAP,
        "line_break_pause_cap_seconds": LINE_BREAK_PAUSE_CAP,
        "gate": gate,
    })
    v78._write_json(state_path, state)
    print("\n===== MASTER RAW LINE PACING V79 =====", flush=True)
    print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
