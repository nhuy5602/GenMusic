"""Connected same-line pacing refinement for the V79 full-mix candidate.

Human listening confirmed that V79 lyrics are intelligible but occasionally
sound interrupted.  The V79 state provides a direct cause: pauses inside one
lyric line may still reach 140 ms.  V80 keeps the exact same line-locked
retrieval, native donor pitch/duration, and cross-song backing, but reduces
same-line pauses to 25 ms and puts the saved musical time at lyric-line rests.

There is no training, target waveform/timing leakage, time stretch, or
pretrained TTS.  PhoWhisper remains evaluation-only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from scripts import run_colab_master_raw_long_phrase_v78 as v78
from scripts.run_colab_master_raw_line_pacing_v79 import (
    select_line_locked_path,
)


STATE_NAME = "master_raw_connected_pacing_v80_state.json"
BASELINE_V79_VOCAL_WA = 0.5369369369369369
BASELINE_V79_MIX_WA = 0.45990990990990993
BASELINE_V79_MAX_SAME_LINE_GAP = 0.14
SAME_LINE_PAUSE_CAP = 0.025
LINE_BREAK_PAUSE_CAP = 2.0

_fill_natural_path_gaps = v78.fill_natural_path_gaps


def fill_connected_line_gaps(
    choices: list[dict[str, Any]],
    **kwargs: Any,
) -> dict[str, Any]:
    """Tighten within-line joins before allocating rests between lines."""
    for index in range(1, len(choices)):
        previous_segment = int(
            choices[index - 1]["target"][-1]["segment_index"]
        )
        current_segment = int(
            choices[index]["target"][0]["segment_index"]
        )
        if previous_segment == current_segment:
            choices[index]["gap_before"] = min(
                float(choices[index]["gap_before"]),
                SAME_LINE_PAUSE_CAP,
            )
    pacing = _fill_natural_path_gaps(
        choices,
        same_line_pause_cap=SAME_LINE_PAUSE_CAP,
        line_break_pause_cap=LINE_BREAK_PAUSE_CAP,
        **kwargs,
    )
    pacing["connected_same_line_policy"] = True
    pacing["human_reported_issue"] = "occasional phrase interruption"
    return pacing


def connected_continuity_gate(
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Require less interruption while preserving V79 semantic quality."""
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
    connected_pass = bool(
        base_gate["objective_pass"]
        and maximum_spanning_groups == 0
        and maximum_same_line_gap <= SAME_LINE_PAUSE_CAP + 1e-6
        and maximum_same_line_gap
        <= BASELINE_V79_MAX_SAME_LINE_GAP - 0.10
        and line_breaks >= 3
        and vocal_wa >= BASELINE_V79_VOCAL_WA - 0.01
        and mix_wa >= BASELINE_V79_MIX_WA - 0.01
        and all(
            sample["generated_full_mix_asr"]["word_accuracy"] >= 0.25
            for sample in samples
        )
    )
    return {
        **base_gate,
        "v80_connected_continuity_pass": connected_pass,
        "baseline_v79_mean_generated_vocal_word_accuracy": (
            BASELINE_V79_VOCAL_WA
        ),
        "baseline_v79_mean_generated_full_mix_word_accuracy": (
            BASELINE_V79_MIX_WA
        ),
        "vocal_word_accuracy_delta_vs_v79": (
            vocal_wa - BASELINE_V79_VOCAL_WA
        ),
        "full_mix_word_accuracy_delta_vs_v79": (
            mix_wa - BASELINE_V79_MIX_WA
        ),
        "baseline_v79_maximum_same_line_gap_seconds": (
            BASELINE_V79_MAX_SAME_LINE_GAP
        ),
        "maximum_same_line_gap_seconds": maximum_same_line_gap,
        "maximum_target_segment_spanning_groups": maximum_spanning_groups,
        "line_break_boundary_count": line_breaks,
    }


def _argument_path(flag: str) -> Path:
    try:
        value = sys.argv[sys.argv.index(flag) + 1]
    except (ValueError, IndexError) as error:
        raise RuntimeError(f"V80 missing required argument {flag}") from error
    return Path(value).resolve()


def main() -> None:
    v78.STATE_NAME = STATE_NAME
    v78.select_long_phrase_path = select_line_locked_path
    v78.fill_natural_path_gaps = fill_connected_line_gaps
    v78.main()

    state_path = _argument_path("--output-root") / STATE_NAME
    state = json.loads(state_path.read_text(encoding="utf-8"))
    samples = list(state.get("samples") or [])
    gate = connected_continuity_gate(samples)
    state.update({
        "version": "V80",
        "status": (
            "connected_candidate_passed"
            if gate["v80_connected_continuity_pass"]
            else "connected_candidate_failed"
        ),
        "human_feedback": "lyrics clear; occasional interruption",
        "design": (
            "V79 line-locked 2048-song exact raw phrases with <=25ms "
            "same-line joins and musical time allocated to <=2.0s line "
            "rests; native donor pitch/duration; cross-song backing"
        ),
        "target_audio_used_at_product_inference": False,
        "target_timing_used_at_product_inference": False,
        "per_unit_time_stretch_used": False,
        "same_line_pause_cap_seconds": SAME_LINE_PAUSE_CAP,
        "line_break_pause_cap_seconds": LINE_BREAK_PAUSE_CAP,
        "gate": gate,
    })
    v78._write_json(state_path, state)
    print("\n===== MASTER RAW CONNECTED PACING V80 =====", flush=True)
    print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
