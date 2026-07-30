"""Generate a connected native Vietnamese full mix from user lyrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.audio import native_waveform as engine

STATE_NAME = "master_raw_connected_pacing_v80_state.json"
SAME_LINE_PAUSE_CAP = 0.025
LINE_BREAK_PAUSE_CAP = 2.0


def select_line_locked_path(
    target_words: list[dict[str, Any]],
    units: list[Any],
    exact_index: dict[tuple[str, ...], list[Any]],
    statistics: dict[str, Any],
    **kwargs: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select donor phrases without crossing target lyric-line boundaries."""
    return engine.select_long_phrase_path(
        target_words,
        units,
        exact_index,
        statistics,
        respect_segment_boundaries=True,
        **kwargs,
    )


def fill_connected_line_gaps(
    choices: list[dict[str, Any]],
    **kwargs: Any,
) -> dict[str, Any]:
    """Cap within-line joins at 25 ms and reserve rests for line breaks."""
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
    pacing = engine.fill_natural_path_gaps(
        choices,
        same_line_pause_cap=SAME_LINE_PAUSE_CAP,
        line_break_pause_cap=LINE_BREAK_PAUSE_CAP,
        **kwargs,
    )
    pacing["connected_same_line_policy"] = True
    return pacing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--genre", default="")
    parser.add_argument("--duration", type=float, default=16.0)
    parser.add_argument("--backing-ratio", type=float, default=0.45)
    parser.add_argument(
        "--presence-attenuation-db",
        type=float,
        default=10.0,
    )
    parser.add_argument("--expected-records", type=int, default=2_048)
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    records = [
        json.loads(line)
        for line in (dataset / "records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    if len(records) != args.expected_records:
        raise RuntimeError(
            f"Expected {args.expected_records} waveform records, "
            f"got {len(records)}"
        )
    state = engine.generate_text_candidate(
        records,
        dataset,
        output_root,
        text=args.text.strip(),
        genre=args.genre.strip(),
        duration_seconds=float(args.duration),
        backing_ratio=float(args.backing_ratio),
        presence_attenuation_db=float(args.presence_attenuation_db),
        path_selector=select_line_locked_path,
        gap_filler=fill_connected_line_gaps,
    )
    state.update({
        "version": "V80",
        "same_line_pause_cap_seconds": SAME_LINE_PAUSE_CAP,
        "line_break_pause_cap_seconds": LINE_BREAK_PAUSE_CAP,
    })
    state_path = output_root / STATE_NAME
    engine.write_json(state_path, state)
    print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
