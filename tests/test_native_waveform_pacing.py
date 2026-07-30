from scripts.generate_native_waveform import (
    LINE_BREAK_PAUSE_CAP,
    SAME_LINE_PAUSE_CAP,
    connected_continuity_gate,
    fill_connected_line_gaps,
)
from src.audio.waveform_units import Unit
from scripts.run_kaggle_native_waveform import (
    DEFAULT_RAW_KERNEL_REF,
    PATCH_FILES,
    _kernel_code,
)


def _choice(index: int, segment: int, previous_segment: int | None) -> dict:
    return {
        "target": [{
            "word": str(index),
            "normalized": str(index),
            "segment_index": segment,
        }],
        "unit": Unit(
            song_id=f"donor-{index}",
            record_id=f"donor-{index}_001",
            waveform_path=f"waveforms/donor-{index}.pt",
            words=(str(index),),
            normalized_words=(str(index),),
            start=index * 0.4,
            end=index * 0.4 + 0.35,
        ),
        "gap_before": (
            0.0
            if previous_segment is None
            else 0.24
            if segment != previous_segment
            else 0.14
        ),
    }


def test_v80_tightens_same_line_joins_and_moves_time_to_line_rests() -> None:
    segments = (0, 0, 0, 1, 1, 2, 2)
    choices = [
        _choice(
            index,
            segment,
            None if index == 0 else segments[index - 1],
        )
        for index, segment in enumerate(segments)
    ]
    pacing = fill_connected_line_gaps(choices, duration_seconds=5.5)
    assert pacing["connected_same_line_policy"] is True
    assert pacing["maximum_same_line_gap_seconds"] <= (
        SAME_LINE_PAUSE_CAP + 1e-6
    )
    assert pacing["maximum_line_break_gap_seconds"] > (
        pacing["maximum_same_line_gap_seconds"]
    )
    assert pacing["maximum_line_break_gap_seconds"] <= (
        LINE_BREAK_PAUSE_CAP + 1e-6
    )
    assert pacing["line_break_boundary_count"] == 2
    assert pacing["same_line_boundary_count"] == 4
    assert pacing["per_unit_time_stretch_used"] is False


def _sample(index: int) -> dict:
    return {
        "song_id": f"prompt-{index}",
        "backing_song_id": f"backing-{index}",
        "generated_vocal_asr": {"word_accuracy": 0.54},
        "generated_full_mix_asr": {
            "word_accuracy": 0.47,
            "hypothesis": f"distinct Vietnamese hypothesis {index}",
        },
        "generated_vocal_acoustics": {"voiced_ratio": 0.72},
        "target_vocal_acoustics": {"voiced_ratio": 0.90},
        "generated_full_mix_acoustics": {
            "duration_seconds": 16.0,
            "clip_ratio": 0.0,
        },
        "retrieval": {
            "exact_word_fraction": 1.0,
            "mean_similarity": 1.0,
            "target_segment_spanning_groups": 0,
            "pacing": {
                "maximum_same_line_gap_seconds": 0.025,
                "line_break_boundary_count": 2,
            },
        },
    }


def test_v80_gate_requires_connected_joins_without_asr_regression() -> None:
    gate = connected_continuity_gate([
        _sample(0),
        _sample(1),
        _sample(2),
    ])
    assert gate["objective_pass"] is True
    assert gate["v80_connected_continuity_pass"] is True
    failed = [_sample(0), _sample(1), _sample(2)]
    failed[0]["retrieval"]["pacing"][
        "maximum_same_line_gap_seconds"
    ] = 0.14
    assert (
        connected_continuity_gate(failed)[
            "v80_connected_continuity_pass"
        ]
        is False
    )


def test_v80_kaggle_bundle_is_goal_eligible_and_token_free() -> None:
    assert DEFAULT_RAW_KERNEL_REF.endswith("1785338959")
    assert (
        "scripts/generate_native_waveform.py"
        in PATCH_FILES
    )
    code = _kernel_code(
        patch_sha256="A" * 64,
        patch_tree_sha256="B" * 64,
    )
    assert "STARTING_NATIVE_WAVEFORM_V80" in code
    assert (
        "scripts/generate_native_waveform.py"
        in code
    )
    assert (
        code.count(
            "'scripts/generate_native_waveform.py'"
        )
        == 2
    )
    assert (
        "'src/audio/native_waveform.py'"
        in code
    )
    assert "master_raw_multishard_v71_state.json" in code
    assert "KAGGLE_API_TOKEN" not in code
    assert "KGAT_" not in code
