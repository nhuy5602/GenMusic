from scripts.generation.generate_native import (
    LINE_BREAK_PAUSE_CAP,
    SAME_LINE_PAUSE_CAP,
    fill_connected_line_gaps,
)
from scripts.generation.run_kaggle_native_waveform import (
    DEFAULT_RAW_KERNEL_REF,
    PATCH_FILES,
    _kernel_code,
)
from src.audio.waveform_units import Unit


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


def test_v80_kaggle_bundle_is_goal_eligible_and_token_free() -> None:
    assert DEFAULT_RAW_KERNEL_REF.endswith("1785338959")
    assert (
        "scripts/generate_native_waveform.py"
        in PATCH_FILES
    )
    code = _kernel_code(
        patch_sha256="A" * 64,
        patch_tree_sha256="B" * 64,
        text="một chiều mưa tôi nhớ về con phố cũ",
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
