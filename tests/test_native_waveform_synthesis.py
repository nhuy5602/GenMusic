import numpy as np
import pytest
import torch

from src.audio.native_waveform import (
    build_long_inventory,
    fill_natural_path_gaps,
    render_long_phrase_path,
    select_long_phrase_path,
)
from src.audio.vocal_mix import span_duck_envelope
from src.audio.waveform_path import extract_source_unit, resolve_source_unit_bounds
from src.audio.waveform_units import Unit


def test_span_duck_envelope_only_reduces_novel_word_region() -> None:
    envelope = span_duck_envelope(
        24_000,
        24_000,
        [(0.40, 0.60)],
        attenuation_db=8.0,
        ramp_ms=50.0,
    )
    assert envelope.shape == (24_000,)
    assert np.isfinite(envelope).all()
    assert float(envelope[:6_000].min()) == 1.0
    assert float(envelope[11_000:13_000].max()) < 0.41
    assert float(envelope[-6_000:].min()) == 1.0


def test_source_unit_rejects_timestamp_outside_clipped_waveform() -> None:
    unit = Unit(
        song_id="song",
        record_id="record",
        waveform_path="waveform.pt",
        words=("mÆ°a",),
        normalized_words=("mÆ°a",),
        start=1.5,
        end=1.8,
    )
    with pytest.raises(
        RuntimeError,
        match="outside the available audio",
    ):
        extract_source_unit(torch.zeros(24_000), unit)


def test_source_unit_bounds_are_the_pre_fade_bounds_used_by_renderer() -> None:
    waveform = torch.zeros(24_000)
    waveform[5_100:6_900] = torch.linspace(-0.25, 0.25, 1_800)
    unit = Unit(
        song_id="song",
        record_id="record",
        waveform_path="waveform.pt",
        words=("mua",),
        normalized_words=("mua",),
        start=0.22,
        end=0.31,
    )
    start, end = resolve_source_unit_bounds(
        waveform,
        unit,
        zero_crossing_search_ms=10.0,
    )
    rendered = extract_source_unit(
        waveform,
        unit,
        zero_crossing_search_ms=10.0,
        fade_ms=0.0,
    )
    assert 0 <= start < end <= waveform.numel()
    assert rendered.numel() == end - start


def _record(song: str, words: list[str]) -> dict:
    return {
        "id": f"{song}_001",
        "song_id": song,
        "vocal_wav_path": f"waveforms/{song}_vocal.pt",
        "segments": [{
            "words": [
                {
                    "word": word,
                    "start": index * 0.4,
                    "end": index * 0.4 + 0.35,
                }
                for index, word in enumerate(words)
            ]
        }],
    }


def test_long_inventory_indexes_up_to_eight_words_and_excludes_heldout() -> None:
    words = ["một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
    units, index = build_long_inventory(
        [_record("train", words), _record("held", words)],
        heldout_song_ids={"held"},
    )
    assert units
    assert all(unit.song_id == "train" for unit in units)
    assert tuple(words[:8]) in index
    assert tuple(words[:9]) not in index
    assert len(index[tuple(words[:8])][0].words) == 8


def test_beam_prefers_complete_long_exact_phrase_path() -> None:
    words = [
        "một",
        "hai",
        "ba",
        "bốn",
        "năm",
        "sáu",
        "bảy",
        "tám",
        "chín",
        "mười",
    ]
    records = [_record("donor", words)]
    units, index = build_long_inventory(records, heldout_song_ids=set())
    target = [
        {
            "word": word,
            "normalized": word,
            "segment_index": 0,
        }
        for word in words
    ]
    statistics = {
        "global": 0.35,
        "by_word": {word: 0.35 for word in words},
        "by_length": {},
    }
    selected, diagnostics = select_long_phrase_path(
        target,
        units,
        index,
        statistics,
    )
    assert sum(len(item["target"]) for item in selected) == len(words)
    assert diagnostics["maximum_selected_phrase_words"] == 8
    assert diagnostics["exact_word_fraction"] == 1.0
    assert diagnostics["exact_long_phrase_word_fraction"] >= 0.8
    assert diagnostics["per_unit_time_stretch_used"] is False
    pacing = fill_natural_path_gaps(selected)
    assert pacing["span_after_gap_fill_seconds"] >= (
        pacing["span_before_gap_fill_seconds"]
    )
    assert pacing["per_unit_time_stretch_used"] is False


def test_beam_can_use_bounded_path_compatibility_tiebreaker() -> None:
    words = [f"word{index}" for index in range(10)]
    records = [_record("alpha", words), _record("zulu", words)]
    units, index = build_long_inventory(records, heldout_song_ids=set())
    target = [
        {"word": word, "normalized": word, "segment_index": 0}
        for word in words
    ]
    statistics = {
        "global": 0.35,
        "by_word": {word: 0.35 for word in words},
        "by_length": {},
    }

    def favor_alpha(choices):
        values = [
            None,
            *[
                1.0 if choice["unit"].song_id == "alpha" else -1.0
                for choice in choices[1:]
            ],
        ]
        return {
            "score": sum(value or 0.0 for value in values),
            "evaluated_transition_count": len(choices) - 1,
            "transition_scores": values,
        }

    selected, diagnostics = select_long_phrase_path(
        target,
        units,
        index,
        statistics,
        path_scorer=favor_alpha,
        path_score_weight=4.0,
    )
    assert diagnostics["compatibility_scorer_used"] is True
    assert diagnostics["compatibility_evaluated_transitions"] >= 1
    assert any(choice["unit"].song_id == "alpha" for choice in selected)
    assert any(
        choice.get("transition_compatibility_score") == 1.0
        for choice in selected
    )


def test_beam_can_use_target_free_content_scores_for_fuzzy_words() -> None:
    target_words = [f"novel{index}" for index in range(8)]
    donor_words = [f"novel{index}x" for index in range(8)]
    records = [
        _record("alpha", donor_words),
        _record("zulu", donor_words),
    ]
    units, index = build_long_inventory(records, heldout_song_ids=set())
    target = [
        {"word": word, "normalized": word, "segment_index": 0}
        for word in target_words
    ]
    statistics = {
        "global": 0.35,
        "by_word": {word: 0.35 for word in target_words},
        "by_length": {},
    }

    def favor_zulu(_normalized, candidates):
        return [
            1.0 if candidate.song_id == "zulu" else -1.0
            for candidate in candidates
        ]

    selected, diagnostics = select_long_phrase_path(
        target,
        units,
        index,
        statistics,
        candidate_scorer=favor_zulu,
        candidate_score_weight=4.0,
    )
    assert diagnostics["content_candidate_scorer_used"] is True
    assert diagnostics["content_scored_selected_groups"] == len(target_words)
    assert all(choice["unit"].song_id == "zulu" for choice in selected)


def test_unit_transform_is_duration_preserving_and_opt_in(tmp_path) -> None:
    waveform_dir = tmp_path / "waveforms"
    waveform_dir.mkdir()
    waveform = torch.linspace(-0.2, 0.2, 24_000)
    torch.save(waveform, waveform_dir / "donor_vocal.pt")
    record = _record("donor", ["mot", "hai"])
    record["vocal_wav_path"] = "waveforms/donor_vocal.pt"
    units, _ = build_long_inventory(
        [record],
        heldout_song_ids=set(),
    )
    unit = next(unit for unit in units if len(unit.words) == 1)
    choice = {
        "target": [
            {"word": "ba", "normalized": "ba", "segment_index": 0}
        ],
        "unit": unit,
        "exact": False,
        "similarity": 0.5,
        "gap_before": 0.0,
    }
    baseline, _ = render_long_phrase_path(
        tmp_path,
        [choice],
        duration_seconds=2.0,
    )
    transformed, _ = render_long_phrase_path(
        tmp_path,
        [choice],
        duration_seconds=2.0,
        unit_transform=lambda _choice, audio: torch.zeros_like(audio),
    )
    assert baseline.shape == transformed.shape
    assert float(baseline.abs().sum()) > 0.0
    assert float(transformed.abs().sum()) == 0.0
