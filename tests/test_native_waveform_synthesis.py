from src.audio.native_waveform import (
    build_long_inventory,
    fill_natural_path_gaps,
    goal_gate,
    select_long_phrase_path,
)
from src.audio.waveform_units import Unit


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
    words = "một hai ba bốn năm sáu bảy tám chín".split()
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
    words = "một hai ba bốn năm sáu bảy tám chín mười".split()
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


def _sample(vocal: float, mix: float, voiced: float = 0.8) -> dict:
    return {
        "song_id": "prompt",
        "backing_song_id": "backing",
        "generated_vocal_asr": {"word_accuracy": vocal},
        "generated_full_mix_asr": {
            "word_accuracy": mix,
            "hypothesis": f"unique hypothesis {vocal}",
        },
        "generated_vocal_acoustics": {"voiced_ratio": voiced},
        "target_vocal_acoustics": {"voiced_ratio": 0.9},
        "generated_full_mix_acoustics": {
            "duration_seconds": 16.0,
            "clip_ratio": 0.0,
        },
        "retrieval": {
            "exact_word_fraction": 1.0,
            "mean_similarity": 1.0,
        },
    }


def test_goal_gate_requires_real_full_mix_thresholds() -> None:
    samples = [_sample(0.50, 0.45), _sample(0.40, 0.35), _sample(0.35, 0.30)]
    gate = goal_gate(samples)
    assert gate["pilot_pass"] is True
    assert gate["objective_pass"] is True
    failed = [_sample(0.40, 0.20), _sample(0.35, 0.20), _sample(0.30, 0.20)]
    assert goal_gate(failed)["objective_pass"] is False
