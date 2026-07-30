from src.audio.native_waveform import (
    build_long_inventory,
    fill_natural_path_gaps,
    select_long_phrase_path,
)


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
