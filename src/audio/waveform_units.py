"""Core aligned-waveform units used by native singing synthesis."""

from __future__ import annotations

import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from statistics import median
from typing import Any

SAMPLE_RATE = 24_000


def normalize_word(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    return "".join(
        character
        for character in normalized
        if character.isalnum()
        or unicodedata.category(character).startswith("L")
    )


def accentless(text: str) -> str:
    text = normalize_word(text).replace("đ", "d")
    return "".join(
        character
        for character in unicodedata.normalize("NFD", text)
        if unicodedata.category(character) != "Mn"
    )


def similarity(left: str, right: str) -> float:
    """Accent-aware lexical similarity used only as retrieval fallback."""
    if left == right:
        return 1.0
    plain_left = accentless(left)
    plain_right = accentless(right)
    return max(
        SequenceMatcher(None, left, right).ratio(),
        0.92 * SequenceMatcher(
            None,
            plain_left,
            plain_right,
        ).ratio(),
    )


# Backward-compatible internal alias for the previously versioned pipeline.
_similarity = similarity


@dataclass(frozen=True)
class Unit:
    song_id: str
    record_id: str
    waveform_path: str
    words: tuple[str, ...]
    normalized_words: tuple[str, ...]
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(1e-3, self.end - self.start)


def duration_statistics(units: list[Unit]) -> dict[str, Any]:
    word_units = [unit for unit in units if len(unit.words) == 1]
    if not word_units:
        raise ValueError("Waveform inventory has no single-word units")
    by_word: dict[str, list[float]] = defaultdict(list)
    by_length: dict[int, list[float]] = defaultdict(list)
    for unit in word_units:
        duration = min(1.25, max(0.12, unit.duration))
        token = unit.normalized_words[0]
        by_word[token].append(duration)
        by_length[min(8, len(accentless(token)))].append(duration)
    global_duration = median(
        value
        for values in by_word.values()
        for value in values
    )
    return {
        "global": float(global_duration),
        "by_word": {
            key: float(median(values))
            for key, values in by_word.items()
        },
        "by_length": {
            int(key): float(median(values))
            for key, values in by_length.items()
        },
    }
