from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from .text_utils import tokenize_words


VI_ONSETS = (
    "ngh",
    "ng",
    "gh",
    "gi",
    "qu",
    "kh",
    "ch",
    "ph",
    "th",
    "tr",
    "nh",
    "b",
    "c",
    "d",
    "g",
    "h",
    "k",
    "l",
    "m",
    "n",
    "p",
    "q",
    "r",
    "s",
    "t",
    "v",
    "x",
)
VIETNAMESE_VOWELS = frozenset("aeiouy")
DEFAULT_RHYME_PROFILE_PATH = Path(__file__).resolve().parents[2] / "models" / "rhyme_profile.json"


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    return stripped.replace("đ", "d").replace("Đ", "D")


def rhyme_key_word(word: str) -> str:
    normalized = re.sub(r"[^a-z]", "", strip_accents(word).lower())
    for onset in VI_ONSETS:
        if normalized.startswith(onset) and len(normalized) > len(onset):
            return normalized[len(onset) :]
    return normalized


def assonance_key_word(word: str) -> str:
    """Return a softer vowel-family key for natural near-rhyme detection."""
    normalized = rhyme_key_word(word)
    vowels = [char for char in normalized if char in VIETNAMESE_VOWELS]
    if vowels:
        return "".join(vowels[-2:])
    return normalized[-2:]


def word_at(line: str, index: int) -> str:
    words = tokenize_words(line)
    if not words:
        return ""
    if index < 0:
        index = len(words) + index
    if index < 0 or index >= len(words):
        return ""
    return words[index]


def end_rhyme_key(line: str) -> str:
    return rhyme_key_word(word_at(line, -1))


def first_rhyme_key(line: str) -> str:
    return rhyme_key_word(word_at(line, 0))


def sixth_rhyme_key(line: str) -> str:
    return rhyme_key_word(word_at(line, 5))


def rhyme_match(first: str, second: str) -> bool:
    first_key = rhyme_key_word(first)
    second_key = rhyme_key_word(second)
    return bool(first_key and first_key == second_key)


def assonance_match(first: str, second: str) -> bool:
    first_key = assonance_key_word(first)
    second_key = assonance_key_word(second)
    return bool(first_key and first_key == second_key)


def end_pair_rhyme_rate(lines: list[str]) -> float:
    pairs = [(lines[index], lines[index + 1]) for index in range(0, len(lines) - 1, 2)]
    if not pairs:
        return 1.0
    hits = sum(1 for first, second in pairs if end_rhyme_key(first) == end_rhyme_key(second))
    return hits / len(pairs)


def end_pair_assonance_rate(lines: list[str]) -> float:
    pairs = [(lines[index], lines[index + 1]) for index in range(0, len(lines) - 1, 2)]
    if not pairs:
        return 1.0
    hits = sum(
        1
        for first, second in pairs
        if assonance_key_word(word_at(first, -1)) == assonance_key_word(word_at(second, -1))
    )
    return hits / len(pairs)


def head_tail_rhyme_rate(lines: list[str]) -> float:
    links = [(lines[index], lines[index + 1]) for index in range(0, len(lines) - 1)]
    if not links:
        return 1.0
    hits = sum(1 for first, second in links if end_rhyme_key(first) == first_rhyme_key(second))
    return hits / len(links)


def luc_bat_rhyme_rate(lines: list[str]) -> float:
    hits = 0
    total = 0
    for index in range(0, len(lines) - 1, 2):
        luc = lines[index]
        bat = lines[index + 1]
        luc_count = len(tokenize_words(luc))
        bat_count = len(tokenize_words(bat))
        if 5 <= luc_count <= 7 and 7 <= bat_count <= 9:
            total += 1
            if end_rhyme_key(luc) == sixth_rhyme_key(bat):
                hits += 1
            if index + 2 < len(lines):
                total += 1
                if end_rhyme_key(bat) == end_rhyme_key(lines[index + 2]):
                    hits += 1
    if total == 0:
        return 0.0
    return hits / total


def vietnamese_rhyme_profile(lines: list[str]) -> dict[str, float]:
    cleaned = [line for line in lines if line.strip()]
    return {
        "end_pair": round(end_pair_rhyme_rate(cleaned), 4),
        "assonance": round(end_pair_assonance_rate(cleaned), 4),
        "head_tail": round(head_tail_rhyme_rate(cleaned), 4),
        "luc_bat": round(luc_bat_rhyme_rate(cleaned), 4),
    }


def dominant_rhyme_scheme(lines: list[str]) -> str:
    profile = vietnamese_rhyme_profile(lines)
    scheme, score = max(profile.items(), key=lambda item: item[1])
    if score <= 0:
        return "free_verse"
    return scheme


def vietnamese_rhyme_rate(lines: list[str]) -> float:
    profile = vietnamese_rhyme_profile(lines)
    return max(profile.values()) if profile else 1.0


def section_vietnamese_rhyme_rate(sections: list[list[str]]) -> float:
    scored = [vietnamese_rhyme_rate(section) for section in sections if len(section) >= 2]
    if not scored:
        return 1.0
    return round(sum(scored) / len(scored), 4)


def section_end_pair_rhyme_rate(sections: list[list[str]]) -> float:
    scored = [end_pair_rhyme_rate(section) for section in sections if len(section) >= 2]
    if not scored:
        return 1.0
    return round(sum(scored) / len(scored), 4)


def section_head_tail_rhyme_rate(sections: list[list[str]]) -> float:
    scored = [head_tail_rhyme_rate(section) for section in sections if len(section) >= 2]
    if not scored:
        return 1.0
    return round(sum(scored) / len(scored), 4)


def section_luc_bat_rhyme_rate(sections: list[list[str]]) -> float:
    scored = [luc_bat_rhyme_rate(section) for section in sections if len(section) >= 2]
    if not scored:
        return 0.0
    return round(sum(scored) / len(scored), 4)


def section_assonance_rate(sections: list[list[str]]) -> float:
    scored = [end_pair_assonance_rate(section) for section in sections if len(section) >= 2]
    if not scored:
        return 1.0
    return round(sum(scored) / len(scored), 4)


def natural_rhyme_score(lines: list[str]) -> float:
    """Score musical vowel cohesion without requiring every adjacent pair to rhyme."""
    cleaned = [line for line in lines if line.strip()]
    if len(cleaned) < 2:
        return 1.0
    profile = vietnamese_rhyme_profile(cleaned)
    return round(
        0.60 * profile["assonance"]
        + 0.20 * profile["head_tail"]
        + 0.20 * profile["luc_bat"],
        4,
    )


def learned_assonance_families(path: str | Path = DEFAULT_RHYME_PROFILE_PATH) -> list[str]:
    """Read learned vowel families as a soft prior for future lyric drafts."""
    profile_path = Path(path)
    if not profile_path.exists():
        return []
    try:
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return [
        str(item.get("key"))
        for item in payload.get("top_assonance_families", [])
        if isinstance(item, dict) and item.get("key")
    ]
