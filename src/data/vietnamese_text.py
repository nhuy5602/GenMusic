"""Text normalization utilities for Vietnamese lyric data.

The normalizer is deliberately deterministic and keeps section boundaries. It
does not invent lyric content; the caller remains responsible for provenance.
"""

from __future__ import annotations

import re
import unicodedata


_ABBREVIATIONS = {
    "ko": "không",
    "kh": "không",
    "k": "không",
    "dc": "được",
    "đc": "được",
    "ntn": "như thế nào",
    "vs": "với",
    "mk": "mình",
    "mn": "mọi người",
    "bt": "biết",
    "bn": "bao nhiêu",
}

_ONES = (
    "không",
    "một",
    "hai",
    "ba",
    "bốn",
    "năm",
    "sáu",
    "bảy",
    "tám",
    "chín",
)


def _read_two_digits(value: int) -> str:
    tens, ones = divmod(value, 10)
    if tens == 0:
        return _ONES[ones]
    if tens == 1:
        if ones == 0:
            return "mười"
        return f"mười {_ONES[ones]}"
    result = f"{_ONES[tens]} mươi"
    if ones == 0:
        return result
    if ones == 1:
        return f"{result} mốt"
    if ones == 5:
        return f"{result} lăm"
    return f"{result} {_ONES[ones]}"


def vietnamese_number(value: int) -> str:
    """Expand an integer into a compact Vietnamese spoken form."""

    if value < 0:
        return f"âm {vietnamese_number(-value)}"
    if value < 10:
        return _ONES[value]
    if value < 100:
        return _read_two_digits(value)
    if value < 1000:
        hundreds, remainder = divmod(value, 100)
        result = f"{_ONES[hundreds]} trăm"
        if remainder:
            if remainder < 10:
                return f"{result} lẻ {_ONES[remainder]}"
            return f"{result} {_read_two_digits(remainder)}"
        return result
    if value < 1_000_000:
        thousands, remainder = divmod(value, 1000)
        result = f"{vietnamese_number(thousands)} nghìn"
        if remainder:
            return f"{result} {vietnamese_number(remainder)}"
        return result
    return str(value)


def _replace_number(match: re.Match[str]) -> str:
    raw = match.group(0)
    try:
        return vietnamese_number(int(raw))
    except ValueError:
        return raw


def _replace_abbreviation(match: re.Match[str]) -> str:
    token = match.group(0)
    replacement = _ABBREVIATIONS.get(token.casefold())
    if replacement is None:
        return token
    if token[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def normalize_vietnamese_lyrics(text: str) -> str:
    """Normalize Unicode, numbers, abbreviations and punctuation spacing.

    Bracketed section labels such as ``[Chorus]`` are preserved because they
    are useful conditioning signals for a lyric-aware music model.
    """

    if not isinstance(text, str):
        raise TypeError("lyrics must be a string")
    value = unicodedata.normalize("NFKC", text)
    value = value.replace("\u2018", "'").replace("\u2019", "'")
    value = value.replace("\u201c", '"').replace("\u201d", '"')
    value = value.replace("\u2013", "-").replace("\u2014", "-")
    value = re.sub(r"\b\d+\b", _replace_number, value)
    value = re.sub(r"(?<![\wÀ-ỹĐđ])(?:[A-Za-zÀ-ỹĐđ]+)(?![\wÀ-ỹĐđ])", _replace_abbreviation, value)
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"[ \t]*\n[ \t]*", "\n", value)
    value = re.sub(r"\s+([,.;:!?])", r"\1", value)
    value = re.sub(r"([,.;:!?])(?=[^\s\]\)])", r"\1 ", value)
    return value.strip()


def lyric_lines(text: str) -> list[str]:
    """Return non-empty lyric lines while retaining section labels."""

    normalized = normalize_vietnamese_lyrics(text)
    return [line.strip() for line in normalized.splitlines() if line.strip()]


def lyric_content_lines(text: str) -> list[str]:
    """Return sung lines and omit bracketed section headings."""

    return [line for line in lyric_lines(text) if not re.fullmatch(r"\[[^\]]+\]", line)]
