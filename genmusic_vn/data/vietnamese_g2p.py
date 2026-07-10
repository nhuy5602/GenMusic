"""Vietnamese grapheme-to-phoneme conversion with tone-aware tokens."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any

from .vietnamese_text import normalize_vietnamese_lyrics


_SYLLABLE_RE = re.compile(r"[0-9A-Za-zÀ-ỹĐđ]+", re.UNICODE)
_TONE_BY_MARK = {
    "\u0300": 2,  # huyền
    "\u0301": 3,  # sắc
    "\u0309": 4,  # hỏi
    "\u0303": 5,  # ngã
    "\u0323": 6,  # nặng
}
_BASE_VOWELS = {
    "a": "a",
    "ă": "ă",
    "â": "ə",
    "e": "ɛ",
    "ê": "e",
    "i": "i",
    "o": "ɔ",
    "ô": "o",
    "ơ": "ɤ",
    "u": "u",
    "ư": "ɯ",
    "y": "i",
}
_ONSETS = (
    ("ngh", "ŋ"),
    ("ng", "ŋ"),
    ("nh", "ɲ"),
    ("ch", "c"),
    ("tr", "tʂ"),
    ("th", "tʰ"),
    ("ph", "f"),
    ("kh", "x"),
    ("gh", "ɣ"),
    ("gi", "z"),
    ("qu", "kw"),
    ("đ", "ɗ"),
    ("d", "z"),
    ("r", "z"),
    ("x", "s"),
    ("s", "s"),
    ("v", "v"),
    ("b", "ɓ"),
    ("m", "m"),
    ("n", "n"),
    ("t", "t"),
    ("l", "l"),
    ("c", "k"),
    ("k", "k"),
    ("g", "ɣ"),
    ("h", "h"),
    ("q", "k"),
)
_CODAS = {
    "c": "k",
    "ch": "k",
    "m": "m",
    "n": "n",
    "ng": "ŋ",
    "nh": "ɲ",
    "p": "p",
    "t": "t",
}


@dataclass(frozen=True)
class G2PResult:
    text: str
    tokens: list[str]
    tone_digits: list[int]
    backend: str

    @property
    def phoneme_string(self) -> str:
        return " ".join(self.tokens)

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["phoneme_string"] = self.phoneme_string
        return result


def tone_digit(syllable: str) -> int:
    """Map Vietnamese tone marks to digits 1..6; ngang is 1."""

    decomposed = unicodedata.normalize("NFD", syllable.casefold())
    for character in decomposed:
        if character in _TONE_BY_MARK:
            return _TONE_BY_MARK[character]
    return 1


def _strip_tone(syllable: str) -> str:
    decomposed = unicodedata.normalize("NFD", syllable.casefold())
    return unicodedata.normalize("NFC", "".join(char for char in decomposed if char not in _TONE_BY_MARK))


def _fallback_syllable_to_ipa(syllable: str) -> str:
    plain = _strip_tone(syllable)
    if not plain:
        return ""
    onset = ""
    remainder = plain
    for spelling, ipa in _ONSETS:
        if remainder.startswith(spelling):
            onset = ipa
            remainder = remainder[len(spelling) :]
            break
    if not remainder:
        return onset

    vowel_positions = [index for index, char in enumerate(remainder) if char in _BASE_VOWELS]
    if not vowel_positions:
        return onset + remainder
    first_vowel = vowel_positions[0]
    prefix = remainder[:first_vowel]
    vowel_part = remainder[first_vowel:]
    coda = ""
    for spelling in sorted(_CODAS, key=len, reverse=True):
        if vowel_part.endswith(spelling) and len(vowel_part) > len(spelling):
            coda = _CODAS[spelling]
            vowel_part = vowel_part[: -len(spelling)]
            break
    vowel_ipa = "".join(_BASE_VOWELS.get(char, char) for char in vowel_part)
    return f"{onset}{prefix}{vowel_ipa}{coda}"


def _phonemizer_syllable(syllable: str) -> str | None:
    try:
        from phonemizer import phonemize  # type: ignore
    except ImportError:
        return None
    try:
        value = phonemize(
            syllable,
            language="vi",
            backend="espeak",
            strip=True,
            preserve_punctuation=False,
            with_stress=True,
        )
    except Exception:
        return None
    value = re.sub(r"\s+", "", value or "")
    return value or None


def vietnamese_g2p(text: str, *, use_phonemizer: bool = True) -> G2PResult:
    """Convert text into one IPA-like token per Vietnamese syllable.

    When ``phonemizer`` and Vietnamese eSpeak support are installed, the
    result is tagged ``phonemizer-espeak``. Otherwise a deterministic fallback
    is used and explicitly reported as ``rule-based-ipa``.
    """

    normalized = normalize_vietnamese_lyrics(text)
    tokens: list[str] = []
    tones: list[int] = []
    used_phonemizer = False
    for syllable in _SYLLABLE_RE.findall(normalized):
        digit = tone_digit(syllable)
        ipa = _phonemizer_syllable(syllable) if use_phonemizer else None
        if ipa is not None:
            used_phonemizer = True
        else:
            ipa = _fallback_syllable_to_ipa(syllable)
        tokens.append(f"{ipa}{digit}")
        tones.append(digit)
    return G2PResult(
        text=normalized,
        tokens=tokens,
        tone_digits=tones,
        backend="phonemizer-espeak" if used_phonemizer else "rule-based-ipa",
    )
