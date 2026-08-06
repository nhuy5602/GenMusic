"""Deterministic, open-vocabulary Vietnamese syllable decomposition.

Vietnamese orthography is highly compositional. A word can be represented by
an onset, a short vowel nucleus, a coda, and one of six tones. These parts are
reused across words, so a recognizer can be evaluated on lexical heldout words
without introducing a closed word vocabulary or a pretrained speech model.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

TONE_MARK_TO_NAME = {
    "\u0301": "sac",
    "\u0300": "huyen",
    "\u0309": "hoi",
    "\u0303": "nga",
    "\u0323": "nang",
}
TONE_NAMES = ("ngang", "sac", "huyen", "hoi", "nga", "nang")
ONSET_NAMES = (
    "none",
    "ngh",
    "ch",
    "gh",
    "gi",
    "kh",
    "ng",
    "nh",
    "ph",
    "qu",
    "th",
    "tr",
    "b",
    "c",
    "d",
    "đ",
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
CODA_NAMES = ("none", "ch", "ng", "nh", "c", "m", "n", "p", "t")
NUCLEUS_GRAPHEMES = (
    "a",
    "ă",
    "â",
    "e",
    "ê",
    "i",
    "o",
    "ô",
    "ơ",
    "u",
    "ư",
    "y",
)
MAX_NUCLEUS_GRAPHEMES = 3

ONSET_TO_ID = {value: index for index, value in enumerate(ONSET_NAMES)}
CODA_TO_ID = {value: index for index, value in enumerate(CODA_NAMES)}
TONE_TO_ID = {value: index for index, value in enumerate(TONE_NAMES)}
NUCLEUS_TO_ID = {
    value: index for index, value in enumerate(NUCLEUS_GRAPHEMES)
}


@dataclass(frozen=True)
class VietnameseSyllable:
    """One normalized orthographic Vietnamese syllable."""

    onset: str
    nucleus: tuple[str, ...]
    coda: str
    tone: str

    @property
    def onset_id(self) -> int:
        return ONSET_TO_ID[self.onset]

    @property
    def nucleus_ids(self) -> tuple[int, ...]:
        return tuple(NUCLEUS_TO_ID[value] for value in self.nucleus)

    @property
    def coda_id(self) -> int:
        return CODA_TO_ID[self.coda]

    @property
    def tone_id(self) -> int:
        return TONE_TO_ID[self.tone]


def decompose_vietnamese_syllable(
    word: str,
) -> VietnameseSyllable | None:
    """Split one Vietnamese word without consulting a corpus vocabulary.

    Foreign or malformed tokens are rejected when their nucleus contains a
    non-vowel. This retains ordinary Vietnamese syllables while preventing
    arbitrary English strings from becoming accidental giant "nuclei".
    """

    decomposed = unicodedata.normalize("NFD", str(word).casefold())
    tone = "ngang"
    tone_free: list[str] = []
    for character in decomposed:
        resolved_tone = TONE_MARK_TO_NAME.get(character)
        if resolved_tone is not None:
            tone = resolved_tone
        else:
            tone_free.append(character)
    base = unicodedata.normalize("NFC", "".join(tone_free))
    if not base or not base.isalpha():
        return None

    onset = next(
        (
            candidate
            for candidate in ONSET_NAMES[1:]
            if base.startswith(candidate)
        ),
        "none",
    )
    body = base if onset == "none" else base[len(onset) :]
    coda = next(
        (
            candidate
            for candidate in CODA_NAMES[1:]
            if body.endswith(candidate) and len(body) > len(candidate)
        ),
        "none",
    )
    if coda != "none":
        body = body[: -len(coda)]
    if (
        not body
        or len(body) > MAX_NUCLEUS_GRAPHEMES
        or any(character not in NUCLEUS_TO_ID for character in body)
    ):
        return None
    return VietnameseSyllable(
        onset=onset,
        nucleus=tuple(body),
        coda=coda,
        tone=tone,
    )
