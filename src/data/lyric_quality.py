"""Conservative quality filters for aligned Vietnamese lyric labels."""

from __future__ import annotations

import re
import unicodedata


_VIETNAMESE_MARKED_CHARS = frozenset(
    "ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệ"
    "íìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
)
_VIETNAMESE_COMMON_WORDS = frozenset(
    "ai anh ba bao bay ben biet bon buoc ca cac can chang chi cho chung co con cua "
    "da dang dau day dem den dieu doi dung duoc em gi giua hay hon khi khong "
    "hai la lai lam len long luc ma mai minh mot mua nam nay nghe ngay nguoi nhau "
    "nhieu nhung noi o qua ra rang roi sau se ta thay thi theo them tren trong "
    "troi tu van ve vi voi yeu".split()
)
_ENGLISH_COMMON_WORDS = frozenset(
    "a an and are be been but by can come comes do for from have i in is it "
    "let love me my of on open our say scale sign since subscribe taking the "
    "this to was we with you your".split()
)
_TRANSCRIPT_NOISE_PHRASES = (
    "dang ky kenh",
    "hay subscribe",
    "subscribe cho kenh",
    "cam on cac ban",
)
_MOJIBAKE_SEQUENCES = ("Æ°", "Æ¡", "Ä‘", "Äƒ", "áº", "á»", "â€", "ðŸ")
_UTF8_LATIN1_PREFIX = re.compile(r"Ã[\u00a0-\u00bf]")


def text_has_mojibake(text: str) -> bool:
    """Detect common broken UTF-8 signatures before tokenization."""
    value = str(text or "")
    if "\ufffd" in value or any(
        sequence in value
        for sequence in _MOJIBAKE_SEQUENCES
    ):
        return True
    if _UTF8_LATIN1_PREFIX.search(value):
        return True
    return any("\u0080" <= char <= "\u009f" for char in value)


def _accentless_token(value: str) -> str:
    decomposed = unicodedata.normalize(
        "NFD",
        str(value).casefold(),
    ).replace("đ", "d")
    return "".join(
        char
        for char in decomposed
        if unicodedata.category(char) != "Mn"
    )


def clean_vietnamese_lyric(text: str) -> str:
    """Return a conservative clean Vietnamese label or reject it as empty."""
    normalized = unicodedata.normalize("NFKC", str(text or "")).strip()
    if not normalized or text_has_mojibake(normalized):
        return ""

    letters = [char for char in normalized if char.isalpha()]
    non_latin_letters = [
        char
        for char in letters
        if "LATIN" not in unicodedata.name(char, "")
        and char.casefold() != "đ"
    ]
    if (
        non_latin_letters
        and len(non_latin_letters) / max(1, len(letters)) > 0.02
    ):
        return ""

    cleaned = re.sub(
        r"\s+",
        " ",
        "".join(
            char
            if char.isalpha() or char.isspace() or char in "'-.,!?"
            else " "
            for char in normalized
        ),
    ).strip(" -'.,!?")
    words = re.findall(
        r"[^\W\d_]+",
        cleaned.casefold(),
        flags=re.UNICODE,
    )
    if len(words) < 2:
        return ""

    folded_words = [_accentless_token(word) for word in words]
    folded_text = " ".join(folded_words)
    if any(
        phrase in folded_text
        for phrase in _TRANSCRIPT_NOISE_PHRASES
    ):
        return ""
    vietnamese_hits = sum(
        word in _VIETNAMESE_COMMON_WORDS
        for word in folded_words
    )
    english_hits = sum(
        word in _ENGLISH_COMMON_WORDS
        for word in folded_words
    )
    marked_count = sum(
        char.casefold() in _VIETNAMESE_MARKED_CHARS
        for char in cleaned
    )
    if vietnamese_hits < 2 and marked_count < 1:
        return ""
    if english_hits >= 2 and english_hits >= vietnamese_hits:
        return ""
    return cleaned
