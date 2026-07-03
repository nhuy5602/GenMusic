from __future__ import annotations

import re
from collections import Counter


VI_STOPWORDS = {
    "anh",
    "ấy",
    "bằng",
    "bị",
    "các",
    "cái",
    "cần",
    "càng",
    "chỉ",
    "cho",
    "chưa",
    "có",
    "còn",
    "của",
    "cùng",
    "cũng",
    "đã",
    "đang",
    "để",
    "đến",
    "đi",
    "đó",
    "được",
    "em",
    "gì",
    "hay",
    "hơn",
    "khi",
    "không",
    "là",
    "lại",
    "làm",
    "lên",
    "mà",
    "mình",
    "một",
    "này",
    "nên",
    "nếu",
    "người",
    "như",
    "những",
    "nơi",
    "nữa",
    "ở",
    "qua",
    "ra",
    "rằng",
    "rất",
    "rồi",
    "sẽ",
    "ta",
    "tại",
    "thì",
    "trên",
    "trong",
    "tôi",
    "từ",
    "và",
    "vẫn",
    "vào",
    "về",
    "vì",
    "với",
}


VI_STOPWORDS.update(
    {
        "ấy",
        "bằng",
        "bị",
        "các",
        "cái",
        "cần",
        "càng",
        "chỉ",
        "chưa",
        "có",
        "còn",
        "của",
        "cùng",
        "cũng",
        "đã",
        "đang",
        "để",
        "đến",
        "đi",
        "đó",
        "được",
        "gì",
        "hơn",
        "không",
        "là",
        "lại",
        "làm",
        "lên",
        "mà",
        "mình",
        "một",
        "này",
        "nên",
        "nếu",
        "người",
        "như",
        "những",
        "nơi",
        "nữa",
        "ở",
        "rằng",
        "rất",
        "rồi",
        "sẽ",
        "tại",
        "thì",
        "trên",
        "tôi",
        "từ",
        "và",
        "vẫn",
        "vào",
        "về",
        "vì",
        "với",
    }
)


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    parts = re.split(r"(?<=[.!?…])\s+|\n+", normalized)
    return [part.strip(" \t\n,;:-") for part in parts if part.strip(" \t\n,;:-")]


def tokenize_words(text: str) -> list[str]:
    return re.findall(r"[^\W\d_]+", text.lower(), flags=re.UNICODE)


def extract_keywords(text: str, limit: int = 10) -> list[str]:
    tokens = [word for word in tokenize_words(text) if word not in VI_STOPWORDS and len(word) > 1]
    counts = Counter(tokens)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], tokens.index(item[0])))
    return [word for word, _ in ranked[:limit]]


def compact_line(text: str, max_words: int = 10) -> str:
    words = tokenize_words(text)
    if not words:
        return ""
    if len(words) <= max_words:
        return " ".join(words)

    meaningful = [word for word in words if word not in VI_STOPWORDS]
    if len(meaningful) >= max_words // 2:
        picked = meaningful[:max_words]
    else:
        picked = words[:max_words]
    return " ".join(picked)
