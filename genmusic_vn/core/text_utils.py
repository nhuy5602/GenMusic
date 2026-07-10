from __future__ import annotations

import re
from collections import Counter


FLATTENED_LYRIC_LINE_STARTS = {
    "ai",
    "ánh",
    "anh",
    "bao",
    "bình",
    "chẳng",
    "cho",
    "còn",
    "cuộc",
    "dẫu",
    "để",
    "đêm",
    "đến",
    "đời",
    "em",
    "gọi",
    "hẹn",
    "hoa",
    "hóa",
    "họa",
    "khi",
    "lặng",
    "lời",
    "lòng",
    "một",
    "mưa",
    "ngày",
    "nếu",
    "nhìn",
    "phố",
    "sao",
    "ta",
    "tháng",
    "thương",
    "tiếng",
    "tình",
    "trăng",
    "trời",
    "từ",
    "và",
    "vẫn",
    "về",
    "yêu",
}


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


def extract_lyric_lines(text: str, *, recover_flattened: bool = True) -> list[str]:
    normalized = normalize_text(text)
    lines: list[str] = []
    for raw_line in normalized.splitlines():
        line = raw_line.strip(" \t,;:-")
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            continue
        if tokenize_words(line):
            lines.append(line)

    if len(lines) >= 2 or not recover_flattened:
        return lines

    recovered = recover_flattened_lyric_lines(normalized)
    return recovered or lines


def recover_flattened_lyric_lines(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized or "\n" in normalized or re.search(r"[.!?…]", normalized):
        return []

    words = re.findall(r"[^\W\d_]+", normalized, flags=re.UNICODE)
    if len(words) < 24 or len(words) > 180:
        return []

    lines: list[str] = []
    current: list[str] = []
    for word in words:
        lower_word = word.lower()
        should_break = current and (
            (len(current) >= 5 and lower_word in FLATTENED_LYRIC_LINE_STARTS)
            or len(current) >= 8
        )
        if should_break:
            lines.append(" ".join(current))
            current = []
        current.append(word)

    if current:
        if lines and len(current) < 3:
            lines[-1] = f"{lines[-1]} {' '.join(current)}"
        else:
            lines.append(" ".join(current))

    if len(lines) < 4:
        return []

    lengths = [len(tokenize_words(line)) for line in lines]
    short_ratio = sum(4 <= length <= 10 for length in lengths) / len(lengths)
    average_length = sum(lengths) / len(lengths)
    if short_ratio < 0.75 or not 5 <= average_length <= 9:
        return []
    return lines


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
