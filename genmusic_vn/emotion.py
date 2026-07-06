from __future__ import annotations

from collections import defaultdict

from .schemas import EmotionProfile
from .text_utils import extract_keywords, normalize_text, tokenize_words


LABELS = {
    "joy": ("vui tươi", 0.82, 0.72),
    "sadness": ("trầm buồn", -0.72, 0.34),
    "anger": ("mãnh liệt", -0.48, 0.88),
    "fear": ("bất an", -0.62, 0.62),
    "calm": ("bình yên", 0.42, 0.26),
    "romantic": ("lãng mạn", 0.58, 0.46),
    "hope": ("hy vọng", 0.72, 0.58),
    "nostalgic": ("hoài niệm", 0.02, 0.32),
}

PHRASE_LEXICON = {
    "joy": {
        "hạnh phúc": 2.0,
        "vui vẻ": 1.7,
        "rộn ràng": 1.7,
        "tươi sáng": 1.5,
        "yêu đời": 1.5,
        "nụ cười": 1.4,
    },
    "sadness": {
        "cô đơn": 2.0,
        "nước mắt": 1.8,
        "vỡ tan": 1.7,
        "lặng im": 1.3,
        "mưa rơi": 1.3,
        "chia xa": 1.6,
    },
    "anger": {
        "phản bội": 2.0,
        "tức giận": 1.9,
        "bốc cháy": 1.5,
        "gào thét": 1.7,
        "bất công": 2.0,
        "cúi đầu": 1.4,
        "siết chặt": 1.4,
        "trống trận": 1.7,
        "nhịp tim dồn": 1.3,
    },
    "fear": {
        "bóng đêm": 1.6,
        "hoang mang": 1.8,
        "lo lắng": 1.8,
        "run rẩy": 1.5,
    },
    "calm": {
        "bình yên": 2.0,
        "dịu dàng": 1.5,
        "êm đềm": 1.6,
        "nắng nhẹ": 1.3,
        "thở chậm": 1.4,
    },
    "romantic": {
        "yêu thương": 1.9,
        "trái tim": 1.6,
        "bàn tay": 1.3,
        "nhớ em": 1.5,
        "nhớ anh": 1.5,
    },
    "hope": {
        "hy vọng": 2.0,
        "ngày mai": 1.6,
        "ánh sáng": 1.6,
        "đi tiếp": 1.5,
        "đứng lên": 1.5,
    },
    "nostalgic": {
        "ngày xưa": 1.8,
        "ngày ấy": 1.8,
        "tuổi thơ": 1.8,
        "kỷ niệm": 1.7,
        "mùa cũ": 1.5,
        "trở về": 1.3,
    },
}

WORD_LEXICON = {
    "joy": {"vui": 1.2, "cười": 1.1, "nắng": 0.8, "mừng": 1.2, "sáng": 0.8, "rạng": 1.0},
    "sadness": {"buồn": 1.5, "đau": 1.4, "khóc": 1.2, "mưa": 0.8, "xa": 0.9, "nhớ": 0.9, "lạc": 0.8},
    "anger": {"giận": 1.5, "tức": 1.4, "cháy": 1.1, "đập": 0.9, "xé": 1.0, "gắt": 1.0, "bất": 0.6, "công": 0.6, "trống": 1.0, "trận": 0.8, "siết": 0.8},
    "fear": {"sợ": 1.5, "lo": 1.2, "lạnh": 0.9, "tối": 0.9, "run": 1.0, "mất": 0.8},
    "calm": {"êm": 1.0, "dịu": 1.2, "lặng": 1.0, "hiền": 1.0, "yên": 1.1, "ru": 0.8},
    "romantic": {"yêu": 1.5, "thương": 1.3, "tim": 1.1, "môi": 0.8, "hôn": 1.0, "hẹn": 0.9},
    "hope": {"tin": 1.0, "mơ": 1.1, "mong": 0.9, "mai": 0.8, "sáng": 0.7, "vươn": 1.1},
    "nostalgic": {"xưa": 1.2, "cũ": 1.1, "ký": 0.7, "ức": 0.7, "nhớ": 1.1, "mùa": 0.7},
}

NEGATORS = {"không", "chẳng", "chả", "chưa", "đừng", "khó"}
INTENSIFIERS = {"rất": 1.35, "quá": 1.25, "thật": 1.15, "cực": 1.4, "lắm": 1.2}


def analyze_emotion(text: str) -> EmotionProfile:
    normalized = normalize_text(text).lower()
    tokens = tokenize_words(normalized)
    scores: dict[str, float] = defaultdict(float)

    for label, phrases in PHRASE_LEXICON.items():
        for phrase, weight in phrases.items():
            if phrase in normalized:
                scores[label] += weight

    for index, token in enumerate(tokens):
        for label, words in WORD_LEXICON.items():
            if token not in words:
                continue

            weight = words[token]
            previous = tokens[max(0, index - 2) : index]
            if previous and previous[-1] in INTENSIFIERS:
                weight *= INTENSIFIERS[previous[-1]]
            if any(item in NEGATORS for item in previous):
                if label in {"joy", "hope", "romantic", "calm"}:
                    scores["sadness"] += weight * 0.7
                    continue
                weight *= 0.45
            scores[label] += weight

    if not scores:
        keywords = extract_keywords(text)
        return EmotionProfile(
            label="calm",
            label_vi=LABELS["calm"][0],
            valence=0.25,
            energy=0.28,
            confidence=0.35,
            keywords=keywords,
            scores={"calm": 0.35},
        )

    dominant = max(scores.items(), key=lambda item: item[1])[0]
    total = sum(max(0.0, value) for value in scores.values()) or 1.0
    valence = sum(LABELS[label][1] * max(0.0, value) for label, value in scores.items()) / total
    energy = sum(LABELS[label][2] * max(0.0, value) for label, value in scores.items()) / total
    confidence = min(0.95, 0.38 + total / (len(tokens) + 5))

    return EmotionProfile(
        label=dominant,
        label_vi=LABELS[dominant][0],
        valence=round(valence, 3),
        energy=round(energy, 3),
        confidence=round(confidence, 3),
        keywords=extract_keywords(text),
        scores={label: round(score, 3) for label, score in sorted(scores.items())},
    )
