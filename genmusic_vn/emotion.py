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
        "anh gặp em": 2.1,
        "đèn vàng": 1.4,
        "lời chào": 1.3,
        "thành phố mềm": 1.5,
        "mềm lại": 1.3,
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
    "romantic": {"yêu": 1.5, "thương": 1.3, "tim": 1.1, "môi": 0.8, "hôn": 1.0, "hẹn": 0.9, "mềm": 0.8},
    "hope": {"tin": 1.0, "mơ": 1.1, "mong": 0.9, "mai": 0.8, "sáng": 0.7, "vươn": 1.1, "dậy": 0.8, "khởi": 0.8, "đầu": 0.5},
    "nostalgic": {"xưa": 1.2, "cũ": 1.1, "ký": 0.7, "ức": 0.7, "nhớ": 1.1, "mùa": 0.7},
}

PHRASE_LEXICON["joy"].update(
    {
        "tiếng cười": 1.5,
        "pháo hoa": 1.7,
        "hân hoan": 1.8,
        "lễ hội": 1.7,
        "chúng ta thắng": 1.7,
        "cheerful": 1.5,
        "upbeat": 1.4,
        "festival": 1.5,
        "celebration": 1.6,
    }
)
PHRASE_LEXICON["sadness"].update(
    {
        "bỏ lỡ": 1.6,
        "không bao giờ quay lại": 1.8,
        "nhớ nhà": 1.7,
        "mất mát": 2.0,
        "missed love": 1.5,
        "sad vietnamese piano ballad": 2.4,
        "sad piano ballad": 2.2,
        "sad": 1.6,
        "minimal piano": 1.0,
        "melancholic": 1.5,
    }
)
PHRASE_LEXICON["anger"].update(
    {
        "giận dữ": 2.0,
        "hét lên": 1.7,
        "phá tan": 1.8,
        "rock cinematic": 1.2,
        "breakthrough": 1.2,
        "angry": 1.8,
    }
)
PHRASE_LEXICON["fear"].update(
    {
        "linh cảm không lành": 2.1,
        "mây đen": 1.3,
        "bí mật": 1.2,
        "sắp vỡ tung": 1.7,
        "màn đêm": 1.0,
        "chưa sẵn sàng đối mặt": 1.5,
        "nửa đêm": 1.3,
        "dòng sông đen": 1.7,
        "đồng hồ đang chạy ngược": 2.0,
        "cinematic suspense": 2.0,
        "thriller": 1.8,
        "mystery": 1.7,
        "mysterious fantasy": 2.1,
        "fantasy ambient": 1.9,
        "magical atmosphere": 1.7,
        "không thuộc về thế giới này": 2.0,
        "surreal": 1.5,
        "dark ambient": 1.7,
        "ominous": 1.8,
        "tense": 1.6,
    }
)
PHRASE_LEXICON["calm"].update(
    {
        "ngồi yên": 1.7,
        "cửa sổ": 0.9,
        "trôi thật chậm": 1.6,
        "pha một tách trà": 1.8,
        "nghỉ ngơi": 1.6,
        "tĩnh lặng": 1.6,
        "mặt hồ": 1.2,
        "thiền định": 2.0,
        "tập trung": 1.5,
        "lo-fi": 1.5,
        "chill": 1.6,
        "meditation": 1.8,
        "focus": 1.2,
        "healing": 1.4,
        "warm storytelling": 2.0,
        "small alley lights": 1.2,
        "acoustic guitar": 0.9,
        "soft flute": 0.9,
        "gentle percussion": 0.8,
        "intimate atmosphere": 1.0,
    }
)
PHRASE_LEXICON["romantic"].update(
    {
        "mùa yêu thương": 1.5,
        "nụ cười của em": 2.0,
        "ngày đông": 0.9,
        "bàn tay mẹ": 1.2,
        "bố đặt tay": 1.2,
        "lãng mạn": 2.0,
        "romantic": 1.9,
        "heartfelt": 1.4,
        "emotional": 1.2,
    }
)
PHRASE_LEXICON["hope"].update(
    {
        "bình minh": 1.7,
        "tan ra": 1.2,
        "tiến về phía trước": 1.8,
        "một đội": 1.3,
        "bầu trời mở ra": 1.8,
        "đứng dậy": 1.8,
        "lần nữa": 1.1,
        "chiến thắng": 2.0,
        "lá cờ": 1.5,
        "vẫn còn ở đây": 1.8,
        "tự do": 1.7,
        "hoàn toàn tự do": 2.0,
        "epic sports anthem": 2.1,
        "victory": 2.0,
        "heroic": 1.8,
        "triumphant": 1.8,
        "adventure": 1.4,
        "hopeful": 1.8,
        "inspiring": 1.5,
        "modern": 0.9,
    }
)
PHRASE_LEXICON["nostalgic"].update(
    {
        "bản nhạc cũ": 1.5,
        "mái trường": 1.5,
        "năm ấy": 1.7,
        "hoài niệm": 2.0,
        "old letter": 1.8,
        "nostalgic": 1.9,
    }
)

WORD_LEXICON["joy"].update({"cười": 1.2, "thắng": 1.4, "lễ": 0.7, "hội": 0.7, "upbeat": 1.2})
WORD_LEXICON["sadness"].update({"mất": 1.0, "khép": 0.9, "lỡ": 1.0, "missed": 0.9})
WORD_LEXICON["anger"].update({"trống": 0.15, "trận": 0.25, "hét": 1.3, "phá": 1.2, "giới": 0.2})
WORD_LEXICON["fear"].update({"mờ": 0.8, "lành": 0.4, "bí": 0.7, "mật": 0.7, "đen": 0.9, "ngược": 1.0, "dark": 1.0, "suspense": 1.4, "mystery": 1.2})
WORD_LEXICON["fear"]["tối"] = 0.2
WORD_LEXICON["calm"].update({"chill": 1.2, "trà": 1.0, "sách": 0.7, "hồ": 0.8, "focus": 1.0, "thiền": 1.2})
WORD_LEXICON["calm"].update({"warm": 1.0, "gentle": 0.8, "intimate": 0.8})
WORD_LEXICON["romantic"].update({"mẹ": 0.7, "bố": 0.7, "em": 0.6, "romantic": 1.3, "heartfelt": 1.0})
WORD_LEXICON["hope"].update({"đội": 1.0, "tiến": 1.0, "thắng": 1.3, "cờ": 0.9, "dậy": 1.1, "tự": 0.6, "do": 0.6, "epic": 1.1, "victory": 1.5, "heroic": 1.3, "adventure": 1.0, "modern": 0.7})
WORD_LEXICON["nostalgic"].update({"trường": 0.7, "nostalgic": 1.4})

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
