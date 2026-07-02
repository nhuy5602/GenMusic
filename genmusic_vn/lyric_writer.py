from __future__ import annotations

from .schemas import EmotionProfile, HarmonyPlan, LyricDraft
from .text_utils import compact_line, extract_keywords, split_sentences, tokenize_words


CHORUS_TEMPLATES = {
    "joy": ["ta nâng niu tiếng cười trong nắng", "để ngày mới hát vang trên môi"],
    "sadness": ["giữ lại chút mưa trong tim", "để nỗi nhớ thôi rơi thật khẽ"],
    "anger": ["ta bước qua lửa đỏ trong lòng", "không cúi đầu trước những vết thương"],
    "fear": ["xin thắp lên một đốm sáng nhỏ", "dẫn ta qua khoảng tối mênh mang"],
    "calm": ["ngồi yên nghe gió ru qua thềm", "để bình yên chạm nhẹ vào tim"],
    "romantic": ["gọi tên nhau giữa mùa rất khẽ", "để yêu thương ở lại thật lâu"],
    "hope": ["ngày mai lên từ trong mắt sáng", "ta đi tiếp qua những ngập ngừng"],
    "nostalgic": ["ngày xưa nghiêng trong màu nắng cũ", "ta trở về bằng một câu ca"],
}


def _title_from_keywords(keywords: list[str], fallback: str) -> str:
    if keywords:
        title_words = keywords[:4]
        return " ".join(title_words).capitalize()
    first = compact_line(fallback, 4)
    return first.capitalize() if first else "Khúc hát chưa đặt tên"


def _polish_line(line: str) -> str:
    line = line.strip(" ,.;:-").lower()
    if not line:
        return ""
    return line


def _make_verse_lines(text: str) -> list[str]:
    sentences = split_sentences(text)
    lines: list[str] = []
    for sentence in sentences:
        compacted = compact_line(sentence, max_words=9)
        if compacted:
            lines.append(_polish_line(compacted))
        if len(lines) >= 4:
            break

    keywords = extract_keywords(text, 8)
    while len(lines) < 4:
        if keywords:
            seed = " ".join(keywords[: min(4, len(keywords))])
            lines.append(_polish_line(f"{seed} còn vang trong ta"))
            keywords = keywords[1:]
        else:
            lines.append("một câu ca đi qua đêm dài")
    return lines[:4]


def _make_bridge(text: str, emotion: EmotionProfile, harmony: HarmonyPlan) -> list[str]:
    keywords = extract_keywords(text, 6)
    center = keywords[0] if keywords else emotion.label_vi
    if emotion.valence < -0.25:
        return [f"nếu {center} còn làm tim nghiêng xuống", "ta xin hát cho lòng nhẹ hơn"]
    if emotion.energy > 0.7:
        return [f"để {center} bật lên như nhịp trống", "ta đi qua giới hạn của mình"]
    return [f"khi {center} nằm yên trong hơi thở", f"{harmony.key} {harmony.scale} dìu ta chậm thôi"]


def rewrite_lyrics(text: str, emotion: EmotionProfile, harmony: HarmonyPlan) -> LyricDraft:
    keywords = extract_keywords(text, 10)
    title = _title_from_keywords(keywords, text)
    verse = _make_verse_lines(text)

    motif = keywords[0] if keywords else emotion.label_vi
    template = CHORUS_TEMPLATES.get(emotion.label, CHORUS_TEMPLATES["calm"])
    chorus = [
        _polish_line(template[0]),
        _polish_line(f"{motif} ơi, ở lại thêm một lần"),
        _polish_line(template[1]),
        _polish_line("cho câu hát tìm thấy đường về"),
    ]
    bridge = _make_bridge(text, emotion, harmony)
    hook_words = tokenize_words(chorus[1])[:6]
    hook = " ".join(hook_words) if hook_words else chorus[1]

    return LyricDraft(title=title, verse=verse, chorus=chorus, bridge=bridge, hook=hook)

