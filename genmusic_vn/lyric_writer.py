from __future__ import annotations

from .schemas import EmotionProfile, HarmonyPlan, LyricDraft
from .stylebank import get_lyric_pattern
from .text_utils import compact_line, extract_keywords, split_sentences, tokenize_words


DEFAULT_CHORUS = {
    "joy": ["ta nang niu tieng cuoi trong nang", "de ngay moi hat vang tren moi"],
    "sadness": ["giu lai chut mua trong tim", "de noi nho thoi roi that khe"],
    "anger": ["ta buoc qua lua do trong long", "khong cui dau truoc nhung vet thuong"],
    "fear": ["xin thap len mot dom sang nho", "dan ta qua khoang toi menh mang"],
    "calm": ["ngoi yen nghe gio ru qua them", "de binh yen cham nhe vao tim"],
    "romantic": ["goi ten nhau giua mua rat khe", "de yeu thuong o lai that lau"],
    "hope": ["ngay mai len tu trong mat sang", "ta di tiep qua nhung ngap ngung"],
    "nostalgic": ["ngay xua nghieng trong mau nang cu", "ta tro ve bang mot cau ca"],
}


def _title_from_keywords(keywords: list[str], fallback: str) -> str:
    if keywords:
        return " ".join(keywords[:4]).capitalize()
    first = compact_line(fallback, 4)
    return first.capitalize() if first else "Khuc hat chua dat ten"


def _polish_line(line: str) -> str:
    return line.strip(" ,.;:-").lower()


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
            lines.append(_polish_line(f"{seed} con vang trong ta"))
            keywords = keywords[1:]
        else:
            lines.append("mot cau ca di qua dem dai")
    return lines[:4]


def _make_bridge(text: str, emotion: EmotionProfile, harmony: HarmonyPlan) -> list[str]:
    keywords = extract_keywords(text, 6)
    center = keywords[0] if keywords else emotion.label_vi
    pattern = get_lyric_pattern(emotion.label)
    bridge_template = pattern.get("bridge", [])
    if len(bridge_template) >= 2:
        return [line.format(motif=center) for line in bridge_template[:2]]
    if emotion.valence < -0.25:
        return [f"neu {center} con lam tim nghieng xuong", "ta xin hat cho long nhe hon"]
    if emotion.energy > 0.7:
        return [f"de {center} bat len nhu nhip trong", "ta di qua gioi han cua minh"]
    return [f"khi {center} nam yen trong hoi tho", f"{harmony.key} {harmony.scale} diu ta cham thoi"]


def rewrite_lyrics(text: str, emotion: EmotionProfile, harmony: HarmonyPlan) -> LyricDraft:
    keywords = extract_keywords(text, 10)
    title = _title_from_keywords(keywords, text)
    verse = _make_verse_lines(text)

    motif = keywords[0] if keywords else emotion.label_vi
    pattern = get_lyric_pattern(emotion.label)
    template = pattern.get("chorus") or DEFAULT_CHORUS.get(emotion.label, DEFAULT_CHORUS["calm"])
    chorus = [
        _polish_line(template[0]),
        _polish_line(f"{motif} oi, o lai them mot lan"),
        _polish_line(template[1]),
        _polish_line("cho cau hat tim thay duong ve"),
    ]
    bridge = _make_bridge(text, emotion, harmony)
    hook_words = tokenize_words(chorus[1])[:6]
    hook = " ".join(hook_words) if hook_words else chorus[1]

    return LyricDraft(title=title, verse=verse, chorus=chorus, bridge=bridge, hook=hook)

