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


def _line_from_sentence(sentence: str, max_words: int = 9) -> str:
    return _polish_line(compact_line(sentence, max_words=max_words))


def _make_verse_lines(text: str, offset: int = 0) -> list[str]:
    sentences = split_sentences(text)
    lines: list[str] = []
    selected = sentences[offset : offset + 4]
    for sentence in selected:
        compacted = _line_from_sentence(sentence, max_words=9)
        if compacted:
            lines.append(compacted)
        if len(lines) >= 4:
            break

    keywords = extract_keywords(text, 10)
    while len(lines) < 4:
        if keywords:
            seed = " ".join(keywords[: min(4, len(keywords))])
            lines.append(_polish_line(f"{seed} con vang trong ta"))
            keywords = keywords[1:]
        else:
            lines.append("mot cau ca di qua dem dai")
    return lines[:4]


def _make_pre_chorus(text: str, emotion: EmotionProfile) -> list[str]:
    keywords = extract_keywords(text, 8)
    motif = keywords[0] if keywords else emotion.label_vi
    if emotion.valence < -0.2:
        return [f"ta giu {motif} o giua lang im", "de trai tim tim lai loi ve"]
    if emotion.energy > 0.65:
        return [f"ta goi {motif} len bang nhip tho", "de buoc chan khong dung lai"]
    return [f"ta nghe {motif} di qua that khe", "roi de long minh cham lai hon"]


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


def _make_chorus(text: str, emotion: EmotionProfile) -> list[str]:
    keywords = extract_keywords(text, 10)
    motif = keywords[0] if keywords else emotion.label_vi
    pattern = get_lyric_pattern(emotion.label)
    template = pattern.get("chorus") or DEFAULT_CHORUS.get(emotion.label, DEFAULT_CHORUS["calm"])
    return [
        _polish_line(template[0]),
        _polish_line(f"{motif} oi, o lai them mot lan"),
        _polish_line(template[1]),
        _polish_line("cho cau hat tim thay duong ve"),
    ]


def _make_outro(chorus: list[str], emotion: EmotionProfile) -> list[str]:
    if emotion.valence < -0.2:
        return [chorus[-1], "roi dem cung hoa thanh binh minh"]
    if emotion.label in {"hope", "joy"}:
        return [chorus[-1], "ngay moi mo ra trong tieng ca"]
    return [chorus[-1], "binh yen nam lai tren doi tay"]


def _build_full_song(
    title: str,
    verse1: list[str],
    pre_chorus: list[str],
    chorus: list[str],
    verse2: list[str],
    bridge: list[str],
    outro: list[str],
) -> tuple[list[str], list[str]]:
    song_form = ["Verse 1", "Pre-Chorus", "Chorus", "Verse 2", "Bridge", "Final Chorus", "Outro"]
    full_song = [
        f"[Title] {title}",
        "",
        "[Verse 1]",
        *verse1,
        "",
        "[Pre-Chorus]",
        *pre_chorus,
        "",
        "[Chorus]",
        *chorus,
        "",
        "[Verse 2]",
        *verse2,
        "",
        "[Bridge]",
        *bridge,
        "",
        "[Final Chorus]",
        *chorus[:2],
        *chorus,
        "",
        "[Outro]",
        *outro,
    ]
    return song_form, full_song


def _build_short_song(title: str, verse: list[str], chorus: list[str], outro: list[str]) -> tuple[list[str], list[str]]:
    short_verse = verse[:2]
    short_chorus = chorus[:2]
    song_form = ["Verse", "Chorus", "Outro"]
    full_song = [
        f"[Title] {title}",
        "",
        "[Verse]",
        *short_verse,
        "",
        "[Chorus]",
        *short_chorus,
        "",
        "[Outro]",
        outro[-1] if outro else short_chorus[-1],
    ]
    return song_form, full_song


def rewrite_lyrics(text: str, emotion: EmotionProfile, harmony: HarmonyPlan) -> LyricDraft:
    keywords = extract_keywords(text, 10)
    title = _title_from_keywords(keywords, text)
    sentence_count = len(split_sentences(text))
    word_count = len(tokenize_words(text))
    verse1 = _make_verse_lines(text, offset=0)
    verse2 = _make_verse_lines(text, offset=4)
    pre_chorus = _make_pre_chorus(text, emotion)
    chorus = _make_chorus(text, emotion)
    bridge = _make_bridge(text, emotion, harmony)
    outro = _make_outro(chorus, emotion)
    hook_words = tokenize_words(chorus[1])[:6]
    hook = " ".join(hook_words) if hook_words else chorus[1]
    if sentence_count <= 2 and word_count <= 40:
        song_form, full_song = _build_short_song(title, verse1, chorus, outro)
    else:
        song_form, full_song = _build_full_song(title, verse1, pre_chorus, chorus, verse2, bridge, outro)

    return LyricDraft(
        title=title,
        verse=verse1,
        chorus=chorus,
        bridge=bridge,
        hook=hook,
        song_form=song_form,
        full_song=full_song,
    )
