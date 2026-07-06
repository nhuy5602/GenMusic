from __future__ import annotations

from .schemas import EmotionProfile, HarmonyPlan, LyricDraft
from .stylebank import get_lyric_pattern
from .text_utils import compact_line, extract_keywords, split_sentences, tokenize_words


DEFAULT_CHORUS = {
    "joy": ["ta nâng niu tiếng cười trong nắng", "để ngày mới hát vang trên môi"],
    "sadness": ["giữ lại chút mưa trong tim", "để nỗi nhớ thôi rơi thật khẽ"],
    "anger": ["ta bước qua lửa đỏ trong lòng", "không cúi đầu trước những vết thương"],
    "fear": ["xin thắp lên một đốm sáng nhỏ", "dẫn ta qua khoảng tối mênh mang"],
    "calm": ["ngồi yên nghe gió ru qua thềm", "để bình yên chạm nhẹ vào tim"],
    "romantic": ["gọi tên nhau giữa mưa rất khẽ", "để yêu thương ở lại thật lâu"],
    "hope": ["ngày mai lên từ trong mắt sáng", "ta đi tiếp qua những ngập ngừng"],
    "nostalgic": ["ngày xưa nghiêng trong màu nắng cũ", "ta trở về bằng một câu ca"],
}


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
            lines.append(_polish_line(f"{seed} còn vang trong ta"))
            keywords = keywords[1:]
        else:
            lines.append("một câu ca đi qua đêm dài")
    return lines[:4]


def _make_pre_chorus(text: str, emotion: EmotionProfile) -> list[str]:
    keywords = extract_keywords(text, 8)
    motif = keywords[0] if keywords else emotion.label_vi
    if emotion.valence < -0.2:
        return [f"ta giữ {motif} ở giữa lặng im", "để trái tim tìm lại lối về"]
    if emotion.energy > 0.65:
        return [f"ta gọi {motif} lên bằng nhịp thở", "để bước chân không dừng lại"]
    return [f"ta nghe {motif} đi qua thật khẽ", "rồi để lòng mình chậm lại hơn"]


def _make_bridge(text: str, emotion: EmotionProfile, harmony: HarmonyPlan) -> list[str]:
    keywords = extract_keywords(text, 6)
    center = keywords[0] if keywords else emotion.label_vi
    pattern = get_lyric_pattern(emotion.label)
    bridge_template = pattern.get("bridge", [])
    if len(bridge_template) >= 2:
        return [line.format(motif=center) for line in bridge_template[:2]]
    if emotion.valence < -0.25:
        return [f"nếu {center} còn làm tim nghiêng xuống", "ta xin hát cho lòng nhẹ hơn"]
    if emotion.energy > 0.7:
        return [f"để {center} bật lên như nhịp trống", "ta đi qua giới hạn của mình"]
    return [f"khi {center} nằm yên trong hơi thở", f"{harmony.key} {harmony.scale} dìu ta chậm thôi"]


def _make_chorus(text: str, emotion: EmotionProfile) -> list[str]:
    keywords = extract_keywords(text, 10)
    motif = keywords[0] if keywords else emotion.label_vi
    pattern = get_lyric_pattern(emotion.label)
    template = pattern.get("chorus") or DEFAULT_CHORUS.get(emotion.label, DEFAULT_CHORUS["calm"])
    return [
        _polish_line(template[0]),
        _polish_line(f"{motif} ơi, ở lại thêm một lần"),
        _polish_line(template[1]),
        _polish_line("cho câu hát tìm thấy đường về"),
    ]


def _make_outro(chorus: list[str], emotion: EmotionProfile) -> list[str]:
    if emotion.valence < -0.2:
        return [chorus[-1], "rồi đêm cũng hóa thành bình minh"]
    if emotion.label in {"hope", "joy"}:
        return [chorus[-1], "ngày mới mở ra trong tiếng ca"]
    return [chorus[-1], "bình yên nằm lại trên đôi tay"]


def _build_full_song(
    verse1: list[str],
    pre_chorus: list[str],
    chorus: list[str],
    verse2: list[str],
    bridge: list[str],
    outro: list[str],
) -> tuple[list[str], list[str]]:
    song_form = ["Verse 1", "Pre-Chorus", "Chorus", "Verse 2", "Bridge", "Final Chorus", "Outro"]
    full_song = [
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


def _build_short_song(verse: list[str], chorus: list[str], outro: list[str]) -> tuple[list[str], list[str]]:
    short_verse = verse[:2]
    short_chorus = chorus[:2]
    song_form = ["Verse", "Chorus", "Outro"]
    full_song = [
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
        song_form, full_song = _build_short_song(verse1, chorus, outro)
    else:
        song_form, full_song = _build_full_song(verse1, pre_chorus, chorus, verse2, bridge, outro)

    return LyricDraft(
        title="",
        verse=verse1,
        chorus=chorus,
        bridge=bridge,
        hook=hook,
        song_form=song_form,
        full_song=full_song,
    )
