from __future__ import annotations

import re

from .rhyme import dominant_rhyme_scheme, end_pair_rhyme_rate, end_rhyme_key, vietnamese_rhyme_rate
from .schemas import EmotionProfile, HarmonyPlan, LyricDraft
from .stylebank import get_lyric_pattern
from .text_utils import compact_line, extract_keywords, extract_lyric_lines, split_sentences, tokenize_words


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


MOTIF_PHRASES = {
    "joy": ["tiếng cười", "ngày mới", "nắng sớm", "con đường mới"],
    "sadness": ["nỗi nhớ", "mưa", "đêm", "người rời xa", "lời chưa nói"],
    "anger": ["bất công", "nhịp tim", "trống trận", "bàn tay", "vết thương"],
    "fear": ["bóng tối", "căn phòng", "đường vắng", "ánh sáng"],
    "calm": ["bờ sông", "gió", "tiếng nước", "bình yên", "hàng cây"],
    "romantic": ["thành phố", "đèn vàng", "lời chào", "mùa hẹn", "yêu thương"],
    "hope": ["ánh sáng", "khởi đầu", "ngày mai", "niềm tin", "đứng dậy"],
    "nostalgic": ["phố cũ", "ánh đèn", "lời hứa", "chiều mưa", "ngày xưa"],
}
BAD_MOTIFS = {
    "sau",
    "nhiều",
    "chúng",
    "muốn",
    "gặp",
    "dưới",
    "hàng",
    "lời",
    "chào",
    "khẽ",
    "đủ",
    "buổi",
    "sáng",
    "một",
    "tôi",
    "anh",
    "em",
}
MOTIF_STOPWORDS = BAD_MOTIFS | {
    "có",
    "còn",
    "của",
    "cùng",
    "đã",
    "đang",
    "đặt",
    "đến",
    "để",
    "đi",
    "đứng",
    "gì",
    "giữa",
    "khi",
    "không",
    "là",
    "làm",
    "lên",
    "lúc",
    "mà",
    "mình",
    "muốn",
    "này",
    "nói",
    "qua",
    "ra",
    "rồi",
    "sẽ",
    "sự",
    "ta",
    "thấy",
    "trong",
    "trước",
    "và",
    "vào",
    "về",
    "với",
    "xuống",
    "ấy",
}
SAFE_FALLBACK_MOTIFS = {
    "joy": "tiếng cười",
    "sadness": "nỗi nhớ",
    "anger": "lửa lòng",
    "fear": "bóng tối",
    "calm": "bình yên",
    "romantic": "yêu thương",
    "hope": "niềm tin",
    "nostalgic": "ngày xưa",
}
RHYME_ENDINGS = {
    "joy": ["trên môi", "sáng ngời", "ngày mới", "đầy vơi"],
    "sadness": ["trong tim", "lặng im", "mưa đêm", "bên thềm"],
    "anger": ["lửa lên", "không quên", "bền gan", "hiên ngang"],
    "fear": ["trong đêm", "lạnh thêm", "âm thầm", "xa xăm"],
    "calm": ["êm đềm", "bên thềm", "trong tim", "nhẹ tênh"],
    "romantic": ["trong tay", "đêm nay", "thật lâu", "bên nhau"],
    "hope": ["chân trời", "sáng ngời", "ngày mới", "không rời"],
    "nostalgic": ["thật lâu", "êm đềm", "trong tim", "ngày xưa"],
}
FALLBACK_RHYME_ENDINGS = ["trong tim", "sáng ngời", "bên thềm", "thật lâu"]
CLAUSE_SPLIT_RE = re.compile(r"[,;]+")
LYRIC_PHRASE_SPLIT_RE = re.compile(r"\s+(đến khi|dẫu|nếu|rồi|từ ngày|từ người)\s+", re.IGNORECASE)


def _polish_line(line: str) -> str:
    return line.strip(" ,.;:-").lower()


def _rhyme_key(line: str) -> str:
    return end_rhyme_key(line)


def _rhyme_endings(emotion: EmotionProfile) -> list[str]:
    if emotion.label in RHYME_ENDINGS:
        return RHYME_ENDINGS[emotion.label]
    if emotion.valence < -0.15:
        return RHYME_ENDINGS["sadness"]
    if emotion.valence > 0.35:
        return RHYME_ENDINGS["hope"]
    return FALLBACK_RHYME_ENDINGS


def _ensure_rhyme_ending(line: str, ending: str, max_words: int = 12) -> str:
    base = _polish_line(line)
    ending = _polish_line(ending)
    if not base:
        return ending
    if _rhyme_key(base) == _rhyme_key(ending):
        return base

    base_words = tokenize_words(base)
    ending_words = tokenize_words(ending)
    if len(base_words) + len(ending_words) <= max_words:
        return _polish_line(f"{base} {ending}")

    stem_limit = max(4, max_words - len(ending_words))
    stem = " ".join(base_words[:stem_limit])
    return _polish_line(f"{stem} {ending}")


def _shape_lines_for_melody(
    lines: list[str],
    emotion: EmotionProfile,
    *,
    start_pair: int = 0,
    max_words: int = 12,
) -> list[str]:
    endings = _rhyme_endings(emotion)
    shaped = [_polish_line(line) for line in lines if _polish_line(line)]
    for index, line in enumerate(shaped):
        if len(shaped) % 2 == 1 and index == len(shaped) - 1 and index > 0:
            pair_index = start_pair + ((index - 1) // 2)
        else:
            pair_index = start_pair + (index // 2)
        ending = endings[pair_index % len(endings)]
        shaped[index] = _ensure_rhyme_ending(line, ending, max_words=max_words)
    return shaped


def _existing_lyric_lines(text: str) -> list[str]:
    lines: list[str] = []
    for line in extract_lyric_lines(text):
        for part in _split_existing_lyric_line(line):
            polished = _polish_line(part)
            if polished:
                lines.append(polished)
    return lines


def _split_existing_lyric_line(line: str) -> list[str]:
    cleaned = line.strip(" \t,;:-")
    if not cleaned:
        return []

    parts = _split_on_outside_commas(cleaned)
    expanded: list[str] = []
    for part in parts:
        expanded.extend(_split_long_lyric_phrase(part))
    return [part.strip(" \t,;:-") for part in expanded if part.strip(" \t,;:-")]


def _split_on_outside_commas(line: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for char in line:
        if char == "(":
            depth += 1
        elif char == ")" and depth > 0:
            depth -= 1
        if char in {",", ";"} and depth == 0:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue
        current.append(char)
    part = "".join(current).strip()
    if part:
        parts.append(part)
    return parts or [line]


def _split_long_lyric_phrase(line: str, max_words: int = 10) -> list[str]:
    words = tokenize_words(line)
    if len(words) <= max_words:
        return [line]

    match = LYRIC_PHRASE_SPLIT_RE.search(line)
    if not match:
        return [line]

    split_at = match.start()
    first = line[:split_at].strip(" \t,;:-")
    second = line[match.start() :].strip(" \t,;:-")
    if not first or not second:
        return [line]
    if len(tokenize_words(first)) < 3 or len(tokenize_words(second)) < 3:
        return [line]
    return [first, second]


def _looks_like_existing_lyrics(text: str) -> bool:
    lines = _existing_lyric_lines(text)
    if len(lines) < 2:
        return False
    short_lines = sum(1 for line in lines if 2 <= len(tokenize_words(line)) <= 14)
    if len(lines) < 6:
        return ("\n" in text or len(lines) >= 4) and short_lines == len(lines)
    return short_lines / len(lines) >= 0.65


def _section_pair_rhyme_rate(lines: list[str]) -> float:
    return end_pair_rhyme_rate(lines)


def _repair_section_if_needed(lines: list[str], emotion: EmotionProfile, *, start_pair: int) -> list[str]:
    cleaned = [_polish_line(line) for line in lines if _polish_line(line)]
    if vietnamese_rhyme_rate(cleaned) >= 0.5:
        return cleaned
    return _shape_lines_for_melody(cleaned, emotion, start_pair=start_pair)


def _rewrite_existing_lyrics(lines: list[str], emotion: EmotionProfile) -> LyricDraft:
    cleaned = [_polish_line(line) for line in lines if _polish_line(line)]
    if len(cleaned) <= 4:
        chorus = cleaned
        hook_words = tokenize_words(chorus[0])[:6]
        hook = " ".join(hook_words) if hook_words else chorus[0]
        detected_scheme = dominant_rhyme_scheme(chorus)
        return LyricDraft(
            title="",
            verse=[],
            chorus=chorus,
            bridge=[],
            hook=hook,
            song_form=["Chorus"],
            full_song=["[Chorus]", *chorus],
            rhyme_scheme=f"selected short chorus input; preserves original user lyric and {detected_scheme} Vietnamese rhyme when present",
        )

    verse = cleaned[:4]
    chorus_source = cleaned[4:8] if len(cleaned) >= 6 else []
    chorus = chorus_source
    if len(chorus) < 2:
        chorus = cleaned[4:6] if len(cleaned) > 4 else cleaned[:2]

    bridge_source = cleaned[8:10]
    bridge = bridge_source if bridge_source else []
    outro_source = cleaned[-2:] if len(cleaned) > 10 else []
    outro = outro_source if outro_source else []

    song_form = ["Verse", "Chorus"]
    full_song = ["[Verse]", *verse, "", "[Chorus]", *chorus]
    if bridge:
        song_form.append("Bridge")
        full_song.extend(["", "[Bridge]", *bridge])
    if outro:
        song_form.append("Outro")
        full_song.extend(["", "[Outro]", *outro])

    hook_words = tokenize_words(chorus[0])[:6]
    hook = " ".join(hook_words) if hook_words else chorus[0]
    detected_scheme = dominant_rhyme_scheme(verse + chorus + bridge + outro)
    return LyricDraft(
        title="",
        verse=verse,
        chorus=chorus,
        bridge=bridge,
        hook=hook,
        song_form=song_form,
        full_song=full_song,
        rhyme_scheme=f"selected lyric input excerpt; preserves original user lyric and {detected_scheme} Vietnamese rhyme when present",
    )


def _line_from_sentence(sentence: str, max_words: int = 10) -> str:
    return _polish_line(compact_line(sentence, max_words=max_words))


def _line_chunks_from_sentence(sentence: str, max_words: int = 10, max_lines: int = 2) -> list[str]:
    clauses = [part.strip() for part in CLAUSE_SPLIT_RE.split(sentence) if part.strip()]
    if len(clauses) > 1:
        clause_lines: list[str] = []
        for clause in clauses:
            remaining = max_lines - len(clause_lines)
            if remaining <= 0:
                break
            clause_lines.extend(
                _line_chunks_from_sentence(clause, max_words=max_words, max_lines=remaining)
            )
        return clause_lines[:max_lines]

    words = tokenize_words(sentence)
    if not words:
        return []
    if len(words) <= max_words:
        return [_line_from_sentence(sentence, max_words=max_words)]

    lines: list[str] = []
    line_count = min(max_lines, (len(words) + max_words - 1) // max_words)
    chunk_size = max(1, (len(words) + line_count - 1) // line_count)
    for start in range(0, len(words), chunk_size):
        chunk = _polish_line(" ".join(words[start : start + chunk_size]))
        if chunk:
            lines.append(chunk)
        if len(lines) >= max_lines:
            break
    return lines


def _make_verse_lines(text: str, offset: int = 0) -> list[str]:
    sentences = split_sentences(text)
    lines: list[str] = []
    selected = sentences[offset : offset + 4]
    for sentence in selected:
        remaining = 4 - len(lines)
        word_len = len(tokenize_words(sentence))
        max_lines = min(remaining, 3 if len(selected) == 1 else (2 if word_len > 9 else 1))
        for compacted in _line_chunks_from_sentence(sentence, max_words=10, max_lines=max_lines):
            if compacted:
                lines.append(compacted)
        if len(lines) >= 4:
            break

    if offset == 0 and len(sentences) <= 2 and lines:
        return lines[:4]

    keywords = extract_keywords(text, 10)
    while len(lines) < 4:
        if keywords:
            seed = " ".join(keywords[: min(4, len(keywords))])
            lines.append(_polish_line(f"{seed} còn vang trong ta"))
            keywords = keywords[1:]
        else:
            lines.append("một câu ca đi qua đêm dài")
    return lines[:4]


def _select_motif(text: str, emotion: EmotionProfile, limit: int = 10) -> str:
    lowered = text.lower()
    for phrase in MOTIF_PHRASES.get(emotion.label, []):
        if phrase in lowered:
            return phrase
    for phrase_list in MOTIF_PHRASES.values():
        for phrase in phrase_list:
            if phrase in lowered:
                return phrase

    candidate_phrase = _candidate_motif_phrase(lowered)
    if candidate_phrase:
        return candidate_phrase

    for keyword in extract_keywords(text, limit):
        if keyword not in MOTIF_STOPWORDS and len(keyword) > 3:
            return keyword
    return SAFE_FALLBACK_MOTIFS.get(emotion.label, emotion.label_vi)


def _candidate_motif_phrase(text: str) -> str:
    words = tokenize_words(text)
    for size in (3, 2):
        for start in range(0, max(0, len(words) - size + 1)):
            phrase_words = words[start : start + size]
            if any(word in MOTIF_STOPWORDS for word in phrase_words):
                continue
            phrase = " ".join(phrase_words)
            if len(phrase) >= 6:
                return phrase
    return ""


def _make_pre_chorus(text: str, emotion: EmotionProfile) -> list[str]:
    motif = _select_motif(text, emotion, 8)
    if emotion.valence < -0.2:
        return [f"ta giữ {motif} ở giữa lặng im", "để trái tim tìm lại lối về"]
    if emotion.energy > 0.65:
        return [f"ta gọi {motif} lên bằng nhịp thở", "để bước chân không dừng lại"]
    return [f"ta nghe {motif} đi qua thật khẽ", "rồi để lòng mình chậm lại hơn"]


def _make_bridge(text: str, emotion: EmotionProfile, harmony: HarmonyPlan) -> list[str]:
    center = _select_motif(text, emotion, 6)
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
    motif = _select_motif(text, emotion, 10)
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
    short_verse = verse[:4]
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
    if _looks_like_existing_lyrics(text):
        return _rewrite_existing_lyrics(_existing_lyric_lines(text), emotion)

    sentence_count = len(split_sentences(text))
    word_count = len(tokenize_words(text))
    verse1 = _shape_lines_for_melody(_make_verse_lines(text, offset=0), emotion, start_pair=0)
    verse2 = _shape_lines_for_melody(_make_verse_lines(text, offset=4), emotion, start_pair=3)
    pre_chorus = _shape_lines_for_melody(_make_pre_chorus(text, emotion), emotion, start_pair=1)
    chorus = _shape_lines_for_melody(_make_chorus(text, emotion), emotion, start_pair=2)
    bridge = _shape_lines_for_melody(_make_bridge(text, emotion, harmony), emotion, start_pair=4)
    outro = _shape_lines_for_melody(_make_outro(chorus, emotion), emotion, start_pair=2)
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
        rhyme_scheme="Vietnamese mixed rhyme: paired end rhymes, head-tail links, and luc-bat-aware detection",
    )
