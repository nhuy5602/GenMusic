from __future__ import annotations

from collections import Counter
import unicodedata

from .schemas import TextPlan
from .text_utils import extract_keywords, normalize_text, split_sentences, tokenize_words


def build_text_plan(
    text: str,
    max_sentences: int = 8,
    max_chars: int = 1800,
    duration_seconds: int | None = None,
) -> TextPlan:
    normalized = normalize_text(text)
    if _looks_like_lyrics(normalized):
        return _build_lyric_text_plan(normalized, duration_seconds=duration_seconds)

    sentences = split_sentences(normalized)
    words = tokenize_words(normalized)
    keywords = extract_keywords(normalized, limit=14)

    if not sentences:
        sentences = [normalized] if normalized else []

    mode = _mode(len(sentences), len(words))
    if len(sentences) <= max_sentences:
        representatives = list(sentences)
    else:
        representatives = _pick_representative_sentences(sentences, keywords, max_sentences=max_sentences)

    condensed = _fit_to_chars(" ".join(representatives), max_chars=max_chars)
    sections = {
        "opening": sentences[: min(2, len(sentences))],
        "development": _middle_sentences(sentences, keywords, limit=4),
        "ending": sentences[-min(2, len(sentences)) :] if sentences else [],
    }

    return TextPlan(
        mode=mode,
        sentence_count=len(sentences),
        word_count=len(words),
        keywords=keywords,
        representative_sentences=representatives,
        condensed_text=condensed or normalized,
        sections=sections,
        input_kind="prose",
    )


def _build_lyric_text_plan(text: str, *, duration_seconds: int | None = None) -> TextPlan:
    lines = _lyric_lines(text)
    words = tokenize_words(text)
    keywords = extract_keywords(text, limit=14)
    max_lines = _lyric_line_budget(duration_seconds)
    selected = _select_lyric_excerpt(lines, max_lines=max_lines)
    condensed = "\n".join(selected)
    sections = {
        "opening": selected[: min(4, len(selected))],
        "chorus_candidate": _find_repeated_block(lines, max_lines=4),
        "ending": selected[-min(2, len(selected)) :] if selected else [],
        "selected_lyric_lines": selected,
    }
    return TextPlan(
        mode="lyrics_long" if len(lines) > max_lines else "lyrics",
        sentence_count=len(lines),
        word_count=len(words),
        keywords=keywords,
        representative_sentences=selected,
        condensed_text=condensed or text,
        sections=sections,
        input_kind="lyrics",
    )


def _looks_like_lyrics(text: str) -> bool:
    lines = _lyric_lines(text)
    if len(lines) < 6:
        return False
    short_lines = sum(1 for line in lines if 2 <= len(tokenize_words(line)) <= 14)
    return short_lines / len(lines) >= 0.65


def _lyric_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw_line in normalize_text(text).splitlines():
        line = raw_line.strip(" \t,;:-")
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            continue
        if tokenize_words(line):
            lines.append(line)
    return lines


def _lyric_line_budget(duration_seconds: int | None) -> int:
    if duration_seconds is None:
        return 12
    return max(6, min(16, int(max(12, duration_seconds) / 5)))


def _select_lyric_excerpt(lines: list[str], *, max_lines: int) -> list[str]:
    if len(lines) <= max_lines:
        return lines

    opening_count = min(4, max_lines)
    selected = list(lines[:opening_count])

    chorus_block = _find_repeated_block(lines, max_lines=min(4, max_lines - len(selected)))
    for line in chorus_block:
        if len(selected) >= max_lines:
            break
        if line not in selected:
            selected.append(line)

    ending_room = min(2, max_lines - len(selected))
    ending = lines[-ending_room:] if ending_room > 0 else []
    middle_slots = max_lines - len(selected) - len([line for line in ending if line not in selected])
    if middle_slots > 0:
        for line in _emotionally_dense_lines(lines, limit=middle_slots):
            if len(selected) >= max_lines - len(ending):
                break
            if line not in selected and line not in ending:
                selected.append(line)

    for line in ending:
        if len(selected) >= max_lines:
            break
        if line not in selected:
            selected.append(line)
    return selected[:max_lines]


def _find_repeated_block(lines: list[str], *, max_lines: int) -> list[str]:
    if max_lines <= 0:
        return []
    normalized = [_normalize_key(line) for line in lines]
    counts = Counter(normalized)
    for index, key in enumerate(normalized):
        if counts[key] >= 2 and len(tokenize_words(lines[index])) >= 4:
            return lines[index : index + max_lines]
    return []


def _emotionally_dense_lines(lines: list[str], *, limit: int) -> list[str]:
    keywords = extract_keywords("\n".join(lines), limit=18)
    scored = _score_sentences(lines, keywords)
    return [lines[index] for index, _score in scored[:limit]]


def _normalize_key(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.lower())
    stripped = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    return " ".join(tokenize_words(stripped.replace("đ", "d")))


def _mode(sentence_count: int, word_count: int) -> str:
    if sentence_count <= 2 and word_count <= 60:
        return "short"
    if sentence_count <= 8 and word_count <= 220:
        return "medium"
    return "long"


def _pick_representative_sentences(sentences: list[str], keywords: list[str], max_sentences: int) -> list[str]:
    chosen_indexes = {0, min(1, len(sentences) - 1), len(sentences) - 1}
    if len(sentences) > 2:
        chosen_indexes.add(len(sentences) - 2)

    remaining_slots = max(0, max_sentences - len(chosen_indexes))
    scored = _score_sentences(sentences, keywords)
    for index, _score in scored:
        if remaining_slots <= 0:
            break
        if index in chosen_indexes:
            continue
        chosen_indexes.add(index)
        remaining_slots -= 1

    return [sentences[index] for index in sorted(chosen_indexes)]


def _middle_sentences(sentences: list[str], keywords: list[str], limit: int) -> list[str]:
    if len(sentences) <= 4:
        return sentences
    scored = _score_sentences(sentences[1:-1], keywords)
    picked = [sentences[index + 1] for index, _score in scored[:limit]]
    return picked


def _score_sentences(sentences: list[str], keywords: list[str]) -> list[tuple[int, float]]:
    keyword_set = set(keywords)
    keyword_weights = Counter(keywords)
    scored: list[tuple[int, float]] = []
    for index, sentence in enumerate(sentences):
        tokens = tokenize_words(sentence)
        if not tokens:
            scored.append((index, 0.0))
            continue
        keyword_hits = sum(1.0 + keyword_weights[token] * 0.15 for token in tokens if token in keyword_set)
        emotional_terms = sum(1 for token in tokens if token in {"buon", "vui", "nho", "yeu", "so", "hy", "vong", "binh", "yen", "mua", "dem", "sang"})
        length_bonus = min(len(tokens), 18) / 18.0
        position_bonus = 0.15 if index in {0, len(sentences) - 1} else 0.0
        scored.append((index, keyword_hits + emotional_terms * 0.4 + length_bonus + position_bonus))
    return sorted(scored, key=lambda item: (-item[1], item[0]))


def _fit_to_chars(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    sentences = split_sentences(text)
    fitted: list[str] = []
    current = 0
    for sentence in sentences:
        added = len(sentence) + (1 if fitted else 0)
        if current + added > max_chars:
            break
        fitted.append(sentence)
        current += added
    return " ".join(fitted) if fitted else text[:max_chars].rstrip()
