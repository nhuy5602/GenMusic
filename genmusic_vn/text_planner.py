from __future__ import annotations

from collections import Counter

from .schemas import TextPlan
from .text_utils import extract_keywords, normalize_text, split_sentences, tokenize_words


def build_text_plan(text: str, max_sentences: int = 8, max_chars: int = 1800) -> TextPlan:
    normalized = normalize_text(text)
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
    )


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

