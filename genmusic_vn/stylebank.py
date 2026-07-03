from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


STYLEBANK_DIR = Path(__file__).resolve().parents[1] / "datasets" / "vn_music_stylebank"


@lru_cache(maxsize=1)
def load_stylebank() -> dict[str, Any]:
    return {
        "emotion_to_music": _read_json("emotion_to_music.json"),
        "vietnamese_instruments": _read_json("vietnamese_instruments.json"),
        "genre_templates": _read_json("genre_templates.json"),
        "chord_presets": _read_json("chord_presets.json"),
        "lyric_patterns": _read_json("lyric_patterns.json"),
    }


def get_emotion_music(label: str) -> dict[str, Any]:
    bank = load_stylebank()["emotion_to_music"]
    emotions = bank.get("emotions", {})
    return dict(emotions.get(label) or emotions.get("calm") or {})


def get_lyric_pattern(label: str) -> dict[str, Any]:
    patterns = load_stylebank()["lyric_patterns"].get("patterns", {})
    return dict(patterns.get(label) or patterns.get("calm") or {})


def get_instrument_prompt_tokens(instrument_ids: list[str]) -> list[str]:
    instruments = load_stylebank()["vietnamese_instruments"].get("instruments", {})
    tokens: list[str] = []
    for instrument_id in instrument_ids:
        entry = instruments.get(instrument_id, {})
        tokens.extend(entry.get("prompt_tokens", [])[:2])
    return tokens


def match_genre_template(genre: str | None, emotion_label: str) -> dict[str, Any]:
    genres = load_stylebank()["genre_templates"].get("genres", {})
    if not genres:
        return {}

    normalized = (genre or "").lower().replace("-", " ").replace("_", " ")
    for key, entry in genres.items():
        key_text = key.replace("_", " ")
        if key_text in normalized:
            return dict(entry)

    for entry in genres.values():
        if emotion_label in entry.get("recommended_emotions", []):
            return dict(entry)
    return dict(next(iter(genres.values())))


def stylebank_prompt_context(emotion_label: str, instrument_ids: list[str], genre: str | None) -> str:
    emotion_style = get_emotion_music(emotion_label)
    genre_style = match_genre_template(genre, emotion_label)
    tokens = []
    tokens.extend(emotion_style.get("prompt_keywords", [])[:4])
    tokens.extend(get_instrument_prompt_tokens(instrument_ids)[:4])
    if genre_style:
        tokens.extend([genre_style.get("rhythm", ""), genre_style.get("mix", "")])
    return "; ".join(token for token in tokens if token)


def _read_json(filename: str) -> dict[str, Any]:
    path = STYLEBANK_DIR / filename
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

