from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any


@dataclass(frozen=True)
class EmotionProfile:
    label: str
    label_vi: str
    valence: float
    energy: float
    confidence: float
    keywords: list[str]
    scores: dict[str, float]


@dataclass(frozen=True)
class HarmonyPlan:
    key: str
    scale: str
    bpm: int
    time_signature: str
    chord_progression: list[str]
    chord_notes: dict[str, list[str]]
    note_pool: list[str]
    melody_register: str
    instruments: list[str]
    arrangement: list[str]
    music_traits: list[str]


@dataclass(frozen=True)
class NoteEvent:
    start: float
    duration: float
    note: str
    midi: int
    velocity: int
    lyric: str = ""


@dataclass(frozen=True)
class LyricDraft:
    title: str
    verse: list[str]
    chorus: list[str]
    bridge: list[str]
    hook: str
    song_form: list[str] = field(default_factory=list)
    full_song: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TextPlan:
    mode: str
    sentence_count: int
    word_count: int
    keywords: list[str]
    representative_sentences: list[str]
    condensed_text: str
    sections: dict[str, list[str]]


@dataclass(frozen=True)
class GeneratedFile:
    kind: str
    path: str
    description: str
    url: str = ""


@dataclass(frozen=True)
class MusicResult:
    run_id: str
    input_text: str
    backend: str
    duration_seconds: int
    emotion: EmotionProfile
    harmony: HarmonyPlan
    melody: list[NoteEvent]
    lyrics: LyricDraft
    text_plan: TextPlan
    prompt: str
    negative_prompt: str
    files: list[GeneratedFile] = field(default_factory=list)


def to_plain_data(value: Any) -> Any:
    if is_dataclass(value):
        return {key: to_plain_data(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: to_plain_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_plain_data(item) for item in value]
    return value
