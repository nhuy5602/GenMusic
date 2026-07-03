from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from genmusic_vn.schemas import EmotionProfile, GeneratedFile, HarmonyPlan, LyricDraft, NoteEvent, VocalPlan


class GeneratorUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class GeneratorInput:
    run_id: str
    text: str
    prompt: str
    negative_prompt: str
    emotion: EmotionProfile
    harmony: HarmonyPlan
    lyrics: LyricDraft
    vocal: VocalPlan
    melody: list[NoteEvent]
    duration_seconds: int


class MusicGenerator:
    backend_name = "base"

    def generate(self, data: GeneratorInput, output_dir: Path) -> list[GeneratedFile]:
        raise NotImplementedError
