from __future__ import annotations

import os
from pathlib import Path

from genmusic_vn.schemas import GeneratedFile

from .base import GeneratorInput, GeneratorUnavailable, MusicGenerator


class MusicGenGenerator(MusicGenerator):
    backend_name = "musicgen"

    def generate(self, data: GeneratorInput, output_dir: Path) -> list[GeneratedFile]:
        try:
            from audiocraft.data.audio import audio_write
            from audiocraft.models import MusicGen
        except ImportError as exc:
            raise GeneratorUnavailable(
                "AudioCraft/MusicGen is not installed. Run this backend on Kaggle GPU with `pip install -U audiocraft`."
            ) from exc

        output_dir.mkdir(parents=True, exist_ok=True)
        model_name = os.getenv("MUSICGEN_MODEL", "facebook/musicgen-small")
        model = MusicGen.get_pretrained(model_name)
        model.set_generation_params(duration=data.duration_seconds)
        wav = model.generate([data.prompt])
        stem = output_dir / "musicgen"
        audio_write(str(stem), wav[0].cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
        return [GeneratedFile(kind="audio", path=str(stem.with_suffix(".wav")), description=f"MusicGen output: {model_name}")]

