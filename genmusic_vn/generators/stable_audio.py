from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from genmusic_vn.schemas import GeneratedFile

from .base import GeneratorInput, GeneratorUnavailable, MusicGenerator


class StableAudioCliGenerator(MusicGenerator):
    backend_name = "stable-audio"

    def generate(self, data: GeneratorInput, output_dir: Path) -> list[GeneratedFile]:
        executable = shutil.which("stable-audio")
        if not executable:
            raise GeneratorUnavailable(
                "Stable Audio CLI is not installed. Run this backend on Kaggle after installing Stable Audio tooling."
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "stable_audio.wav"
        model_name = os.getenv("STABLE_AUDIO_MODEL", "small-music")
        cmd = [
            executable,
            "--model",
            model_name,
            "-p",
            data.prompt,
            "--duration",
            str(data.duration_seconds),
            "-o",
            str(output_path),
        ]
        subprocess.run(cmd, check=True)
        return [GeneratedFile(kind="audio", path=str(output_path), description=f"Stable Audio CLI output: {model_name}")]

