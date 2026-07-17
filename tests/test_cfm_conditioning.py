from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.models.cfm_flow import sample_cfm
from src.models.text_to_music_diffusion import MusicDiffusionConfig, generate_audio


class _RecordingFlowModel:
    def __init__(self) -> None:
        self.calls = []
        self.style_dim = 512

    def eval(self):
        return self

    def to(self, _device):
        return self

    def __call__(self, x, texts, timestep, style_prompt=None):
        self.calls.append(
            {
                "style": style_prompt.detach().cpu().clone() if style_prompt is not None else None,
                "texts": texts,
                "timestep": timestep.detach().cpu().clone(),
            }
        )
        return torch.zeros_like(x)


class ConditioningParityTests(unittest.TestCase):
    def test_sample_cfm_forwards_real_style(self) -> None:
        config = MusicDiffusionConfig()
        model = _RecordingFlowModel()
        style = torch.arange(512, dtype=torch.float32)

        generated = sample_cfm(
            model,
            ["lyrics"],
            5,
            config,
            "cpu",
            steps=1,
            seed=1,
            style_prompt=style,
        )

        self.assertEqual(generated.dim(), 3)
        self.assertEqual(generated.shape[1], config.n_mels)
        call = model.calls[0]
        self.assertTrue(torch.equal(call["style"], style.unsqueeze(0)))
        self.assertEqual(call["texts"], ["lyrics"])

    def test_generate_audio_passes_style_anchor(self) -> None:
        config = MusicDiffusionConfig()
        model = _RecordingFlowModel()
        style = torch.ones(1, 512)
        sampled_texts = []

        def fake_sample(_model, _texts, frames, **kwargs):
            sampled_texts.append(_texts)
            self.assertTrue(torch.equal(kwargs["style_prompt"], style))
            return torch.zeros((1, config.n_mels, frames), dtype=torch.float32)

        def fake_render(_mel, destination, _config, vocoder_type="vocos"):
            path = Path(destination)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"wav")
            return path.resolve()

        with tempfile.TemporaryDirectory() as temp, patch(
            "src.models.cfm_flow.sample_cfm", side_effect=fake_sample
        ), patch(
            "src.models.text_to_music_diffusion.render_mel_to_wav", side_effect=fake_render
        ):
            report = generate_audio(
                model,
                "mot cau",
                "Vietnamese pop",
                Path(temp) / "conditioned.wav",
                duration_seconds=8.0,
                config=config,
                style_prompt=style,
                steps=2,
            )

        self.assertEqual(sampled_texts, [["mot cau"]])
        self.assertTrue(report["muq_style_conditioned"])


if __name__ == "__main__":
    unittest.main()
