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

    def eval(self):
        return self

    def to(self, _device):
        return self

    def __call__(self, *, x, cond, texts, timestep, style_prompt):
        self.calls.append(
            {
                "cond": cond.detach().cpu().clone(),
                "style": style_prompt.detach().cpu().clone() if style_prompt is not None else None,
                "texts": texts,
                "timestep": timestep.detach().cpu().clone(),
            }
        )
        return torch.zeros_like(x)


class ConditioningParityTests(unittest.TestCase):
    def test_sample_cfm_forwards_real_backing_and_style(self) -> None:
        config = MusicDiffusionConfig()
        model = _RecordingFlowModel()
        backing = torch.arange(config.n_mels * 3, dtype=torch.float32).reshape(1, config.n_mels, 3)
        style = torch.arange(512, dtype=torch.float32)

        generated = sample_cfm(
            model,
            ["lyrics"],
            5,
            config,
            "cpu",
            steps=1,
            seed=1,
            backing_mel=backing,
            style_prompt=style,
        )

        self.assertEqual(tuple(generated.shape), (1, config.n_mels, 5))
        call = model.calls[0]
        self.assertEqual(tuple(call["cond"].shape), (1, 5, config.n_mels))
        self.assertTrue(torch.equal(call["cond"][:, :3], backing.transpose(1, 2)))
        self.assertTrue(torch.count_nonzero(call["cond"][:, 3:]) == 0)
        self.assertTrue(torch.equal(call["style"], style.unsqueeze(0)))

    def test_generate_audio_slices_backing_at_matching_chunk_offsets(self) -> None:
        config = MusicDiffusionConfig()
        model = _RecordingFlowModel()
        duration_seconds = 8.0
        total_frames = int(duration_seconds * config.sample_rate / config.hop_length) + 8
        backing = (torch.arange(total_frames, dtype=torch.float32) / 1000.0).view(1, 1, -1).expand(1, config.n_mels, -1)
        style = torch.ones(1, 512)
        sampled_backing = []
        sampled_texts = []

        def fake_sample(_model, _texts, frames, **kwargs):
            sampled_backing.append(kwargs["backing_mel"].detach().cpu().clone())
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
                duration_seconds=duration_seconds,
                config=config,
                backing_mel=backing,
                style_prompt=style,
                steps=2,
            )

        self.assertEqual(len(sampled_backing), 2)
        first_chunk_frames = sampled_backing[0].shape[-1]
        self.assertEqual(float(sampled_backing[0][0, 0, 0]), 0.0)
        self.assertAlmostEqual(
            float(sampled_backing[1][0, 0, 0]),
            float(backing[0, 0, first_chunk_frames]),
            places=6,
        )
        self.assertEqual(sampled_texts, [["mot cau"], ["mot cau"]])
        self.assertTrue(report["backing_conditioned"])
        self.assertTrue(report["muq_style_conditioned"])


if __name__ == "__main__":
    unittest.main()
