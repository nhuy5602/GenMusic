from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.models.dit_transformer import align_text_embeddings_to_frames
from src.models.text_to_music_diffusion import MusicDiffusionConfig, denormalize_mel, normalize_mel
from src.training.self_diffusion import MusicDiffusionDataset, lyric_text_for_window


class ModelImprovementTests(unittest.TestCase):
    def test_text_alignment_ignores_padding_embeddings(self) -> None:
        embeddings = torch.tensor(
            [
                [[0.0], [1.0], [2.0], [3.0], [999.0]],
                [[0.0], [4.0], [999.0], [999.0], [999.0]],
            ]
        )
        mask = torch.tensor(
            [
                [True, True, True, True, False],
                [True, True, False, False, False],
            ]
        )

        aligned = align_text_embeddings_to_frames(embeddings, mask, frames=8)

        self.assertEqual(tuple(aligned.shape), (2, 8, 1))
        self.assertLess(float(aligned.max()), 10.0)

    def test_old_segment_records_get_approximate_word_alignment(self) -> None:
        segments = [{"start": 0.0, "end": 4.0, "text": "mot hai ba bon"}]

        selected = lyric_text_for_window("mot hai ba bon", segments, 1.1, 2.9)
        silence = lyric_text_for_window("mot hai ba bon", segments, 5.0, 6.0)

        self.assertEqual(selected, "hai ba")
        self.assertEqual(silence, "")

    def test_mel_normalization_round_trip(self) -> None:
        config = MusicDiffusionConfig(mel_mean=-4.0, mel_std=2.0)
        raw = torch.tensor([[-6.0, -4.0, 0.0]])

        restored = denormalize_mel(normalize_mel(raw, config), config)

        self.assertTrue(torch.allclose(restored, raw))

    def test_dataset_returns_lyrics_without_style_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "mels").mkdir()
            config = MusicDiffusionConfig(frames_per_chunk=100)
            vocal = torch.linspace(-6.0, 1.0, 100 * 400).reshape(100, 400)
            backing = vocal + 0.25
            torch.save(vocal, root / "mels" / "vocal.pt")
            torch.save(backing, root / "mels" / "backing.pt")
            record = {
                "id": "song",
                "text": "mot hai ba bon",
                "style": "Vietnamese pop, 120 BPM",
                "frames": 400,
                "has_vocal": True,
                "vocal_mel_path": "mels/vocal.pt",
                "backing_mel_path": "mels/backing.pt",
                "segments": [{"start": 0.0, "end": 4.0, "text": "mot hai ba bon"}],
            }
            (root / "records.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

            dataset = MusicDiffusionDataset(root, config)
            with patch("src.training.self_diffusion.random.randint", return_value=100):
                item = dataset[0]

            self.assertNotIn("Vietnamese pop", item["text"])
            self.assertEqual(item["text"], "hai ba")
            self.assertEqual(tuple(item["vocal_mel"].shape), (100, 100))


if __name__ == "__main__":
    unittest.main()
