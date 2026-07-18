from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.models.cfm_flow import build_mismatched_texts, cfm_loss
from src.models.dit_transformer import InputEmbedding
from src.models.text_to_music_diffusion import MusicDiffusionConfig, denormalize_mel, normalize_mel
from src.training.self_diffusion import (
    MusicDiffusionDataset,
    _is_checkpoint_improvement,
    clean_vietnamese_lyric,
    lyric_text_for_window,
    split_training_records,
    validate_dataset,
)
from scripts.evaluate_generation_quality import (
    evenly_spaced_records,
    generation_candidate_rank,
    transcription_metrics,
)
class ModelImprovementTests(unittest.TestCase):
    def test_content_negative_uses_a_different_nonempty_lyric(self) -> None:
        mismatched, valid = build_mismatched_texts(
            ["em yeu anh", "mua roi", "", "em yeu anh"]
        )

        self.assertEqual(mismatched, ["mua roi", "em yeu anh", "", "mua roi"])
        self.assertEqual(valid, [True, True, False, True])

    def test_cfg_tie_prefers_realistic_vocal_over_confident_drone(self) -> None:
        real = {
            "pitch_std_semitones": 2.6,
            "spectral_flatness": 0.022,
            "voiced_ratio": 0.86,
        }
        natural = {
            "asr": {"word_accuracy": 0.0, "cer": 1.0},
            "metrics": {
                "pitch_std_semitones": 2.7,
                "spectral_flatness": 0.029,
                "voiced_ratio": 0.60,
                "mean_voiced_prob": 0.05,
            },
        }
        drone = {
            "asr": {"word_accuracy": 0.0, "cer": 1.0},
            "metrics": {
                "pitch_std_semitones": 0.35,
                "spectral_flatness": 0.00004,
                "voiced_ratio": 1.0,
                "mean_voiced_prob": 0.70,
            },
        }

        self.assertGreater(
            generation_candidate_rank(natural, real),
            generation_candidate_rank(drone, real),
        )

    def test_quality_samples_cover_the_full_combined_dataset(self) -> None:
        records = [{"id": str(index)} for index in range(10)]

        selected = evenly_spaced_records(records, 4)

        self.assertEqual([record["id"] for record in selected], ["0", "3", "6", "9"])

    def test_best_checkpoint_rejects_conditioning_collapse(self) -> None:
        self.assertFalse(
            _is_checkpoint_improvement(0.20, 0.30, 0.08, 0.18, 0.001)
        )
        self.assertTrue(
            _is_checkpoint_improvement(0.20, 0.30, 0.22, 0.18, 0.001)
        )

    def test_resume_validation_samples_tensor_shapes_but_checks_all_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "mels").mkdir()
            (root / "config.json").write_text(
                json.dumps({"n_mels": 4}), encoding="utf-8"
            )
            records = []
            for index in range(3):
                vocal_path = root / "mels" / f"vocal-{index}.pt"
                backing_path = root / "mels" / f"backing-{index}.pt"
                torch.save(torch.zeros(4, 8), vocal_path)
                torch.save(torch.zeros(4, 8), backing_path)
                records.append({
                    "id": str(index),
                    "frames": 8,
                    "vocal_mel_path": f"mels/{vocal_path.name}",
                    "backing_mel_path": f"mels/{backing_path.name}",
                })
            (root / "records.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )

            report = validate_dataset(root, max_tensor_records=1)

        self.assertEqual(report["status"], "valid")
        self.assertEqual(report["path_checked_records"], 3)
        self.assertEqual(report["tensor_shape_checked_records"], 1)

    def test_frame_aligned_text_gate_has_nonzero_floor(self) -> None:
        embedding = InputEmbedding(mel_dim=4, text_dim=8, out_dim=8)
        with torch.no_grad():
            embedding.text_frame_gate.fill_(-100.0)

        self.assertGreaterEqual(float(embedding.text_frame_strength().detach()), 0.20)

    def test_text_sensitivity_loss_penalizes_identical_empty_branch(self) -> None:
        class TextIgnoringModel(torch.nn.Module):
            def forward(self, *, x, texts, timestep, style_prompt):
                return torch.zeros_like(x)

        clean = torch.randn(2, 12, 4)
        backing = torch.zeros_like(clean)
        style = torch.zeros(2, 8)
        config = MusicDiffusionConfig(n_mels=4)
        common = {
            "text_contrastive_weight": 0.0,
            "text_contrastive_prob": 1.0,
            "text_sensitivity_target": 0.20,
        }
        torch.manual_seed(17)
        baseline = cfm_loss(
            TextIgnoringModel(), clean, backing, style, ["xin chao", "viet nam"], config,
            lambda_vocal=0.0, text_sensitivity_weight=0.0, **common,
        )[0]
        torch.manual_seed(17)
        penalized = cfm_loss(
            TextIgnoringModel(), clean, backing, style, ["xin chao", "viet nam"], config,
            lambda_vocal=0.0, text_sensitivity_weight=2.0, **common,
        )[0]

        # Shortfall=0.20 is in Huber's linear region: (0.20 - 0.025) * 2.
        self.assertAlmostEqual(float(penalized - baseline), 0.35, places=5)

    def test_content_loss_stays_finite_for_large_fp16_velocities(self) -> None:
        class LargeVelocityModel(torch.nn.Module):
            def forward(self, *, x, texts, timestep, style_prompt):
                # Squaring this in float16 overflows; cfm_loss must promote its
                # numerical objective to float32 before doing so.
                return torch.full_like(x, 400.0, dtype=torch.float16)

        clean = torch.randn(2, 12, 4)
        backing = torch.zeros_like(clean)
        style = torch.zeros(2, 8)
        loss = cfm_loss(
            LargeVelocityModel(),
            clean,
            backing,
            style,
            ["em yeu anh", "mua roi"],
            MusicDiffusionConfig(n_mels=4),
            lambda_vocal=0.0,
            text_contrastive_weight=0.5,
            text_contrastive_prob=1.0,
            text_sensitivity_weight=1.0,
            text_sensitivity_target=0.08,
        )[0]

        self.assertTrue(torch.isfinite(loss))

    def test_collapsed_text_response_has_finite_backward(self) -> None:
        class TextIgnoringModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.bias = torch.nn.Parameter(torch.tensor(0.0))

            def forward(self, *, x, texts, timestep, style_prompt):
                # Both lyric branches are exactly equal. This is the collapse
                # state the sensitivity objective must escape without sqrt(0)
                # creating a NaN gradient.
                return torch.zeros_like(x) + self.bias

        model = TextIgnoringModel()
        clean = torch.randn(2, 12, 4)
        loss = cfm_loss(
            model,
            clean,
            torch.zeros_like(clean),
            torch.zeros(2, 8),
            ["em yeu anh", "mua roi"],
            MusicDiffusionConfig(n_mels=4),
            lambda_vocal=0.0,
            text_contrastive_weight=0.5,
            text_contrastive_prob=1.0,
            text_sensitivity_weight=1.0,
            text_sensitivity_target=0.08,
        )[0]

        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.bias.grad)
        self.assertTrue(torch.isfinite(model.bias.grad))

    def test_validation_split_is_stable_and_disjoint(self) -> None:
        records = [{"id": f"song-{index}", "text": "lời bài hát"} for index in range(20)]
        train_a, validation_a = split_training_records(records, validation_fraction=0.2, seed=123)
        train_b, validation_b = split_training_records(
            [dict(record) for record in records], validation_fraction=0.2, seed=123
        )

        train_ids = {record["id"] for record in train_a}
        validation_ids = {record["id"] for record in validation_a}
        self.assertFalse(train_ids & validation_ids)
        self.assertEqual(len(validation_ids), 4)
        self.assertEqual(validation_ids, {record["id"] for record in validation_b})
        self.assertEqual(train_ids, {record["id"] for record in train_b})

    def test_transcription_metrics_are_accent_insensitive(self) -> None:
        metrics = transcription_metrics(
            "Đêm nay mưa rơi trên lối cũ",
            "dem nay mua roi tren loi cu",
        )

        self.assertEqual(metrics["wer"], 0.0)
        self.assertEqual(metrics["cer"], 0.0)
        self.assertEqual(metrics["word_accuracy"], 1.0)


    def test_old_segment_records_get_approximate_word_alignment(self) -> None:
        segments = [{"start": 0.0, "end": 4.0, "text": "mot hai ba bon"}]

        selected = lyric_text_for_window("mot hai ba bon", segments, 1.1, 2.9)
        silence = lyric_text_for_window("mot hai ba bon", segments, 5.0, 6.0)

        self.assertEqual(selected, "hai ba")
        self.assertEqual(silence, "")

    def test_training_label_filter_rejects_mixed_asr_noise(self) -> None:
        self.assertEqual(
            clean_vietnamese_lyric("thì Since my love comes to the past và tủy trùr"),
            "",
        )
        self.assertEqual(clean_vietnamese_lyric("Hãy subscribe cho kênh của mình"), "")
        self.assertTrue(clean_vietnamese_lyric("Anh sẽ chờ em dù biển xanh kia cạn khô"))

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

            dataset = MusicDiffusionDataset(root, config, lyric_aligned_crop_prob=0.0)
            with (
                patch("src.training.self_diffusion.random.random", return_value=1.0),
                patch("src.training.self_diffusion.random.randint", return_value=100),
            ):
                item = dataset[0]

            self.assertNotIn("Vietnamese pop", item["text"])
            self.assertEqual(item["text"], "hai ba")
            self.assertEqual(tuple(item["vocal_mel"].shape), (100, 100))


if __name__ == "__main__":
    unittest.main()
