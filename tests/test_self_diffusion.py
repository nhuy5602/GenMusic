from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from genmusic_vn.data.vietnamese_g2p import vietnamese_g2p
from genmusic_vn.data.vietnamese_text import normalize_vietnamese_lyrics
from genmusic_vn.integrations.kaggle_auto import DEFAULT_MODEL, KaggleJobConfig, run_local_generation, stage_text_to_music_job
from genmusic_vn.training.self_diffusion import create_random_dataset, train_model, validate_dataset


class SelfDiffusionTests(unittest.TestCase):
    def test_random_dataset_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            report = create_random_dataset(Path(temp) / "dataset", count=2, frames=32)
            self.assertEqual(report["backend"], "genmusic-vn-self-diffusion")
            validation = validate_dataset(Path(temp) / "dataset")
            self.assertEqual(validation["status"], "valid")

    def test_training_and_local_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = root / "dataset"
            create_random_dataset(dataset, count=2, frames=32)
            report = train_model(dataset, root / "model.pt", epochs=1, batch_size=2, max_records=2)
            self.assertEqual(report["status"], "complete")
            generated = run_local_generation(text="Mưa rơi nhẹ nhàng.", style="soft piano", output_dir=root / "audio", duration_seconds=1, checkpoint=root / "model.pt", steps=1)
            self.assertEqual(generated["status"], "complete")
            self.assertTrue(Path(generated["audio_path"]).exists())

    def test_kaggle_job_contains_only_project_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            state = stage_text_to_music_job(text="Một ngày mới.", output_root=temp, duration_seconds=4, genre="acoustic pop", config=KaggleJobConfig(submit=False))
            self.assertEqual(state["backend"], "genmusic-vn-self-diffusion")
            self.assertEqual(state["model"], DEFAULT_MODEL)
            script = Path(state["kernel_dir"], "run_genmusic.py").read_text(encoding="utf-8")
            source_zip = Path(state["dataset_dir"], "genmusic_vn_source.zip")
            with zipfile.ZipFile(source_zip) as archive:
                self.assertIn("genmusic_vn/models/text_to_music_diffusion.py", archive.namelist())
            self.assertIn("genmusic_vn_source.zip", script)
            self.assertIn("train-self", script)
            self.assertNotIn("git clone", script.lower())
            self.assertNotIn("raw.", script.lower())

    def test_vietnamese_text_contract(self) -> None:
        self.assertIn("mười hai", normalize_vietnamese_lyrics("Mưa 12 ngày, ko về."))
        result = vietnamese_g2p("Sóng gió", use_phonemizer=False)
        self.assertEqual(result.backend, "rule-based-ipa")
        self.assertEqual(len(result.tokens), 2)


if __name__ == "__main__":
    unittest.main()
