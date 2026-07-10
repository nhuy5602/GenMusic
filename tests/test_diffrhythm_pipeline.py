from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from genmusic_vn.data.vietnamese_g2p import vietnamese_g2p
from genmusic_vn.data.vietnamese_text import normalize_vietnamese_lyrics
from genmusic_vn.integrations.diffrhythm_official import (
    DEFAULT_MODEL_REF,
    create_random_official_dataset,
    ensure_official_checkout,
    official_train_command,
    validate_official_dataset,
)
from genmusic_vn.integrations.kaggle_auto import KaggleJobConfig, stage_text_to_music_job


class DiffRhythmPipelineTests(unittest.TestCase):
    def test_project_uses_official_diff_rhythm_job(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            state = stage_text_to_music_job(
                text="Mưa rơi trên phố cũ.\nEm còn nhớ anh.",
                output_root=temp,
                duration_seconds=30,
                genre="Vietnamese pop ballad, piano",
                config=KaggleJobConfig(submit=False),
            )
            self.assertEqual(state["backend"], "ASLP-lab/DiffRhythm")
            self.assertEqual(state["model"], DEFAULT_MODEL_REF)
            self.assertEqual(state["audio_length"], 95)
            self.assertTrue(Path(state["dataset_dir"], "lyrics.lrc").exists())
            script = Path(state["kernel_dir"], "run_diffrhythm.py").read_text(encoding="utf-8")
            self.assertIn("genmusic_vn_source.zip", script)
            self.assertIn("third_party", script)
            self.assertNotIn("git clone", script)
            self.assertIn("infer.py", script)

    def test_official_source_is_vendored(self) -> None:
        source = ensure_official_checkout()
        self.assertTrue((source / "infer" / "infer.py").exists())
        self.assertTrue((source / "requirements.txt").exists())

    def test_random_dataset_reports_dependency_or_creates_official_format(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            report = create_random_official_dataset(Path(temp) / "random", count=2, max_frames=64)
            self.assertIn(report["status"], {"needs-torch", "created"})
            if report["status"] == "created":
                self.assertTrue(Path(temp, "random", "train.scp").exists())
            else:
                self.assertIn("torch", report["message"])

    def test_official_dataset_preflight_and_train_command(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for folder in ("lrc", "latent", "style"):
                (root / folder).mkdir()
            for folder in ("lrc", "latent", "style"):
                (root / folder / "one.pt").write_bytes(b"placeholder")
            (root / "train.scp").write_text("one|lrc/one.pt|latent/one.pt|style/one.pt\n", encoding="utf-8")
            (root / "random_dataset_report.json").write_text(json.dumps({"max_frames": 64}), encoding="utf-8")
            report = validate_official_dataset(root)
            command = official_train_command("third_party/DiffRhythm", root, epochs=1, batch_size=1)
        self.assertEqual(report["status"], "valid")
        self.assertIn("train.py", " ".join(command))
        self.assertIn("--model-config", command)

    def test_vietnamese_text_contract(self) -> None:
        self.assertIn("mười hai", normalize_vietnamese_lyrics("Mưa 12 ngày, ko về."))
        result = vietnamese_g2p("Sóng gió", use_phonemizer=False)
        self.assertEqual(result.backend, "rule-based-ipa")
        self.assertEqual(len(result.tokens), 2)


if __name__ == "__main__":
    unittest.main()
