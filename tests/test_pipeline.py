from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from genmusic_vn.emotion import analyze_emotion
from genmusic_vn.kaggle_auto import KaggleJobConfig, run_or_stage_kaggle_job, slugify
from genmusic_vn.music_theory import build_harmony, chord_notes
from genmusic_vn.pipeline import create_music_project


class PipelineTests(unittest.TestCase):
    def test_vietnamese_emotion_detects_sadness(self) -> None:
        profile = analyze_emotion("Một chiều mưa rất buồn, tôi cô đơn nhớ con phố cũ.")
        self.assertIn(profile.label, {"sadness", "nostalgic"})
        self.assertLess(profile.valence, 0.2)

    def test_chord_notes_support_common_progression(self) -> None:
        self.assertEqual(chord_notes("Am"), ["A4", "C5", "E5"])
        self.assertEqual(chord_notes("Fmaj7"), ["F4", "A4", "C5", "E5"])

    def test_pipeline_exports_prompt_pack_without_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            result = create_music_project(
                "Đêm thành phố sáng lên, lòng người vẫn tìm một nơi bình yên.",
                output_root=temp,
                duration_seconds=8,
                render_audio=False,
            )
            run_dir = Path(temp) / result.run_id
            pack = json.loads((run_dir / "prompt_pack.json").read_text(encoding="utf-8"))
            self.assertIn("prompt", pack)
            self.assertEqual(pack["duration_seconds"], 8)

    def test_pipeline_renders_guide_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            result = create_music_project(
                "Ngày mai ta đi tiếp qua những ngập ngừng và giữ ánh sáng trong tim.",
                output_root=temp,
                duration_seconds=6,
                render_audio=True,
            )
            paths = {Path(file.path).name for file in result.files}
            self.assertIn("guide.wav", paths)
            self.assertIn("guide.mid", paths)

    def test_kaggle_job_can_be_staged_without_submit(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            result = create_music_project(
                "Một đoạn văn yên bình để demo tự động hóa Kaggle.",
                output_root=temp,
                duration_seconds=6,
                render_audio=False,
            )
            job = run_or_stage_kaggle_job(
                result,
                temp,
                KaggleJobConfig(username="demo-user", submit=False),
            )
            self.assertEqual(job["status"], "staged")
            self.assertTrue((Path(job["dataset_dir"]) / "prompt_pack.json").exists())
            self.assertTrue((Path(job["kernel_dir"]) / "kernel-metadata.json").exists())

    def test_slugify_keeps_kaggle_safe_slug(self) -> None:
        self.assertEqual(slugify("GenMusic Việt Nam Demo!!!", 50), "genmusic-viet-nam-demo")


if __name__ == "__main__":
    unittest.main()
