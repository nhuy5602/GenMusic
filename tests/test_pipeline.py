from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from genmusic_vn.emotion import analyze_emotion
from genmusic_vn.kaggle_auto import KaggleJobConfig, slugify, submit_text_to_music_job
from genmusic_vn.music_theory import chord_notes
from genmusic_vn.pipeline import create_music_project
from genmusic_vn.stylebank import get_emotion_music, load_stylebank


class PipelineTests(unittest.TestCase):
    def test_vietnamese_emotion_detects_sadness(self) -> None:
        profile = analyze_emotion("Một chiều mưa rất buồn, tôi cô đơn nhớ con phố cũ.")
        self.assertIn(profile.label, {"sadness", "nostalgic"})
        self.assertLess(profile.valence, 0.2)

    def test_chord_notes_support_common_progression(self) -> None:
        self.assertEqual(chord_notes("Am"), ["A4", "C5", "E5"])
        self.assertEqual(chord_notes("Fmaj7"), ["F4", "A4", "C5", "E5"])

    def test_pipeline_exports_prompt_pack_without_audio_for_kaggle_source(self) -> None:
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

    def test_long_text_is_planned_and_rewritten_as_full_song(self) -> None:
        long_text = " ".join(
            [
                f"Cau chuyen thu {index} noi ve con pho cu, mua dem va nhung loi hua con do."
                for index in range(1, 26)
            ]
            + [
                "Sau tat ca, nhan vat tim thay hy vong, anh sang va mot ngay moi dang mo ra."
            ]
        )
        with tempfile.TemporaryDirectory() as temp:
            result = create_music_project(
                long_text,
                output_root=temp,
                duration_seconds=12,
                render_audio=False,
            )
            self.assertEqual(result.text_plan.mode, "long")
            self.assertGreaterEqual(result.text_plan.sentence_count, 26)
            self.assertLess(len(result.text_plan.representative_sentences), result.text_plan.sentence_count)
            self.assertIn("hy vong", result.text_plan.condensed_text)
            self.assertIn("Final Chorus", result.lyrics.song_form)
            self.assertTrue(any("[Verse 2]" in line for line in result.lyrics.full_song))
            self.assertIn("song form:", result.prompt)

    def test_kaggle_job_stages_raw_text_request_and_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            job = submit_text_to_music_job(
                text="Một đoạn văn yên bình để demo tự động hóa Kaggle.",
                output_root=temp,
                duration_seconds=6,
                genre="Vietnamese cinematic pop",
                config=KaggleJobConfig(username="demo-user", submit=False),
            )
            self.assertEqual(job["status"], "staged")
            self.assertEqual(job["backend"], "musicgen")
            self.assertTrue((Path(job["dataset_dir"]) / "request.json").exists())
            source_zip = Path(job["dataset_dir"]) / "genmusic_vn_source.zip"
            self.assertTrue(source_zip.exists())
            with zipfile.ZipFile(source_zip) as archive:
                names = set(archive.namelist())
            self.assertIn("datasets/vn_music_stylebank/emotion_to_music.json", names)
            self.assertIn("genmusic_vn/stylebank.py", names)
            self.assertTrue((Path(job["kernel_dir"]) / "kernel-metadata.json").exists())

    def test_slugify_keeps_kaggle_safe_slug(self) -> None:
        self.assertEqual(slugify("GenMusic Việt Nam Demo!!!", 50), "genmusic-viet-nam-demo")

    def test_stylebank_loads_emotion_music_dataset(self) -> None:
        bank = load_stylebank()
        self.assertIn("emotion_to_music", bank)
        sadness = get_emotion_music("sadness")
        self.assertEqual(sadness["scale"], "minor")
        self.assertIn("sao_truc", sadness["vietnamese_instruments"])


if __name__ == "__main__":
    unittest.main()
