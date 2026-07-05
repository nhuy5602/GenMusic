from __future__ import annotations

import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from genmusic_vn.emotion import analyze_emotion
from genmusic_vn.kaggle_auto import (
    KaggleJobConfig,
    _is_dataset_status_forbidden,
    kaggle_cli_command,
    load_kaggle_api_tokens,
    make_run_id,
    slugify,
    submit_text_to_music_job,
)
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
            self.assertIn("vocal plan:", result.prompt)
            self.assertIn("singer-ready melody", result.prompt)
            self.assertNotIn("without lead vocal", result.prompt)
            self.assertIn(result.vocal.gender, {"female", "male", "duet"})
            self.assertIn("vocal", json.loads((Path(temp) / result.run_id / "prompt_pack.json").read_text(encoding="utf-8")))

    def test_short_text_is_rewritten_as_short_song_with_vocal_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            result = create_music_project(
                "Một chiều mưa, tôi nhớ về những con phố cũ. Có lời hứa chưa kịp nói, có ánh đèn vẫn chờ trong tim.",
                output_root=temp,
                duration_seconds=8,
                render_audio=False,
            )
            lyrics_text = "\n".join(result.lyrics.full_song)
            self.assertEqual(result.lyrics.song_form, ["Verse", "Chorus", "Outro"])
            self.assertFalse(any("[Verse 2]" in line for line in result.lyrics.full_song))
            self.assertIn("ngày xưa nghiêng trong màu nắng cũ", lyrics_text)
            self.assertIn("chiều ơi, ở lại thêm một lần", lyrics_text)
            self.assertIn("bình yên nằm lại trên đôi tay", lyrics_text)
            self.assertNotIn("ngay xua", lyrics_text)
            self.assertNotIn("o lai", lyrics_text)
            self.assertNotIn("binh yen", lyrics_text)
            self.assertIn("vocal plan:", result.prompt)
            self.assertNotIn("no lead vocal", result.prompt)
            self.assertNotIn("lyric lines:", result.prompt)
            self.assertTrue(result.vocal.pitch_center)
            self.assertIn(result.vocal.gender, {"female", "male", "duet"})

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
            self.assertIn("genmusic_vn/vocal_planner.py", names)
            self.assertTrue((Path(job["kernel_dir"]) / "kernel-metadata.json").exists())
            self.assertEqual(job["tts_model"], "facebook/mms-tts-vie")
            kernel_script = (Path(job["kernel_dir"]) / "run_genmusic_vn.py").read_text(encoding="utf-8")
            self.assertIn("lyrics.txt", kernel_script)
            self.assertIn("vocal_plan", kernel_script)
            self.assertIn("facebook/mms-tts-vie", kernel_script)
            self.assertIn("render_mms_tts_vocal", kernel_script)
            self.assertIn("mix_vocal_with_backing", kernel_script)
            self.assertIn("_backing.mp3", kernel_script)
            self.assertTrue((Path(job["job_dir"]) / "run_commands.sh").exists())
            self.assertTrue(job["shell_commands_path"].endswith("run_commands.sh"))
            shell_commands = (Path(job["job_dir"]) / "run_commands.sh").read_text(encoding="utf-8")
            self.assertIn("#!/usr/bin/env bash", shell_commands)
            self.assertIn("python3 -m pip install --user -U kaggle", shell_commands)

    def test_slugify_keeps_kaggle_safe_slug(self) -> None:
        self.assertEqual(slugify("GenMusic Việt Nam Demo!!!", 50), "genmusic-viet-nam-demo")

    def test_make_run_id_avoids_fast_duplicate_submissions(self) -> None:
        first = make_run_id("same vietnamese text")
        second = make_run_id("same vietnamese text")
        self.assertNotEqual(first, second)

    def test_stylebank_loads_emotion_music_dataset(self) -> None:
        bank = load_stylebank()
        self.assertIn("emotion_to_music", bank)
        sadness = get_emotion_music("sadness")
        self.assertEqual(sadness["scale"], "minor")
        self.assertIn("sao_truc", sadness["vietnamese_instruments"])
        self.assertIn("ngày xưa nghiêng", load_stylebank()["lyric_patterns"]["patterns"]["nostalgic"]["chorus"][0])

    def test_kaggle_api_tokens_can_be_read_from_environment(self) -> None:
        old_username = os.environ.get("KAGGLE_USERNAME")
        old_key = os.environ.get("KAGGLE_KEY")
        old_api_token = os.environ.get("KAGGLE_API_TOKEN")
        try:
            os.environ["KAGGLE_USERNAME"] = "demo_user"
            os.environ["KAGGLE_KEY"] = "demo_key"
            os.environ["KAGGLE_API_TOKEN"] = "demo_access_token"
            tokens = load_kaggle_api_tokens()
            self.assertEqual(tokens["KAGGLE_USERNAME"], "demo_user")
            self.assertEqual(tokens["KAGGLE_KEY"], "demo_key")
            self.assertEqual(tokens["KAGGLE_API_TOKEN"], "demo_access_token")
        finally:
            if old_username is None:
                os.environ.pop("KAGGLE_USERNAME", None)
            else:
                os.environ["KAGGLE_USERNAME"] = old_username
            if old_key is None:
                os.environ.pop("KAGGLE_KEY", None)
            else:
                os.environ["KAGGLE_KEY"] = old_key
            if old_api_token is None:
                os.environ.pop("KAGGLE_API_TOKEN", None)
            else:
                os.environ["KAGGLE_API_TOKEN"] = old_api_token

    def test_kaggle_api_token_can_be_read_from_access_token_file(self) -> None:
        old_api_token = os.environ.get("KAGGLE_API_TOKEN")
        try:
            os.environ.pop("KAGGLE_API_TOKEN", None)
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                config_dir = root / ".kaggle"
                config_dir.mkdir()
                (config_dir / "access_token").write_text("demo_file_access_token\n", encoding="utf-8")
                with patch("genmusic_vn.kaggle_auto.Path.home", return_value=root):
                    tokens = load_kaggle_api_tokens()
            self.assertEqual(tokens["KAGGLE_API_TOKEN"], "demo_file_access_token")
        finally:
            if old_api_token is None:
                os.environ.pop("KAGGLE_API_TOKEN", None)
            else:
                os.environ["KAGGLE_API_TOKEN"] = old_api_token

    def test_dataset_status_forbidden_is_detected(self) -> None:
        self.assertTrue(
            _is_dataset_status_forbidden(
                {
                    "stdout": "",
                    "stderr": "403 Client Error: Forbidden for url: https://api.kaggle.com/v1/datasets.DatasetApiService/GetDatasetStatus",
                }
            )
        )

    def test_kaggle_cli_command_finds_user_site_scripts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            user_site = root / "Python314" / "site-packages"
            scripts_dir = user_site.parent / ("Scripts" if os.name == "nt" else "bin")
            scripts_dir.mkdir(parents=True)
            executable = scripts_dir / ("kaggle.exe" if os.name == "nt" else "kaggle")
            executable.write_text("", encoding="utf-8")

            with (
                patch("genmusic_vn.kaggle_auto.shutil.which", return_value=None),
                patch("genmusic_vn.kaggle_auto.site.USER_BASE", str(root)),
                patch("genmusic_vn.kaggle_auto.site.USER_SITE", str(user_site)),
                patch("genmusic_vn.kaggle_auto.sys.executable", str(root / "python.exe")),
            ):
                self.assertEqual(kaggle_cli_command(), [str(executable)])


if __name__ == "__main__":
    unittest.main()
