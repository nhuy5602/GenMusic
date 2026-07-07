from __future__ import annotations

import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from genmusic_vn.evaluation import _rhyme_pair_rate, evaluate_dataset, load_eval_dataset
from genmusic_vn.emotion import analyze_emotion
from genmusic_vn.kaggle_auto import (
    KaggleJobConfig,
    kaggle_cli_command,
    load_kaggle_api_tokens,
    make_run_id,
    slugify,
    submit_text_to_music_job,
)
from genmusic_vn.music_theory import chord_notes
from genmusic_vn.pipeline import create_music_project
from genmusic_vn.scene_planner import build_scene_plan
from genmusic_vn.stylebank import get_emotion_music, load_stylebank
from genmusic_vn.synthetic_dataset import generate_synthetic_records, write_jsonl


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
            self.assertIn("scene cues:", result.prompt)
            self.assertIn("source keywords:", result.prompt)
            self.assertIn("source text images:", result.prompt)
            self.assertIn("singer-ready melody", result.prompt)
            self.assertNotIn("titled", result.prompt)
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
            self.assertEqual(result.lyrics.title, "")
            self.assertIn("rain", result.scene.labels)
            self.assertIn("old_street", result.scene.labels)
            self.assertIn("love_promise", result.scene.labels)
            self.assertEqual(result.lyrics.song_form, ["Verse", "Chorus", "Outro"])
            self.assertFalse(any("[Title]" in line for line in result.lyrics.full_song))
            self.assertFalse(any("[Verse 2]" in line for line in result.lyrics.full_song))
            self.assertIn("ngày xưa nghiêng trong màu nắng cũ", lyrics_text)
            self.assertIn("phố cũ ơi, ở lại thêm một lần", lyrics_text)
            self.assertIn("bình yên nằm lại trên đôi tay", lyrics_text)
            self.assertNotIn("ngay xua", lyrics_text)
            self.assertNotIn("o lai", lyrics_text)
            self.assertNotIn("binh yen", lyrics_text)
            self.assertIn("vocal plan:", result.prompt)
            self.assertIn("rainy atmosphere", result.prompt)
            self.assertIn("nostalgic old streets", result.prompt)
            self.assertIn("unspoken promise", result.prompt)
            self.assertIn("wide stereo", result.prompt)
            self.assertNotIn("titled", result.prompt)
            self.assertNotIn("no lead vocal", result.prompt)
            self.assertNotIn("lyric lines:", result.prompt)
            self.assertTrue(result.vocal.pitch_center)
            self.assertIn(result.vocal.gender, {"female", "male", "duet"})

    def test_non_rhyming_input_is_shaped_into_singable_rhymed_lines(self) -> None:
        text = (
            "Tôi mở cửa đi qua thành phố. "
            "Chiếc xe dừng lại dưới ánh đèn. "
            "Một người lạ gọi tên tôi. "
            "Ngày mai chưa biết sẽ ra sao."
        )
        with tempfile.TemporaryDirectory() as temp:
            result = create_music_project(text, output_root=temp, duration_seconds=12, render_audio=False)

        content_lines = [
            line
            for line in result.lyrics.full_song
            if line.strip() and not line.startswith("[")
        ]
        self.assertEqual(result.lyrics.rhyme_scheme, "paired A-A / B-B Vietnamese end rhymes")
        self.assertGreaterEqual(_rhyme_pair_rate(content_lines), 0.75)
        self.assertTrue(all(4 <= len(line.split()) <= 12 for line in content_lines))
        self.assertIn("lyric rhyme scheme:", result.prompt)

    def test_existing_long_lyrics_are_arranged_as_duration_limited_excerpt(self) -> None:
        lyric_text = "\n".join(
            [
                "Một con đường mở ra trong nắng",
                "Tôi nghe nhịp tim còn vang",
                "Bàn tay ai chạm vào ký ức",
                "Cho đêm dài bỗng dịu dàng",
                "Ánh đèn đang gọi tên ta",
                "Giữ lại một mùa đi xa",
                "Nếu ngày mai còn nhiều giông gió",
                "Ta vẫn về chung một nhà",
                "Tôi qua những ngày rất vội",
                "Nhặt từng câu hát chưa phai",
                "Ánh đèn đang gọi tên ta",
                "Giữ lại một mùa đi xa",
                "Nếu ngày mai còn nhiều giông gió",
                "Ta vẫn về chung một nhà",
                "Ngoài kia mưa rơi thật lâu",
                "Trong tim còn nguyên nhiệm màu",
                "Ánh đèn đang gọi tên ta",
                "Giữ lại một mùa đi xa",
                "Nếu ngày mai còn nhiều giông gió",
                "Ta vẫn về chung một nhà",
            ]
        )
        with tempfile.TemporaryDirectory() as temp:
            result = create_music_project(lyric_text, output_root=temp, duration_seconds=60, render_audio=False)

        content_lines = [
            line
            for line in result.lyrics.full_song
            if line.strip() and not line.startswith("[")
        ]
        self.assertEqual(result.text_plan.input_kind, "lyrics")
        self.assertEqual(result.text_plan.mode, "lyrics_long")
        self.assertLessEqual(len(result.text_plan.representative_sentences), 12)
        self.assertLessEqual(len(content_lines), 12)
        self.assertIn("selected lyric input excerpt", result.lyrics.rhyme_scheme)
        self.assertIn("một con đường mở ra", "\n".join(result.lyrics.full_song))
        self.assertIn("ánh đèn đang gọi tên ta", "\n".join(result.lyrics.full_song))
        self.assertIn("lyric rhyme scheme:", result.prompt)

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
            self.assertEqual(job["duration_policy"], "soft_target")
            self.assertEqual(job["target_duration_seconds"], 6)
            self.assertEqual(job["tts_voice_actual"], "fixed_mms_vietnamese_voice")
            self.assertIn("fixed single-speaker", job["tts_voice_note"])
            request_pack = json.loads((Path(job["dataset_dir"]) / "request.json").read_text(encoding="utf-8"))
            self.assertEqual(request_pack["duration_policy"], "soft_target")
            self.assertEqual(request_pack["target_duration_seconds"], 6)
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
            self.assertIn("tts_voice_actual", kernel_script)
            self.assertIn("build_duration_plan", kernel_script)
            self.assertIn("planned_backing_duration_seconds", kernel_script)
            self.assertIn("duration_plan.json", kernel_script)
            self.assertIn("scene_plan", kernel_script)
            self.assertIn("select_tts_lines_for_duration", kernel_script)
            self.assertIn("duration_ceiling_seconds", kernel_script)
            self.assertIn("enforce_audio_duration", kernel_script)
            self.assertIn("duration=first", kernel_script)
            self.assertIn("normalize=0", kernel_script)
            self.assertIn("anoisesrc", kernel_script)
            self.assertIn("aformat=channel_layouts=stereo", kernel_script)

    def test_scene_plan_handles_multiple_input_types(self) -> None:
        hopeful_text = "Sau thất bại, tôi đứng dậy đón bình minh và tin vào ngày mai."
        hopeful = analyze_emotion(hopeful_text)
        hopeful_scene = build_scene_plan(hopeful_text, hopeful)
        self.assertIn("morning_sun", hopeful_scene.labels)
        self.assertIn("hope_rise", hopeful_scene.labels)

        tense_text = "Tôi giận dữ trước bất công, tim như lửa và vẫn muốn đấu tranh."
        tense = analyze_emotion(tense_text)
        tense_scene = build_scene_plan(tense_text, tense)
        self.assertIn("conflict_fire", tense_scene.labels)
        with tempfile.TemporaryDirectory() as temp:
            tense_result = create_music_project(tense_text, output_root=temp, duration_seconds=12, render_audio=False)
        self.assertIn("dark Vietnamese cinematic pop cue", tense_result.prompt)

        summer_text = "Mùa hè có tiếng cười và sân trường rực nắng."
        summer = analyze_emotion(summer_text)
        summer_scene = build_scene_plan(summer_text, summer)
        self.assertNotIn("rain", summer_scene.labels)

    def test_xlsx_mood_cases_are_covered_by_prompt_hints(self) -> None:
        cases = [
            (
                "Tiếng trống vang lên giữa sân vận động. Chúng tôi không còn là những cá nhân riêng lẻ, mà là một đội cùng tiến về phía trước.",
                "epic sports anthem, stadium drums, team spirit, heroic brass, powerful percussion",
                {"hope", "joy"},
            ),
            (
                "Mây đen kéo đến rất nhanh. Con đường phía trước mờ đi, còn trong lòng tôi là một linh cảm không lành.",
                "cinematic suspense, dark clouds, approaching storm, low drones, tense strings",
                {"fear"},
            ),
            (
                "Trong khu rừng cổ, có một ánh sáng xanh le lói sau màn sương. Tôi biết mình đã bước vào một nơi không thuộc về thế giới này.",
                "mysterious fantasy ambient orchestral, ancient forest, blue light, mist, magical atmosphere",
                {"fear"},
            ),
            (
                "Có một con hẻm nhỏ luôn sáng đèn vào mỗi tối. Ở đó, người ta kể cho nhau nghe những câu chuyện chưa từng được viết thành sách.",
                "warm storytelling Vietnamese folk, small alley lights at night, acoustic guitar, soft flute",
                {"calm", "romantic"},
            ),
        ]
        with tempfile.TemporaryDirectory() as temp:
            for text, genre, expected in cases:
                result = create_music_project(text, output_root=temp, duration_seconds=20, genre=genre, render_audio=False)
                self.assertIn(result.emotion.label, expected)

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

    def test_evaluation_dataset_reports_metrics(self) -> None:
        records = load_eval_dataset()
        self.assertGreaterEqual(len(records), 6)
        with tempfile.TemporaryDirectory() as temp:
            report = evaluate_dataset(output_root=temp, duration_seconds=8)
        self.assertGreaterEqual(report["sample_count"], 6)
        self.assertIn("emotion_match", report["summary"])
        self.assertIn("prompt_keyword_recall", report["summary"])
        self.assertIn("scene_cue_density", report["summary"])
        self.assertIn("no_title", report["summary"])
        self.assertIn("rhyme_pair_rate", report["summary"])
        self.assertIn("melody_line_rate", report["summary"])
        self.assertIn("romanized_violation_count", report["summary"])
        self.assertIn("unknown", report["by_length"])
        self.assertIn("nostalgic", report["by_expected_emotion"])

    def test_synthetic_evaluation_dataset_can_be_generated(self) -> None:
        records = generate_synthetic_records(6, seed=7, lengths=["short"])
        self.assertEqual(len(records), 6)
        self.assertTrue(all(record["length_bucket"] == "short" for record in records))
        with tempfile.TemporaryDirectory() as temp:
            dataset_path = write_jsonl(records, Path(temp) / "synthetic.jsonl")
            report = evaluate_dataset(dataset_path, output_root=Path(temp) / "runs", duration_seconds=8)
        self.assertEqual(report["sample_count"], 6)
        self.assertIn("short", report["by_length"])
        self.assertGreaterEqual(report["summary"]["no_title"], 1.0)

    def test_kaggle_api_tokens_can_be_read_from_environment(self) -> None:
        old_username = os.environ.get("KAGGLE_USERNAME")
        old_key = os.environ.get("KAGGLE_KEY")
        try:
            os.environ["KAGGLE_USERNAME"] = "demo_user"
            os.environ["KAGGLE_KEY"] = "demo_key"
            tokens = load_kaggle_api_tokens()
            self.assertEqual(tokens["KAGGLE_USERNAME"], "demo_user")
            self.assertEqual(tokens["KAGGLE_KEY"], "demo_key")
        finally:
            if old_username is None:
                os.environ.pop("KAGGLE_USERNAME", None)
            else:
                os.environ["KAGGLE_USERNAME"] = old_username
            if old_key is None:
                os.environ.pop("KAGGLE_KEY", None)
            else:
                os.environ["KAGGLE_KEY"] = old_key

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
