from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from genmusic_vn.data.lyric_alignment import AlignedLine, align_lyrics_to_segments, read_lrc, write_lrc
from genmusic_vn.data.vietnamese_g2p import tone_digit, vietnamese_g2p
from genmusic_vn.data.vietnamese_text import normalize_vietnamese_lyrics
from genmusic_vn.evaluation.jam_metrics import subjective_summary, word_error_rate
from genmusic_vn.models.jam_diffrhythm import DiTConfig, model_config
from genmusic_vn.training.preference import build_preference_pairs, tone_contour_agreement


class JamPipelineTests(unittest.TestCase):
    def test_normalization_and_tone_aware_g2p(self) -> None:
        normalized = normalize_vietnamese_lyrics("Mưa 12 ngày, ko về.")
        self.assertIn("mười hai", normalized)
        self.assertIn("không", normalized)
        self.assertEqual([tone_digit(item) for item in ["ma", "mà", "má", "mả", "mã", "mạ"]], [1, 2, 3, 4, 5, 6])
        result = vietnamese_g2p("Sóng gió", use_phonemizer=False)
        self.assertEqual(result.backend, "rule-based-ipa")
        self.assertEqual(len(result.tokens), 2)
        self.assertTrue(result.tokens[-1][-1].isdigit())

    def test_alignment_writes_lrc_without_losing_source(self) -> None:
        lines = align_lyrics_to_segments("Mưa rơi\nEm về", [{"start": 0, "end": 2, "text": "Mưa rơi"}, {"start": 2, "end": 4, "text": "Em về"}])
        with tempfile.TemporaryDirectory() as temp:
            path = write_lrc(lines, Path(temp) / "song.lrc")
            loaded = read_lrc(path)
        self.assertEqual([line.text for line in loaded], ["Mưa rơi", "Em về"])
        self.assertEqual(loaded[0].source, "lrc")

    def test_preference_score_and_pair_builder(self) -> None:
        self.assertEqual(tone_contour_agreement([100, 110, 105], [80, 90, 85]), 1.0)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "pairs.jsonl"
            report = build_preference_pairs(
                [{"id": "one", "candidates": [{"id": "win", "tone_score": 0.9}, {"id": "loss", "tone_score": 0.2}]}],
                output,
            )
            pair = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(report["pair_count"], 1)
        self.assertEqual(pair["preferred"], "win")
        self.assertEqual(pair["dispreferred"], "loss")

    def test_objective_and_subjective_metric_contracts(self) -> None:
        self.assertEqual(word_error_rate("mưa rơi bên thềm", "mưa bên thềm"), 1 / 4)
        summary = subjective_summary(
            [
                {"listener_id": "a", "model_variant": "teacher", "musicality": 4, "intelligibility": 3, "comparison": 1},
                {"listener_id": "b", "model_variant": "teacher", "musicality": 2, "intelligibility": 5, "comparison": -1},
            ]
        )
        self.assertEqual(summary["listener_count"], 2)
        self.assertEqual(summary["variants"]["teacher"]["MOS_musicality"], 3.0)
        self.assertEqual(summary["CMOS"], 0.0)

    def test_model_presets_keep_parameter_count_explicit(self) -> None:
        self.assertIsInstance(model_config("demo"), DiTConfig)
        self.assertIsNone(model_config("diffrhythm").declared_parameter_count)


if __name__ == "__main__":
    unittest.main()
