from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

from scripts.run_colab_full_training import merge_processed_datasets
from src.integrations.colab_auto import (
    DEFAULT_COLAB_NOTEBOOK_URL,
    DEFAULT_KERNEL_REFS,
    build_colab_notebook,
    write_colab_notebook,
)
from src.models.text_to_music_diffusion import MusicDiffusionConfig


class ColabIntegrationTests(unittest.TestCase):
    def _make_part(self, root: Path, part: int) -> Path:
        mel_dir = root / "mels"
        mel_dir.mkdir(parents=True)
        tensor = torch.randn(64, 32)
        for name in ("backing", "vocal"):
            torch.save(tensor, mel_dir / f"song_{part}_{name}.pt")
        torch.save(torch.randn(512), mel_dir / f"song_{part}_style.pt")
        record = {
            "id": f"song-{part}",
            "text": "Một câu hát tiếng Việt.",
            "style": "pop",
            "frames": 32,
            "backing_mel_path": f"mels/song_{part}_backing.pt",
            "vocal_mel_path": f"mels/song_{part}_vocal.pt",
            "style_embed_path": f"mels/song_{part}_style.pt",
        }
        (root / "records.jsonl").write_text(
            json.dumps(record, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (root / "config.json").write_text(
            json.dumps(asdict(MusicDiffusionConfig(frames_per_chunk=32))),
            encoding="utf-8",
        )
        return root

    def test_notebook_uses_drive_secret_and_preserves_kaggle(self) -> None:
        notebook = build_colab_notebook()
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
        )
        self.assertEqual(notebook["metadata"]["accelerator"], "GPU")
        self.assertIn(DEFAULT_COLAB_NOTEBOOK_URL, source)
        self.assertIn("drive.mount", source)
        self.assertIn('userdata.get("KAGGLE_API_TOKEN")', source)
        self.assertIn("run_colab_full_training.py", source)
        self.assertIn("--save-every-epoch", Path("cli.py").read_text(encoding="utf-8"))
        for kernel_ref in DEFAULT_KERNEL_REFS:
            self.assertIn(kernel_ref, source)
        self.assertNotIn('os.environ["KAGGLE_API_TOKEN"] = "KGAT', source)
        self.assertIn("Kaggle", "".join(notebook["cells"][0]["source"]))

    def test_write_notebook_creates_valid_ipynb(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "GenMusic.ipynb"
            report = write_colab_notebook(destination)
            payload = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(report["backend"], "google-colab")
            self.assertTrue(report["kaggle_backend_preserved"])
            self.assertEqual(payload["nbformat"], 4)
            self.assertEqual(len(payload["cells"]), 7)

    def test_merge_processed_parts_rewrites_paths_without_collisions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            part_one = self._make_part(root / "part1", 1)
            part_two = self._make_part(root / "part2", 2)
            combined = root / "work" / "combined"
            summary = merge_processed_datasets(
                [part_one, part_two],
                combined,
                expected_records=2,
            )
            records = [
                json.loads(line)
                for line in (combined / "records.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual(summary["combined_records"], 2)
            self.assertEqual(records[0]["id"], "source01_song-1")
            self.assertEqual(records[1]["id"], "source02_song-2")
            for record in records:
                for field in (
                    "backing_mel_path",
                    "vocal_mel_path",
                    "style_embed_path",
                ):
                    self.assertTrue((combined / record[field]).is_file())


if __name__ == "__main__":
    unittest.main()
