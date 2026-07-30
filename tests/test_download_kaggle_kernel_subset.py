from __future__ import annotations

import unittest
from pathlib import Path

from scripts.download_kaggle_kernel_subset import _safe_destination


class TestDownloadKaggleKernelSubset(unittest.TestCase):
    def test_safe_destination_keeps_nested_output_under_root(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            destination = _safe_destination(
                tmp_path,
                "result/held_out/example.wav",
            )
            self.assertEqual(
                destination,
                tmp_path.resolve() / "result" / "held_out" / "example.wav",
            )

    def test_safe_destination_rejects_unsafe_output_names(self) -> None:
        import tempfile
        invalid_names = [
            "../outside.wav",
            "result/../../outside.wav",
            "/absolute.wav",
            r"result\outside.wav",
            "",
            ".",
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            for name in invalid_names:
                with self.subTest(name=name):
                    with self.assertRaisesRegex(ValueError, "Unsafe kernel output path"):
                        _safe_destination(tmp_path, name)


if __name__ == "__main__":
    unittest.main()
