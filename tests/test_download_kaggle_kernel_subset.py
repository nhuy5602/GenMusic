from __future__ import annotations

from pathlib import Path

import pytest

from scripts.download_kaggle_kernel_subset import _safe_destination


def test_safe_destination_keeps_nested_output_under_root(tmp_path: Path) -> None:
    destination = _safe_destination(
        tmp_path,
        "result/held_out/example.wav",
    )
    assert destination == (
        tmp_path.resolve() / "result" / "held_out" / "example.wav"
    )


@pytest.mark.parametrize(
    "name",
    [
        "../outside.wav",
        "result/../../outside.wav",
        "/absolute.wav",
        r"result\outside.wav",
        "",
        ".",
    ],
)
def test_safe_destination_rejects_unsafe_output_names(
    tmp_path: Path,
    name: str,
) -> None:
    with pytest.raises(ValueError, match="Unsafe kernel output path"):
        _safe_destination(tmp_path, name)
