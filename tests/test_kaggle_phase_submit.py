from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.kaggle.phase_submit import ensure_source_dataset, new_run_dir


def test_new_run_dir_does_not_reuse_same_second_directory(tmp_path: Path) -> None:
    context = SimpleNamespace(project_root=tmp_path)

    with patch("scripts.kaggle.phase_submit.time.time", return_value=1234.5):
        first_timestamp, first_dir = new_run_dir(context, "quality")
        second_timestamp, second_dir = new_run_dir(context, "quality")

    assert first_timestamp == 1234
    assert second_timestamp == 1235
    assert first_dir.is_dir()
    assert second_dir.is_dir()
    assert first_dir != second_dir


def test_existing_source_can_skip_legacy_visibility_poll(tmp_path: Path) -> None:
    context = SimpleNamespace(
        cli=["kaggle"],
        environment={},
        username="owner",
        project_root=tmp_path,
    )

    with patch(
        "scripts.kaggle.phase_submit._wait_for_dataset_visible"
    ) as wait_for_visible:
        result = ensure_source_dataset(
            context,
            source_ref="owner/already-verified-source",
            run_dir=tmp_path / "run",
            timestamp=1234,
            phase="native-vocal-pretrain",
            verify_existing=False,
        )

    assert result == "owner/already-verified-source"
    wait_for_visible.assert_not_called()
