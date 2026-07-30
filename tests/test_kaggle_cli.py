from subprocess import CompletedProcess
from unittest.mock import patch

from scripts.kaggle_cli import kaggle_cli, wait_for_dataset


def test_kaggle_cli_uses_project_standard_command() -> None:
    with patch(
        "scripts.kaggle_cli.kaggle_cli_command",
        return_value=["python", "-m", "kaggle"],
    ):
        assert kaggle_cli() == ["python", "-m", "kaggle"]


def test_wait_for_dataset_settles_after_ready() -> None:
    result = CompletedProcess(
        args=["kaggle", "datasets", "status", "owner/source"],
        returncode=0,
        stdout="Ready",
        stderr="",
    )
    with (
        patch("scripts.kaggle_cli.run_cli", return_value=result) as run,
        patch("scripts.kaggle_cli.time.sleep") as sleep,
    ):
        wait_for_dataset(
            ["kaggle"],
            "owner/source",
            {},
            attempts=1,
            settle_seconds=2,
        )

    run.assert_called_once()
    sleep.assert_called_once_with(2)
