"""Shared Kaggle CLI process helpers.

Keep authentication and CLI selection in ``src.integrations.kaggle_auto`` so
every launcher follows the same modern access-token and legacy-key behavior.
This module only owns subprocess execution and readiness polling.
"""

from __future__ import annotations

import subprocess
import sys
import time

from src.integrations.kaggle_auto import kaggle_cli_command


def kaggle_cli() -> list[str]:
    """Return the project-standard Kaggle CLI command."""
    command = kaggle_cli_command()
    if not command:
        raise RuntimeError("Kaggle CLI is unavailable; run `uv sync` first")
    return command


def run_cli(
    cli: list[str],
    args: list[str],
    env: dict[str, str],
    *,
    timeout: int = 900,
) -> subprocess.CompletedProcess[str]:
    """Run a Kaggle command with deterministic UTF-8 output handling."""
    result = subprocess.run(
        cli + args,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    return result


def wait_for_dataset(
    cli: list[str],
    ref: str,
    env: dict[str, str],
    *,
    attempts: int = 180,
    poll_seconds: float = 5.0,
    settle_seconds: float = 15.0,
) -> None:
    """Wait until a newly uploaded Kaggle dataset is ready and mountable."""
    for _ in range(attempts):
        result = run_cli(cli, ["datasets", "status", ref], env, timeout=120)
        status = (result.stdout + result.stderr).casefold()
        if result.returncode == 0 and "ready" in status:
            time.sleep(settle_seconds)
            return
        time.sleep(poll_seconds)
    raise TimeoutError(f"Kaggle source dataset did not become ready: {ref}")
