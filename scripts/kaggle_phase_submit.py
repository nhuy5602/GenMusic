"""Local submission helpers for independently rerunnable Kaggle phases."""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.run_kaggle_all_parts import _old_kaggle_cli, _run_cli
from scripts.run_kaggle_iterative_self import _create_dataset, _wait_for_dataset_visible
from src.integrations.kaggle_auto import (
    kaggle_access_token,
    kaggle_auth_available,
    kaggle_auth_environment,
    load_kaggle_api_tokens,
    resolve_kaggle_username,
    write_source_zip,
)


@dataclass(frozen=True)
class SubmitContext:
    project_root: Path
    username: str
    tokens: dict[str, str]
    environment: dict[str, str]
    cli: list[str]


def submit_context() -> SubmitContext:
    project_root = Path(__file__).resolve().parents[1]
    tokens = kaggle_auth_environment(load_kaggle_api_tokens())
    username = resolve_kaggle_username(tokens.get("KAGGLE_USERNAME"))
    if not username or not kaggle_auth_available(tokens):
        raise RuntimeError("Missing Kaggle username/access token")
    environment = {
        **os.environ,
        **tokens,
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
    }
    return SubmitContext(
        project_root=project_root,
        username=username,
        tokens=tokens,
        environment=environment,
        cli=_old_kaggle_cli(tokens),
    )


def require_complete_kernels(context: SubmitContext, refs: list[str]) -> None:
    for ref in refs:
        result = _run_cli(
            context.cli,
            ["kernels", "status", ref],
            context.environment,
            timeout=120,
        )
        status = (result.stdout + result.stderr).casefold()
        if result.returncode != 0 or "complete" not in status:
            raise RuntimeError(f"Required Kaggle kernel is not complete: {ref}")


def ensure_source_dataset(
    context: SubmitContext,
    *,
    source_ref: str,
    run_dir: Path,
    timestamp: int,
    phase: str,
) -> str:
    resolved = source_ref.strip()
    if resolved:
        _wait_for_dataset_visible(
            context.cli,
            resolved,
            context.environment,
            expected_marker="kaggle_phase_runtime.py",
        )
        return resolved

    resolved = f"{context.username}/genmusic-source-{phase}-{timestamp}"
    upload_dir = run_dir / "source_dataset"
    upload_dir.mkdir(parents=True, exist_ok=True)
    write_source_zip(context.project_root, upload_dir / "genmusic_vn_source.zip")
    _create_dataset(
        cli=context.cli,
        env=context.environment,
        upload_dir=upload_dir,
        dataset_ref=resolved,
        title=f"GenMusic {phase} source {timestamp}",
        expected_marker="kaggle_phase_runtime.py",
    )
    return resolved


def create_small_dataset(
    context: SubmitContext,
    *,
    upload_dir: Path,
    dataset_ref: str,
    title: str,
    expected_marker: str,
) -> None:
    _create_dataset(
        cli=context.cli,
        env=context.environment,
        upload_dir=upload_dir,
        dataset_ref=dataset_ref,
        title=title,
        expected_marker=expected_marker,
    )


def submit_phase_kernel(
    context: SubmitContext,
    *,
    phase: str,
    run_dir: Path,
    kernel_slug: str,
    code: str,
    dataset_sources: list[str],
    kernel_sources: list[str],
    enable_gpu: bool,
    enable_internet: bool,
    accelerator: str,
    timeout_seconds: int,
    state: dict,
) -> Path:
    kernel_ref = f"{context.username}/{kernel_slug}"
    kernel_dir = run_dir / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    code_file = f"run_{phase}.py"
    (kernel_dir / code_file).write_text(code, encoding="utf-8")
    metadata = {
        "id": kernel_ref,
        "title": kernel_slug,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true" if enable_gpu else "false",
        "enable_internet": "true" if enable_internet else "false",
        "dataset_sources": dataset_sources,
        "kernel_sources": kernel_sources,
    }
    if enable_gpu:
        metadata["machine_shape"] = accelerator
    (kernel_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    state_path = run_dir / "state.json"
    full_state = {
        **state,
        "phase": phase,
        "kernel_ref": kernel_ref,
        "kernel_url": f"https://www.kaggle.com/code/{kernel_ref}",
        "status": "prepared",
    }
    state_path.write_text(json.dumps(full_state, indent=2), encoding="utf-8")

    push_args = [
        "kernels",
        "push",
        "-p",
        str(kernel_dir),
        "--timeout",
        str(timeout_seconds),
    ]
    if enable_gpu and kaggle_access_token(context.tokens):
        push_args.extend(["--accelerator", accelerator])
    result = _run_cli(context.cli, push_args, context.environment, timeout=900)
    text = result.stdout + result.stderr
    if result.returncode != 0 or "kernel push error" in text.casefold():
        full_state.update(
            {
                "status": "submit_failed",
                "submit_returncode": result.returncode,
                "submit_output_tail": text[-4000:],
            }
        )
        state_path.write_text(json.dumps(full_state, indent=2), encoding="utf-8")
        raise RuntimeError(f"Kaggle rejected {phase} phase; state: {state_path}")

    full_state["status"] = "submitted"
    state_path.write_text(json.dumps(full_state, indent=2), encoding="utf-8")
    print(f"Submitted {phase}: {full_state['kernel_url']}")
    print(f"State: {state_path}")
    return state_path


def new_run_dir(context: SubmitContext, phase: str) -> tuple[int, Path]:
    timestamp = int(time.time())
    run_dir = context.project_root / "outputs" / "kaggle_phases" / f"{phase}-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return timestamp, run_dir
