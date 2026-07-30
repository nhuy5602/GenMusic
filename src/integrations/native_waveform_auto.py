"""Web-facing Kaggle orchestration for native waveform generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.generation.run_kaggle_native_waveform import (
    DEFAULT_RAW_KERNEL_REF,
    DEFAULT_SOURCE_REF,
    PATCH_BUNDLE_NAME,
    _build_patch_bundle,
    _kernel_code,
    _patch_tree_sha256,
)
from src.audio.native_waveform import target_words_from_text
from src.data.lyric_alignment import write_lrc
from src.data.vietnamese_text import normalize_vietnamese_lyrics
from src.integrations.kaggle_auto import (
    PROJECT_ROOT,
    _commands,
    _dataset_url,
    _fail,
    _make_lrc,
    _now,
    _run,
    _write_state,
    kaggle_cli_command,
    kaggle_dataset_exists,
    kaggle_readiness,
    make_run_id,
    resolve_kaggle_username,
    slugify,
    submit_kaggle_job,
)

NATIVE_WAVEFORM_MODEL = "native-waveform-v80"
NATIVE_WAVEFORM_BACKEND = "genmusic-native-waveform-v80"
NATIVE_DURATION_SECONDS = 16
MAX_NATIVE_WORDS = 32


@dataclass(frozen=True)
class NativeWaveformJobConfig:
    username: str | None = None
    machine_shape: str = "NvidiaTeslaT4"
    submit: bool = True
    wait: bool = False
    poll_seconds: int = 30
    timeout_seconds: int = 7_200
    source_ref: str = DEFAULT_SOURCE_REF
    raw_kernel_ref: str = DEFAULT_RAW_KERNEL_REF


def stage_native_waveform_job(
    *,
    text: str,
    output_root: str | Path,
    duration_seconds: int,
    genre: str | None,
    config: NativeWaveformJobConfig,
) -> dict[str, Any]:
    """Stage a private Kaggle request using the validated V80 pipeline."""
    normalized = normalize_vietnamese_lyrics(text).strip()
    target_words = target_words_from_text(normalized)
    if len(target_words) > MAX_NATIVE_WORDS:
        raise ValueError(
            f"Native waveform generation accepts at most "
            f"{MAX_NATIVE_WORDS} lyric words per 16-second request"
        )

    # V80 was validated at 16 seconds. Keep the product route on that exact
    # duration instead of silently compressing lyrics to satisfy the slider.
    requested_duration = NATIVE_DURATION_SECONDS
    run_id = make_run_id(normalized)
    username = (
        resolve_kaggle_username(config.username)
        or "YOUR_KAGGLE_USERNAME"
    )
    run_dir = Path(output_root) / run_id
    job_dir = run_dir / "kaggle_job"
    dataset_dir = job_dir / "dataset"
    kernel_dir = job_dir / "kernel"
    download_dir = job_dir / "downloaded_output"
    for path in (dataset_dir, kernel_dir, download_dir):
        path.mkdir(parents=True, exist_ok=True)

    dataset_slug = slugify(
        f"genmusic-native-request-{run_id}",
        max_length=48,
    )
    kernel_slug = slugify(
        f"genmusic-native-waveform-{run_id}",
        max_length=48,
    )
    request_dataset_ref = f"{username}/{dataset_slug}"
    kernel_ref = f"{username}/{kernel_slug}"
    patch_path = dataset_dir / PATCH_BUNDLE_NAME
    patch_sha256 = _build_patch_bundle(PROJECT_ROOT, patch_path)
    patch_tree_sha256 = _patch_tree_sha256(PROJECT_ROOT)
    request = {
        "run_id": run_id,
        "text": normalized,
        "lyrics": normalized,
        "genre": (genre or "").strip(),
        "requested_duration_seconds": int(duration_seconds),
        "duration_seconds": requested_duration,
        "model": NATIVE_WAVEFORM_MODEL,
        "backend": NATIVE_WAVEFORM_BACKEND,
        "source_ref": config.source_ref,
        "raw_kernel_ref": config.raw_kernel_ref,
        "created_at": _now(),
    }
    (run_dir / "request.json").write_text(
        json.dumps(request, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (dataset_dir / "request.json").write_text(
        json.dumps(request, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_lrc(
        _make_lrc(normalized, requested_duration),
        dataset_dir / "lyrics.lrc",
    )
    (dataset_dir / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": dataset_slug,
                "id": request_dataset_ref,
                "licenses": [{"name": "other"}],
                "subtitle": "Private native waveform generation request.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    code_file = "run_native_waveform.py"
    (kernel_dir / code_file).write_text(
        _kernel_code(
            patch_sha256=patch_sha256,
            patch_tree_sha256=patch_tree_sha256,
            text=normalized,
            genre=request["genre"],
            duration_seconds=requested_duration,
        ),
        encoding="utf-8",
    )
    (kernel_dir / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": kernel_ref,
                "title": kernel_slug,
                "code_file": code_file,
                "language": "python",
                "kernel_type": "script",
                "is_private": "true",
                "enable_gpu": "true",
                "enable_internet": "true",
                "machine_shape": config.machine_shape,
                "dataset_sources": [
                    config.source_ref,
                    request_dataset_ref,
                ],
                "kernel_sources": [config.raw_kernel_ref],
                "competition_sources": [],
                "model_sources": [],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    commands = _commands(
        dataset_dir,
        kernel_dir,
        download_dir,
        kernel_ref,
    )
    state = {
        "run_id": run_id,
        "job_kind": "native_waveform_generation",
        "status": "staged",
        "created_at": _now(),
        "backend": NATIVE_WAVEFORM_BACKEND,
        "model": NATIVE_WAVEFORM_MODEL,
        "lyrics": normalized,
        "genre": request["genre"],
        "duration_seconds": requested_duration,
        "requested_duration_seconds": int(duration_seconds),
        "dataset_ref": config.source_ref,
        "training_dataset_ref": config.source_ref,
        "raw_kernel_ref": config.raw_kernel_ref,
        "request_dataset_ref": request_dataset_ref,
        "kernel_ref": kernel_ref,
        "run_dir": str(run_dir),
        "job_dir": str(job_dir),
        "dataset_dir": str(dataset_dir),
        "kernel_dir": str(kernel_dir),
        "download_dir": str(download_dir),
        "state_path": str(job_dir / "job_state.json"),
        "dataset_url": _dataset_url(config.source_ref),
        "request_dataset_url": _dataset_url(request_dataset_ref),
        "kernel_url": (
            f"https://www.kaggle.com/code/{kernel_ref}"
            if username != "YOUR_KAGGLE_USERNAME"
            else ""
        ),
        "commands": commands,
        "messages": [
            "Web app is using native waveform V80.",
            "Duration is fixed at the validated 16-second setting.",
            "The request dataset and Kaggle kernel are private.",
        ],
        "history": [],
        "downloaded_files": [],
        "last_error": "",
        "patch_sha256": patch_sha256,
        "patch_tree_sha256": patch_tree_sha256,
    }
    _write_state(state)
    return state


def submit_native_waveform_job(
    *,
    text: str,
    output_root: str | Path = "outputs",
    duration_seconds: int = NATIVE_DURATION_SECONDS,
    genre: str | None = None,
    config: NativeWaveformJobConfig | None = None,
) -> dict[str, Any]:
    config = config or NativeWaveformJobConfig()
    state = stage_native_waveform_job(
        text=text,
        output_root=output_root,
        duration_seconds=duration_seconds,
        genre=genre,
        config=config,
    )
    if not config.submit:
        state["messages"].append("Native waveform job staged only.")
        _write_state(state)
        return state

    readiness = kaggle_readiness(config.username)
    state["kaggle_ready"] = readiness["ready"]
    state["messages"].extend(readiness["messages"])
    if not readiness["ready"]:
        state["status"] = "needs_setup"
        _write_state(state)
        return state
    if not kaggle_dataset_exists(config.source_ref):
        return _fail(
            state,
            f"Native waveform source dataset is unavailable: "
            f"{config.source_ref}",
        )
    cli = kaggle_cli_command()
    if cli is None:
        return _fail(state, "Kaggle CLI is unavailable.")
    raw_status = _run(
        cli + ["kernels", "status", config.raw_kernel_ref],
        timeout=120,
    )
    raw_text = (
        raw_status.get("stdout", "")
        + "\n"
        + raw_status.get("stderr", "")
    ).casefold()
    if raw_status["returncode"] != 0 or "complete" not in raw_text:
        return _fail(
            state,
            f"Native waveform corpus kernel is not complete: "
            f"{config.raw_kernel_ref}",
        )
    return submit_kaggle_job(
        state,
        wait=config.wait,
        poll_seconds=config.poll_seconds,
        timeout_seconds=config.timeout_seconds,
    )
