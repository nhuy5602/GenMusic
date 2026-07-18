"""Local and Kaggle orchestration for the GenMusic diffusion model."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import time
import unicodedata
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Any

from ..data.lyric_alignment import AlignedLine, write_lrc
from ..data.vietnamese_text import normalize_vietnamese_lyrics
from ..models.text_to_music_diffusion import (
    MusicDiffusionConfig,
    estimate_minimum_lyric_duration,
    generate_audio,
    load_checkpoint,
    render_mel_to_wav,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "genmusic-vn-self-diffusion-v1"
DEFAULT_KAGGLE_DATASET_SLUG = "genmusic-vn-self-diffusion-training"
KAGGLE_DATASET_ENV = "GENMUSIC_KAGGLE_DATASET_REF"
KAGGLE_CHECKPOINT_ENV = "GENMUSIC_KAGGLE_CHECKPOINT_REF"
KAGGLE_BACKING_ENV = "GENMUSIC_KAGGLE_BACKING_REF"
KAGGLE_API_TOKEN_ENV = "KAGGLE_API_TOKEN"
KAGGLE_ACCESS_TOKEN_ALIAS_ENV = "KAGGLE_ACCESS_TOKEN"
KAGGLE_AUTH_ENV_KEYS = (KAGGLE_API_TOKEN_ENV, KAGGLE_ACCESS_TOKEN_ALIAS_ENV, "KAGGLE_KEY")

# Shared by every scripts/run_kaggle_*.py: directories that must never end up in the
# "source code" zip pushed to Kaggle. `ckpt` in particular is a HuggingFace download
# cache (the real DiffRhythm2 teacher checkpoint, ~4.3GB) that _load_teacher() recreates
# locally on demand -- previously missing here, this once ballooned a "source code"
# upload to ~4GB with no functional benefit (the preprocess/train kernels never read it).
SOURCE_ZIP_EXCLUDED_DIRS = {".git", "outputs", "__pycache__", ".pytest_cache", ".venv", ".kaggle", "dataset", "datasets", "scratch", "ckpt", "DiffRhythm2-main"}


def write_source_zip(project_root: Path, destination: Path) -> None:
    """Zips the project source (excluding SOURCE_ZIP_EXCLUDED_DIRS, .env*, kaggle.json)
    for upload as a Kaggle Dataset so a pushed kernel can copy it into /kaggle/working."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in project_root.rglob("*"):
            relative = path.relative_to(project_root)
            if not path.is_file() or any(part in SOURCE_ZIP_EXCLUDED_DIRS for part in relative.parts):
                continue
            if relative.name.startswith(".env") or relative.name == "kaggle.json":
                continue
            archive.write(path, relative.as_posix())


class SelfMusicError(RuntimeError):
    pass


@dataclass(frozen=True)
class KaggleJobConfig:
    model: str = DEFAULT_MODEL
    username: str | None = None
    machine_shape: str = "NvidiaTeslaT4"
    submit: bool = True
    wait: bool = False
    poll_seconds: int = 60
    timeout_seconds: int = 21_600
    checkpoint_kernel_ref: str | None = None
    backing_kernel_ref: str | None = None
    pronunciation_prior_strength: float = 0.8
    diffusion_steps: int = 48
    backing_to_vocal_rms: float = 0.45


def _fit_backing_mel(backing_mel, target_frames: int, *, seed: int):
    """Crop or repeat a real backing stem to the generated vocal duration."""
    import torch

    mel = torch.as_tensor(backing_mel, dtype=torch.float32)
    if mel.dim() == 3 and mel.shape[0] == 1:
        mel = mel.squeeze(0)
    if mel.dim() != 2 or mel.shape[1] < 1:
        raise ValueError(f"Expected backing mel shaped (n_mels, frames), got {tuple(mel.shape)}")
    frames = max(1, int(target_frames))
    if mel.shape[1] >= frames:
        available = mel.shape[1] - frames
        start = int(seed) % (available + 1) if available else 0
        return mel[:, start:start + frames]
    repeats = math.ceil(frames / mel.shape[1])
    return mel.repeat(1, repeats)[:, :frames]


def mix_vocal_with_backing(
    vocal_path: str | Path,
    backing_path: str | Path,
    destination: str | Path,
    *,
    backing_to_vocal_rms: float = 0.45,
) -> dict[str, Any]:
    """Mix a real instrumental stem under the generated vocal without masking it.

    The ratio is RMS based rather than a fixed amplitude multiplier because the
    source stems have very different mastering levels. Keeping backing below the
    vocal preserves Vietnamese consonants while still producing a full-song mix.
    """
    import librosa
    import numpy as np
    import soundfile as sf

    vocal, sample_rate = sf.read(str(vocal_path), dtype="float32", always_2d=True)
    backing, backing_rate = sf.read(str(backing_path), dtype="float32", always_2d=True)
    vocal = vocal.mean(axis=1)
    backing = backing.mean(axis=1)
    if backing_rate != sample_rate:
        backing = librosa.resample(backing, orig_sr=backing_rate, target_sr=sample_rate)
    if backing.size < vocal.size:
        backing = np.tile(backing, math.ceil(vocal.size / max(1, backing.size)))
    backing = backing[:vocal.size]

    epsilon = 1e-8
    vocal_rms = float(np.sqrt(np.mean(np.square(vocal, dtype=np.float64)) + epsilon))
    backing_rms = float(np.sqrt(np.mean(np.square(backing, dtype=np.float64)) + epsilon))
    ratio = max(0.0, float(backing_to_vocal_rms))
    backing_scale = (vocal_rms * ratio / backing_rms) if backing_rms > epsilon else 0.0
    mixed = vocal + backing * backing_scale
    peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    limiter_scale = min(1.0, 0.98 / peak) if peak > 0.0 else 1.0
    mixed = (mixed * limiter_scale).astype("float32")

    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output), mixed, sample_rate, subtype="PCM_16")
    return {
        "audio_path": str(output),
        "duration_seconds": float(mixed.size / sample_rate),
        "sample_rate": int(sample_rate),
        "vocal_rms": vocal_rms,
        "backing_rms_before_scale": backing_rms,
        "backing_scale": float(backing_scale),
        "backing_to_vocal_rms": ratio,
        "limiter_scale": float(limiter_scale),
        "clip_ratio": float(np.mean(np.abs(mixed) >= 0.999)) if mixed.size else 0.0,
    }


def _load_generation_condition(path: str | Path | None, *, keys: tuple[str, ...]):
    if path is None:
        return None
    import torch

    condition_path = Path(path)
    payload = torch.load(condition_path, map_location="cpu", weights_only=True)
    if isinstance(payload, torch.Tensor):
        return payload
    if isinstance(payload, dict):
        for key in keys:
            value = payload.get(key)
            if isinstance(value, torch.Tensor):
                return value
    raise ValueError(f"Condition file does not contain a tensor: {condition_path}")


def run_local_generation(
    *,
    text: str,
    style: str,
    output_dir: str | Path,
    duration_seconds: float,
    checkpoint: str | Path | None = None,
    steps: int = 6,
    guidance_scale: float = 1.0,
    pronunciation_prior_strength: float = 0.0,
    pronunciation_prior_model: str = "facebook/mms-tts-vie",
    seed: int = 5602,
    device: str | None = None,
    mel_output: str | Path | None = None,
    vocoder: str = "vocos",
    roberta_model: str = "vinai/xphonebert-base",
    reference_dataset: str | Path | None = None,
    reference_id: str | None = None,
    use_reference_style: bool = True,
    backing_mel: str | Path | None = None,
    style_anchor: str | Path | None = None,
    mix_backing: bool = False,
    backing_to_vocal_rms: float = 0.45,
) -> dict[str, Any]:
    normalized = normalize_vietnamese_lyrics(text).strip()
    if not normalized:
        raise SelfMusicError("Văn bản input đang trống.")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    selected_device = device or _default_device()

    backing_condition = None
    style_condition = None
    reference_info = None
    if reference_dataset:
        from ..training.self_diffusion import load_reference_conditioning

        reference = load_reference_conditioning(reference_dataset, reference_id)
        backing_condition = reference["backing_mel"]
        style_condition = reference["style_anchor"] if use_reference_style else None
        reference_info = {
            "id": reference["id"],
            "has_backing_mel": backing_condition is not None,
            "has_style_anchor": style_condition is not None,
            "reference_style_enabled": bool(use_reference_style),
        }

    # Explicit condition files intentionally override the selected dataset
    # record, which makes it possible to mix a chosen backing and style anchor.
    if backing_mel:
        backing_condition = _load_generation_condition(backing_mel, keys=("mel", "backing_mel"))
    if style_anchor:
        style_condition = _load_generation_condition(
            style_anchor, keys=("style", "style_anchor", "embedding")
        )
    if checkpoint and Path(checkpoint).exists():
        model, config, payload = load_checkpoint(checkpoint, device=selected_device, roberta_model=roberta_model)
        checkpoint_path = str(Path(checkpoint).resolve())
        checkpoint_epoch = payload.get("epoch", 0)
    else:
        from ..models.dit_transformer import MicroDiT
        config = MusicDiffusionConfig()
        model = MicroDiT(config, roberta_model=roberta_model).to(selected_device)
        checkpoint_path = ""
        checkpoint_epoch = 0
    requested_duration = max(1.0, float(duration_seconds))
    minimum_duration = estimate_minimum_lyric_duration(normalized)
    effective_duration = max(requested_duration, minimum_duration)
    if effective_duration > requested_duration:
        print(f"[INFO] Duration tang tu {requested_duration:.2f}s len {effective_duration:.2f}s de tranh ep lyric.", flush=True)
    vocal_destination = destination / ("vocal.wav" if mix_backing else "final.wav")
    report = generate_audio(
        model,
        normalized,
        style or "Vietnamese pop, warm piano, clear melody",
        vocal_destination,
        duration_seconds=effective_duration,
        config=config,
        device=selected_device,
        steps=max(1, int(steps)),
        guidance_scale=float(guidance_scale),
        pronunciation_prior_strength=float(pronunciation_prior_strength),
        pronunciation_prior_model=pronunciation_prior_model,
        seed=int(seed),
        mel_output=mel_output,
        vocoder_type=vocoder,
        backing_mel=backing_condition,
        style_anchor=style_condition,
    )
    if float(pronunciation_prior_strength) > 0.0:
        from ..models.pronunciation_prior import trim_wav_silence

        untrimmed_path = Path(report["audio_path"])
        trimmed_path = trim_wav_silence(
            untrimmed_path,
            destination / ("vocal_trimmed.wav" if mix_backing else "final_trimmed.wav"),
        )
        report["untrimmed_audio_path"] = str(untrimmed_path)
        report["audio_path"] = str(trimmed_path)
    if mix_backing:
        if backing_condition is None:
            raise SelfMusicError(
                "--mix-backing requires --reference-dataset or --backing-mel with a real backing stem."
            )
        target_frames = max(
            1,
            int(effective_duration * config.sample_rate / config.hop_length),
        )
        fitted_backing = _fit_backing_mel(
            backing_condition,
            target_frames,
            seed=int(seed),
        )
        backing_path = render_mel_to_wav(
            fitted_backing,
            destination / "backing.wav",
            config,
            vocoder_type=vocoder,
        )
        vocal_path = Path(report["audio_path"])
        mix_report = mix_vocal_with_backing(
            vocal_path,
            backing_path,
            destination / "final.wav",
            backing_to_vocal_rms=float(backing_to_vocal_rms),
        )
        report["vocal_audio_path"] = str(vocal_path)
        report["backing_audio_path"] = str(backing_path)
        report["audio_path"] = mix_report["audio_path"]
        report["backing_mixed"] = True
        report["mix"] = mix_report
    else:
        report["backing_mixed"] = False
    report.update(
        {
            "text": normalized,
            "style": style,
            "checkpoint": checkpoint_path,
            "checkpoint_epoch": checkpoint_epoch,
            "device": selected_device,
            "requested_duration_seconds": requested_duration,
            "minimum_lyric_duration_seconds": minimum_duration,
            "duration_auto_adjusted": effective_duration > requested_duration,
            "reference_conditioning": reference_info,
            "backing_mel_path": str(Path(backing_mel).resolve()) if backing_mel else None,
            "style_anchor_path": str(Path(style_anchor).resolve()) if style_anchor else None,
            "backing_to_vocal_rms": (
                float(backing_to_vocal_rms) if mix_backing else None
            ),
        }
    )
    mp3_path = _convert_to_mp3(Path(report["audio_path"]))
    if mp3_path:
        report["mp3_path"] = str(mp3_path)
    (destination / "generation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def stage_text_to_music_job(*, text: str, output_root: str | Path, duration_seconds: int, genre: str | None, config: KaggleJobConfig, input_received_at: str | None = None) -> dict[str, Any]:
    normalized = normalize_vietnamese_lyrics(text).strip()
    if not normalized:
        raise ValueError("Văn bản input đang trống.")
    requested_duration = max(1, min(120, int(duration_seconds)))
    requested_duration = min(120, max(requested_duration, math.ceil(estimate_minimum_lyric_duration(text))))
    run_id = make_run_id(normalized)
    username = resolve_kaggle_username(config.username) or "YOUR_KAGGLE_USERNAME"
    run_dir = Path(output_root) / run_id
    job_dir = run_dir / "kaggle_job"
    dataset_dir = job_dir / "dataset"
    kernel_dir = job_dir / "kernel"
    download_dir = job_dir / "downloaded_output"
    for path in (dataset_dir, kernel_dir, download_dir):
        path.mkdir(parents=True, exist_ok=True)
    request_dataset_slug = slugify(f"genmusic-self-diffusion-{run_id}", max_length=48)
    kernel_slug = slugify(f"genmusic-self-diffusion-kernel-{run_id}", max_length=48)
    request_dataset_ref = f"{username}/{request_dataset_slug}"
    checkpoint_kernel_ref = resolve_checkpoint_kernel_ref(
        config.checkpoint_kernel_ref,
        config.username,
    )
    backing_kernel_ref = resolve_backing_kernel_ref(
        config.backing_kernel_ref,
        config.username,
    )
    kernel_ref = f"{username}/{kernel_slug}"
    request_dataset_url = _dataset_url(request_dataset_ref)
    kernel_url = f"https://www.kaggle.com/code/{kernel_ref}" if username != "YOUR_KAGGLE_USERNAME" else ""
    received = input_received_at or _now()
    style_prompt = genre or "Vietnamese pop ballad, warm piano, emotional strings, clear melody"
    request = {
        "run_id": run_id,
        "text": normalized,
        "lyrics": normalized,
        "duration_seconds": requested_duration,
        "checkpoint_kernel_ref": checkpoint_kernel_ref,
        "backing_kernel_ref": backing_kernel_ref,
        "request_dataset_ref": request_dataset_ref,
        "checkpoint_url": _kernel_url(checkpoint_kernel_ref),
        "backing_url": _kernel_url(backing_kernel_ref),
        "request_dataset_url": request_dataset_url,
        "kernel_url": kernel_url,
        "style_prompt": style_prompt,
        "diffusion_steps": max(1, int(config.diffusion_steps)),
        "guidance_scale": 1.0,
        "pronunciation_prior_strength": float(config.pronunciation_prior_strength),
        "backing_to_vocal_rms": float(config.backing_to_vocal_rms),
        "model": config.model or DEFAULT_MODEL,
        "backend": "genmusic-vn-self-diffusion",
        "source": "project-local",
        "input_received_at": received,
        "created_at": _now(),
    }
    (run_dir / "request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
    (dataset_dir / "request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
    write_lrc(_make_lrc(normalized, requested_duration), dataset_dir / "lyrics.lrc")
    _write_source_zip(dataset_dir / "genmusic_vn_source.zip")
    (dataset_dir / "dataset-metadata.json").write_text(json.dumps({"title": request_dataset_slug, "id": request_dataset_ref, "licenses": [{"name": "other"}], "subtitle": "Request cho model GenMusic.", "description": "Private request dataset for the GenMusic conditional diffusion model."}, ensure_ascii=False, indent=2), encoding="utf-8")
    (kernel_dir / "run_genmusic.py").write_text(_kernel_script(request_dataset_slug), encoding="utf-8")
    (kernel_dir / "kernel-metadata.json").write_text(json.dumps({"id": kernel_ref, "title": kernel_slug, "code_file": "run_genmusic.py", "language": "python", "kernel_type": "script", "is_private": "true", "enable_gpu": "true", "enable_internet": "true", "machine_shape": config.machine_shape, "dataset_sources": [request_dataset_ref], "competition_sources": [], "kernel_sources": [checkpoint_kernel_ref, backing_kernel_ref], "model_sources": []}, ensure_ascii=False, indent=2), encoding="utf-8")
    commands = _commands(dataset_dir, kernel_dir, download_dir, kernel_ref)
    (job_dir / "run_commands.ps1").write_text("\n".join(commands) + "\n", encoding="utf-8")
    state = {
        "run_id": run_id,
        "job_kind": "self_diffusion_generation",
        "status": "staged",
        "created_at": _now(),
        "input_received_at": received,
        "backend": "genmusic-vn-self-diffusion",
        "model": request["model"],
        "lyrics": normalized,
        "checkpoint_kernel_ref": checkpoint_kernel_ref,
        "backing_kernel_ref": backing_kernel_ref,
        "request_dataset_ref": request_dataset_ref,
        "kernel_ref": kernel_ref,
        "run_dir": str(run_dir),
        "job_dir": str(job_dir),
        "dataset_dir": str(dataset_dir),
        "kernel_dir": str(kernel_dir),
        "download_dir": str(download_dir),
        "state_path": str(job_dir / "job_state.json"),
        "duration_seconds": requested_duration,
        "machine_shape": config.machine_shape,
        "diffusion_steps": request["diffusion_steps"],
        "pronunciation_prior_strength": request["pronunciation_prior_strength"],
        "backing_to_vocal_rms": request["backing_to_vocal_rms"],
        "checkpoint_url": _kernel_url(checkpoint_kernel_ref),
        "backing_url": _kernel_url(backing_kernel_ref),
        "request_dataset_url": request_dataset_url,
        "kernel_url": kernel_url,
        "commands": commands,
        "messages": [
            f"Checkpoint cố định: {checkpoint_kernel_ref}.",
            f"Backing stem: {backing_kernel_ref}.",
            "Job chỉ generate và mix; không train lại theo từng request.",
        ],
        "history": [],
        "generation_backend": "genmusic-vn-self-diffusion",
        "downloaded_files": [],
        "last_error": "",
    }
    _write_state(state)
    return state


def submit_text_to_music_job(*, text: str, output_root: str | Path = "outputs", duration_seconds: int = 12, genre: str | None = None, config: KaggleJobConfig | None = None) -> dict[str, Any]:
    config = config or KaggleJobConfig()
    state = stage_text_to_music_job(text=text, output_root=output_root, duration_seconds=duration_seconds, genre=genre, config=config)
    if not config.submit:
        state["messages"].append("Đã stage job self-diffusion; chưa submit Kaggle.")
        _write_state(state)
        return state
    readiness = kaggle_readiness(config.username)
    state["kaggle_ready"] = readiness["ready"]
    state["messages"].extend(readiness["messages"])
    if not readiness["ready"]:
        state["status"] = "needs_setup"
        _write_state(state)
        return state
    for label, ref in (
        ("checkpoint", state["checkpoint_kernel_ref"]),
        ("backing", state["backing_kernel_ref"]),
    ):
        if not kaggle_kernel_complete(ref):
            return _fail(
                state,
                f"Kaggle {label} kernel '{ref}' chưa COMPLETE hoặc không tồn tại.",
            )
    return submit_kaggle_job(state, wait=config.wait, poll_seconds=config.poll_seconds, timeout_seconds=config.timeout_seconds)


def submit_kaggle_job(state: dict[str, Any], *, wait: bool, poll_seconds: int, timeout_seconds: int) -> dict[str, Any]:
    cli = kaggle_cli_command()
    if cli is None:
        return _fail(state, "Không tìm thấy Kaggle CLI.")
    created = _run(cli + ["datasets", "create", "-p", state["dataset_dir"], "-r", "zip"], timeout=900)
    state["history"].append(_history_item("datasets create", created))
    if created["returncode"] != 0:
        # Kaggle đôi khi upload xong nhưng CLI lỗi khi parse JSON phản hồi.
        # Xác nhận trạng thái resource trước khi coi job là thất bại.
        if not _wait_for_dataset_ready(cli, state["request_dataset_ref"], timeout_seconds=900):
            verified = _run(cli + ["datasets", "status", state["request_dataset_ref"]], timeout=120)
            state["history"].append(_history_item("datasets status", verified))
            return _fail(state, _summarize_cli_error(created))
        state["messages"].append("Kaggle đã nhận dataset dù CLI báo lỗi parse phản hồi; đã xác nhận trạng thái ready.")
    elif not _wait_for_dataset_ready(cli, state["request_dataset_ref"], timeout_seconds=900):
        return _fail(state, f"Dataset request Kaggle '{state['request_dataset_ref']}' chưa chuyển sang trạng thái ready.")
    state["status"] = "dataset_uploaded"
    state["last_error"] = ""
    _write_state(state)
    push_command = cli + ["kernels", "push", "-p", state["kernel_dir"]]
    # Modern KGAT authentication accepts the accelerator as an explicit push
    # argument; keeping it here avoids silently falling back to a CPU session.
    if kaggle_access_token():
        push_command.extend(["--accelerator", state["machine_shape"]])
    pushed = _run(push_command, timeout=900)
    state["history"].append(_history_item("kernels push", pushed))
    if pushed["returncode"] != 0:
        return _fail(state, _summarize_cli_error(pushed))
    state["status"] = "submitted"
    state["messages"].append("Kernel Kaggle đã được submit; có thể mở link để xem tiến trình.")
    state["last_error"] = ""
    state["submitted_at"] = _now()
    _write_state(state)
    if not wait:
        return state
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        refreshed = refresh_kaggle_job(state)
        if refreshed["status"] in {"complete", "failed"}:
            return refreshed
        time.sleep(max(5, poll_seconds))
    state["status"] = "timeout"
    _write_state(state)
    return state


def refresh_kaggle_job(state_or_path: dict[str, Any] | str | Path) -> dict[str, Any]:
    state = state_or_path if isinstance(state_or_path, dict) else _load_state(state_or_path)
    cli = kaggle_cli_command()
    if cli is None:
        return _fail(state, "Không tìm thấy Kaggle CLI.")
    access_token = kaggle_access_token()
    if access_token:
        status = _modern_kernel_status(state["kernel_ref"], access_token)
    else:
        status = _run(cli + ["kernels", "status", state["kernel_ref"]], timeout=120)
    state["history"].append(_history_item("kernels status", status))
    status_text = f"{status['stdout']}\n{status['stderr']}".lower()
    if status["returncode"] != 0:
        return _fail(state, _summarize_cli_error(status))
    if any(value in status_text for value in ("complete", "completed", "succeeded")):
        state["status"] = "complete"
        if access_token:
            download = _modern_kernel_output(
                state["kernel_ref"],
                Path(state["download_dir"]),
                access_token,
            )
        else:
            download = _run(cli + ["kernels", "output", state["kernel_ref"], "-p", state["download_dir"], "-o"], timeout=1200)
        state["history"].append(_history_item("kernels output", download))
        if download["returncode"] != 0:
            return _fail(state, _summarize_cli_error(download))
        state["downloaded_files"] = [str(path) for path in Path(state["download_dir"]).rglob("*") if path.is_file()]
        _attach_artifact_urls(state)
    elif any(value in status_text for value in ("error", "failed", "cancelled", "canceled")):
        state["status"] = "failed"
        state["last_error"] = status_text[-2000:]
    else:
        state["status"] = "running" if "running" in status_text else "submitted"
        state["last_error"] = ""
    state["checked_at"] = _now()
    _write_state(state)
    return state


def _modern_kaggle_rpc(
    method: str,
    payload: dict[str, Any],
    access_token: str,
    *,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Call the www.kaggle.com RPC endpoint required by modern KGAT tokens."""
    import requests

    response = requests.post(
        f"https://www.kaggle.com/api/v1/kernels.KernelsApiService/{method}",
        headers={"Authorization": f"Bearer {access_token}"},
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def _modern_kernel_status(kernel_ref: str, access_token: str) -> dict[str, Any]:
    owner, slug = validate_kernel_ref(kernel_ref).split("/", 1)
    try:
        payload = _modern_kaggle_rpc(
            "GetKernelSessionStatus",
            {"userName": owner, "kernelSlug": slug},
            access_token,
        )
        return {"returncode": 0, "stdout": str(payload.get("status", "")), "stderr": ""}
    except Exception as exc:
        return {"returncode": 1, "stdout": "", "stderr": f"{type(exc).__name__}: {exc}"}


def _modern_kernel_output(
    kernel_ref: str,
    destination: Path,
    access_token: str,
) -> dict[str, Any]:
    """Download generated web artifacts without copying the whole source tree."""
    import requests

    owner, slug = validate_kernel_ref(kernel_ref).split("/", 1)
    destination.mkdir(parents=True, exist_ok=True)
    downloaded: list[str] = []
    page_token = ""
    try:
        while True:
            request: dict[str, Any] = {
                "userName": owner,
                "kernelSlug": slug,
                "pageSize": 100,
            }
            if page_token:
                request["pageToken"] = page_token
            payload = _modern_kaggle_rpc(
                "ListKernelSessionOutput",
                request,
                access_token,
                timeout=180.0,
            )
            for item in payload.get("files", []):
                name = str(item.get("fileName", "")).replace("\\", "/")
                # The web kernel copies source into /kaggle/working as well. Only
                # generation artifacts belong in a request's downloaded output.
                if not name.startswith("genmusic_output/"):
                    continue
                relative = Path(name)
                target = (destination / relative).resolve()
                if destination.resolve() not in target.parents:
                    raise ValueError(f"Unsafe Kaggle output path: {name}")
                url = str(item.get("url", ""))
                if not url:
                    continue
                response = requests.get(url, stream=True, timeout=300.0)
                response.raise_for_status()
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                downloaded.append(name)
            page_token = str(payload.get("nextPageToken") or "")
            if not page_token:
                break
        return {
            "returncode": 0,
            "stdout": "Downloaded: " + ", ".join(downloaded),
            "stderr": "",
        }
    except Exception as exc:
        return {"returncode": 1, "stdout": "", "stderr": f"{type(exc).__name__}: {exc}"}


def kaggle_readiness(username: str | None = None) -> dict[str, Any]:
    tokens = load_kaggle_api_tokens()
    if username and not tokens.get("KAGGLE_USERNAME"):
        tokens["KAGGLE_USERNAME"] = username
    ready = bool(tokens.get("KAGGLE_USERNAME") and kaggle_auth_available(tokens) and kaggle_cli_command())
    if not ready:
        return {
            "ready": False,
            "messages": ["Cần KAGGLE_USERNAME, KAGGLE_API_TOKEN=KGAT_... (hoặc KAGGLE_KEY cũ) và Kaggle CLI."],
        }
    return {"ready": True, "messages": []}


def kaggle_dataset_exists(dataset_ref: str) -> bool:
    cli = kaggle_cli_command()
    if cli is None or dataset_ref.startswith("YOUR_KAGGLE_USERNAME/"):
        return False
    result = _run(cli + ["datasets", "status", dataset_ref], timeout=120)
    return _dataset_status_is_ready(result)


def kaggle_kernel_complete(kernel_ref: str) -> bool:
    cli = kaggle_cli_command()
    if cli is None or kernel_ref.startswith("YOUR_KAGGLE_USERNAME/"):
        return False
    result = _run(cli + ["kernels", "status", kernel_ref], timeout=120)
    status_text = f"{result['stdout']}\n{result['stderr']}".lower()
    return result["returncode"] == 0 and "complete" in status_text


def _wait_for_dataset_ready(cli: list[str], dataset_ref: str, *, timeout_seconds: int) -> bool:
    deadline = time.time() + max(1, timeout_seconds)
    while time.time() < deadline:
        result = _run(cli + ["datasets", "status", dataset_ref], timeout=120)
        status_text = f"{result['stdout']}\n{result['stderr']}".lower()
        if _dataset_status_is_ready(result):
            time.sleep(15)
            return True
        if result["returncode"] == 0 and any(value in status_text for value in ("failed", "deleted", "cancelled", "canceled")):
            return False
        time.sleep(5)
    return False


def _dataset_status_is_ready(result: dict[str, Any]) -> bool:
    if result["returncode"] != 0:
        return False
    status_text = result["stdout"].strip().lower()
    return status_text == "ready" or 'status "ready"' in status_text or status_text.endswith("status ready")


def upload_dataset_to_kaggle(dataset_dir: str | Path, *, username: str | None = None, slug: str | None = None, dataset_ref: str | None = None, timeout_seconds: int = 3_600) -> dict[str, Any]:
    root = Path(dataset_dir).resolve()
    if not root.exists() or not (root / "records.jsonl").exists():
        raise ValueError(f"Không tìm thấy dataset self-diffusion hợp lệ tại {root}.")
    cli = kaggle_cli_command()
    resolved_username = resolve_kaggle_username(username)
    if cli is None or not resolved_username or not kaggle_auth_available():
        report = {
            "status": "needs_setup",
            "dataset": str(root),
            "message": "Cần Kaggle CLI, KAGGLE_USERNAME và KAGGLE_API_TOKEN=KGAT_... (hoặc KAGGLE_KEY cũ).",
        }
        (root / "kaggle_upload_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report
    if dataset_ref:
        dataset_ref = validate_dataset_ref(dataset_ref)
    elif slug:
        dataset_ref = f"{resolved_username}/{slugify(slug, max_length=50)}"
    else:
        dataset_ref = resolve_training_dataset_ref(None, username)
    metadata_path = root / "dataset-metadata.json"
    metadata_path.write_text(json.dumps({"title": "GenMusic VN self-diffusion training", "id": dataset_ref, "licenses": [{"name": "other"}], "subtitle": "Synthetic mel dataset for the GenMusic text-to-music model.", "description": "Synthetic structured mel tensors and Vietnamese text/style conditions for pipeline training smoke tests."}, ensure_ascii=False, indent=2), encoding="utf-8")
    started = time.perf_counter()
    result = _run(cli + ["datasets", "create", "-p", str(root), "-r", "zip"], timeout=timeout_seconds)
    dataset_ready = result["returncode"] == 0 and _wait_for_dataset_ready(cli, dataset_ref, timeout_seconds=min(timeout_seconds, 900))
    status = "uploaded" if dataset_ready else ("pending" if result["returncode"] == 0 else "failed")
    report = {"status": status, "dataset": str(root), "dataset_ref": dataset_ref, "dataset_url": _dataset_url(dataset_ref), "dataset_ready": dataset_ready, "returncode": result["returncode"], "elapsed_seconds": round(time.perf_counter() - started, 3), "stdout_tail": result["stdout"][-4000:], "stderr_tail": result["stderr"][-4000:]}
    (root / "kaggle_upload_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _read_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _kaggle_file_credentials(config_dir: Path) -> dict[str, str]:
    """Read Kaggle's config files only as a fallback behind project dotenv files."""
    values: dict[str, str] = {}
    kaggle_json = config_dir / "kaggle.json"
    if kaggle_json.exists():
        try:
            payload = json.loads(kaggle_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        if payload.get("username"):
            values["KAGGLE_USERNAME"] = str(payload["username"]).strip()
        if payload.get("key"):
            values["KAGGLE_KEY"] = str(payload["key"]).strip()

    access_token = config_dir / "access_token"
    if access_token.exists():
        try:
            token = access_token.read_text(encoding="utf-8").strip()
        except OSError:
            token = ""
        # Do not reinterpret a legacy key accidentally written here as a new token.
        if token.startswith(("KGAT_", "KGAT-")):
            values[KAGGLE_API_TOKEN_ENV] = token
    return values


def _replace_auth_credentials(values: dict[str, str], preferred: dict[str, str]) -> None:
    """Replace lower-priority auth as a group, avoiding mixed credential types."""
    if any(preferred.get(key) for key in KAGGLE_AUTH_ENV_KEYS):
        for key in KAGGLE_AUTH_ENV_KEYS:
            values.pop(key, None)
    values.update({key: value for key, value in preferred.items() if value})


def load_kaggle_api_tokens() -> dict[str, str]:
    """Load Kaggle settings with OS env > project dotenv > ~/.kaggle files."""
    config_dir = Path(os.getenv("KAGGLE_CONFIG_DIR") or Path.home() / ".kaggle").expanduser()
    values = _kaggle_file_credentials(config_dir)

    project_values: dict[str, str] = {}
    # .env.local intentionally overrides .env while both outrank ~/.kaggle.
    for path in (PROJECT_ROOT / ".env", PROJECT_ROOT / ".env.local"):
        _replace_auth_credentials(project_values, _read_dotenv(path))
    _replace_auth_credentials(values, project_values)

    environment_values = {
        key: str(os.environ[key])
        for key in (
            "KAGGLE_USERNAME",
            "KAGGLE_KEY",
            KAGGLE_API_TOKEN_ENV,
            KAGGLE_ACCESS_TOKEN_ALIAS_ENV,
            KAGGLE_DATASET_ENV,
            KAGGLE_CHECKPOINT_ENV,
            KAGGLE_BACKING_ENV,
        )
        if os.getenv(key)
    }
    _replace_auth_credentials(values, environment_values)

    # Accept a readable alias in .env but expose Kaggle's official variable name.
    if not values.get(KAGGLE_API_TOKEN_ENV) and values.get(KAGGLE_ACCESS_TOKEN_ALIAS_ENV):
        values[KAGGLE_API_TOKEN_ENV] = values[KAGGLE_ACCESS_TOKEN_ALIAS_ENV]
    # Existing .env files used KAGGLE_KEY. Detect a newly pasted KGAT token there
    # and pass it to Kaggle under the modern variable instead of legacy-key auth.
    key_value = str(values.get("KAGGLE_KEY", "")).strip()
    if not values.get(KAGGLE_API_TOKEN_ENV) and key_value.startswith(("KGAT_", "KGAT-")):
        values[KAGGLE_API_TOKEN_ENV] = key_value
        values.pop("KAGGLE_KEY", None)
    return values


def kaggle_access_token(values: dict[str, str] | None = None) -> str | None:
    credentials = values if values is not None else load_kaggle_api_tokens()
    token = str(credentials.get(KAGGLE_API_TOKEN_ENV, "")).strip()
    return token if token.startswith(("KGAT_", "KGAT-")) else None


def kaggle_auth_available(values: dict[str, str] | None = None) -> bool:
    credentials = values if values is not None else load_kaggle_api_tokens()
    return bool(
        kaggle_access_token(credentials)
        or (credentials.get("KAGGLE_USERNAME") and credentials.get("KAGGLE_KEY"))
    )


def kaggle_auth_environment(values: dict[str, str] | None = None) -> dict[str, str]:
    credentials = dict(values if values is not None else load_kaggle_api_tokens())
    token = kaggle_access_token(credentials)
    if token:
        credentials[KAGGLE_API_TOKEN_ENV] = token
    return credentials


def kaggle_cli_command() -> list[str] | None:
    return [sys.executable, "-m", "kaggle"]


def resolve_kaggle_username(username: str | None) -> str | None:
    return username or load_kaggle_api_tokens().get("KAGGLE_USERNAME")


def validate_dataset_ref(dataset_ref: str) -> str:
    value = str(dataset_ref or "").strip().strip("/")
    parts = value.split("/")
    if len(parts) != 2 or not all(parts) or any(char.isspace() for char in value):
        raise ValueError("Dataset ref phải có dạng owner/slug, ví dụ user/genmusic-vn-self-diffusion-training.")
    return value


def validate_kernel_ref(kernel_ref: str) -> str:
    value = str(kernel_ref or "").strip().strip("/")
    parts = value.split("/")
    if len(parts) != 2 or not all(parts) or any(char.isspace() for char in value):
        raise ValueError(
            "Kernel ref phải có dạng owner/slug, ví dụ user/genmusic-checkpoint."
        )
    return value


def _resolve_required_kernel_ref(
    configured: str | None,
    *,
    env_key: str,
    username: str | None,
) -> str:
    value = configured or os.getenv(env_key) or load_kaggle_api_tokens().get(env_key)
    if value:
        return validate_kernel_ref(value)
    owner = resolve_kaggle_username(username) or "YOUR_KAGGLE_USERNAME"
    suffix = "genmusic-latest-checkpoint" if env_key == KAGGLE_CHECKPOINT_ENV else "genmusic-backing-source"
    return f"{owner}/{suffix}"


def resolve_checkpoint_kernel_ref(
    kernel_ref: str | None = None,
    username: str | None = None,
) -> str:
    return _resolve_required_kernel_ref(
        kernel_ref,
        env_key=KAGGLE_CHECKPOINT_ENV,
        username=username,
    )


def resolve_backing_kernel_ref(
    kernel_ref: str | None = None,
    username: str | None = None,
) -> str:
    return _resolve_required_kernel_ref(
        kernel_ref,
        env_key=KAGGLE_BACKING_ENV,
        username=username,
    )


def resolve_training_dataset_ref(dataset_ref: str | None = None, username: str | None = None) -> str:
    configured = dataset_ref or os.getenv(KAGGLE_DATASET_ENV) or load_kaggle_api_tokens().get(KAGGLE_DATASET_ENV)
    if configured:
        return validate_dataset_ref(configured)
    owner = resolve_kaggle_username(username) or "YOUR_KAGGLE_USERNAME"
    return f"{owner}/{DEFAULT_KAGGLE_DATASET_SLUG}"


def _dataset_url(dataset_ref: str) -> str:
    return f"https://www.kaggle.com/datasets/{dataset_ref}" if not dataset_ref.startswith("YOUR_KAGGLE_USERNAME/") else ""


def _kernel_url(kernel_ref: str) -> str:
    return (
        f"https://www.kaggle.com/code/{kernel_ref}"
        if not kernel_ref.startswith("YOUR_KAGGLE_USERNAME/")
        else ""
    )


def make_run_id(text: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{sha1(text.encode('utf-8')).hexdigest()[:10]}"


def slugify(value: str, *, max_length: int = 48) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii").lower()
    normalized = "".join(char if char.isalnum() else "-" for char in normalized).strip("-")
    return normalized[:max_length].strip("-") or "genmusic-vn"


def _make_lrc(text: str, duration: int):
    lines = [line.strip() for line in text.splitlines() if line.strip()] or [text]
    span = max(1.0, duration / len(lines))
    return [AlignedLine(line, index * span, (index + 1) * span, "generated") for index, line in enumerate(lines)]


def _kernel_script(request_dataset_slug: str) -> str:
    return f'''import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

input_root = Path("/kaggle/input/{request_dataset_slug}")
if not (input_root / "request.json").exists():
    request_files = list(Path("/kaggle/input").rglob("request.json"))
    if request_files:
        input_root = request_files[0].parent
request = json.loads((input_root / "request.json").read_text(encoding="utf-8"))

checkpoint_candidates = list(Path("/kaggle/input").rglob("self_all_parts.pt"))
if not checkpoint_candidates:
    raise RuntimeError("Checkpoint kernel is missing self_all_parts.pt.")
checkpoint = max(checkpoint_candidates, key=lambda path: path.stat().st_size)

backing_dataset = None
for records_path in Path("/kaggle/input").rglob("records.jsonl"):
    try:
        record = json.loads(next(line for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()))
    except (OSError, StopIteration, json.JSONDecodeError):
        continue
    backing_path = record.get("backing_mel_path")
    if backing_path and (records_path.parent / backing_path).exists():
        backing_dataset = records_path.parent
        break
if backing_dataset is None:
    raise RuntimeError("Backing kernel is missing a usable records.jsonl/backing_mel_path.")

source_root = Path("/kaggle/working/GenMusic")
source_zip = next(input_root.rglob("genmusic_vn_source.zip"), None)
source_dir = next(input_root.rglob("genmusic_vn_source"), None)
if source_zip and source_zip.is_file():
    with zipfile.ZipFile(source_zip) as archive:
        archive.extractall(source_root)
elif source_dir and source_dir.is_dir():
    shutil.copytree(source_dir, source_root, dirs_exist_ok=True)
else:
    raise RuntimeError("Không tìm thấy source model trong request dataset Kaggle.")
os.environ["PYTHONPATH"] = str(source_root) + os.pathsep + os.environ.get("PYTHONPATH", "")
subprocess.run(["apt-get", "update", "-qq"], check=True)
subprocess.run(["apt-get", "install", "-y", "-qq", "espeak-ng"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch", "torchaudio", "librosa", "soundfile", "imageio-ffmpeg", "vocos", "transformers", "sentencepiece", "text2phonemesequence"], check=True)
output = Path("/kaggle/working/genmusic_output")
command = [
    sys.executable, "cli.py", "generate-local",
    "--text", request["lyrics"],
    "--style", request["style_prompt"],
    "--duration", str(request["duration_seconds"]),
    "--checkpoint", str(checkpoint),
    "--steps", str(request["diffusion_steps"]),
    "--guidance-scale", str(request["guidance_scale"]),
    "--pronunciation-prior-strength", str(request["pronunciation_prior_strength"]),
    "--reference-dataset", str(backing_dataset),
    "--no-reference-style",
    "--mix-backing",
    "--backing-to-vocal-rms", str(request["backing_to_vocal_rms"]),
    "--vocoder", "vocos",
    "--device", "cuda",
    "--out", str(output),
]
print("Generation-only web job:", " ".join(command), flush=True)
subprocess.run(command, cwd=source_root, check=True)
output.joinpath("request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
'''


def _write_source_zip(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    excluded = {".git", "outputs", "__pycache__", ".pytest_cache", ".venv", ".kaggle"}
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in PROJECT_ROOT.rglob("*"):
            relative = path.relative_to(PROJECT_ROOT)
            if not path.is_file() or any(part in excluded for part in relative.parts):
                continue
            if relative.name.startswith(".env") or relative.name == "kaggle.json":
                continue
            if relative.as_posix().startswith(("models/", "outputs/", "datasets/")):
                continue
            archive.write(path, relative.as_posix())


def _commands(dataset_dir: Path, kernel_dir: Path, download_dir: Path, kernel_ref: str) -> list[str]:
    return [
        "pip install -U kaggle",
        "# Đặt KAGGLE_USERNAME và KAGGLE_API_TOKEN=KGAT_... trong .env hoặc environment.",
        f'kaggle datasets create -p "{dataset_dir}" -r zip',
        f'kaggle kernels push -p "{kernel_dir}"',
        f'kaggle kernels status "{kernel_ref}"',
        f'kaggle kernels output "{kernel_ref}" -p "{download_dir}" -o',
    ]


def _default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _convert_to_mp3(wav_path: Path) -> Path | None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        try:
            import imageio_ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            return None
    if not ffmpeg:
        return None
    destination = wav_path.with_suffix(".mp3")
    result = subprocess.run([ffmpeg, "-y", "-i", str(wav_path), str(destination)], capture_output=True, text=True, check=False)
    return destination.resolve() if result.returncode == 0 and destination.exists() else None


def _run(command: list[str], *, timeout: int) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env={
                **os.environ,
                **load_kaggle_api_tokens(),
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUTF8": "1",
            },
        )
        return {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout or "",
            "stderr": result.stderr or "",
        }
    except Exception as exc:
        return {"command": command, "returncode": -1, "stdout": "", "stderr": str(exc)}


def _history_item(stage: str, result: dict[str, Any]) -> dict[str, Any]:
    return {"stage": stage, "returncode": result["returncode"], "stdout_tail": result["stdout"][-1000:], "stderr_tail": result["stderr"][-1000:], "at": _now()}


def _summarize_cli_error(result: dict[str, Any]) -> str:
    return (result.get("stderr") or result.get("stdout") or "Kaggle command failed")[-2000:]


def _fail(state: dict[str, Any], message: str) -> dict[str, Any]:
    state["status"] = "failed"
    state["last_error"] = message
    _write_state(state)
    return state


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_state(state: dict[str, Any]) -> None:
    path = Path(state["state_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_state(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _attach_artifact_urls(state: dict[str, Any]) -> None:
    output_root = PROJECT_ROOT / "outputs"
    audio_candidates: dict[str, list[tuple[int, Path, str]]] = {".mp3": [], ".wav": []}
    for path_text in state.get("downloaded_files", []):
        path = Path(path_text)
        try:
            relative = path.resolve().relative_to(output_root.resolve()).as_posix()
        except ValueError:
            continue
        url = "/outputs/" + relative
        suffix = path.suffix.lower()
        if suffix in audio_candidates:
            normalized = path.as_posix().lower()
            priority = 0 if normalized.endswith("genmusic_output/final" + suffix) else 1
            audio_candidates[suffix].append((priority, path, url))
    for suffix, key in ((".mp3", "mp3"), (".wav", "wav")):
        if not audio_candidates[suffix]:
            continue
        _, path, url = min(audio_candidates[suffix], key=lambda item: (item[0], item[1].as_posix()))
        state[f"{key}_url"] = url
        state[f"{key}_path"] = str(path)
    lrc = Path(state["dataset_dir"]) / "lyrics.lrc"
    if lrc.exists():
        try:
            state["lrc_url"] = "/outputs/" + lrc.resolve().relative_to(output_root.resolve()).as_posix()
        except ValueError:
            pass
