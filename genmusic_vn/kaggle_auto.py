from __future__ import annotations

import json
import os
import re
import shutil
import site
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
from uuid import uuid4


class KaggleAutoError(RuntimeError):
    pass


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MUSICGEN_MODEL = "facebook/musicgen-medium"
DEFAULT_TTS_MODEL = "hynt/F5-TTS-Vietnamese-ViVoice"
DEFAULT_MMS_TTS_MODEL = "facebook/mms-tts-vie"
DEFAULT_TTS_VOICE_ACTUAL = "f5_vietnamese_vivoice_reference"
DEFAULT_TTS_VOICE_NOTE = "F5-TTS Vietnamese uses a short Vietnamese reference voice; MMS Vietnamese is kept as fallback."


@dataclass(frozen=True)
class KaggleJobConfig:
    model: str = DEFAULT_MUSICGEN_MODEL
    username: str | None = None
    machine_shape: str = "NvidiaTeslaT4"
    submit: bool = True
    wait: bool = False
    poll_seconds: int = 60
    timeout_seconds: int = 10_800


def submit_text_to_music_job(
    *,
    text: str,
    output_root: str | Path = "outputs",
    duration_seconds: int = 30,
    genre: str | None = None,
    config: KaggleJobConfig | None = None,
) -> dict[str, Any]:
    config = config or KaggleJobConfig()
    state = stage_text_to_music_job(
        text=text,
        output_root=output_root,
        duration_seconds=duration_seconds,
        genre=genre,
        config=config,
    )
    if not config.submit:
        state["status"] = "staged"
        state["messages"].append("Job staged locally. Submit is disabled.")
        _write_state(state)
        return state

    readiness = kaggle_readiness(config.username)
    state["kaggle_ready"] = readiness["ready"]
    state["messages"].extend(readiness["messages"])
    _write_state(state)
    if not readiness["ready"]:
        state["status"] = "needs_setup"
        state["messages"].append("Install/configure Kaggle API, then rerun the generated commands.")
        _write_state(state)
        return state

    return submit_kaggle_job(
        state,
        wait=config.wait,
        poll_seconds=config.poll_seconds,
        timeout_seconds=config.timeout_seconds,
    )


def submit_tts_retry_job(
    state_or_path: dict[str, Any] | str | Path,
    *,
    config: KaggleJobConfig | None = None,
) -> dict[str, Any]:
    config = config or KaggleJobConfig()
    parent_state = _load_state(state_or_path)
    state = stage_tts_retry_job(parent_state, config=config)
    if not config.submit:
        state["status"] = "staged"
        state["messages"].append("TTS retry job staged locally. Submit is disabled.")
        _write_state(state)
        return state

    readiness = kaggle_readiness(config.username)
    state["kaggle_ready"] = readiness["ready"]
    state["messages"].extend(readiness["messages"])
    _write_state(state)
    if not readiness["ready"]:
        state["status"] = "needs_setup"
        state["messages"].append("Install/configure Kaggle API, then retry TTS again.")
        _write_state(state)
        return state

    return submit_kaggle_job(
        state,
        wait=config.wait,
        poll_seconds=config.poll_seconds,
        timeout_seconds=config.timeout_seconds,
    )


def stage_text_to_music_job(
    *,
    text: str,
    output_root: str | Path,
    duration_seconds: int,
    genre: str | None,
    config: KaggleJobConfig,
) -> dict[str, Any]:
    normalized = _normalize_request_text(text)
    if not normalized:
        raise ValueError("Input text is empty.")

    duration_seconds = max(6, min(180, int(duration_seconds)))
    run_id = make_run_id(normalized)
    username = resolve_kaggle_username(config.username) or "YOUR_KAGGLE_USERNAME"
    run_dir = Path(output_root) / run_id
    job_dir = run_dir / "kaggle_job"
    dataset_dir = job_dir / "dataset"
    kernel_dir = job_dir / "kernel"
    download_dir = job_dir / "downloaded_output"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    kernel_dir.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)

    dataset_slug = slugify(f"genmusic-vn-job-{run_id}", max_length=48)
    kernel_slug = slugify(f"genmusic-vn-musicgen-{run_id}", max_length=48)
    dataset_ref = f"{username}/{dataset_slug}"
    kernel_ref = f"{username}/{kernel_slug}"

    request = {
        "run_id": run_id,
        "text": normalized,
        "duration_seconds": duration_seconds,
        "target_duration_seconds": duration_seconds,
        "duration_policy": "soft_target",
        "genre": genre or "Vietnamese cinematic pop text-to-song",
        "model": config.model,
        "tts_model": DEFAULT_TTS_MODEL,
        "mms_tts_model": DEFAULT_MMS_TTS_MODEL,
        "backend": "musicgen",
        "created_at": _now(),
    }
    (run_dir / "request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
    (dataset_dir / "request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_source_zip(dataset_dir / "genmusic_vn_source.zip")
    (dataset_dir / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": dataset_slug,
                "id": dataset_ref,
                "licenses": [{"name": "other"}],
                "subtitle": "Raw Vietnamese text request and pipeline source for Kaggle MusicGen generation.",
                "description": "Private automation dataset generated by the GenMusic VN local app.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    (kernel_dir / "run_genmusic_vn.py").write_text(_kernel_script(dataset_slug=dataset_slug), encoding="utf-8")
    (kernel_dir / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": kernel_ref,
                "title": kernel_slug,
                "code_file": "run_genmusic_vn.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": "true",
                "enable_gpu": "true",
                "enable_internet": "true",
                "machine_shape": config.machine_shape,
                "dataset_sources": [dataset_ref],
                "competition_sources": [],
                "kernel_sources": [],
                "model_sources": [],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    commands = _commands(dataset_dir, kernel_dir, download_dir, kernel_ref)
    (job_dir / "run_commands.ps1").write_text("\n".join(commands) + "\n", encoding="utf-8")

    state = {
        "run_id": run_id,
        "status": "staged",
        "created_at": _now(),
        "kaggle_ready": False,
        "backend": "musicgen",
        "model": config.model,
        "duration_policy": "soft_target",
        "target_duration_seconds": duration_seconds,
        "duration_plan": {
            "policy": "soft_target",
            "target_duration_seconds": duration_seconds,
        },
        "dataset_ref": dataset_ref,
        "kernel_ref": kernel_ref,
        "run_dir": str(run_dir),
        "job_dir": str(job_dir),
        "dataset_dir": str(dataset_dir),
        "kernel_dir": str(kernel_dir),
        "download_dir": str(download_dir),
        "request_path": str(run_dir / "request.json"),
        "state_path": str(job_dir / "job_state.json"),
        "mp3_path": "",
        "mp3_url": "",
        "lyrics_path": "",
        "lyrics_url": "",
        "lyrics_text": "",
        "lyrics": {},
        "vocal_plan": {},
        "backing_path": "",
        "backing_url": "",
        "vocal_path": "",
        "vocal_url": "",
        "musicgen_failed": False,
        "vocal_failed": False,
        "tts_model": DEFAULT_TTS_MODEL,
        "mms_tts_model": DEFAULT_MMS_TTS_MODEL,
        "tts_voice_actual": DEFAULT_TTS_VOICE_ACTUAL,
        "tts_voice_note": DEFAULT_TTS_VOICE_NOTE,
        "commands": commands,
        "messages": ["Kaggle MusicGen job files prepared."],
        "last_error": "",
        "musicgen_error": "",
        "f5_tts_error": "",
        "tts_error": "",
        "history": [],
        "downloaded_files": [],
    }
    _write_state(state)
    return state


def stage_tts_retry_job(parent_state: dict[str, Any], *, config: KaggleJobConfig) -> dict[str, Any]:
    backing_source = _find_retry_backing_path(parent_state)
    request = _load_retry_request(parent_state)
    parent_run_id = str(parent_state.get("run_id") or request.get("run_id") or "genmusic-vn")
    retry_run_id = make_retry_run_id(parent_run_id)
    username = resolve_kaggle_username(config.username) or "YOUR_KAGGLE_USERNAME"

    output_root = Path(parent_state.get("run_dir", "outputs")).parent
    run_dir = output_root / retry_run_id
    job_dir = run_dir / "kaggle_job"
    dataset_dir = job_dir / "dataset"
    kernel_dir = job_dir / "kernel"
    download_dir = job_dir / "downloaded_output"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    kernel_dir.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)

    dataset_slug = slugify(f"genmusic-vn-tts-data-{retry_run_id}", max_length=48)
    kernel_slug = slugify(f"genmusic-vn-tts-code-{retry_run_id}", max_length=48)
    dataset_ref = f"{username}/{dataset_slug}"
    kernel_ref = f"{username}/{kernel_slug}"

    request["run_id"] = retry_run_id
    request["parent_run_id"] = parent_run_id
    request["backend"] = "tts_retry"
    request["model"] = parent_state.get("model") or request.get("model") or config.model
    previous_tts_model = request.get("tts_model")
    if previous_tts_model == DEFAULT_MMS_TTS_MODEL:
        request["mms_tts_model"] = request.get("mms_tts_model") or previous_tts_model
        request["tts_model"] = DEFAULT_TTS_MODEL
    else:
        request["tts_model"] = previous_tts_model or DEFAULT_TTS_MODEL
        request["mms_tts_model"] = request.get("mms_tts_model") or parent_state.get("mms_tts_model") or DEFAULT_MMS_TTS_MODEL
    if "target_duration_seconds" not in request and parent_state.get("target_duration_seconds"):
        request["target_duration_seconds"] = parent_state["target_duration_seconds"]
    if "duration_seconds" not in request and parent_state.get("target_duration_seconds"):
        request["duration_seconds"] = parent_state["target_duration_seconds"]

    (run_dir / "request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
    (dataset_dir / "request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(backing_source, dataset_dir / "backing_input.mp3")
    _write_source_zip(dataset_dir / "genmusic_vn_source.zip")
    (dataset_dir / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": dataset_slug,
                "id": dataset_ref,
                "licenses": [{"name": "other"}],
                "subtitle": "TTS-only retry dataset for GenMusic VN.",
                "description": "Private retry dataset containing the previous backing MP3 and request metadata.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    (kernel_dir / "run_genmusic_vn_tts_retry.py").write_text(
        _tts_retry_kernel_script(dataset_slug=dataset_slug),
        encoding="utf-8",
    )
    (kernel_dir / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": kernel_ref,
                "title": kernel_slug,
                "code_file": "run_genmusic_vn_tts_retry.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": "true",
                "enable_gpu": "true",
                "enable_internet": "true",
                "machine_shape": config.machine_shape,
                "dataset_sources": [dataset_ref],
                "competition_sources": [],
                "kernel_sources": [],
                "model_sources": [],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    commands = _commands(dataset_dir, kernel_dir, download_dir, kernel_ref)
    (job_dir / "run_commands.ps1").write_text("\n".join(commands) + "\n", encoding="utf-8")

    state = {
        "run_id": retry_run_id,
        "parent_run_id": parent_run_id,
        "job_kind": "tts_retry",
        "status": "staged",
        "created_at": _now(),
        "kaggle_ready": False,
        "backend": "tts_retry",
        "model": request.get("model") or config.model,
        "duration_policy": request.get("duration_policy", "soft_target"),
        "target_duration_seconds": int(request.get("target_duration_seconds", request.get("duration_seconds", 30))),
        "duration_plan": {
            "policy": request.get("duration_policy", "soft_target"),
            "target_duration_seconds": int(request.get("target_duration_seconds", request.get("duration_seconds", 30))),
        },
        "dataset_ref": dataset_ref,
        "kernel_ref": kernel_ref,
        "run_dir": str(run_dir),
        "job_dir": str(job_dir),
        "dataset_dir": str(dataset_dir),
        "kernel_dir": str(kernel_dir),
        "download_dir": str(download_dir),
        "request_path": str(run_dir / "request.json"),
        "state_path": str(job_dir / "job_state.json"),
        "mp3_path": "",
        "mp3_url": "",
        "lyrics_path": "",
        "lyrics_url": "",
        "lyrics_text": parent_state.get("lyrics_text", ""),
        "lyrics": parent_state.get("lyrics", {}),
        "vocal_plan": parent_state.get("vocal_plan", {}),
        "backing_path": str(backing_source),
        "backing_url": parent_state.get("backing_url", "") or parent_state.get("mp3_url", ""),
        "vocal_path": "",
        "vocal_url": "",
        "musicgen_failed": bool(parent_state.get("musicgen_failed")),
        "vocal_failed": False,
        "tts_model": request.get("tts_model") or DEFAULT_TTS_MODEL,
        "mms_tts_model": request.get("mms_tts_model") or DEFAULT_MMS_TTS_MODEL,
        "tts_voice_actual": DEFAULT_TTS_VOICE_ACTUAL,
        "tts_voice_note": DEFAULT_TTS_VOICE_NOTE,
        "commands": commands,
        "messages": [
            f"TTS-only retry prepared from parent job {parent_run_id}.",
            "The previous backing MP3 will be reused; MusicGen will not run again.",
        ],
        "last_error": "",
        "musicgen_error": parent_state.get("musicgen_error", ""),
        "f5_tts_error": "",
        "tts_error": "",
        "history": [],
        "downloaded_files": [],
    }
    _write_state(state)
    return state


def submit_kaggle_job(
    state: dict[str, Any],
    *,
    wait: bool = False,
    poll_seconds: int = 60,
    timeout_seconds: int = 10_800,
) -> dict[str, Any]:
    cli = kaggle_cli_command()
    if cli is None:
        state["status"] = "needs_setup"
        state["messages"].append("Kaggle CLI was not found.")
        _write_state(state)
        return state

    dataset = _run(cli + ["datasets", "create", "-p", state["dataset_dir"], "-r", "zip"], timeout=600)
    state["history"].append(_history_item("datasets create", dataset))
    if dataset["returncode"] != 0:
        state["status"] = "failed"
        state["last_error"] = _summarize_cli_error(dataset)
        state["messages"].append("Dataset upload failed.")
        _write_state(state)
        return state

    state["status"] = "dataset_uploaded"
    state["messages"].append("Raw text request and source uploaded to Kaggle.")
    _write_state(state)

    if not _wait_for_dataset_ready(state, cli):
        _write_state(state)
        return state

    pushed = _run(cli + ["kernels", "push", "-p", state["kernel_dir"]], timeout=600)
    state["history"].append(_history_item("kernels push", pushed))
    if pushed["returncode"] != 0:
        state["status"] = "failed"
        state["last_error"] = _summarize_cli_error(pushed)
        state["messages"].append("Kernel submit failed.")
        _write_state(state)
        return state

    state["status"] = "submitted"
    state["submitted_at"] = _now()
    if state.get("job_kind") == "tts_retry":
        state["messages"].append("Kaggle TTS-only retry kernel submitted.")
    else:
        state["messages"].append("Kaggle MusicGen kernel submitted with GPU enabled.")
    _write_state(state)

    if wait:
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            state = refresh_kaggle_job(state)
            if state["status"] in {"complete", "failed"}:
                return state
            time.sleep(max(5, poll_seconds))
        state["status"] = "timeout"
        state["messages"].append("Timed out while waiting for Kaggle job.")
        _write_state(state)
    return state


def refresh_kaggle_job(state_or_path: dict[str, Any] | str | Path) -> dict[str, Any]:
    state = _load_state(state_or_path)
    if state.get("status") == "complete" and state.get("mp3_path") and Path(state["mp3_path"]).exists():
        return state

    cli = kaggle_cli_command()
    if cli is None:
        state["status"] = "needs_setup"
        state["messages"].append("Kaggle CLI was not found.")
        _write_state(state)
        return state

    status = _run(cli + ["kernels", "status", state["kernel_ref"]], timeout=120)
    state["history"].append(_history_item("kernels status", status))
    text = f"{status['stdout']}\n{status['stderr']}".lower()
    state["last_status_output"] = status["stdout"] or status["stderr"]

    if status["returncode"] != 0:
        state["status"] = "failed"
        state["last_error"] = _summarize_cli_error(status)
        state["messages"].append("Could not read Kaggle kernel status.")
        _write_state(state)
        return state

    if any(marker in text for marker in ["complete", "completed", "succeeded"]):
        state["status"] = "complete"
        _download_kernel_output(state, cli, expect_mp3=True)
    elif any(marker in text for marker in ["error", "failed", "cancelled", "canceled"]):
        state["status"] = "failed"
        _download_kernel_output(state, cli, expect_mp3=False)
        if not state.get("last_error"):
            state["last_error"] = state.get("last_status_output", "")
    elif "running" in text:
        state["status"] = "running"
    else:
        state["status"] = "submitted"

    state["checked_at"] = _now()
    _write_state(state)
    return state


def sync_kaggle_artifact(*, source: str, ref: str, output_dir: str | Path = "models/current") -> dict[str, Any]:
    cli = kaggle_cli_command()
    if cli is None:
        raise KaggleAutoError("Kaggle CLI was not found. Install with `pip install kaggle`.")

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    if source == "dataset":
        command = cli + ["datasets", "download", ref, "-p", str(target), "--unzip"]
    elif source == "kernel":
        command = cli + ["kernels", "output", ref, "-p", str(target)]
    else:
        raise KaggleAutoError("Unsupported source. Use `dataset` or `kernel`.")

    result = _run(command, timeout=1800)
    manifest = {
        "source": source,
        "ref": ref,
        "output_dir": str(target),
        "synced_at": _now(),
        "command": command,
        "returncode": result["returncode"],
        "stdout": result["stdout"],
        "stderr": result["stderr"],
        "files": [str(path) for path in sorted(target.rglob("*")) if path.is_file()],
    }
    (target / "model_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if result["returncode"] != 0:
        raise KaggleAutoError(result["stderr"] or result["stdout"] or "Kaggle sync failed.")
    return manifest


def kaggle_readiness(username: str | None = None) -> dict[str, Any]:
    messages: list[str] = []
    cli = kaggle_cli_command()
    if cli is None:
        messages.append("Kaggle CLI missing. Install with `pip install kaggle`.")

    user = resolve_kaggle_username(username)
    if not user:
        messages.append("Kaggle username missing. Set KAGGLE_USERNAME in .env or environment.")

    tokens = load_kaggle_api_tokens()
    has_key = bool(tokens.get("KAGGLE_KEY"))
    if not has_key:
        messages.append("Kaggle API key missing. Set KAGGLE_KEY in .env or environment.")

    return {"ready": cli is not None and bool(user) and has_key, "username": user, "messages": messages}


def resolve_kaggle_username(username: str | None = None) -> str | None:
    if username:
        return username.strip()
    tokens = load_kaggle_api_tokens()
    env_user = tokens.get("KAGGLE_USERNAME")
    if env_user:
        return env_user.strip()
    return None


def kaggle_cli_command() -> list[str] | None:
    executable = shutil.which("kaggle")
    if executable:
        return [executable]

    script_name = "kaggle.exe" if os.name == "nt" else "kaggle"
    script_dirs = [
        Path(sys.executable).parent / "Scripts",
        Path(site.USER_BASE) / ("Scripts" if os.name == "nt" else "bin"),
        Path(site.USER_SITE).parent / ("Scripts" if os.name == "nt" else "bin"),
    ]
    seen: set[Path] = set()
    for scripts_dir in script_dirs:
        candidate = scripts_dir / script_name
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return [str(candidate)]
    return None


def load_kaggle_api_tokens() -> dict[str, str]:
    tokens: dict[str, str] = {}
    tokens.update(_read_kaggle_json(Path.home() / ".kaggle" / "kaggle.json"))
    tokens.update(_read_env_file(PROJECT_ROOT / ".env"))
    tokens.update(_read_env_file(PROJECT_ROOT / ".env.local"))
    for key in ("KAGGLE_USERNAME", "KAGGLE_KEY"):
        value = os.getenv(key) or tokens.get(key)
        if value:
            tokens[key] = _clean_env_value(value)
    return {key: value for key, value in tokens.items() if key in {"KAGGLE_USERNAME", "KAGGLE_KEY"} and value}


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in _read_text_flexible(path).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = _clean_env_value(value)
        if key:
            values[key] = value
    return values


def _read_kaggle_json(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        data = json.loads(_read_text_flexible(path))
    except (OSError, json.JSONDecodeError):
        return {}
    values: dict[str, str] = {}
    username = data.get("username")
    key = data.get("key")
    if isinstance(username, str):
        values["KAGGLE_USERNAME"] = _clean_env_value(username)
    if isinstance(key, str):
        values["KAGGLE_KEY"] = _clean_env_value(key)
    return values


def _read_text_flexible(path: Path) -> str:
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-16", "utf-8"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def _clean_env_value(value: str) -> str:
    return value.replace("\x00", "").strip().strip('"').strip("'").lstrip("\ufeff").strip()


def _normalize_request_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _load_retry_request(parent_state: dict[str, Any]) -> dict[str, Any]:
    request_path = parent_state.get("request_path")
    if isinstance(request_path, str) and Path(request_path).exists():
        try:
            data = json.loads(Path(request_path).read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except (OSError, json.JSONDecodeError):
            pass
    text = parent_state.get("input_text") or parent_state.get("text") or ""
    if not text:
        lyrics_text = parent_state.get("lyrics_text")
        text = lyrics_text if isinstance(lyrics_text, str) else ""
    return {
        "run_id": parent_state.get("run_id", "genmusic-vn"),
        "text": text,
        "duration_seconds": int(parent_state.get("target_duration_seconds", 30)),
        "target_duration_seconds": int(parent_state.get("target_duration_seconds", 30)),
        "duration_policy": parent_state.get("duration_policy", "soft_target"),
        "genre": parent_state.get("genre") or "Vietnamese cinematic pop text-to-song",
        "model": parent_state.get("model") or DEFAULT_MUSICGEN_MODEL,
        "tts_model": parent_state.get("tts_model") or DEFAULT_TTS_MODEL,
        "mms_tts_model": parent_state.get("mms_tts_model") or DEFAULT_MMS_TTS_MODEL,
        "backend": "tts_retry",
        "created_at": _now(),
    }


def _find_retry_backing_path(parent_state: dict[str, Any]) -> Path:
    candidates = [
        parent_state.get("backing_path"),
        parent_state.get("mp3_path"),
    ]
    for downloaded in parent_state.get("downloaded_files") or []:
        if isinstance(downloaded, str) and downloaded.lower().endswith(("_backing.mp3", "_fallback.mp3", ".mp3")):
            candidates.append(downloaded)
    for value in candidates:
        if not isinstance(value, str) or not value:
            continue
        path = Path(value)
        if path.exists() and path.is_file() and path.suffix.lower() == ".mp3":
            return path
    raise KaggleAutoError("Không tìm thấy backing MP3 local để retry TTS. Hãy chờ job cũ tải output xong trước.")


def make_run_id(text: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    nonce = uuid4().hex[:6]
    digest = sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"{timestamp}-{nonce}-{digest}"


def make_retry_run_id(parent_run_id: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    nonce = uuid4().hex[:6]
    base = slugify(parent_run_id, max_length=28) or "genmusic-vn"
    return f"{timestamp}-tts-{nonce}-{base}"


def slugify(value: str, max_length: int = 50) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", normalized).strip("-").lower()
    slug = re.sub(r"-{2,}", "-", slug)
    if not slug:
        slug = "genmusic-vn"
    return slug[:max_length].strip("-")


def _kernel_script(dataset_slug: str) -> str:
    return f'''from __future__ import annotations

import json
import shutil
import subprocess
import sys
import traceback
import zipfile
from pathlib import Path


INPUT_ROOT = Path("/kaggle/input")
DATASET_DIR = INPUT_ROOT / "{dataset_slug}"
SOURCE_DIR = Path("/kaggle/working/genmusic_vn_source")
PIPELINE_DIR = Path("/kaggle/working/pipeline_output")
OUTPUT_DIR = Path("/kaggle/working/genmusic_vn")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ASSET_DIR = Path("/kaggle/working/genmusic_vn_assets")
ASSET_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_TTS_MODEL = "hynt/F5-TTS-Vietnamese-ViVoice"
DEFAULT_F5_TTS_MODEL = "hynt/F5-TTS-Vietnamese-ViVoice"
DEFAULT_MMS_TTS_MODEL = "facebook/mms-tts-vie"
DEFAULT_TTS_VOICE_ACTUAL = "f5_vietnamese_vivoice_reference"
DEFAULT_TTS_VOICE_NOTE = "F5-TTS Vietnamese uses a short Vietnamese reference voice; MMS Vietnamese is kept as fallback."
F5_REPO_URL = "https://github.com/nguyenthienhy/F5-TTS-Vietnamese.git"
F5_REF_AUDIO_URL = "https://raw.githubusercontent.com/nguyenthienhy/F5-TTS-Vietnamese/main/ref.wav"
F5_REF_TEXT = "cả hai bên hãy cố gắng hiểu cho nhau"


def ensure(import_name: str, *pip_specs: str) -> None:
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", *pip_specs])


def ensure_transformers_version(min_version: str = "4.33.0") -> None:
    ensure("packaging", "packaging")
    ensure("transformers", f"transformers>={{min_version}}")

    import transformers
    from packaging.version import parse

    if parse(transformers.__version__) < parse(min_version):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", f"transformers>={{min_version}}"])


def find_input_file(name: str, *, required: bool = True) -> Path | None:
    direct = DATASET_DIR / name
    if direct.exists():
        return direct
    matches = sorted(INPUT_ROOT.rglob(name))
    if matches:
        return matches[0]
    if not required:
        return None
    available = [str(path) for path in sorted(INPUT_ROOT.rglob("*")) if path.is_file()]
    raise FileNotFoundError(
        f"Could not find {{name}} under {{INPUT_ROOT}}. Available files: {{available[:50]}}"
    )


def find_extracted_source_root() -> Path:
    candidates = sorted(
        path.parent
        for path in INPUT_ROOT.rglob("pyproject.toml")
        if (path.parent / "genmusic_vn").is_dir()
    )
    if candidates:
        return candidates[0]
    available = [str(path) for path in sorted(INPUT_ROOT.rglob("*")) if path.is_file()]
    raise FileNotFoundError(
        f"Could not find extracted genmusic_vn source under {{INPUT_ROOT}}. Available files: {{available[:50]}}"
    )


def prepare_source() -> None:
    if SOURCE_DIR.exists():
        shutil.rmtree(SOURCE_DIR)
    source_zip = find_input_file("genmusic_vn_source.zip", required=False)
    if source_zip is not None:
        SOURCE_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(source_zip) as archive:
            archive.extractall(SOURCE_DIR)
    else:
        shutil.copytree(find_extracted_source_root(), SOURCE_DIR)
    sys.path.insert(0, str(SOURCE_DIR))


def convert_wav_to_mp3(wav_path: Path, mp3_path: Path) -> Path:
    mp3_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(wav_path), "-codec:a", "libmp3lame", "-qscale:a", "0", str(mp3_path)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return mp3_path


def lyric_lines_for_duration(result) -> list[str]:
    lines: list[str] = []
    for line in result.lyrics.full_song:
        stripped = line.strip()
        if not stripped or stripped.startswith("["):
            continue
        lines.append(stripped)
    if not lines:
        lines = result.lyrics.verse + result.lyrics.chorus + result.lyrics.bridge
    return [line for line in lines if line.strip()]


def build_duration_plan(request: dict, result) -> dict:
    target = max(6, min(180, int(request.get("target_duration_seconds", request.get("duration_seconds", 30)))))
    model_name = request.get("model") or "facebook/musicgen-medium"
    lines = lyric_lines_for_duration(result)
    word_count = sum(len(line.split()) for line in lines)
    section_count = sum(1 for line in result.lyrics.full_song if line.strip().startswith("["))
    is_existing_lyrics = getattr(getattr(result, "text_plan", None), "input_kind", "") == "lyrics"
    seconds_per_word = 0.56 if is_existing_lyrics else 0.42
    line_pause = 0.48 if is_existing_lyrics else 0.35
    section_pause = 0.45 if is_existing_lyrics else 0.35
    estimated_vocal = word_count * seconds_per_word + max(0, len(lines) - 1) * line_pause + section_count * section_pause
    outro_tail = 2.5 if target <= 30 else 4.0
    breathing_room = min(10.0, max(2.0, len(lines) * (0.35 if is_existing_lyrics else 0.25)))
    soft_overrun = max(2, min(8, int(round(target * 0.18))))
    duration_ceiling = min(180, target + soft_overrun)
    natural_needed = int(round(max(target, estimated_vocal + breathing_room + outro_tail)))
    planned = min(natural_needed, duration_ceiling)
    planned = max(6, min(180, planned))
    musicgen_render = musicgen_render_duration_seconds(model_name, planned)
    return {{
        "policy": "soft_target",
        "target_duration_seconds": target,
        "planned_backing_duration_seconds": planned,
        "musicgen_render_duration_seconds": musicgen_render,
        "musicgen_duration_was_chunked": musicgen_render < planned,
        "duration_ceiling_seconds": duration_ceiling,
        "duration_was_capped": natural_needed > duration_ceiling,
        "estimated_vocal_duration_seconds": round(estimated_vocal, 2),
        "outro_tail_seconds": outro_tail,
        "lyric_line_count": len(lines),
        "lyric_word_count": word_count,
        "tts_seconds_per_word": seconds_per_word,
        "input_kind": getattr(getattr(result, "text_plan", None), "input_kind", ""),
    }}


def musicgen_render_duration_seconds(model_name: str, planned_duration: int) -> int:
    model = (model_name or "").lower()
    if "large" in model:
        cap = 16
    elif "medium" in model:
        cap = 24
    else:
        cap = 30
    return max(6, min(int(planned_duration), cap))


def render_musicgen_backing_mp3(request: dict, prompt: str, negative_prompt: str, duration_plan: dict) -> Path:
    ensure_transformers_version("4.33.0")
    ensure("accelerate", "accelerate")
    ensure("scipy", "scipy")
    ensure("numpy", "numpy")

    import torch
    from scipy.io import wavfile
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    model_name = request.get("model") or "facebook/musicgen-medium"
    planned_duration = max(6, min(180, int(duration_plan.get("planned_backing_duration_seconds", request.get("duration_seconds", 30)))))
    render_duration = max(6, min(planned_duration, int(duration_plan.get("musicgen_render_duration_seconds", planned_duration))))
    device = "cpu"
    if torch.cuda.is_available():
        major, _minor = torch.cuda.get_device_capability()
        if major >= 7:
            device = "cuda"
    dtype = torch.float16 if device == "cuda" else torch.float32

    processor = AutoProcessor.from_pretrained(model_name)
    model = MusicgenForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype).to(device)
    model.eval()

    backing_prompt = (
        prompt
        + "; instrumental backing track only; leave space for a separate Vietnamese vocal track; "
        "audible instrumental bed starts immediately under the vocal; natural short ending; "
        "no lead singing, no spoken words, no garbled vocals"
    )

    inputs = processor(text=[backing_prompt], padding=True, return_tensors="pt")
    inputs = {{key: value.to(device) for key, value in inputs.items()}}
    frame_rate = getattr(model.config.audio_encoder, "frame_rate", 50)
    sampling_rate = getattr(model.config.audio_encoder, "sampling_rate", 32000)
    max_new_tokens = max(32, int(render_duration * frame_rate))

    with torch.inference_mode():
        audio_values = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            guidance_scale=2.2,
            temperature=0.85,
            top_k=250,
        )

    audio = audio_values[0].detach().cpu().float()
    if audio.ndim == 2:
        audio = audio[0]
    audio_np = postprocess_audio(audio.numpy(), sampling_rate)
    audio_np = enforce_audio_duration(audio_np, sampling_rate, planned_duration)

    wav_path = OUTPUT_DIR / f"{{request.get('run_id', 'genmusic_vn')}}_musicgen.wav"
    wavfile.write(str(wav_path), rate=sampling_rate, data=audio_np)
    mp3_path = OUTPUT_DIR / f"{{request.get('run_id', 'genmusic_vn')}}_backing.mp3"
    return convert_wav_to_mp3(wav_path, mp3_path)


def postprocess_audio(audio_np, sampling_rate: int):
    import numpy as np

    audio_np = audio_np.astype("float32")
    if audio_np.size == 0:
        return audio_np
    audio_np = audio_np - float(np.mean(audio_np))

    fade = min(int(0.08 * sampling_rate), max(1, audio_np.size // 8))
    if fade > 1:
        ramp = np.linspace(0.0, 1.0, fade, dtype="float32")
        audio_np[:fade] *= ramp
        audio_np[-fade:] *= ramp[::-1]

    drive = 1.15
    audio_np = np.tanh(audio_np * drive) / np.tanh(drive)
    peak = max(1e-6, float(np.max(np.abs(audio_np))))
    return (audio_np / peak * 0.78).astype("float32")


def enforce_audio_duration(audio_np, sampling_rate: int, duration_seconds: int):
    import numpy as np

    target_samples = max(1, int(duration_seconds * sampling_rate))
    if audio_np.size > target_samples:
        audio_np = audio_np[:target_samples].copy()
        fade = min(int(0.45 * sampling_rate), max(1, audio_np.size // 6))
        if fade > 1:
            ramp = np.linspace(1.0, 0.0, fade, dtype="float32")
            audio_np[-fade:] *= ramp
    elif audio_np.size < target_samples:
        if audio_np.size <= 1:
            audio_np = np.zeros(target_samples, dtype="float32")
        else:
            crossfade = min(int(0.75 * sampling_rate), max(1, audio_np.size // 8))
            chunks = [audio_np.astype("float32")]
            while sum(chunk.size for chunk in chunks) < target_samples + crossfade:
                next_chunk = audio_np.astype("float32").copy()
                if crossfade > 1 and chunks[-1].size > crossfade and next_chunk.size > crossfade:
                    fade_out = np.linspace(1.0, 0.0, crossfade, dtype="float32")
                    fade_in = np.linspace(0.0, 1.0, crossfade, dtype="float32")
                    overlapped = chunks[-1][-crossfade:] * fade_out + next_chunk[:crossfade] * fade_in
                    chunks[-1] = chunks[-1][:-crossfade]
                    chunks.append(overlapped)
                    chunks.append(next_chunk[crossfade:])
                else:
                    chunks.append(next_chunk)
            audio_np = np.concatenate(chunks)[:target_samples]
            fade = min(int(0.65 * sampling_rate), max(1, audio_np.size // 10))
            if fade > 1:
                ramp = np.linspace(1.0, 0.0, fade, dtype="float32")
                audio_np[-fade:] *= ramp
    return audio_np.astype("float32")


def lyric_lines_for_tts(result) -> list[str]:
    return lyric_lines_for_duration(result)


def select_tts_lines_for_duration(result, duration_plan: dict) -> list[str]:
    all_lines = lyric_lines_for_tts(result)
    if not all_lines:
        return []
    planned = max(6, int(duration_plan.get("planned_backing_duration_seconds", 30)))
    is_existing_lyrics = getattr(getattr(result, "text_plan", None), "input_kind", "") == "lyrics"
    seconds_per_word = float(duration_plan.get("tts_seconds_per_word") or (0.56 if is_existing_lyrics else 0.42))
    word_budget = max(18, int(max(6, planned - 4) / seconds_per_word))
    selected: list[str] = []
    used_words = 0
    for line in all_lines:
        line_words = len(line.split())
        if selected and used_words + line_words > word_budget:
            break
        selected.append(line)
        used_words += line_words
        if len(selected) >= 18:
            break
    if not is_existing_lyrics and all_lines[-1] not in selected and len(selected) >= 2:
        selected[-1] = all_lines[-1]
    return selected or all_lines[:1]


def render_mms_tts_vocal(request: dict, result, duration_plan: dict) -> Path:
    ensure_transformers_version("4.33.0")
    ensure("accelerate", "accelerate")
    ensure("scipy", "scipy")
    ensure("numpy", "numpy")

    import numpy as np
    import torch
    from scipy.io import wavfile
    from transformers import AutoModelForTextToWaveform, AutoTokenizer

    model_name = request.get("mms_tts_model") or DEFAULT_MMS_TTS_MODEL
    # Keep MMS TTS on CPU. On Kaggle, CUDA can be left in a bad state after MusicGen
    # generation, which makes the TTS model fail before it can render any vocal.
    device = "cpu"
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTextToWaveform.from_pretrained(model_name).to(device)
    model.eval()
    sampling_rate = int(getattr(model.config, "sampling_rate", 16000))

    pieces = []
    silence = np.zeros(int(sampling_rate * 0.42), dtype="float32")
    for line in select_tts_lines_for_duration(result, duration_plan):
        clean_line = clean_tts_line(line)
        if not clean_line:
            continue
        inputs = tokenizer(clean_line, return_tensors="pt")
        inputs = {{key: value.to(device) for key, value in inputs.items()}}
        with torch.inference_mode():
            waveform = model(**inputs).waveform
        audio = waveform[0].detach().cpu().float().numpy()
        audio = postprocess_vocal_audio(audio)
        pieces.append(audio)
        pieces.append(silence)

    if not pieces:
        pieces = [np.zeros(int(sampling_rate * 1.0), dtype="float32")]
    else:
        pieces.append(np.zeros(int(sampling_rate * 1.4), dtype="float32"))

    vocal_np = np.concatenate(pieces).astype("float32")
    raw_path = OUTPUT_DIR / f"{{request.get('run_id', 'genmusic_vn')}}_vocal_mms_raw.wav"
    wavfile.write(str(raw_path), rate=sampling_rate, data=vocal_np)

    profiled_path = OUTPUT_DIR / f"{{request.get('run_id', 'genmusic_vn')}}_vocal_mms.wav"
    apply_vocal_profile(raw_path, profiled_path, result.vocal, sampling_rate)
    return profiled_path


def clean_tts_line(line: str) -> str:
    return " ".join(line.replace("|", " ").replace("/", " ").split()).strip(" ,.;:-")


def f5_gen_text_from_lines(lines: list[str]) -> str:
    cleaned = [clean_tts_line(line).lower() for line in lines if clean_tts_line(line)]
    return ". ".join(cleaned).strip()


def ensure_f5_tts_assets(request: dict) -> tuple[Path, Path, Path]:
    ensure("huggingface_hub", "huggingface_hub")
    import urllib.request
    from huggingface_hub import snapshot_download

    install_dir = ASSET_DIR / "f5_tts_vietnamese_repo"
    if not install_dir.exists():
        subprocess.check_call(["git", "clone", "--depth", "1", F5_REPO_URL, str(install_dir)])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", str(install_dir)])

    model_id = request.get("tts_model") or DEFAULT_F5_TTS_MODEL
    model_dir = ASSET_DIR / "f5_tts_vietnamese_model"
    snapshot_download(repo_id=model_id, local_dir=str(model_dir), local_dir_use_symlinks=False)

    vocab_file = model_dir / "vocab.txt"
    config_file = model_dir / "config.json"
    if not vocab_file.exists() and config_file.exists():
        shutil.copy2(config_file, vocab_file)
    ckpt_file = model_dir / "model_last.pt"
    if not ckpt_file.exists():
        matches = sorted(model_dir.glob("*.pt")) + sorted(model_dir.glob("*.safetensors"))
        if not matches:
            raise FileNotFoundError(f"F5 checkpoint not found in {{model_dir}}")
        ckpt_file = matches[0]

    ref_audio = ASSET_DIR / "f5_ref.wav"
    if not ref_audio.exists():
        urllib.request.urlretrieve(F5_REF_AUDIO_URL, ref_audio)
    return ref_audio, vocab_file, ckpt_file


def render_f5_tts_vocal(request: dict, result, duration_plan: dict) -> Path:
    ref_audio, vocab_file, ckpt_file = ensure_f5_tts_assets(request)
    gen_text = f5_gen_text_from_lines(select_tts_lines_for_duration(result, duration_plan))
    if not gen_text:
        raise ValueError("No lyric text available for F5-TTS.")

    output_file = f"{{request.get('run_id', 'genmusic_vn')}}_vocal_f5.wav"
    command = [
        "f5-tts_infer-cli",
        "--model",
        "F5TTS_Base",
        "--ref_audio",
        str(ref_audio),
        "--ref_text",
        F5_REF_TEXT,
        "--gen_text",
        gen_text,
        "--speed",
        "1.0",
        "--vocoder_name",
        "vocos",
        "--vocab_file",
        str(vocab_file),
        "--ckpt_file",
        str(ckpt_file),
        "--output_dir",
        str(OUTPUT_DIR),
        "--output_file",
        output_file,
    ]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    vocal_path = OUTPUT_DIR / output_file
    if not vocal_path.exists():
        candidates = sorted(OUTPUT_DIR.rglob(output_file))
        if candidates:
            vocal_path = candidates[0]
    if not vocal_path.exists():
        raise FileNotFoundError(f"F5-TTS output was not found: {{vocal_path}}")
    return vocal_path


def postprocess_vocal_audio(audio_np):
    import numpy as np

    audio_np = audio_np.astype("float32")
    if audio_np.size == 0:
        return audio_np
    audio_np = audio_np - float(np.mean(audio_np))
    peak = max(1e-6, float(np.max(np.abs(audio_np))))
    audio_np = audio_np / peak * 0.72
    fade = min(240, max(1, audio_np.size // 10))
    if fade > 1:
        ramp = np.linspace(0.0, 1.0, fade, dtype="float32")
        audio_np[:fade] *= ramp
        audio_np[-fade:] *= ramp[::-1]
    return audio_np.astype("float32")


def apply_vocal_profile(raw_path: Path, output_path: Path, vocal, sampling_rate: int) -> Path:
    gender = getattr(vocal, "gender", "")
    if gender == "male":
        pitch_filter = f"asetrate={{sampling_rate}}*0.90,aresample={{sampling_rate}},atempo=1.111"
    elif gender == "female":
        pitch_filter = f"asetrate={{sampling_rate}}*1.10,aresample={{sampling_rate}},atempo=0.909"
    else:
        pitch_filter = f"aresample={{sampling_rate}}"
    filter_chain = (
        f"{{pitch_filter}},highpass=f=90,lowpass=f=8500,"
        "acompressor=threshold=-18dB:ratio=2.2:attack=10:release=160,"
        "aecho=0.8:0.88:45:0.06,alimiter=limit=0.86,volume=0.98"
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(raw_path), "-af", filter_chain, str(output_path)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return output_path


def build_mix_filter(duration_plan: dict, scene_plan: dict) -> str:
    layers = set(scene_plan.get("ambience_layers") or [])
    base = (
        "[0:a]volume=0.74,afade=t=in:st=0:d=0.08,"
        "aformat=channel_layouts=stereo,aecho=0.8:0.88:18:0.06[bg];"
        "[1:a]adelay=280|280,volume=0.86,highpass=f=90,lowpass=f=8500,"
        "acompressor=threshold=-18dB:ratio=2.5:attack=12:release=180,"
        "aformat=channel_layouts=stereo[vox];"
    )
    ambience_layers = layers.intersection({{"rain", "street", "night", "water", "air", "room"}})
    if ambience_layers:
        duration = max(1.0, float(duration_plan.get("planned_backing_duration_seconds", 30)))
        fade_out = max(0.0, duration - 1.2)
        amplitude = ambience_amplitude(ambience_layers)
        lowpass = 5200 if "rain" in ambience_layers else 3600
        highpass = 750 if "rain" in ambience_layers else 120
        return (
            base
            + f"anoisesrc=color=pink:amplitude={{amplitude:.3f}}:duration={{duration:.2f}}[amb0];"
            + f"[amb0]highpass=f={{highpass}},lowpass=f={{lowpass}},volume=0.42,"
            + f"afade=t=in:st=0:d=1.0,afade=t=out:st={{fade_out:.2f}}:d=1.0,"
            + "aformat=channel_layouts=stereo[amb];"
            + "[bg][vox][amb]amix=inputs=3:duration=first:dropout_transition=1:normalize=0,"
            + "alimiter=limit=0.90[out]"
        )
    return (
        base
        + "[bg][vox]amix=inputs=2:duration=first:dropout_transition=1:normalize=0,"
        + "alimiter=limit=0.90[out]"
    )


def ambience_amplitude(layers: set[str]) -> float:
    if "rain" in layers:
        return 0.026
    if "water" in layers:
        return 0.020
    if "street" in layers or "night" in layers:
        return 0.016
    return 0.012


def mix_vocal_with_backing(request: dict, backing_mp3_path: Path, vocal_wav_path: Path, duration_plan: dict, scene_plan: dict) -> Path:
    final_mp3_path = OUTPUT_DIR / f"{{request.get('run_id', 'genmusic_vn')}}.mp3"
    filter_complex = build_mix_filter(duration_plan, scene_plan)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(backing_mp3_path),
            "-i",
            str(vocal_wav_path),
            "-filter_complex",
            filter_complex,
            "-map",
            "[out]",
            "-codec:a",
            "libmp3lame",
            "-qscale:a",
            "0",
            str(final_mp3_path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return final_mp3_path


def render_guide_fallback_mp3(request: dict, result) -> Path:
    from genmusic_vn.generators.base import GeneratorInput
    from genmusic_vn.generators.guide_track import GuideTrackGenerator

    generator_input = GeneratorInput(
        run_id=result.run_id,
        text=result.input_text,
        prompt=result.prompt,
        negative_prompt=result.negative_prompt,
        emotion=result.emotion,
        harmony=result.harmony,
        lyrics=result.lyrics,
        vocal=result.vocal,
        melody=result.melody,
        duration_seconds=int(request.get("planned_duration_seconds", request.get("duration_seconds", 30))),
    )
    fallback_dir = OUTPUT_DIR / "fallback"
    generated = GuideTrackGenerator().generate(generator_input, fallback_dir)
    wav_path = next(Path(item.path) for item in generated if item.kind == "audio")
    mp3_path = OUTPUT_DIR / f"{{request.get('run_id', 'genmusic_vn')}}_fallback.mp3"
    return convert_wav_to_mp3(wav_path, mp3_path)


def main() -> None:
    request_path = find_input_file("request.json")
    if request_path is None:
        raise FileNotFoundError("request.json was not found in Kaggle input.")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    prepare_source()

    from genmusic_vn.pipeline import create_music_project
    from genmusic_vn.schemas import to_plain_data

    result = create_music_project(
        text=request["text"],
        output_root=PIPELINE_DIR,
        backend="guide",
        duration_seconds=int(request.get("duration_seconds", 30)),
        genre=request.get("genre"),
        render_audio=False,
    )
    musicgen_error = ""
    tts_error = ""
    backing_mp3_path = None
    vocal_path = None
    report = to_plain_data(result)
    scene_plan = report.get("scene", {{}})
    duration_plan = build_duration_plan(request, result)
    request["planned_duration_seconds"] = duration_plan["planned_backing_duration_seconds"]
    (OUTPUT_DIR / "duration_plan.json").write_text(
        json.dumps(duration_plan, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    try:
        backing_mp3_path = render_musicgen_backing_mp3(request, result.prompt, result.negative_prompt, duration_plan)
        generation_backend = "musicgen"
    except Exception as exc:
        musicgen_error = "".join(traceback.format_exception(exc))[-4000:]
        (OUTPUT_DIR / "musicgen_error.txt").write_text(musicgen_error, encoding="utf-8")
        backing_mp3_path = render_guide_fallback_mp3(request, result)
        generation_backend = "guide_fallback"

    if musicgen_error:
        tts_error = "Skipped vocal TTS because MusicGen failed in the same Kaggle kernel; CUDA may be unstable after the MusicGen error."
        (OUTPUT_DIR / "tts_error.txt").write_text(tts_error, encoding="utf-8")
        mp3_path = backing_mp3_path
        generation_backend = generation_backend + "+tts_skipped_backing_only"
    else:
        f5_tts_error = ""
        try:
            vocal_path = render_f5_tts_vocal(request, result, duration_plan)
            mp3_path = mix_vocal_with_backing(request, backing_mp3_path, vocal_path, duration_plan, scene_plan)
            generation_backend = generation_backend + "+f5_tts_vocal_mix"
        except Exception as exc:
            f5_tts_error = "".join(traceback.format_exception(exc))[-4000:]
            (OUTPUT_DIR / "f5_tts_error.txt").write_text(f5_tts_error, encoding="utf-8")
            try:
                vocal_path = render_mms_tts_vocal(request, result, duration_plan)
                mp3_path = mix_vocal_with_backing(request, backing_mp3_path, vocal_path, duration_plan, scene_plan)
                generation_backend = generation_backend + "+f5_failed_mms_tts_vocal_mix"
            except Exception as mms_exc:
                tts_error = "".join(traceback.format_exception(mms_exc))[-4000:]
                (OUTPUT_DIR / "tts_error.txt").write_text(tts_error, encoding="utf-8")
                mp3_path = backing_mp3_path
                generation_backend = generation_backend + "+tts_failed_backing_only"

    lyrics_text = "\\n".join(result.lyrics.full_song)
    (OUTPUT_DIR / "lyrics.txt").write_text(lyrics_text, encoding="utf-8")
    (OUTPUT_DIR / "lyrics.json").write_text(
        json.dumps(report.get("lyrics", {{}}), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    final = {{
        "run_id": request.get("run_id"),
        "backend": generation_backend,
        "model": request.get("model"),
        "duration_policy": "soft_target",
        "target_duration_seconds": duration_plan["target_duration_seconds"],
        "planned_backing_duration_seconds": duration_plan["planned_backing_duration_seconds"],
        "duration_ceiling_seconds": duration_plan["duration_ceiling_seconds"],
        "duration_was_capped": duration_plan["duration_was_capped"],
        "estimated_vocal_duration_seconds": duration_plan["estimated_vocal_duration_seconds"],
        "outro_tail_seconds": duration_plan["outro_tail_seconds"],
        "duration_plan": duration_plan,
        "tts_model": request.get("tts_model") or DEFAULT_TTS_MODEL,
        "mms_tts_model": request.get("mms_tts_model") or DEFAULT_MMS_TTS_MODEL,
        "tts_voice_actual": DEFAULT_TTS_VOICE_ACTUAL,
        "tts_voice_note": DEFAULT_TTS_VOICE_NOTE,
        "mp3_path": str(mp3_path),
        "backing_path": str(backing_mp3_path),
        "vocal_path": str(vocal_path) if vocal_path else "",
        "musicgen_failed": bool(musicgen_error),
        "vocal_failed": bool(tts_error),
        "musicgen_error": musicgen_error,
        "f5_tts_error": f5_tts_error if 'f5_tts_error' in locals() else "",
        "tts_error": tts_error,
        "prompt": result.prompt,
        "negative_prompt": result.negative_prompt,
        "lyrics_text": lyrics_text,
        "lyrics": report.get("lyrics"),
        "vocal_plan": report.get("vocal"),
        "scene_plan": scene_plan,
        "analysis": report,
    }}
    (OUTPUT_DIR / "kaggle_result.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.rmtree(ASSET_DIR, ignore_errors=True)
    print(json.dumps({{"mp3_path": str(mp3_path), "prompt": result.prompt}}, ensure_ascii=False, indent=2))


main()
'''


def _tts_retry_kernel_script(dataset_slug: str) -> str:
    return f'''from __future__ import annotations

import json
import shutil
import subprocess
import sys
import traceback
import zipfile
from pathlib import Path


INPUT_ROOT = Path("/kaggle/input")
DATASET_DIR = INPUT_ROOT / "{dataset_slug}"
SOURCE_DIR = Path("/kaggle/working/genmusic_vn_source")
PIPELINE_DIR = Path("/kaggle/working/pipeline_output")
OUTPUT_DIR = Path("/kaggle/working/genmusic_vn")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ASSET_DIR = Path("/kaggle/working/genmusic_vn_assets")
ASSET_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_TTS_MODEL = "hynt/F5-TTS-Vietnamese-ViVoice"
DEFAULT_F5_TTS_MODEL = "hynt/F5-TTS-Vietnamese-ViVoice"
DEFAULT_MMS_TTS_MODEL = "facebook/mms-tts-vie"
DEFAULT_TTS_VOICE_ACTUAL = "f5_vietnamese_vivoice_reference"
DEFAULT_TTS_VOICE_NOTE = "F5-TTS Vietnamese uses a short Vietnamese reference voice; MMS Vietnamese is kept as fallback."
F5_REPO_URL = "https://github.com/nguyenthienhy/F5-TTS-Vietnamese.git"
F5_REF_AUDIO_URL = "https://raw.githubusercontent.com/nguyenthienhy/F5-TTS-Vietnamese/main/ref.wav"
F5_REF_TEXT = "cả hai bên hãy cố gắng hiểu cho nhau"


def ensure(import_name: str, *pip_specs: str) -> None:
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", *pip_specs])


def ensure_transformers_version(min_version: str = "4.33.0") -> None:
    ensure("packaging", "packaging")
    ensure("transformers", f"transformers>={{min_version}}")
    import transformers
    from packaging.version import parse
    if parse(transformers.__version__) < parse(min_version):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", f"transformers>={{min_version}}"])


def find_input_file(name: str, *, required: bool = True) -> Path | None:
    direct = DATASET_DIR / name
    if direct.exists():
        return direct
    matches = sorted(INPUT_ROOT.rglob(name))
    if matches:
        return matches[0]
    if not required:
        return None
    available = [str(path) for path in sorted(INPUT_ROOT.rglob("*")) if path.is_file()]
    raise FileNotFoundError(f"Could not find {{name}} under {{INPUT_ROOT}}. Available files: {{available[:50]}}")


def find_extracted_source_root() -> Path:
    candidates = sorted(path.parent for path in INPUT_ROOT.rglob("pyproject.toml") if (path.parent / "genmusic_vn").is_dir())
    if candidates:
        return candidates[0]
    available = [str(path) for path in sorted(INPUT_ROOT.rglob("*")) if path.is_file()]
    raise FileNotFoundError(f"Could not find extracted genmusic_vn source under {{INPUT_ROOT}}. Available files: {{available[:50]}}")


def prepare_source() -> None:
    if SOURCE_DIR.exists():
        shutil.rmtree(SOURCE_DIR)
    source_zip = find_input_file("genmusic_vn_source.zip", required=False)
    if source_zip is not None:
        SOURCE_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(source_zip) as archive:
            archive.extractall(SOURCE_DIR)
    else:
        shutil.copytree(find_extracted_source_root(), SOURCE_DIR)
    sys.path.insert(0, str(SOURCE_DIR))


def lyric_lines_for_duration(result) -> list[str]:
    lines: list[str] = []
    for line in result.lyrics.full_song:
        stripped = line.strip()
        if not stripped or stripped.startswith("["):
            continue
        lines.append(stripped)
    if not lines:
        lines = result.lyrics.verse + result.lyrics.chorus + result.lyrics.bridge
    return [line for line in lines if line.strip()]


def build_duration_plan(request: dict, result) -> dict:
    target = max(6, min(180, int(request.get("target_duration_seconds", request.get("duration_seconds", 30)))))
    lines = lyric_lines_for_duration(result)
    word_count = sum(len(line.split()) for line in lines)
    section_count = sum(1 for line in result.lyrics.full_song if line.strip().startswith("["))
    is_existing_lyrics = getattr(getattr(result, "text_plan", None), "input_kind", "") == "lyrics"
    seconds_per_word = 0.56 if is_existing_lyrics else 0.42
    line_pause = 0.48 if is_existing_lyrics else 0.35
    section_pause = 0.45 if is_existing_lyrics else 0.35
    estimated_vocal = word_count * seconds_per_word + max(0, len(lines) - 1) * line_pause + section_count * section_pause
    outro_tail = 2.5 if target <= 30 else 4.0
    breathing_room = min(10.0, max(2.0, len(lines) * (0.35 if is_existing_lyrics else 0.25)))
    soft_overrun = max(2, min(8, int(round(target * 0.18))))
    duration_ceiling = min(180, target + soft_overrun)
    natural_needed = int(round(max(target, estimated_vocal + breathing_room + outro_tail)))
    planned = max(6, min(180, min(natural_needed, duration_ceiling)))
    return {{
        "policy": "soft_target",
        "target_duration_seconds": target,
        "planned_backing_duration_seconds": planned,
        "duration_ceiling_seconds": duration_ceiling,
        "duration_was_capped": natural_needed > duration_ceiling,
        "estimated_vocal_duration_seconds": round(estimated_vocal, 2),
        "outro_tail_seconds": outro_tail,
        "lyric_line_count": len(lines),
        "lyric_word_count": word_count,
        "tts_seconds_per_word": seconds_per_word,
        "input_kind": getattr(getattr(result, "text_plan", None), "input_kind", ""),
    }}


def select_tts_lines_for_duration(result, duration_plan: dict) -> list[str]:
    all_lines = lyric_lines_for_duration(result)
    if not all_lines:
        return []
    planned = max(6, int(duration_plan.get("planned_backing_duration_seconds", 30)))
    is_existing_lyrics = getattr(getattr(result, "text_plan", None), "input_kind", "") == "lyrics"
    seconds_per_word = float(duration_plan.get("tts_seconds_per_word") or (0.56 if is_existing_lyrics else 0.42))
    word_budget = max(18, int(max(6, planned - 4) / seconds_per_word))
    selected: list[str] = []
    used_words = 0
    for line in all_lines:
        line_words = len(line.split())
        if selected and used_words + line_words > word_budget:
            break
        selected.append(line)
        used_words += line_words
        if len(selected) >= 18:
            break
    if not is_existing_lyrics and all_lines[-1] not in selected and len(selected) >= 2:
        selected[-1] = all_lines[-1]
    return selected or all_lines[:1]


def clean_tts_line(line: str) -> str:
    return " ".join(line.replace("|", " ").replace("/", " ").split()).strip(" ,.;:-")


def f5_gen_text_from_lines(lines: list[str]) -> str:
    cleaned = [clean_tts_line(line).lower() for line in lines if clean_tts_line(line)]
    return ". ".join(cleaned).strip()


def ensure_f5_tts_assets(request: dict) -> tuple[Path, Path, Path]:
    ensure("huggingface_hub", "huggingface_hub")
    import urllib.request
    from huggingface_hub import snapshot_download

    install_dir = ASSET_DIR / "f5_tts_vietnamese_repo"
    if not install_dir.exists():
        subprocess.check_call(["git", "clone", "--depth", "1", F5_REPO_URL, str(install_dir)])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", str(install_dir)])

    model_id = request.get("tts_model") or DEFAULT_F5_TTS_MODEL
    model_dir = ASSET_DIR / "f5_tts_vietnamese_model"
    snapshot_download(repo_id=model_id, local_dir=str(model_dir), local_dir_use_symlinks=False)

    vocab_file = model_dir / "vocab.txt"
    config_file = model_dir / "config.json"
    if not vocab_file.exists() and config_file.exists():
        shutil.copy2(config_file, vocab_file)
    ckpt_file = model_dir / "model_last.pt"
    if not ckpt_file.exists():
        matches = sorted(model_dir.glob("*.pt")) + sorted(model_dir.glob("*.safetensors"))
        if not matches:
            raise FileNotFoundError(f"F5 checkpoint not found in {{model_dir}}")
        ckpt_file = matches[0]

    ref_audio = ASSET_DIR / "f5_ref.wav"
    if not ref_audio.exists():
        urllib.request.urlretrieve(F5_REF_AUDIO_URL, ref_audio)
    return ref_audio, vocab_file, ckpt_file


def render_f5_tts_vocal(request: dict, result, duration_plan: dict) -> Path:
    ref_audio, vocab_file, ckpt_file = ensure_f5_tts_assets(request)
    gen_text = f5_gen_text_from_lines(select_tts_lines_for_duration(result, duration_plan))
    if not gen_text:
        raise ValueError("No lyric text available for F5-TTS.")

    output_file = f"{{request.get('run_id', 'genmusic_vn')}}_vocal_f5.wav"
    command = [
        "f5-tts_infer-cli",
        "--model",
        "F5TTS_Base",
        "--ref_audio",
        str(ref_audio),
        "--ref_text",
        F5_REF_TEXT,
        "--gen_text",
        gen_text,
        "--speed",
        "1.0",
        "--vocoder_name",
        "vocos",
        "--vocab_file",
        str(vocab_file),
        "--ckpt_file",
        str(ckpt_file),
        "--output_dir",
        str(OUTPUT_DIR),
        "--output_file",
        output_file,
    ]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    vocal_path = OUTPUT_DIR / output_file
    if not vocal_path.exists():
        candidates = sorted(OUTPUT_DIR.rglob(output_file))
        if candidates:
            vocal_path = candidates[0]
    if not vocal_path.exists():
        raise FileNotFoundError(f"F5-TTS output was not found: {{vocal_path}}")
    return vocal_path


def postprocess_vocal_audio(audio_np):
    import numpy as np
    audio_np = audio_np.astype("float32")
    if audio_np.size == 0:
        return audio_np
    audio_np = audio_np - float(np.mean(audio_np))
    peak = max(1e-6, float(np.max(np.abs(audio_np))))
    audio_np = audio_np / peak * 0.72
    fade = min(240, max(1, audio_np.size // 10))
    if fade > 1:
        ramp = np.linspace(0.0, 1.0, fade, dtype="float32")
        audio_np[:fade] *= ramp
        audio_np[-fade:] *= ramp[::-1]
    return audio_np.astype("float32")


def apply_vocal_profile(raw_path: Path, output_path: Path, vocal, sampling_rate: int) -> Path:
    gender = getattr(vocal, "gender", "")
    if gender == "male":
        pitch_filter = f"asetrate={{sampling_rate}}*0.90,aresample={{sampling_rate}},atempo=1.111"
    elif gender == "female":
        pitch_filter = f"asetrate={{sampling_rate}}*1.10,aresample={{sampling_rate}},atempo=0.909"
    else:
        pitch_filter = f"aresample={{sampling_rate}}"
    filter_chain = (
        f"{{pitch_filter}},highpass=f=90,lowpass=f=8500,"
        "acompressor=threshold=-18dB:ratio=2.2:attack=10:release=160,"
        "aecho=0.8:0.88:45:0.06,alimiter=limit=0.86,volume=0.98"
    )
    subprocess.run(["ffmpeg", "-y", "-i", str(raw_path), "-af", filter_chain, str(output_path)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return output_path


def render_mms_tts_vocal(request: dict, result, duration_plan: dict) -> Path:
    ensure_transformers_version("4.33.0")
    ensure("accelerate", "accelerate")
    ensure("scipy", "scipy")
    ensure("numpy", "numpy")
    import numpy as np
    import torch
    from scipy.io import wavfile
    from transformers import AutoModelForTextToWaveform, AutoTokenizer

    model_name = request.get("mms_tts_model") or DEFAULT_MMS_TTS_MODEL
    # TTS-only retry fallback: keep MMS TTS on CPU so it is isolated from GPU-side failures.
    device = "cpu"
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTextToWaveform.from_pretrained(model_name).to(device)
    model.eval()
    sampling_rate = int(getattr(model.config, "sampling_rate", 16000))

    pieces = []
    silence = np.zeros(int(sampling_rate * 0.42), dtype="float32")
    for line in select_tts_lines_for_duration(result, duration_plan):
        clean_line = clean_tts_line(line)
        if not clean_line:
            continue
        inputs = tokenizer(clean_line, return_tensors="pt")
        inputs = {{key: value.to(device) for key, value in inputs.items()}}
        with torch.inference_mode():
            waveform = model(**inputs).waveform
        audio = waveform[0].detach().cpu().float().numpy()
        pieces.append(postprocess_vocal_audio(audio))
        pieces.append(silence)

    if not pieces:
        pieces = [np.zeros(int(sampling_rate * 1.0), dtype="float32")]
    else:
        pieces.append(np.zeros(int(sampling_rate * 1.4), dtype="float32"))

    vocal_np = np.concatenate(pieces).astype("float32")
    raw_path = OUTPUT_DIR / f"{{request.get('run_id', 'genmusic_vn')}}_vocal_mms_raw.wav"
    wavfile.write(str(raw_path), rate=sampling_rate, data=vocal_np)
    profiled_path = OUTPUT_DIR / f"{{request.get('run_id', 'genmusic_vn')}}_vocal_mms.wav"
    apply_vocal_profile(raw_path, profiled_path, result.vocal, sampling_rate)
    return profiled_path


def build_mix_filter(duration_plan: dict, scene_plan: dict) -> str:
    layers = set(scene_plan.get("ambience_layers") or [])
    base = (
        "[0:a]volume=0.74,afade=t=in:st=0:d=0.08,"
        "aformat=channel_layouts=stereo,aecho=0.8:0.88:18:0.06[bg];"
        "[1:a]adelay=280|280,volume=0.86,highpass=f=90,lowpass=f=8500,"
        "acompressor=threshold=-18dB:ratio=2.5:attack=12:release=180,"
        "aformat=channel_layouts=stereo[vox];"
    )
    ambience_layers = layers.intersection({{"rain", "street", "night", "water", "air", "room"}})
    if ambience_layers:
        duration = max(1.0, float(duration_plan.get("planned_backing_duration_seconds", 30)))
        fade_out = max(0.0, duration - 1.2)
        amplitude = 0.026 if "rain" in ambience_layers else (0.020 if "water" in ambience_layers else 0.016)
        lowpass = 5200 if "rain" in ambience_layers else 3600
        highpass = 750 if "rain" in ambience_layers else 120
        return (
            base
            + f"anoisesrc=color=pink:amplitude={{amplitude:.3f}}:duration={{duration:.2f}}[amb0];"
            + f"[amb0]highpass=f={{highpass}},lowpass=f={{lowpass}},volume=0.42,"
            + f"afade=t=in:st=0:d=1.0,afade=t=out:st={{fade_out:.2f}}:d=1.0,"
            + "aformat=channel_layouts=stereo[amb];"
            + "[bg][vox][amb]amix=inputs=3:duration=first:dropout_transition=1:normalize=0,"
            + "alimiter=limit=0.90[out]"
        )
    return base + "[bg][vox]amix=inputs=2:duration=first:dropout_transition=1:normalize=0,alimiter=limit=0.90[out]"


def mix_vocal_with_backing(request: dict, backing_mp3_path: Path, vocal_wav_path: Path, duration_plan: dict, scene_plan: dict) -> Path:
    final_mp3_path = OUTPUT_DIR / f"{{request.get('run_id', 'genmusic_vn')}}.mp3"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(backing_mp3_path),
            "-i", str(vocal_wav_path),
            "-filter_complex", build_mix_filter(duration_plan, scene_plan),
            "-map", "[out]",
            "-codec:a", "libmp3lame",
            "-qscale:a", "0",
            str(final_mp3_path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return final_mp3_path


def main() -> None:
    request_path = find_input_file("request.json")
    backing_mp3_path = find_input_file("backing_input.mp3")
    if request_path is None or backing_mp3_path is None:
        raise FileNotFoundError("request.json or backing_input.mp3 was not found in Kaggle input.")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    backing_copy_path = OUTPUT_DIR / f"{{request.get('run_id', 'genmusic_vn')}}_backing.mp3"
    shutil.copy2(backing_mp3_path, backing_copy_path)
    backing_mp3_path = backing_copy_path
    prepare_source()

    from genmusic_vn.pipeline import create_music_project
    from genmusic_vn.schemas import to_plain_data

    result = create_music_project(
        text=request["text"],
        output_root=PIPELINE_DIR,
        backend="guide",
        duration_seconds=int(request.get("duration_seconds", 30)),
        genre=request.get("genre"),
        render_audio=False,
    )
    report = to_plain_data(result)
    scene_plan = report.get("scene", {{}})
    duration_plan = build_duration_plan(request, result)
    tts_error = ""
    f5_tts_error = ""
    vocal_path = None
    try:
        vocal_path = render_f5_tts_vocal(request, result, duration_plan)
        mp3_path = mix_vocal_with_backing(request, backing_mp3_path, vocal_path, duration_plan, scene_plan)
        generation_backend = "tts_retry+f5_tts_vocal_mix"
    except Exception as exc:
        f5_tts_error = "".join(traceback.format_exception(exc))[-4000:]
        (OUTPUT_DIR / "f5_tts_error.txt").write_text(f5_tts_error, encoding="utf-8")
        try:
            vocal_path = render_mms_tts_vocal(request, result, duration_plan)
            mp3_path = mix_vocal_with_backing(request, backing_mp3_path, vocal_path, duration_plan, scene_plan)
            generation_backend = "tts_retry+f5_failed_mms_tts_vocal_mix"
        except Exception as mms_exc:
            tts_error = "".join(traceback.format_exception(mms_exc))[-4000:]
            (OUTPUT_DIR / "tts_error.txt").write_text(tts_error, encoding="utf-8")
            mp3_path = backing_mp3_path
            generation_backend = "tts_retry+tts_failed_backing_only"

    lyrics_text = "\\n".join(result.lyrics.full_song)
    (OUTPUT_DIR / "lyrics.txt").write_text(lyrics_text, encoding="utf-8")
    (OUTPUT_DIR / "lyrics.json").write_text(json.dumps(report.get("lyrics", {{}}), ensure_ascii=False, indent=2), encoding="utf-8")
    final = {{
        "run_id": request.get("run_id"),
        "parent_run_id": request.get("parent_run_id"),
        "backend": generation_backend,
        "model": request.get("model"),
        "duration_policy": "soft_target",
        "duration_plan": duration_plan,
        "target_duration_seconds": duration_plan["target_duration_seconds"],
        "planned_backing_duration_seconds": duration_plan["planned_backing_duration_seconds"],
        "duration_ceiling_seconds": duration_plan["duration_ceiling_seconds"],
        "duration_was_capped": duration_plan["duration_was_capped"],
        "estimated_vocal_duration_seconds": duration_plan["estimated_vocal_duration_seconds"],
        "outro_tail_seconds": duration_plan["outro_tail_seconds"],
        "tts_model": request.get("tts_model") or DEFAULT_TTS_MODEL,
        "mms_tts_model": request.get("mms_tts_model") or DEFAULT_MMS_TTS_MODEL,
        "tts_voice_actual": DEFAULT_TTS_VOICE_ACTUAL,
        "tts_voice_note": DEFAULT_TTS_VOICE_NOTE,
        "mp3_path": str(mp3_path),
        "backing_path": str(backing_mp3_path),
        "vocal_path": str(vocal_path) if vocal_path else "",
        "musicgen_failed": False,
        "vocal_failed": bool(tts_error),
        "musicgen_error": "",
        "f5_tts_error": f5_tts_error,
        "tts_error": tts_error,
        "prompt": result.prompt,
        "negative_prompt": result.negative_prompt,
        "lyrics_text": lyrics_text,
        "lyrics": report.get("lyrics"),
        "vocal_plan": report.get("vocal"),
        "scene_plan": scene_plan,
        "analysis": report,
    }}
    (OUTPUT_DIR / "kaggle_result.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.rmtree(ASSET_DIR, ignore_errors=True)
    print(json.dumps({{"mode": "TTS-only retry", "mp3_path": str(mp3_path)}}, ensure_ascii=False, indent=2))


main()
'''


def _write_source_zip(path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    include_roots = [root / "genmusic_vn", root / "datasets" / "vn_music_stylebank"]
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for include_root in include_roots:
            if not include_root.exists():
                continue
            for file_path in include_root.rglob("*"):
                if not file_path.is_file():
                    continue
                if "__pycache__" in file_path.parts or file_path.suffix in {".pyc", ".pyo"}:
                    continue
                archive.write(file_path, file_path.relative_to(root))
        archive.write(root / "pyproject.toml", "pyproject.toml")


def _commands(dataset_dir: Path, kernel_dir: Path, download_dir: Path, kernel_ref: str) -> list[str]:
    return [
        "pip install -U kaggle",
        "# Tao file .env hoac .env.local voi KAGGLE_USERNAME va KAGGLE_KEY truoc khi chay.",
        f'kaggle datasets create -p "{dataset_dir}" -r zip',
        f'kaggle kernels push -p "{kernel_dir}"',
        f'kaggle kernels status "{kernel_ref}"',
        f'kaggle kernels output "{kernel_ref}" -p "{download_dir}"',
    ]


def _wait_for_dataset_ready(
    state: dict[str, Any],
    cli: list[str],
    *,
    timeout_seconds: int = 300,
    poll_seconds: int = 5,
) -> bool:
    state["status"] = "dataset_processing"
    _append_message_once(state, "Waiting for Kaggle dataset to become ready.")
    _write_state(state)

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        result = _run(cli + ["datasets", "status", state["dataset_ref"], "--format", "json"], timeout=120)
        state["history"].append(_history_item("datasets status", result))
        if result["returncode"] != 0:
            error = _summarize_cli_error(result)
            if "403" in error or "GetDatasetStatus" in error:
                state["status"] = "dataset_ready"
                state["dataset_status_warning"] = error
                _append_message_once(
                    state,
                    "Kaggle dataset status check was blocked; continuing after successful upload.",
                )
                _write_state(state)
                return True
            state["status"] = "failed"
            state["last_error"] = error
            _append_message_once(state, "Could not read Kaggle dataset status.")
            return False

        status_text = _dataset_status_from_output(result["stdout"])
        state["last_dataset_status"] = status_text
        if status_text in {"ready", "complete", "completed"}:
            state["status"] = "dataset_ready"
            _append_message_once(state, "Kaggle dataset is ready.")
            _write_state(state)
            return True
        if status_text in {"error", "failed", "cancelled", "canceled"}:
            state["status"] = "failed"
            state["last_error"] = result["stdout"] or "Kaggle dataset processing failed."
            _append_message_once(state, "Kaggle dataset processing failed.")
            return False

        _write_state(state)
        time.sleep(max(1, poll_seconds))

    state["status"] = "timeout"
    state["last_error"] = "Timed out while waiting for Kaggle dataset to become ready."
    _append_message_once(state, state["last_error"])
    return False


def _dataset_status_from_output(output: str) -> str:
    text = output.strip()
    if not text:
        return ""
    try:
        data = json.loads(text)
        status = data.get("status")
        return str(status).strip().lower() if status is not None else ""
    except json.JSONDecodeError:
        return text.lower()


def _download_kernel_output(state: dict[str, Any], cli: list[str], *, expect_mp3: bool) -> None:
    download_dir = Path(state["download_dir"])
    download_dir.mkdir(parents=True, exist_ok=True)
    output = _run(cli + ["kernels", "output", state["kernel_ref"], "-p", str(download_dir)], timeout=1800)
    state["history"].append(_history_item("kernels output", output))
    output_error = _summarize_cli_error(output) if output["returncode"] != 0 else ""
    files = [path for path in sorted(download_dir.rglob("*")) if path.is_file()]
    mp3_files = [path for path in files if path.suffix.lower() == ".mp3"]
    lyrics_files = [path for path in files if path.name == "lyrics.txt"]
    backing_files = [path for path in mp3_files if path.name.endswith("_backing.mp3")]
    vocal_files = [path for path in files if path.name.endswith(("_vocal_f5.wav", "_vocal_mms.wav"))]
    state["downloaded_files"] = [str(path) for path in files]
    _apply_kaggle_result_metadata(state, files)
    if backing_files:
        state["backing_path"] = str(backing_files[0])
        state["backing_url"] = _output_url(state, backing_files[0])
    if vocal_files:
        state["vocal_path"] = str(vocal_files[0])
        state["vocal_url"] = _output_url(state, vocal_files[0])
        if not state.get("generation_backend"):
            prefix = "tts_retry" if state.get("job_kind") == "tts_retry" else "musicgen"
            if vocal_files[0].name.endswith("_vocal_f5.wav"):
                state["generation_backend"] = f"{prefix}+f5_tts_vocal_mix"
            elif vocal_files[0].name.endswith("_vocal_mms.wav"):
                state["generation_backend"] = f"{prefix}+mms_tts_vocal_mix"
    if lyrics_files:
        lyrics_path = lyrics_files[0]
        state["lyrics_path"] = str(lyrics_path)
        state["lyrics_url"] = _output_url(state, lyrics_path)
        if not state.get("lyrics_text"):
            try:
                state["lyrics_text"] = lyrics_path.read_text(encoding="utf-8")
            except OSError:
                pass
    if output_error and not files:
        state["last_error"] = output_error
        _append_message_once(state, "Kaggle output download failed.")
        return
    if not state.get("last_error") and not mp3_files:
        state["last_error"] = _summarize_downloaded_logs(files)
    if mp3_files:
        _remove_messages(
            state,
            "Kaggle output download failed.",
            "Kaggle output downloaded, but no MP3 file was found.",
        )
        preferred = [path for path in mp3_files if path.name == f"{state['run_id']}.mp3"]
        mp3_path = preferred[0] if preferred else mp3_files[0]
        state["mp3_path"] = str(mp3_path)
        state["mp3_url"] = _output_url(state, mp3_path)
        backend = state.get("generation_backend", "")
        if state.get("musicgen_failed") or "guide_fallback" in backend:
            _append_message_once(state, "MusicGen failed on Kaggle; fallback guide backing was downloaded.")
        if "f5_tts_vocal_mix" in backend:
            _append_message_once(state, "Backing track and F5-TTS Vietnamese vocal were mixed into the final MP3.")
        elif "mms_tts_vocal_mix" in backend:
            if "f5_failed" in backend:
                _append_message_once(state, "F5-TTS failed on Kaggle; MMS Vietnamese TTS fallback was mixed into the final MP3.")
            else:
                _append_message_once(state, "Backing track and MMS Vietnamese TTS vocal were mixed into the final MP3.")
        elif state.get("vocal_failed") or "tts_failed" in backend or "tts_skipped" in backend:
            _append_message_once(state, "TTS/Vocal failed; downloaded MP3 is backing track only.")
        else:
            _append_message_once(state, "Kaggle output MP3 downloaded to local machine.")
        if output_error:
            state["download_warning"] = output_error
    elif expect_mp3:
        state["status"] = "failed"
        if output_error and not state.get("last_error"):
            state["last_error"] = output_error
        _append_message_once(state, "Kaggle output downloaded, but no MP3 file was found.")
    else:
        if output_error and not state.get("last_error"):
            state["last_error"] = output_error
        _append_message_once(state, "Kaggle error log downloaded.")


def _apply_kaggle_result_metadata(state: dict[str, Any], files: list[Path]) -> None:
    for path in files:
        if path.name != "kaggle_result.json":
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        backend = data.get("backend")
        if isinstance(backend, str):
            state["generation_backend"] = backend
        tts_model = data.get("tts_model")
        if isinstance(tts_model, str):
            state["tts_model"] = tts_model
        mms_tts_model = data.get("mms_tts_model")
        if isinstance(mms_tts_model, str):
            state["mms_tts_model"] = mms_tts_model
        tts_voice_actual = data.get("tts_voice_actual")
        if isinstance(tts_voice_actual, str):
            state["tts_voice_actual"] = tts_voice_actual
        tts_voice_note = data.get("tts_voice_note")
        if isinstance(tts_voice_note, str):
            state["tts_voice_note"] = tts_voice_note
        duration_plan = data.get("duration_plan")
        if isinstance(duration_plan, dict):
            state["duration_plan"] = duration_plan
            for key in (
                "policy",
                "target_duration_seconds",
                "planned_backing_duration_seconds",
                "musicgen_render_duration_seconds",
                "musicgen_duration_was_chunked",
                "duration_ceiling_seconds",
                "duration_was_capped",
                "estimated_vocal_duration_seconds",
                "outro_tail_seconds",
                "lyric_line_count",
                "lyric_word_count",
            ):
                if key in duration_plan:
                    state_key = "duration_policy" if key == "policy" else key
                    state[state_key] = duration_plan[key]
        for key in (
            "duration_policy",
            "target_duration_seconds",
            "planned_backing_duration_seconds",
            "musicgen_render_duration_seconds",
            "musicgen_duration_was_chunked",
            "duration_ceiling_seconds",
            "duration_was_capped",
            "estimated_vocal_duration_seconds",
            "outro_tail_seconds",
        ):
            if key in data:
                state[key] = data[key]
        backing_path = data.get("backing_path")
        if isinstance(backing_path, str):
            state["kaggle_backing_path"] = backing_path
        vocal_path = data.get("vocal_path")
        if isinstance(vocal_path, str):
            state["kaggle_vocal_path"] = vocal_path
        musicgen_failed = data.get("musicgen_failed")
        if isinstance(musicgen_failed, bool):
            state["musicgen_failed"] = musicgen_failed
        vocal_failed = data.get("vocal_failed")
        if isinstance(vocal_failed, bool):
            state["vocal_failed"] = vocal_failed
        musicgen_error = data.get("musicgen_error")
        if isinstance(musicgen_error, str) and musicgen_error.strip():
            state["musicgen_error"] = musicgen_error.strip()[-1000:]
            state["last_error"] = musicgen_error.strip()[-1000:]
        tts_error = data.get("tts_error")
        if isinstance(tts_error, str) and tts_error.strip():
            state["tts_error"] = tts_error.strip()[-1000:]
            if not state.get("last_error"):
                state["last_error"] = state["tts_error"]
        f5_tts_error = data.get("f5_tts_error")
        if isinstance(f5_tts_error, str) and f5_tts_error.strip():
            state["f5_tts_error"] = f5_tts_error.strip()[-1000:]
        lyrics_text = data.get("lyrics_text")
        if isinstance(lyrics_text, str):
            state["lyrics_text"] = lyrics_text
        lyrics = data.get("lyrics")
        if isinstance(lyrics, dict):
            state["lyrics"] = lyrics
        vocal_plan = data.get("vocal_plan")
        if isinstance(vocal_plan, dict):
            state["vocal_plan"] = vocal_plan
        scene_plan = data.get("scene_plan")
        if isinstance(scene_plan, dict):
            state["scene_plan"] = scene_plan
        return


def _output_url(state: dict[str, Any], path: Path) -> str:
    try:
        relative = path.relative_to(Path(state["run_dir"]))
    except ValueError:
        return ""
    return "/outputs/" + f"{state['run_id']}/{relative.as_posix()}"


def _summarize_downloaded_logs(files: list[Path]) -> str:
    log_files = [path for path in files if path.suffix.lower() == ".log"]
    if not log_files:
        return ""
    return _summarize_log_file(log_files[0])


def _summarize_log_file(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    lines: list[str] = []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    value = item.get("data")
                    if isinstance(value, str):
                        lines.extend(value.splitlines())
    except json.JSONDecodeError:
        lines = text.splitlines()
    cleaned = [line.strip() for line in lines if line.strip()]
    interesting = [
        line
        for line in cleaned
        if any(marker in line.lower() for marker in ["error", "exception", "traceback", "filenotfound", "runtimeerror"])
    ]
    selected = interesting[-8:] if interesting else cleaned[-8:]
    return " | ".join(selected)[-1000:]


def _append_message_once(state: dict[str, Any], message: str) -> None:
    if message not in state.setdefault("messages", []):
        state["messages"].append(message)


def _remove_messages(state: dict[str, Any], *messages: str) -> None:
    blocked = set(messages)
    state["messages"] = [message for message in state.get("messages", []) if message not in blocked]


def _load_state(state_or_path: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(state_or_path, dict):
        return state_or_path
    return json.loads(Path(state_or_path).read_text(encoding="utf-8"))


def _write_state(state: dict[str, Any]) -> None:
    Path(state["state_path"]).write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _run(command: list[str], *, timeout: int) -> dict[str, Any]:
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    tokens = load_kaggle_api_tokens()
    env.update(tokens)
    if tokens.get("KAGGLE_USERNAME") and tokens.get("KAGGLE_KEY"):
        runtime_home = PROJECT_ROOT / ".kaggle_runtime_home"
        config_dir = runtime_home / ".kaggle"
        config_dir.mkdir(parents=True, exist_ok=True)
        env["HOME"] = str(runtime_home)
        env["USERPROFILE"] = str(runtime_home)
        env["KAGGLE_CONFIG_DIR"] = str(config_dir)
        env.pop("KAGGLE_API_TOKEN", None)
        env.pop("KAGGLE_API_V1_TOKEN_PATH", None)
    proc = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=env,
    )
    return {
        "command": command,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def _history_item(label: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": label,
        "at": _now(),
        "command": result["command"],
        "returncode": result["returncode"],
        "stdout": result["stdout"][-4000:],
        "stderr": result["stderr"][-4000:],
    }


def _summarize_cli_error(result: dict[str, Any]) -> str:
    text = "\n".join(part for part in [result.get("stderr", ""), result.get("stdout", "")] if part)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return " | ".join(lines[-4:])[-1000:]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
