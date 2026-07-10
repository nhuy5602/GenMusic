"""Kaggle automation for the official ASLP-lab/DiffRhythm backend."""
from __future__ import annotations

import json
import os
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

from ..data.lyric_alignment import write_lrc
from ..data.vietnamese_text import normalize_vietnamese_lyrics
from .diffrhythm_official import DEFAULT_MODEL_REF, OFFICIAL_REPO_COMMIT, OFFICIAL_REPO_URL, make_lrc_for_diff_rhythm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIFFRHYTHM_MODEL = "ASLP-lab/DiffRhythm-1_2"
DEFAULT_CUSTOM_MUSIC_MODEL = DEFAULT_DIFFRHYTHM_MODEL
DEFAULT_OFFICIAL_REPO_COMMIT = OFFICIAL_REPO_COMMIT


class KaggleAutoError(RuntimeError):
    pass


@dataclass(frozen=True)
class KaggleJobConfig:
    model: str = DEFAULT_DIFFRHYTHM_MODEL
    username: str | None = None
    machine_shape: str = "NvidiaTeslaT4"
    submit: bool = True
    wait: bool = False
    poll_seconds: int = 60
    timeout_seconds: int = 21_600


def submit_text_to_music_job(*, text: str, output_root: str | Path = "outputs", duration_seconds: int = 95, genre: str | None = None, config: KaggleJobConfig | None = None) -> dict[str, Any]:
    config = config or KaggleJobConfig()
    state = stage_text_to_music_job(text=text, output_root=output_root, duration_seconds=duration_seconds, genre=genre, config=config)
    if not config.submit:
        state["status"] = "staged"
        state["messages"].append("Đã stage job DiffRhythm; chưa submit Kaggle.")
        _write_state(state)
        return state
    readiness = kaggle_readiness(config.username)
    state["kaggle_ready"] = readiness["ready"]
    state["messages"].extend(readiness["messages"])
    if not readiness["ready"]:
        state["status"] = "needs_setup"
        _write_state(state)
        return state
    return submit_kaggle_job(state, wait=config.wait, poll_seconds=config.poll_seconds, timeout_seconds=config.timeout_seconds)


def stage_text_to_music_job(*, text: str, output_root: str | Path, duration_seconds: int, genre: str | None, config: KaggleJobConfig, input_received_at: str | None = None) -> dict[str, Any]:
    normalized = normalize_vietnamese_lyrics(text).strip()
    if not normalized:
        raise ValueError("Văn bản input đang trống.")
    requested_duration = max(1, int(duration_seconds))
    audio_length = 95 if requested_duration <= 95 else min(285, requested_duration)
    run_id = make_run_id(normalized)
    username = resolve_kaggle_username(config.username) or "YOUR_KAGGLE_USERNAME"
    run_dir = Path(output_root) / run_id
    job_dir = run_dir / "kaggle_job"
    dataset_dir = job_dir / "dataset"
    kernel_dir = job_dir / "kernel"
    download_dir = job_dir / "downloaded_output"
    for path in (dataset_dir, kernel_dir, download_dir):
        path.mkdir(parents=True, exist_ok=True)
    dataset_slug = slugify(f"genmusic-vn-diffrhythm-{run_id}", max_length=48)
    kernel_slug = slugify(f"genmusic-vn-diffrhythm-kernel-{run_id}", max_length=48)
    dataset_ref = f"{username}/{dataset_slug}"
    kernel_ref = f"{username}/{kernel_slug}"
    received = input_received_at or _now()
    style_prompt = genre or "Vietnamese pop ballad, warm piano, emotional strings, clear vocal melody"
    request = {
        "run_id": run_id,
        "text": normalized,
        "lyrics": normalized,
        "duration_seconds_requested": requested_duration,
        "audio_length": audio_length,
        "genre": genre or "Vietnamese pop ballad",
        "style_prompt": style_prompt,
        "model": config.model or DEFAULT_MODEL_REF,
        "backend": "ASLP-lab/DiffRhythm",
        "official_repo": OFFICIAL_REPO_URL,
        "official_repo_commit": DEFAULT_OFFICIAL_REPO_COMMIT,
        "official_repo_source": "vendored:third_party/DiffRhythm",
        "input_received_at": received,
        "created_at": _now(),
    }
    (run_dir / "request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
    (dataset_dir / "request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
    write_lrc(make_lrc_for_diff_rhythm(normalized, audio_length), dataset_dir / "lyrics.lrc")
    _write_source_zip(dataset_dir / "genmusic_vn_source.zip")
    (dataset_dir / "dataset-metadata.json").write_text(json.dumps({"title": dataset_slug, "id": dataset_ref, "licenses": [{"name": "other"}], "subtitle": "Input LRC và request cho DiffRhythm.", "description": "Private request dataset for official ASLP-lab/DiffRhythm inference."}, ensure_ascii=False, indent=2), encoding="utf-8")
    (kernel_dir / "run_diffrhythm.py").write_text(_kernel_script(dataset_slug), encoding="utf-8")
    (kernel_dir / "kernel-metadata.json").write_text(json.dumps({"id": kernel_ref, "title": kernel_slug, "code_file": "run_diffrhythm.py", "language": "python", "kernel_type": "script", "is_private": "true", "enable_gpu": "true", "enable_internet": "true", "machine_shape": config.machine_shape, "dataset_sources": [dataset_ref], "competition_sources": [], "kernel_sources": [], "model_sources": []}, ensure_ascii=False, indent=2), encoding="utf-8")
    commands = _commands(dataset_dir, kernel_dir, download_dir, kernel_ref)
    (job_dir / "run_commands.ps1").write_text("\n".join(commands) + "\n", encoding="utf-8")
    state = {
        "run_id": run_id,
        "job_kind": "diffrhythm_generation",
        "status": "staged",
        "created_at": _now(),
        "input_received_at": received,
        "backend": "ASLP-lab/DiffRhythm",
        "model": request["model"],
        "lyrics": normalized,
        "dataset_ref": dataset_ref,
        "kernel_ref": kernel_ref,
        "run_dir": str(run_dir),
        "job_dir": str(job_dir),
        "dataset_dir": str(dataset_dir),
        "kernel_dir": str(kernel_dir),
        "download_dir": str(download_dir),
        "state_path": str(job_dir / "job_state.json"),
        "audio_length": audio_length,
        "duration_seconds_requested": requested_duration,
        "commands": commands,
        "messages": ["Đã chuẩn bị request DiffRhythm chính thức; model sẽ tự tải trên Kaggle."],
        "history": [],
        "generation_backend": "ASLP-lab/DiffRhythm",
        "downloaded_files": [],
        "last_error": "",
    }
    _write_state(state)
    return state


def submit_kaggle_job(state: dict[str, Any], *, wait: bool, poll_seconds: int, timeout_seconds: int) -> dict[str, Any]:
    cli = kaggle_cli_command()
    if cli is None:
        return _fail(state, "Không tìm thấy Kaggle CLI.")
    created = _run(cli + ["datasets", "create", "-p", state["dataset_dir"], "-r", "zip"], timeout=900)
    state["history"].append(_history_item("datasets create", created))
    if created["returncode"] != 0:
        return _fail(state, _summarize_cli_error(created))
    state["status"] = "dataset_uploaded"
    _write_state(state)
    pushed = _run(cli + ["kernels", "push", "-p", state["kernel_dir"]], timeout=900)
    state["history"].append(_history_item("kernels push", pushed))
    if pushed["returncode"] != 0:
        return _fail(state, _summarize_cli_error(pushed))
    state["status"] = "submitted"
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
    status = _run(cli + ["kernels", "status", state["kernel_ref"]], timeout=120)
    state["history"].append(_history_item("kernels status", status))
    status_text = f"{status['stdout']}\n{status['stderr']}".lower()
    if status["returncode"] != 0:
        return _fail(state, _summarize_cli_error(status))
    if any(value in status_text for value in ("complete", "completed", "succeeded")):
        state["status"] = "complete"
        download = _run(cli + ["kernels", "output", state["kernel_ref"], "-p", state["download_dir"], "-o"], timeout=1200)
        state["history"].append(_history_item("kernels output", download))
        state["downloaded_files"] = [str(path) for path in Path(state["download_dir"]).rglob("*") if path.is_file()]
        _attach_artifact_urls(state)
    elif any(value in status_text for value in ("error", "failed", "cancelled", "canceled")):
        state["status"] = "failed"
        state["last_error"] = status_text[-2000:]
    else:
        state["status"] = "running" if "running" in status_text else "submitted"
    state["checked_at"] = _now()
    _write_state(state)
    return state


def kaggle_readiness(username: str | None = None) -> dict[str, Any]:
    tokens = load_kaggle_api_tokens()
    if username and not tokens.get("KAGGLE_USERNAME"):
        tokens["KAGGLE_USERNAME"] = username
    ready = bool(tokens.get("KAGGLE_USERNAME") and tokens.get("KAGGLE_KEY") and kaggle_cli_command())
    return {"ready": ready, "messages": [] if ready else ["Cần KAGGLE_USERNAME, KAGGLE_KEY và Kaggle CLI."]}


def load_kaggle_api_tokens() -> dict[str, str]:
    values: dict[str, str] = {}
    for path in (PROJECT_ROOT / ".env", PROJECT_ROOT / ".env.local"):
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    key, value = line.split("=", 1)
                    values[key.strip()] = value.strip().strip('"').strip("'")
    for key in ("KAGGLE_USERNAME", "KAGGLE_KEY"):
        if os.getenv(key):
            values[key] = str(os.getenv(key))
    return values


def kaggle_cli_command() -> list[str] | None:
    candidates = [shutil.which("kaggle")]
    scripts = Path(site.USER_BASE) / ("Scripts" if os.name == "nt" else "bin")
    candidates.append(str(scripts / ("kaggle.exe" if os.name == "nt" else "kaggle")))
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return [candidate]
    return None


def resolve_kaggle_username(username: str | None) -> str | None:
    return username or load_kaggle_api_tokens().get("KAGGLE_USERNAME")


def make_run_id(text: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{sha1(text.encode('utf-8')).hexdigest()[:10]}"


def slugify(value: str, *, max_length: int = 48) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii").lower()
    normalized = "".join(char if char.isalnum() else "-" for char in normalized).strip("-")
    return normalized[:max_length].strip("-") or "genmusic-vn"


def _kernel_script(dataset_slug: str) -> str:
    return f'''import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

input_root = Path("/kaggle/input/{dataset_slug}")
request = json.loads((input_root / "request.json").read_text(encoding="utf-8"))
source_root = Path("/kaggle/working/GenMusic")
with zipfile.ZipFile(input_root / "genmusic_vn_source.zip") as archive:
    archive.extractall(source_root)
repo = source_root / "third_party" / "DiffRhythm"
if not (repo / "infer" / "infer.py").exists():
    raise RuntimeError("Vendored DiffRhythm source is missing from genmusic_vn_source.zip")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(repo / "requirements.txt")], check=True)
output = Path("/kaggle/working/diffrhythm_output")
output.mkdir(parents=True, exist_ok=True)
command = [sys.executable, str(repo / "infer" / "infer.py"), "--lrc-path", str(input_root / "lyrics.lrc"), "--ref-prompt", request["style_prompt"], "--audio-length", str(request["audio_length"]), "--output-dir", str(output), "--batch-infer-num", "1", "--chunked"]
environment = os.environ.copy()
environment["PYTHONPATH"] = os.pathsep.join([str(repo), str(repo / "infer"), environment.get("PYTHONPATH", "")])
subprocess.run(command, cwd=repo, env=environment, check=True)
(output / "request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
if shutil.which("ffmpeg") and list(output.glob("*.wav")):
    wav = list(output.glob("*.wav"))[0]
    subprocess.run(["ffmpeg", "-y", "-i", str(wav), str(output / "final.mp3")], check=False)
'''


def _write_source_zip(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    excluded_parts = {
        ".git",
        "outputs",
        "__pycache__",
        ".pytest_cache",
        ".venv",
        "models",
    }
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in PROJECT_ROOT.rglob("*"):
            relative = path.relative_to(PROJECT_ROOT)
            if not path.is_file() or any(part in excluded_parts for part in relative.parts):
                continue
            if relative.as_posix().startswith(("datasets/random_diffrhythm/", "datasets/processed/")):
                continue
            archive.write(path, relative.as_posix())


def _commands(dataset_dir: Path, kernel_dir: Path, download_dir: Path, kernel_ref: str) -> list[str]:
    return [
        "pip install -U kaggle",
        "# Đặt KAGGLE_USERNAME và KAGGLE_KEY trong .env hoặc environment.",
        f'kaggle datasets create -p "{dataset_dir}" -r zip',
        f'kaggle kernels push -p "{kernel_dir}"',
        f'kaggle kernels status "{kernel_ref}"',
        f'kaggle kernels output "{kernel_ref}" -p "{download_dir}" -o',
    ]


def _run(command: list[str], *, timeout: int) -> dict[str, Any]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, env={**os.environ, **load_kaggle_api_tokens()})
        return {"command": command, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
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
    for path_text in state.get("downloaded_files", []):
        path = Path(path_text)
        try:
            relative = path.resolve().relative_to(output_root.resolve()).as_posix()
        except ValueError:
            continue
        url = "/outputs/" + relative
        suffix = path.suffix.lower()
        if suffix == ".mp3":
            state["mp3_url"] = url
            state["mp3_path"] = str(path)
        elif suffix == ".wav":
            state["wav_url"] = url
            state["wav_path"] = str(path)
    lrc = Path(state["dataset_dir"]) / "lyrics.lrc"
    if lrc.exists():
        try:
            relative = lrc.resolve().relative_to(output_root.resolve()).as_posix()
            state["lrc_url"] = "/outputs/" + relative
        except ValueError:
            pass
