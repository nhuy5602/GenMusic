"""Local and Kaggle orchestration for the self-authored music model."""

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

from ..data.lyric_alignment import AlignedLine, write_lrc
from ..data.vietnamese_text import normalize_vietnamese_lyrics
from ..models.text_to_music_diffusion import MusicDiffusionConfig, generate_audio, load_checkpoint, make_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "genmusic-vn-self-diffusion-v1"
DEFAULT_KAGGLE_DATASET_SLUG = "genmusic-vn-self-diffusion-training"
KAGGLE_DATASET_ENV = "GENMUSIC_KAGGLE_DATASET_REF"


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
    training_dataset_ref: str | None = None


def run_local_generation(*, text: str, style: str, output_dir: str | Path, duration_seconds: float, checkpoint: str | Path | None = None, steps: int = 6, seed: int = 5602, device: str | None = None, mel_output: str | Path | None = None, vocoder: str = "istft", model_type: str = "conv1d", roberta_model: str = "xlm-roberta-base") -> dict[str, Any]:
    normalized = normalize_vietnamese_lyrics(text).strip()
    if not normalized:
        raise SelfMusicError("Văn bản input đang trống.")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    selected_device = device or _default_device()
    if checkpoint and Path(checkpoint).exists():
        model, config, payload = load_checkpoint(checkpoint, device=selected_device, model_type=model_type, roberta_model=roberta_model)
        checkpoint_path = str(Path(checkpoint).resolve())
        checkpoint_epoch = payload.get("epoch", 0)
    else:
        config = MusicDiffusionConfig()
        if model_type == "dit":
            from ..models.dit_transformer import MicroDiT
            model = MicroDiT(config, roberta_model=roberta_model).to(selected_device)
        else:
            model = make_model(config).to(selected_device)
        checkpoint_path = ""
        checkpoint_epoch = 0
    report = generate_audio(
        model,
        normalized,
        style or "Vietnamese pop, warm piano, clear melody",
        destination / "final.wav",
        duration_seconds=max(1.0, float(duration_seconds)),
        config=config,
        device=selected_device,
        steps=max(1, int(steps)),
        seed=int(seed),
        mel_output=mel_output,
        vocoder_type=vocoder,
    )
    report.update({"text": normalized, "style": style, "checkpoint": checkpoint_path, "checkpoint_epoch": checkpoint_epoch, "device": selected_device})
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
    training_dataset_ref = resolve_training_dataset_ref(config.training_dataset_ref, config.username)
    training_dataset_slug = training_dataset_ref.rsplit("/", 1)[-1]
    kernel_ref = f"{username}/{kernel_slug}"
    dataset_url = _dataset_url(training_dataset_ref)
    request_dataset_url = _dataset_url(request_dataset_ref)
    kernel_url = f"https://www.kaggle.com/code/{kernel_ref}" if username != "YOUR_KAGGLE_USERNAME" else ""
    received = input_received_at or _now()
    style_prompt = genre or "Vietnamese pop ballad, warm piano, emotional strings, clear melody"
    request = {
        "run_id": run_id,
        "text": normalized,
        "lyrics": normalized,
        "duration_seconds": requested_duration,
        "training_dataset_ref": training_dataset_ref,
        "request_dataset_ref": request_dataset_ref,
        "dataset_url": dataset_url,
        "request_dataset_url": request_dataset_url,
        "kernel_url": kernel_url,
        "style_prompt": style_prompt,
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
    (dataset_dir / "dataset-metadata.json").write_text(json.dumps({"title": request_dataset_slug, "id": request_dataset_ref, "licenses": [{"name": "other"}], "subtitle": "Request cho model GenMusic tự code.", "description": "Private request dataset for the self-authored GenMusic diffusion model."}, ensure_ascii=False, indent=2), encoding="utf-8")
    (kernel_dir / "run_genmusic.py").write_text(_kernel_script(request_dataset_slug, training_dataset_slug), encoding="utf-8")
    (kernel_dir / "kernel-metadata.json").write_text(json.dumps({"id": kernel_ref, "title": kernel_slug, "code_file": "run_genmusic.py", "language": "python", "kernel_type": "script", "is_private": "true", "enable_gpu": "true", "enable_internet": "true", "machine_shape": config.machine_shape, "dataset_sources": [training_dataset_ref, request_dataset_ref], "competition_sources": [], "kernel_sources": [], "model_sources": []}, ensure_ascii=False, indent=2), encoding="utf-8")
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
        "dataset_ref": training_dataset_ref,
        "training_dataset_ref": training_dataset_ref,
        "request_dataset_ref": request_dataset_ref,
        "kernel_ref": kernel_ref,
        "run_dir": str(run_dir),
        "job_dir": str(job_dir),
        "dataset_dir": str(dataset_dir),
        "kernel_dir": str(kernel_dir),
        "download_dir": str(download_dir),
        "state_path": str(job_dir / "job_state.json"),
        "duration_seconds": requested_duration,
        "dataset_url": dataset_url,
        "request_dataset_url": request_dataset_url,
        "kernel_url": kernel_url,
        "commands": commands,
        "messages": [f"Dataset training cố định: {training_dataset_ref}.", "Đã đóng gói request và source model tự code; chưa submit Kaggle."],
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
    if not kaggle_dataset_exists(state["training_dataset_ref"]):
        return _fail(state, f"Không tìm thấy dataset training Kaggle '{state['training_dataset_ref']}'. Hãy chạy make-and-upload-dataset trước.")
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
    pushed = _run(cli + ["kernels", "push", "-p", state["kernel_dir"]], timeout=900)
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
        state["last_error"] = ""
    state["checked_at"] = _now()
    _write_state(state)
    return state


def kaggle_readiness(username: str | None = None) -> dict[str, Any]:
    tokens = load_kaggle_api_tokens()
    if username and not tokens.get("KAGGLE_USERNAME"):
        tokens["KAGGLE_USERNAME"] = username
    ready = bool(tokens.get("KAGGLE_USERNAME") and tokens.get("KAGGLE_KEY") and kaggle_cli_command())
    return {"ready": ready, "messages": [] if ready else ["Cần KAGGLE_USERNAME, KAGGLE_KEY và Kaggle CLI."]}


def kaggle_dataset_exists(dataset_ref: str) -> bool:
    cli = kaggle_cli_command()
    if cli is None or dataset_ref.startswith("YOUR_KAGGLE_USERNAME/"):
        return False
    result = _run(cli + ["datasets", "status", dataset_ref], timeout=120)
    return _dataset_status_is_ready(result)


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
    if cli is None or not resolved_username or not load_kaggle_api_tokens().get("KAGGLE_KEY"):
        report = {"status": "needs_setup", "dataset": str(root), "message": "Cần Kaggle CLI, KAGGLE_USERNAME và KAGGLE_KEY."}
        (root / "kaggle_upload_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report
    if dataset_ref:
        dataset_ref = validate_dataset_ref(dataset_ref)
    elif slug:
        dataset_ref = f"{resolved_username}/{slugify(slug, max_length=50)}"
    else:
        dataset_ref = resolve_training_dataset_ref(None, username)
    metadata_path = root / "dataset-metadata.json"
    metadata_path.write_text(json.dumps({"title": "GenMusic VN self-diffusion training", "id": dataset_ref, "licenses": [{"name": "other"}], "subtitle": "Synthetic mel dataset for the self-authored text-to-music model.", "description": "Synthetic structured mel tensors and Vietnamese text/style conditions for pipeline training smoke tests."}, ensure_ascii=False, indent=2), encoding="utf-8")
    started = time.perf_counter()
    result = _run(cli + ["datasets", "create", "-p", str(root), "-r", "zip"], timeout=timeout_seconds)
    dataset_ready = result["returncode"] == 0 and _wait_for_dataset_ready(cli, dataset_ref, timeout_seconds=min(timeout_seconds, 900))
    status = "uploaded" if dataset_ready else ("pending" if result["returncode"] == 0 else "failed")
    report = {"status": status, "dataset": str(root), "dataset_ref": dataset_ref, "dataset_url": _dataset_url(dataset_ref), "dataset_ready": dataset_ready, "returncode": result["returncode"], "elapsed_seconds": round(time.perf_counter() - started, 3), "stdout_tail": result["stdout"][-4000:], "stderr_tail": result["stderr"][-4000:]}
    (root / "kaggle_upload_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def load_kaggle_api_tokens() -> dict[str, str]:
    values: dict[str, str] = {}
    for path in (PROJECT_ROOT / ".env", PROJECT_ROOT / ".env.local"):
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    key, value = line.split("=", 1)
                    values[key.strip()] = value.strip().strip('"').strip("'")
    for key in ("KAGGLE_USERNAME", "KAGGLE_KEY", KAGGLE_DATASET_ENV):
        if os.getenv(key):
            values[key] = str(os.getenv(key))
    return values


def kaggle_cli_command() -> list[str] | None:
    candidates = [shutil.which("kaggle")]
    scripts = Path(site.USER_BASE) / ("Scripts" if os.name == "nt" else "bin")
    candidates.append(str(scripts / ("kaggle.exe" if os.name == "nt" else "kaggle")))
    runtime_scripts = Path(sys.executable).resolve().parent / "Scripts"
    candidates.append(str(runtime_scripts / ("kaggle.exe" if os.name == "nt" else "kaggle")))
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return [candidate]
    return None


def resolve_kaggle_username(username: str | None) -> str | None:
    return username or load_kaggle_api_tokens().get("KAGGLE_USERNAME")


def validate_dataset_ref(dataset_ref: str) -> str:
    value = str(dataset_ref or "").strip().strip("/")
    parts = value.split("/")
    if len(parts) != 2 or not all(parts) or any(char.isspace() for char in value):
        raise ValueError("Dataset ref phải có dạng owner/slug, ví dụ user/genmusic-vn-self-diffusion-training.")
    return value


def resolve_training_dataset_ref(dataset_ref: str | None = None, username: str | None = None) -> str:
    configured = dataset_ref or os.getenv(KAGGLE_DATASET_ENV) or load_kaggle_api_tokens().get(KAGGLE_DATASET_ENV)
    if configured:
        return validate_dataset_ref(configured)
    owner = resolve_kaggle_username(username) or "YOUR_KAGGLE_USERNAME"
    return f"{owner}/{DEFAULT_KAGGLE_DATASET_SLUG}"


def _dataset_url(dataset_ref: str) -> str:
    return f"https://www.kaggle.com/datasets/{dataset_ref}" if not dataset_ref.startswith("YOUR_KAGGLE_USERNAME/") else ""


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


def _kernel_script(request_dataset_slug: str, training_dataset_slug: str) -> str:
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
training_dataset = Path("/kaggle/input/{training_dataset_slug}")
if not (training_dataset / "records.jsonl").exists():
    training_records = list(Path("/kaggle/input").rglob("records.jsonl"))
    if training_records:
        training_dataset = training_records[0].parent
if not (training_dataset / "records.jsonl").exists():
    raise RuntimeError("Dataset training Kaggle không tồn tại hoặc thiếu records.jsonl.")
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
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch", "torchaudio", "librosa", "matplotlib"], check=False)
checkpoint = Path("/kaggle/working/self_music.pt")
subprocess.run([sys.executable, "cli.py", "train-self", "--dataset", str(training_dataset), "--checkpoint", str(checkpoint), "--epochs", "1", "--batch-size", "4"], cwd=source_root, check=True)
output = Path("/kaggle/working/genmusic_output")
subprocess.run([sys.executable, "cli.py", "generate-local", "--text", request["lyrics"], "--style", request["style_prompt"], "--duration", str(request["duration_seconds"]), "--checkpoint", str(checkpoint), "--steps", "6", "--out", str(output)], cwd=source_root, check=True)
output.joinpath("request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
if shutil.which("ffmpeg") and list(output.glob("*.wav")):
    subprocess.run(["ffmpeg", "-y", "-i", str(list(output.glob("*.wav"))[0]), str(output / "final.mp3")], check=False)
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
        "# Đặt KAGGLE_USERNAME và KAGGLE_KEY trong .env hoặc environment.",
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
        if path.suffix.lower() == ".mp3":
            state["mp3_url"] = url
            state["mp3_path"] = str(path)
        elif path.suffix.lower() == ".wav":
            state["wav_url"] = url
            state["wav_path"] = str(path)
    lrc = Path(state["dataset_dir"]) / "lyrics.lrc"
    if lrc.exists():
        try:
            state["lrc_url"] = "/outputs/" + lrc.resolve().relative_to(output_root.resolve()).as_posix()
        except ValueError:
            pass
