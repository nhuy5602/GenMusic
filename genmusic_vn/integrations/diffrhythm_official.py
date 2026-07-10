"""Official ASLP-lab/DiffRhythm integration.

The project owns orchestration and dataset preparation, while model code and
checkpoint loading stay in the upstream repository. This avoids maintaining a
second, incompatible DiT implementation.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..data.lyric_alignment import AlignedLine, write_lrc
from ..data.vietnamese_text import lyric_content_lines, normalize_vietnamese_lyrics


OFFICIAL_REPO_URL = "https://github.com/ASLP-lab/DiffRhythm.git"
OFFICIAL_REPO_COMMIT = "28ad63c0f096fe2ee258bcabbcf081d5d9366afd"
DEFAULT_MODEL_REF = "ASLP-lab/DiffRhythm-1_2"
DEFAULT_REPO_PATH = Path("third_party") / "DiffRhythm"


class DiffRhythmError(RuntimeError):
    pass


@dataclass(frozen=True)
class DiffRhythmConfig:
    repo_path: str | None = None
    model_ref: str = DEFAULT_MODEL_REF
    audio_length: int = 95
    chunked: bool = True
    batch_infer_num: int = 1
    timeout_seconds: int = 21_600
    python_executable: str = sys.executable

    def normalized_audio_length(self) -> int:
        requested = int(self.audio_length)
        if requested == 95:
            return 95
        return max(96, min(285, requested))


def ensure_official_checkout(repo_path: str | Path | None = None, *, update: bool = False) -> Path:
    destination = Path(repo_path or DEFAULT_REPO_PATH)
    if not (destination / "infer" / "infer.py").exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(["git", "clone", "--depth", "1", OFFICIAL_REPO_URL, str(destination)], capture_output=True, text=True)
        if result.returncode != 0:
            raise DiffRhythmError(f"Không clone được DiffRhythm: {result.stderr[-1000:]}")
    elif update:
        subprocess.run(["git", "fetch", "--depth", "1", "origin", "main"], cwd=destination, check=False, capture_output=True, text=True)
        subprocess.run(["git", "checkout", "--force", "main"], cwd=destination, check=False, capture_output=True, text=True)
    return destination.resolve()


def make_lrc_for_diff_rhythm(lyrics: str, duration_seconds: int) -> list[AlignedLine]:
    lines = lyric_content_lines(normalize_vietnamese_lyrics(lyrics))
    if not lines:
        lines = [""]
    span = max(1.0, float(duration_seconds)) / len(lines)
    return [AlignedLine(line, index * span, (index + 1) * span, "synthetic-random-or-user-input") for index, line in enumerate(lines)]


def run_official_inference(
    *,
    lyrics: str,
    style_prompt: str,
    output_dir: str | Path,
    config: DiffRhythmConfig | None = None,
) -> dict[str, Any]:
    """Run upstream ``infer/infer.py`` and return the produced audio path."""

    config = config or DiffRhythmConfig()
    repo = ensure_official_checkout(config.repo_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    lrc_path = destination / "input.lrc"
    write_lrc(make_lrc_for_diff_rhythm(lyrics, config.normalized_audio_length()), lrc_path)
    command = [
        config.python_executable,
        str(repo / "infer" / "infer.py"),
        "--lrc-path",
        str(lrc_path),
        "--ref-prompt",
        style_prompt.strip() or "Vietnamese pop ballad, piano, warm strings",
        "--audio-length",
        str(config.normalized_audio_length()),
        "--output-dir",
        str(destination),
        "--batch-infer-num",
        str(max(1, config.batch_infer_num)),
    ]
    if config.chunked:
        command.append("--chunked")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join([str(repo), str(repo / "infer"), environment.get("PYTHONPATH", "")])
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=repo, env=environment, capture_output=True, text=True, timeout=config.timeout_seconds)
    elapsed = time.perf_counter() - started
    report = {
        "backend": "ASLP-lab/DiffRhythm",
        "model_ref": config.model_ref,
        "repo": str(repo),
        "command": command,
        "returncode": completed.returncode,
        "elapsed_seconds": round(elapsed, 3),
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "status": "complete" if completed.returncode == 0 else "failed",
    }
    (destination / "diffrhythm_inference_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if completed.returncode != 0:
        raise DiffRhythmError(f"DiffRhythm inference thất bại: {completed.stderr[-1200:]}")
    outputs = sorted(destination.glob("*.wav")) + sorted(destination.glob("*.mp3"))
    if not outputs:
        raise DiffRhythmError("DiffRhythm chạy xong nhưng không tạo WAV/MP3.")
    report["audio_path"] = str(outputs[0].resolve())
    return report


def create_random_official_dataset(output_dir: str | Path, *, count: int = 4, max_frames: int = 64, seed: int = 5602) -> dict[str, Any]:
    """Create tiny official-format data for a real Kaggle smoke training run."""

    try:
        import torch
    except ImportError:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        report = {"status": "needs-torch", "count": max(1, count), "max_frames": max_frames, "seed": seed, "backend": "ASLP-lab/DiffRhythm", "message": "Cần torch để ghi .pt. Chạy lại trong Kaggle GPU hoặc cài extra diffrhythm.", "next_command": f"python -m genmusic_vn.cli make-random-diffrhythm-dataset --out {root.as_posix()} --count {count} --max-frames {max_frames} --seed {seed}"}
        (root / "random_dataset_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    root = Path(output_dir)
    latent_dir = root / "latent"
    style_dir = root / "style"
    lrc_dir = root / "lrc"
    for directory in (latent_dir, style_dir, lrc_dir):
        directory.mkdir(parents=True, exist_ok=True)
    entries: list[str] = []
    for index in range(max(1, count)):
        item_id = f"random_{index:05d}"
        latent = torch.randn(1, 64, max_frames, dtype=torch.float16)
        style = torch.randn(1, 512, dtype=torch.float16)
        token_a = torch.randint(3, 363, (min(12, max_frames // 2),), dtype=torch.long).tolist()
        token_b = torch.randint(3, 363, (min(12, max_frames // 2),), dtype=torch.long).tolist()
        lrc = {"time": [0.0, max_frames / 43.066], "lrc": [token_a, token_b]}
        torch.save(latent, latent_dir / f"{item_id}.pt")
        torch.save(style, style_dir / f"{item_id}.pt")
        torch.save(lrc, lrc_dir / f"{item_id}.pt")
        entries.append(f"{item_id}|{(lrc_dir / f'{item_id}.pt').relative_to(root).as_posix()}|{(latent_dir / f'{item_id}.pt').relative_to(root).as_posix()}|{(style_dir / f'{item_id}.pt').relative_to(root).as_posix()}")
    (root / "train.scp").write_text("\n".join(entries) + "\n", encoding="utf-8")
    tiny_config = {
        "model_type": "diffrhythm",
        "model": {"dim": 64, "depth": 1, "heads": 1, "ff_mult": 2, "text_dim": 64, "conv_layers": 0, "mel_dim": 64, "text_num_embeds": 363},
    }
    (root / "diffrhythm-random.json").write_text(json.dumps(tiny_config, indent=2), encoding="utf-8")
    report = {"status": "created", "format": "official-diffrhythm", "count": len(entries), "max_frames": max_frames, "seed": seed, "train_scp": str((root / "train.scp").resolve()), "model_config": str((root / "diffrhythm-random.json").resolve()), "backend": "ASLP-lab/DiffRhythm"}
    (root / "random_dataset_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def validate_official_dataset(dataset_dir: str | Path) -> dict[str, Any]:
    root = Path(dataset_dir)
    scp = root / "train.scp"
    if not scp.exists():
        raise DiffRhythmError(f"Thiếu train.scp: {scp}")
    entries = [line.strip() for line in scp.read_text(encoding="utf-8").splitlines() if line.strip()]
    missing: list[str] = []
    for entry in entries:
        parts = entry.split("|")
        if len(parts) != 4:
            missing.append(entry)
            continue
        for path in parts[1:]:
            candidate = Path(path)
            if not candidate.is_absolute():
                candidate = root / candidate
            if not candidate.exists():
                missing.append(str(candidate))
    report = {"status": "valid" if entries and not missing else "invalid", "record_count": len(entries), "missing": missing, "format": "official-diffrhythm-train.scp", "dataset": str(root.resolve())}
    (root / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def official_train_command(repo: str | Path, dataset_dir: str | Path, *, epochs: int = 1, batch_size: int = 1) -> list[str]:
    root = Path(dataset_dir).resolve()
    use_accelerate = False
    try:
        import torch

        use_accelerate = bool(torch.cuda.is_available())
    except ImportError:
        pass
    if use_accelerate:
        accelerate = shutil.which("accelerate")
        launcher = [accelerate, "launch"] if accelerate else [sys.executable, "-m", "accelerate.commands.launch"]
        command = launcher + ["--config-file", str(Path(repo) / "config" / "accelerate_config.yaml"), str(Path(repo) / "train" / "train.py")]
    else:
        command = [sys.executable, str(Path(repo) / "train" / "train.py")]
    command += [
        "--model-config",
        str(root / "diffrhythm-random.json"),
        "--file-path",
        str(root / "train.scp"),
        "--batch-size",
        str(batch_size),
        "--max-frames",
        str(_max_frames_from_config(root)),
        "--min-frames",
        "16",
        "--epochs",
        str(max(1, epochs)),
        "--exp-name",
        "genmusic-diffrhythm-random-smoke",
        "--grad-ckpt",
        "0",
    ]
    if not use_accelerate:
        command += ["--num-workers", "0"]
    return command


def run_official_training(dataset_dir: str | Path, *, repo_path: str | Path | None = None, epochs: int = 1, batch_size: int = 1, timeout_seconds: int = 21_600) -> dict[str, Any]:
    repo = ensure_official_checkout(repo_path)
    validation = validate_official_dataset(dataset_dir)
    if validation["status"] != "valid":
        raise DiffRhythmError("Random/official dataset không hợp lệ, xem validation_report.json.")
    cpu_smoke_patch = _prepare_cpu_smoke_repo(repo)
    command = official_train_command(repo, dataset_dir, epochs=epochs, batch_size=batch_size)
    started = time.perf_counter()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join([str(repo), str(repo / "train"), environment.get("PYTHONPATH", "")])
    try:
        completed = subprocess.run(command, cwd=repo, env=environment, capture_output=True, text=True, timeout=timeout_seconds)
    except FileNotFoundError as exc:
        raise DiffRhythmError("Chưa cài accelerate/torch. Hãy chạy train trên Kaggle với requirements của DiffRhythm.") from exc
    except subprocess.TimeoutExpired as exc:
        report = {
            "backend": "ASLP-lab/DiffRhythm",
            "dataset": str(Path(dataset_dir).resolve()),
            "repo": str(repo),
            "command": command,
            "cpu_smoke_patch": cpu_smoke_patch,
            "returncode": -1,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "stdout_tail": (exc.stdout or "")[-4000:],
            "stderr_tail": (exc.stderr or "")[-4000:],
            "status": "timeout",
        }
        (Path(dataset_dir) / "official_train_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        raise DiffRhythmError(f"Official DiffRhythm training vượt quá giới hạn {timeout_seconds}s; xem official_train_report.json.") from exc
    report = {"backend": "ASLP-lab/DiffRhythm", "dataset": str(Path(dataset_dir).resolve()), "repo": str(repo), "command": command, "cpu_smoke_patch": cpu_smoke_patch, "returncode": completed.returncode, "elapsed_seconds": round(time.perf_counter() - started, 3), "stdout_tail": completed.stdout[-4000:], "stderr_tail": completed.stderr[-4000:], "status": "complete" if completed.returncode == 0 else "failed"}
    destination = Path(dataset_dir) / "official_train_report.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if completed.returncode != 0:
        raise DiffRhythmError(f"Official DiffRhythm training thất bại: {completed.stderr[-1200:]}")
    return report


def _prepare_cpu_smoke_repo(repo: Path) -> bool:
    """Make the upstream trainer usable for a tiny Windows CPU smoke run.

    The official trainer currently hard-codes four persistent DataLoader workers,
    which is appropriate for a GPU environment but can make a one-batch local
    check very expensive on Windows. The checkout is disposable and remains
    untouched when CUDA is available.
    """

    try:
        import torch

        if torch.cuda.is_available():
            return False
    except ImportError:
        return False
    trainer_path = repo / "model" / "trainer.py"
    if not trainer_path.exists():
        return False
    source = trainer_path.read_text(encoding="utf-8")
    patched = source.replace("num_workers=4,", "num_workers=0,").replace("persistent_workers=True", "persistent_workers=False")
    if patched == source:
        return False
    trainer_path.write_text(patched, encoding="utf-8")
    return True


def _max_frames_from_config(dataset_dir: Path) -> int:
    try:
        return int(json.loads((dataset_dir / "random_dataset_report.json").read_text(encoding="utf-8"))["max_frames"])
    except Exception:
        return 64


def write_official_distillation_plan(output_path: str | Path, *, teacher_ref: str = DEFAULT_MODEL_REF, student_steps: int = 4, teacher_steps: int = 32) -> dict[str, Any]:
    """Record a distillation plan without pretending upstream ships a student checkpoint."""

    plan = {"status": "planned", "backend": "ASLP-lab/DiffRhythm", "teacher_ref": teacher_ref, "teacher_steps": teacher_steps, "student_steps": student_steps, "note": "Upstream repository exposes CFM/DiT training; this project does not fabricate a distilled checkpoint. Run distillation on Kaggle with the teacher checkpoint and preference/data artifacts."}
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    return plan
