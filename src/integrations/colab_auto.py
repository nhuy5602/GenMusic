"""Google Colab notebook generation for the full GenMusic training run.

Kaggle remains available as an independent backend. The Colab notebook only
reuses public Kaggle preprocessing artifacts and stores checkpoints on Drive.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_COLAB_NOTEBOOK_URL = (
    "https://colab.research.google.com/drive/"
    "1zGT80eSQdyUjP6rMxY0WWsAo-xEVd8GD?usp=sharing"
)
DEFAULT_REPO_URL = "https://github.com/nhuy5602/GenMusic.git"
DEFAULT_KERNEL_REFS = (
    "ngochuy5602/genmusic-prep-p1-1784095999",
    "ngochuy5602/genmusic-prep-p2-1784096002",
    "ngochuy5602/genmusic-prep-p3-1784107963",
    "ngochuy5602/genmusic-prep-p4-1784108049",
    "ngochuy5602/genmusic-prep-p5-1784120546",
    "ngochuy5602/genmusic-fullexp-1784078352",
)


def _markdown_cell(source: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def _code_cell(source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def build_colab_notebook(
    *,
    colab_url: str = DEFAULT_COLAB_NOTEBOOK_URL,
    repo_url: str = DEFAULT_REPO_URL,
    repo_ref: str = "master",
    kernel_refs: tuple[str, ...] = DEFAULT_KERNEL_REFS,
    expected_records: int = 1843,
    epochs: int = 40,
    batch_size: int = 4,
    cache_data_on_drive: bool = False,
) -> dict[str, Any]:
    config_source = f'''# Colab backend configuration. Kaggle launchers are unchanged.
COLAB_NOTEBOOK_URL = {colab_url!r}
REPO_URL = {repo_url!r}
REPO_REF = {repo_ref!r}
DRIVE_ROOT = "/content/drive/MyDrive/GenMusic"
WORKSPACE = "/content/genmusic_colab"
EXPECTED_RECORDS = {int(expected_records)}
EPOCHS = {int(epochs)}
BATCH_SIZE = {int(batch_size)}
CACHE_DATA_ON_DRIVE = {bool(cache_data_on_drive)!r}
KERNEL_REFS = {list(kernel_refs)!r}
'''
    runtime_source = '''from google.colab import drive
drive.mount("/content/drive")

import shutil
import subprocess

if shutil.which("nvidia-smi") is None:
    raise RuntimeError(
        "Chưa có GPU. Chọn Runtime > Change runtime type > T4 GPU rồi chạy lại."
    )
subprocess.run(["nvidia-smi", "-L"], check=True)
'''
    setup_source = '''import os
import pathlib
import subprocess
import sys

repo_root = pathlib.Path("/content/GenMusic")
if (repo_root / ".git").is_dir():
    subprocess.run(["git", "-C", str(repo_root), "fetch", "origin", REPO_REF], check=True)
    subprocess.run(["git", "-C", str(repo_root), "checkout", REPO_REF], check=True)
    subprocess.run(["git", "-C", str(repo_root), "pull", "--ff-only", "origin", REPO_REF], check=True)
else:
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", REPO_REF, REPO_URL, str(repo_root)],
        check=True,
    )

subprocess.run(
    ["apt-get", "update", "-qq"],
    check=True,
)
subprocess.run(
    ["apt-get", "install", "-y", "-qq", "espeak-ng", "ffmpeg"],
    check=True,
)
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--upgrade",
        "kaggle>=2.0.0",
        "transformers==5.13.1",
        "sentencepiece",
        "librosa",
        "soundfile",
        "vocos",
        "imageio-ffmpeg",
        "matplotlib",
    ],
    check=True,
)
os.environ["PYTHONPATH"] = str(repo_root) + os.pathsep + os.environ.get("PYTHONPATH", "")
print("Source and dependencies are ready:", repo_root)
'''
    auth_source = '''import getpass
import os

try:
    from google.colab import userdata
    kaggle_token = userdata.get("KAGGLE_API_TOKEN")
except Exception:
    kaggle_token = ""

if not kaggle_token:
    kaggle_token = getpass.getpass(
        "Paste KAGGLE_API_TOKEN (KGAT_...). Token is kept only in this runtime: "
    ).strip()
if not kaggle_token.startswith(("KGAT_", "KGAT-")):
    raise ValueError("KAGGLE_API_TOKEN must start with KGAT_ or KGAT-")
os.environ["KAGGLE_API_TOKEN"] = kaggle_token
print("Kaggle token loaded into the current Colab runtime.")
'''
    run_source = '''command = [
    sys.executable,
    str(repo_root / "scripts" / "run_colab_full_training.py"),
    "--workspace",
    WORKSPACE,
    "--drive-root",
    DRIVE_ROOT,
    "--expected-records",
    str(EXPECTED_RECORDS),
    "--epochs",
    str(EPOCHS),
    "--batch-size",
    str(BATCH_SIZE),
]
if CACHE_DATA_ON_DRIVE:
    command.append("--cache-data-on-drive")
for kernel_ref in KERNEL_REFS:
    command.extend(["--kernel", kernel_ref])

print("Starting full Colab pipeline...")
subprocess.run(command, cwd=repo_root, check=True)
'''
    result_source = '''from IPython.display import Audio, display
from pathlib import Path

result_dir = Path(DRIVE_ROOT) / "generated_all_parts"
mp3_files = sorted(result_dir.glob("*.mp3"))
wav_files = sorted(result_dir.glob("*.wav"))
print("Checkpoint:", Path(DRIVE_ROOT) / "checkpoints" / "baseline_all_parts.pt")
print("State:", Path(DRIVE_ROOT) / "colab_training_state.json")
print("Generated files:", [str(path) for path in mp3_files + wav_files])
if mp3_files:
    display(Audio(str(mp3_files[0])))
elif wav_files:
    display(Audio(str(wav_files[0])))
'''
    return {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {
                "name": "GenMusic Full Training - Colab",
                "provenance": [],
                "gpuType": "T4",
            },
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": [
            _markdown_cell(
                "# GenMusic full-dataset training on Google Colab\n\n"
                "Backend này chạy song song với Kaggle, không thay thế các launcher Kaggle. "
                "Notebook tải sáu output preprocessing public, ghép đúng 1.843 record, "
                "train 40 epoch và lưu checkpoint theo từng epoch vào Google Drive.\n\n"
                "Trước khi chạy: chọn **Runtime → Change runtime type → T4 GPU**. "
                "Để tránh paste token mỗi lần, có thể thêm Colab secret "
                "`KAGGLE_API_TOKEN` và bật quyền truy cập notebook."
            ),
            _code_cell(config_source),
            _code_cell(runtime_source),
            _code_cell(setup_source),
            _code_cell(auth_source),
            _code_cell(run_source),
            _code_cell(result_source),
        ],
    }


def write_colab_notebook(
    destination: str | Path,
    **options: Any,
) -> dict[str, Any]:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    notebook = build_colab_notebook(**options)
    path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "status": "created",
        "backend": "google-colab",
        "notebook": str(path.resolve()),
        "colab_url": options.get("colab_url", DEFAULT_COLAB_NOTEBOOK_URL),
        "cell_count": len(notebook["cells"]),
        "kaggle_backend_preserved": True,
    }
