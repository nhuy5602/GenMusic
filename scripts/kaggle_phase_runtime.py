"""Shared runtime helpers for independent Kaggle pipeline phases.

This module is copied through the GenMusic source dataset and imported inside
Kaggle. It deliberately uses only the standard library at import time so the
evaluation phase can install its optional audio dependencies first.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Iterable


WORKING_ROOT = Path("/kaggle/working")
INPUT_ROOT = Path("/kaggle/input")


def run_logged(
    command: list[str],
    label: str,
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    """Stream a command to the Kaggle log and persist the same output."""
    log_path = WORKING_ROOT / f"{label}.log"
    with log_path.open("w", encoding="utf-8") as log_stream:
        process = subprocess.Popen(
            command,
            cwd=str(cwd) if cwd is not None else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        tail: list[str] = []
        for line in process.stdout:
            print(line, end="", flush=True)
            log_stream.write(line)
            log_stream.flush()
            tail.append(line)
            if len(tail) > 300:
                tail.pop(0)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"{label} failed with exit code {return_code}\n{''.join(tail)}"
        )


def install_online_audio_dependencies(
    source_root: Path, *, native_only: bool = False
) -> dict[str, str]:
    """Install only the packages required by checkpoint evaluation."""
    packages = [
        "vocos==0.1.0",
        "encodec==0.1.1",
        "segments==2.4.0",
        "openai-whisper==20250625",
    ]
    if not native_only:
        packages.append("text2phonemesequence==0.1.4")
    run_logged(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            *packages,
        ],
        "install_online_dependencies",
    )
    if not native_only:
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/lingjzhu/CharsiuG2P/main/dicts/vie-c.tsv",
            source_root / "vie-c.tsv",
        )
    environment = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "PYTHONPATH": str(source_root) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    os.environ.update(environment)
    return environment


def gpu_preflight() -> None:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        raise RuntimeError("Kaggle did not allocate a GPU")
    run_logged([nvidia_smi, "-L"], "gpu_hardware")
    probe = (
        "import torch,transformers; "
        "print('torch=' + torch.__version__); "
        "print('transformers=' + transformers.__version__); "
        "assert torch.cuda.is_available(); "
        "print('gpu=' + torch.cuda.get_device_name(0)); "
        "print('capability=' + repr(torch.cuda.get_device_capability())); "
        "print('arches=' + repr(torch.cuda.get_arch_list())); "
        "print('cuda_smoke=' + repr(torch.rand(1, device='cuda').cpu().tolist()))"
    )
    run_logged([sys.executable, "-c", probe], "gpu_preflight")


def build_combined_dataset(*, source_count: int, expected_records: int) -> Path:
    """Combine completed per-part preprocess outputs without copying mel data."""
    records_paths = sorted(
        path
        for path in INPUT_ROOT.rglob("records.jsonl")
        if "genmusic-source-" not in str(path).lower()
    )
    if len(records_paths) != source_count:
        raise RuntimeError(
            f"Expected {source_count} processed inputs, found {len(records_paths)}: "
            f"{[str(path) for path in records_paths]!r}"
        )

    combined_root = WORKING_ROOT / "combined_dataset"
    combined_mels = combined_root / "mels"
    combined_mels.mkdir(parents=True, exist_ok=True)
    combined_records: list[dict] = []
    source_counts: list[dict] = []
    required_fields = ("backing_mel_path", "vocal_mel_path", "style_embed_path")
    for source_index, records_path in enumerate(records_paths, start=1):
        source_dir = records_path.parent
        records = [
            json.loads(line)
            for line in records_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        source_counts.append({"source": str(source_dir), "records": len(records)})
        for record_index, record in enumerate(records, start=1):
            record["id"] = "source%02d_%s" % (
                source_index,
                record.get("id", record_index),
            )
            for field in required_fields:
                relative_path = record.get(field)
                source_file = source_dir / relative_path if relative_path else None
                if source_file is None or not source_file.is_file():
                    raise FileNotFoundError(
                        f"Missing {field} for {record['id']}: {source_file}"
                    )
                destination = combined_mels / (
                    "source%02d_%s" % (source_index, source_file.name)
                )
                if not destination.exists():
                    os.symlink(source_file, destination)
                record[field] = "mels/" + destination.name
            combined_records.append(record)

    if len(combined_records) != expected_records:
        raise RuntimeError(
            f"Expected {expected_records} combined records, found {len(combined_records)}"
        )
    shutil.copy2(records_paths[0].parent / "config.json", combined_root / "config.json")
    (combined_root / "records.jsonl").write_text(
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n"
            for record in combined_records
        ),
        encoding="utf-8",
    )
    (WORKING_ROOT / "combined_summary.json").write_text(
        json.dumps(
            {
                "expected_records": expected_records,
                "combined_records": len(combined_records),
                "sources": source_counts,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return combined_root


def find_checkpoint() -> Path:
    """Choose the explicit final checkpoint, never a preflight/best surrogate."""
    candidates = [
        path
        for path in INPUT_ROOT.rglob("self_all_parts*.pt")
        if path.is_file() and "preflight" not in path.name.casefold()
    ]
    if not candidates:
        raise FileNotFoundError("No mounted self_all_parts checkpoint was found")
    final_candidates = [path for path in candidates if path.name == "self_all_parts.pt"]
    return max(final_candidates or candidates, key=lambda path: path.stat().st_size)


def find_json_reports(filename: str) -> Iterable[tuple[Path, dict]]:
    for path in INPUT_ROOT.rglob(filename):
        if not path.is_file():
            continue
        try:
            yield path, json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
