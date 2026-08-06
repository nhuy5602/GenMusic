"""Account-local configuration for the Kaggle native-waveform corpus."""

from __future__ import annotations

import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / ".genmusic" / "kaggle.json"
RAW_KERNEL_ENV = "GENMUSIC_NATIVE_RAW_KERNEL_REF"
CONFIG_PATH_ENV = "GENMUSIC_CONFIG_PATH"


def config_path() -> Path:
    override = os.environ.get(CONFIG_PATH_ENV, "").strip()
    return Path(override).expanduser() if override else DEFAULT_CONFIG_PATH


def load_native_raw_kernel_ref(explicit: str | None = None) -> str:
    """Resolve a corpus ref without embedding any developer account in Git."""
    if explicit and explicit.strip():
        return explicit.strip()
    environment = os.environ.get(RAW_KERNEL_ENV, "").strip()
    if environment:
        return environment
    path = config_path()
    if not path.is_file():
        return ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return ""
    native = payload.get("native_waveform") or {}
    return str(native.get("raw_kernel_ref") or "").strip()


def save_native_raw_kernel_ref(ref: str) -> Path:
    """Persist the caller's own Kaggle kernel ref in an ignored local file."""
    resolved = ref.strip()
    if "/" not in resolved:
        raise ValueError("Kaggle kernel ref must use owner/slug form")
    path = config_path()
    payload: dict = {}
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            payload = {}
    payload["native_waveform"] = {"raw_kernel_ref": resolved}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def setup_message() -> str:
    return (
        "Native waveform corpus is not configured. Run `uv run python "
        "scripts/run_kaggle_native_waveform_dataset.py`, wait for that Kaggle "
        "kernel to complete, then retry. The generated ref is stored only in "
        "`.genmusic/kaggle.json`."
    )
