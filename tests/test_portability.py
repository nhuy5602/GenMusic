from __future__ import annotations

import json
from pathlib import Path

from scripts.check_portability import audit_portability
from src.integrations.native_waveform_config import (
    load_native_raw_kernel_ref,
    save_native_raw_kernel_ref,
)


def test_portability_audit_passes_for_submission_tree() -> None:
    root = Path(__file__).resolve().parents[1]
    assert audit_portability(root)["status"] == "passed"


def test_native_kernel_ref_is_local_and_account_agnostic(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = tmp_path / "kaggle.json"
    monkeypatch.setenv("GENMUSIC_CONFIG_PATH", str(path))
    monkeypatch.delenv("GENMUSIC_NATIVE_RAW_KERNEL_REF", raising=False)
    assert load_native_raw_kernel_ref() == ""
    saved = save_native_raw_kernel_ref("new-account/native-corpus")
    assert saved == path
    assert load_native_raw_kernel_ref() == "new-account/native-corpus"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["native_waveform"]["raw_kernel_ref"] == (
        "new-account/native-corpus"
    )
