from __future__ import annotations

import json
from pathlib import Path

from src.data.raw_multishard import (
    raw_materialization_gate,
)
from scripts.materialize_native_waveform_dataset import DEFAULT_SHARDS
from scripts.run_kaggle_native_waveform_dataset import (
    PATCH_FILES,
    _build_patch_bundle,
    _kernel_code,
    _patch_tree_sha256,
)


def test_v71_has_32_unique_balanced_shards() -> None:
    assert len(DEFAULT_SHARDS) == 32
    assert len(set(DEFAULT_SHARDS)) == 32
    assert DEFAULT_SHARDS[0].endswith("00000-of-00063.parquet")
    assert DEFAULT_SHARDS[-1].endswith("00031-of-00063.parquet")


def test_v71_gate_requires_exactly_2048_clean_records() -> None:
    validation = {
        "records": 2_048,
        "unique_songs": 2_048,
        "exact_timestamp_fraction": 1.0,
        "finite_fraction": 1.0,
        "mono_waveform_fraction": 1.0,
        "duration_valid_fraction": 1.0,
        "non_silent_vocal_fraction": 1.0,
        "non_silent_backing_fraction": 1.0,
    }
    assert raw_materialization_gate(
        validation,
        target_records=2_048,
    )["pass"] is True
    validation["records"] = 2_047
    assert raw_materialization_gate(
        validation,
        target_records=2_048,
    )["pass"] is False


def test_v71_bundle_is_token_free_and_requests_2048(
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    bundle = tmp_path / "v71.zip"
    digest = _build_patch_bundle(project_root, bundle)
    tree = _patch_tree_sha256(project_root)
    code = _kernel_code(
        patch_sha256=digest,
        patch_tree_sha256=tree,
        repo_id="owner/repo",
        shards=DEFAULT_SHARDS,
    )
    assert bundle.is_file()
    assert '"--target-records", "2048"' in code
    assert '"--reserve-records", "64"' in code
    assert "STARTING_MASTER_RAW_MULTISHARD_V71" in code
    assert "KGAT_" not in code
    assert "src/data/lyric_quality.py" in PATCH_FILES
    assert ".env" not in PATCH_FILES
    with __import__("zipfile").ZipFile(bundle) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    assert manifest["contains_tokens"] is False
