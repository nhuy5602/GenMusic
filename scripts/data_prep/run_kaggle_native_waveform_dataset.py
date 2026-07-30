"""Submit native waveform dataset materialization to Kaggle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.kaggle.phase_submit import (
    create_small_dataset,
    new_run_dir,
    submit_context,
    submit_phase_kernel,
)
from scripts.data_prep.materialize_native import DEFAULT_SHARDS
from scripts.generation.master_waveform_pipeline import (
    BASE_SOURCE_DATASET_REF as DEFAULT_SOURCE_REF,
)
from src.data.preprocess_aligned_vietnamese import DEFAULT_REPO_ID


PATCH_BUNDLE_NAME = (
    "colab_genmusic_master_raw_multishard_v71_20260729.zip"
)
PATCH_FILES = (
    "scripts/data_prep/materialize_native.py",
    "src/data/lyric_quality.py",
    "src/data/aligned_raw.py",
    "src/data/preprocess_aligned_vietnamese.py",
    "src/data/preprocess_raw_vietnamese.py",
    "src/data/raw_multishard.py",
)
OUTPUT_NAME = "master_raw_multishard_v71_20260729"
STATE_NAME = "master_raw_multishard_v71_state.json"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _patch_tree_sha256(project_root: Path) -> str:
    digest = hashlib.sha256()
    for relative in sorted(PATCH_FILES):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update((project_root / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest().upper()


def _build_patch_bundle(project_root: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    manifest = {
        "format": "genmusic-master-raw-multishard-v71",
        "files": list(PATCH_FILES),
        "contains_tokens": False,
    }
    with zipfile.ZipFile(
        temporary,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for relative in PATCH_FILES:
            archive.writestr(
                PurePosixPath(relative).as_posix(),
                (project_root / relative).read_bytes(),
            )
        archive.writestr(
            "manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
        )
    temporary.replace(destination)
    return _file_sha256(destination)


def _kernel_code(
    *,
    patch_sha256: str,
    patch_tree_sha256: str,
    repo_id: str,
    shards: tuple[str, ...],
) -> str:
    shard_arguments = "\n".join(
        f'    "--shard", {shard!r},'
        for shard in shards
    )
    return f'''from pathlib import Path, PurePosixPath
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile

PATCH_NAME = {PATCH_BUNDLE_NAME!r}
PATCH_SHA256 = {patch_sha256!r}
PATCH_TREE_SHA256 = {patch_tree_sha256!r}
PATCH_FILES = {list(PATCH_FILES)!r}
working = Path("/kaggle/working")
output_root = working / {OUTPUT_NAME!r}
output_root.mkdir(parents=True, exist_ok=True)

source_runners = sorted(
    Path("/kaggle/input/datasets").rglob(
        "run_colab_master_segment_quality_recovery.py"
    )
)
if len(source_runners) != 1:
    raise RuntimeError(f"Expected one source tree, found {{source_runners}}")
source_repo = source_runners[0].parent.parent
repo = working / "GenMusic"
if repo.exists():
    shutil.rmtree(repo)
shutil.copytree(
    source_repo,
    repo,
    ignore=shutil.ignore_patterns(".venv", "__pycache__", "*.pyc", "*.pyo"),
)

patch_matches = sorted(Path("/kaggle/input").rglob(PATCH_NAME))
if patch_matches:
    if len(patch_matches) != 1:
        raise RuntimeError(f"Ambiguous V71 patch zips: {{patch_matches}}")
    patch_zip = patch_matches[0]
    actual = hashlib.sha256(patch_zip.read_bytes()).hexdigest().upper()
    if actual != PATCH_SHA256:
        raise RuntimeError(
            f"V71 patch SHA mismatch actual={{actual}} expected={{PATCH_SHA256}}"
        )
    with zipfile.ZipFile(patch_zip) as archive:
        for member in archive.infolist():
            normalized = member.filename.replace("\\\\", "/")
            path = PurePosixPath(normalized)
            if (
                member.filename != normalized
                or path.is_absolute()
                or ".." in path.parts
            ):
                raise RuntimeError(f"Unsafe V71 patch path: {{member.filename!r}}")
        archive.extractall(repo)
else:
    manifests = []
    for candidate in Path("/kaggle/input").rglob("manifest.json"):
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if payload.get("format") == "genmusic-master-raw-multishard-v71":
            manifests.append(candidate)
    if len(manifests) != 1:
        raise FileNotFoundError(
            f"Could not uniquely locate expanded V71 patch: {{manifests}}"
        )
    patch_root = manifests[0].parent
    for relative in PATCH_FILES:
        source = patch_root / PurePosixPath(relative)
        if not source.is_file():
            raise FileNotFoundError(f"Missing expanded V71 file: {{source}}")
        destination = repo / PurePosixPath(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

tree_digest = hashlib.sha256()
for relative in sorted(PATCH_FILES):
    source = repo / PurePosixPath(relative)
    tree_digest.update(relative.encode("utf-8"))
    tree_digest.update(b"\\0")
    tree_digest.update(source.read_bytes())
    tree_digest.update(b"\\0")
if tree_digest.hexdigest().upper() != PATCH_TREE_SHA256:
    raise RuntimeError("V71 extracted source tree SHA mismatch")

try:
    subprocess.run(["uv", "--version"], check=True, capture_output=True)
except (FileNotFoundError, subprocess.CalledProcessError):
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "uv"],
        check=True,
    )
subprocess.run(["uv", "sync"], cwd=repo, check=True)
environment = os.environ.copy()
environment.update({{
    "PYTHONPATH": str(repo),
    "PYTHONUTF8": "1",
    "PYTHONIOENCODING": "utf-8",
    "PYTHONUNBUFFERED": "1",
    "LC_ALL": "C.UTF-8",
    "HF_HOME": str(working / "hf_cache"),
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}})
command = [
    "uv", "run", "--with", "pyarrow", "python",
    "scripts/materialize_native_waveform_dataset.py",
    "--repo-id", {repo_id!r},
    "--output-root", str(output_root),
    "--target-records", "2048",
    "--records-per-shard", "64",
    "--candidate-records-per-shard", "128",
    "--reserve-records", "64",
    "--batch-size", "8",
    "--demucs-device", "cuda",
{shard_arguments}
]
print("STARTING_MASTER_RAW_MULTISHARD_V71:", " ".join(command), flush=True)
result = subprocess.run(command, cwd=repo, env=environment, check=False)
state_path = output_root / {STATE_NAME!r}
if state_path.exists():
    state = json.loads(state_path.read_text(encoding="utf-8"))
else:
    state = {{"status": "failed_without_state", "returncode": result.returncode}}
    state_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
print("\\n===== MASTER RAW MULTISHARD V71 =====")
print(json.dumps(state, ensure_ascii=False, indent=2))
if result.returncode != 0:
    print(f"V71_PROCESS_FAILED_BUT_STATE_RETAINED={{result.returncode}}")
shutil.rmtree(repo / ".venv", ignore_errors=True)
shutil.rmtree(working / "hf_cache", ignore_errors=True)
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accelerator", default="NvidiaTeslaT4")
    parser.add_argument("--timeout-seconds", type=int, default=14_400)
    parser.add_argument("--source-ref", default=DEFAULT_SOURCE_REF)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--shard", action="append", dest="shards")
    args = parser.parse_args()
    shards = tuple(args.shards or DEFAULT_SHARDS)
    if len(shards) != 32 or len(set(shards)) != 32:
        raise ValueError("V71 requires exactly 32 unique shards")

    context = submit_context()
    timestamp, run_dir = new_run_dir(
        context,
        "master-raw-multishard-v71",
    )
    bundle = (
        context.project_root
        / "outputs"
        / "colab_bundles"
        / PATCH_BUNDLE_NAME
    )
    patch_sha256 = _build_patch_bundle(context.project_root, bundle)
    tree_sha256 = _patch_tree_sha256(context.project_root)
    patch_slug = f"genmusic-raw-wide-v71-a-{timestamp}"
    kernel_slug = f"genmusic-raw-wide-v71-{timestamp}"
    patch_ref = f"{context.username}/{patch_slug}"
    upload_dir = run_dir / "patch_dataset"
    upload_dir.mkdir(parents=True)
    shutil.copy2(bundle, upload_dir / bundle.name)
    create_small_dataset(
        context,
        upload_dir=upload_dir,
        dataset_ref=patch_ref,
        title=f"GenMusic raw wide V71 assets {timestamp}",
        expected_marker=PATCH_BUNDLE_NAME,
    )
    state_path = submit_phase_kernel(
        context,
        phase="master-raw-multishard-v71",
        run_dir=run_dir,
        kernel_slug=kernel_slug,
        code=_kernel_code(
            patch_sha256=patch_sha256,
            patch_tree_sha256=tree_sha256,
            repo_id=args.repo_id,
            shards=shards,
        ),
        dataset_sources=[args.source_ref, patch_ref],
        kernel_sources=[],
        enable_gpu=True,
        enable_internet=True,
        accelerator=args.accelerator,
        timeout_seconds=args.timeout_seconds,
        state={
            "status": "preparing",
            "training": False,
            "generator_training": False,
            "data_gate_only": True,
            "repo_id": args.repo_id,
            "shards": list(shards),
            "target_records": 2_048,
            "patch_ref": patch_ref,
            "patch_sha256": patch_sha256,
            "patch_tree_sha256": tree_sha256,
            "pretrained_tts_used": False,
            "pretrained_asr_used": False,
            "pretrained_source_separator_used": True,
        },
    )
    print(json.dumps({
        "status": "MASTER_RAW_MULTISHARD_V71_SUBMITTED",
        "kernel_ref": f"{context.username}/{kernel_slug}",
        "patch_ref": patch_ref,
        "patch_sha256": patch_sha256,
        "patch_tree_sha256": tree_sha256,
        "state_path": str(state_path),
    }, indent=2))


if __name__ == "__main__":
    main()
