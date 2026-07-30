"""Submit the final native waveform generation evaluation."""

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

from scripts.kaggle_phase_submit import (
    create_small_dataset,
    new_run_dir,
    require_complete_kernels,
    submit_context,
    submit_phase_kernel,
)
from scripts.master_waveform_pipeline import (
    BASE_SOURCE_DATASET_REF as DEFAULT_SOURCE_REF,
)


PATCH_BUNDLE_NAME = (
    "colab_genmusic_native_waveform_v80_20260730.zip"
)
PATCH_FILES = (
    "scripts/generate_native_waveform.py",
    "scripts/evaluate_generation_quality.py",
    "src/audio/__init__.py",
    "src/audio/native_waveform.py",
    "src/audio/vocal_mix.py",
    "src/audio/waveform_path.py",
    "src/audio/waveform_units.py",
    "src/data/aligned_raw.py",
)
OUTPUT_NAME = "master_raw_connected_pacing_v80_20260730"
STATE_NAME = "master_raw_connected_pacing_v80_state.json"
COLAB_RUNNER = "scripts/generate_native_waveform.py"
DEFAULT_RAW_KERNEL_REF = (
    "ngochuy5602/genmusic-raw-wide-v71-1785338959"
)
MANIFEST_FORMAT = "genmusic-native-waveform-v80"


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
        "format": MANIFEST_FORMAT,
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


def _kernel_code(*, patch_sha256: str, patch_tree_sha256: str) -> str:
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
MANIFEST_FORMAT = {MANIFEST_FORMAT!r}
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
        raise RuntimeError(f"Ambiguous native waveform patches: {{patch_matches}}")
    patch_zip = patch_matches[0]
    actual = hashlib.sha256(patch_zip.read_bytes()).hexdigest().upper()
    if actual != PATCH_SHA256:
        raise RuntimeError(
            "Native waveform patch SHA mismatch "
            f"actual={{actual}} expected={{PATCH_SHA256}}"
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
                raise RuntimeError(
                    f"Unsafe native waveform patch path: {{member.filename!r}}"
                )
        archive.extractall(repo)
else:
    manifests = []
    for candidate in Path("/kaggle/input").rglob("manifest.json"):
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if payload.get("format") == MANIFEST_FORMAT:
            manifests.append(candidate)
    if len(manifests) != 1:
        raise FileNotFoundError(
            "Could not uniquely locate expanded native waveform patch: "
            f"{{manifests}}"
        )
    patch_root = manifests[0].parent
    for relative in PATCH_FILES:
        source = patch_root / PurePosixPath(relative)
        if not source.is_file():
            raise FileNotFoundError(
                f"Missing expanded native waveform file: {{source}}"
            )
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
    raise RuntimeError("Native waveform source tree SHA mismatch")

states = sorted(
    Path("/kaggle/input").rglob("master_raw_multishard_v71_state.json")
)
if len(states) != 1:
    raise RuntimeError(f"Expected one V71 state, found {{states}}")
source_state = json.loads(states[0].read_text(encoding="utf-8"))
if (
    source_state.get("status") != "materialization_passed"
    or not bool((source_state.get("gate") or {{}}).get("pass"))
):
    raise RuntimeError(f"V71 source gate did not pass: {{source_state}}")
dataset = states[0].parent / "dataset"
if not (dataset / "records.jsonl").is_file():
    raise FileNotFoundError(f"V71 dataset missing records: {{dataset}}")

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
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
}})
command = [
    "uv", "run", "python",
    {COLAB_RUNNER!r},
    "--dataset", str(dataset),
    "--output-root", str(output_root),
    "--duration", "16.0",
    "--backing-ratio", "0.45",
    "--presence-attenuation-db", "10.0",
    "--expected-records", "2048",
]
print("STARTING_NATIVE_WAVEFORM_V80:", " ".join(command), flush=True)
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
print("\\n===== NATIVE WAVEFORM V80 =====")
print(json.dumps(state, ensure_ascii=False, indent=2))
if result.returncode != 0:
    print(f"V80_PROCESS_FAILED_BUT_STATE_RETAINED={{result.returncode}}")
shutil.rmtree(repo / ".venv", ignore_errors=True)
shutil.rmtree(working / "hf_cache", ignore_errors=True)
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accelerator", default="NvidiaTeslaT4")
    parser.add_argument("--timeout-seconds", type=int, default=7_200)
    parser.add_argument("--source-ref", default=DEFAULT_SOURCE_REF)
    parser.add_argument("--raw-kernel-ref", default=DEFAULT_RAW_KERNEL_REF)
    args = parser.parse_args()
    context = submit_context()
    require_complete_kernels(context, (args.raw_kernel_ref,))
    timestamp, run_dir = new_run_dir(
        context,
        "native-waveform-v80",
    )
    bundle = (
        context.project_root
        / "outputs"
        / "colab_bundles"
        / PATCH_BUNDLE_NAME
    )
    patch_sha256 = _build_patch_bundle(context.project_root, bundle)
    tree_sha256 = _patch_tree_sha256(context.project_root)
    patch_slug = f"genmusic-native-waveform-v80-a-{timestamp}"
    kernel_slug = f"genmusic-native-waveform-v80-{timestamp}"
    patch_ref = f"{context.username}/{patch_slug}"
    upload_dir = run_dir / "patch_dataset"
    upload_dir.mkdir(parents=True)
    shutil.copy2(bundle, upload_dir / bundle.name)
    create_small_dataset(
        context,
        upload_dir=upload_dir,
        dataset_ref=patch_ref,
        title=f"GenMusic native waveform V80 assets {timestamp}",
        expected_marker=PATCH_BUNDLE_NAME,
    )
    state_path = submit_phase_kernel(
        context,
        phase="native-waveform-v80",
        run_dir=run_dir,
        kernel_slug=kernel_slug,
        code=_kernel_code(
            patch_sha256=patch_sha256,
            patch_tree_sha256=tree_sha256,
        ),
        dataset_sources=[args.source_ref, patch_ref],
        kernel_sources=[args.raw_kernel_ref],
        enable_gpu=True,
        enable_internet=True,
        accelerator=args.accelerator,
        timeout_seconds=args.timeout_seconds,
        state={
            "status": "preparing",
            "training": False,
            "goal_eligible_prediction": True,
            "raw_kernel_ref": args.raw_kernel_ref,
            "patch_ref": patch_ref,
            "patch_sha256": patch_sha256,
            "patch_tree_sha256": tree_sha256,
            "records_expected": 2_048,
            "duration_seconds": 16.0,
            "same_line_pause_cap_seconds": 0.025,
            "line_break_pause_cap_seconds": 2.0,
            "respect_segment_boundaries": True,
            "cross_song_backing": True,
            "per_unit_time_stretch": False,
            "pretrained_tts_used": False,
            "pretrained_asr_used": True,
            "asr_evaluation_only": True,
        },
    )
    print(json.dumps({
        "status": "NATIVE_WAVEFORM_V80_SUBMITTED",
        "kernel_ref": f"{context.username}/{kernel_slug}",
        "patch_ref": patch_ref,
        "patch_sha256": patch_sha256,
        "patch_tree_sha256": tree_sha256,
        "state_path": str(state_path),
    }, indent=2))


if __name__ == "__main__":
    main()
