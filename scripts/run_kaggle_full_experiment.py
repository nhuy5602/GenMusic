"""Local orchestrator: pushes a single Kaggle GPU kernel that runs the full
experiment (scripts/run_full_experiment.py) end to end -- preprocessing,
vocoder sanity check, baseline training, distillation attempt, generation,
evaluation -- in one kernel to minimize GPU-quota round trips.
"""
import argparse
import json
import os
import sys
import time
import subprocess
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.integrations.kaggle_auto import (
    kaggle_auth_available,
    kaggle_auth_environment,
    kaggle_cli_command,
    load_kaggle_api_tokens,
    resolve_kaggle_username,
    write_source_zip,
)


def _kernel_script_content(raw_dataset_slug: str, max_files: int, whisper_model: str, baseline_epochs: int, distill_epochs: int) -> str:
    return f'''import os
import shutil
import subprocess
import sys
import tarfile
import traceback
import urllib.request
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

try:
    print("--- STEP 1: Locating raw dataset ---")
    input_dir = Path("/kaggle/input")
    raw_dataset = next((d for d in input_dir.rglob("*") if d.is_dir() and "{raw_dataset_slug}" in d.name.lower()), None)
    if not raw_dataset:
        raw_dataset = next((d for d in input_dir.rglob("*") if d.is_dir() and "vietnamese-music-dataset" in d.name.lower()), None)
    if not raw_dataset:
        raise RuntimeError("Could not find the raw music dataset in /kaggle/input.")
    print(f"Raw dataset path: {{raw_dataset.resolve()}}")

    print("--- STEP 2: Setting up GenMusic source code ---")
    source_dataset_dir = next((d for d in input_dir.rglob("*") if d.is_dir() and "genmusic-source-" in d.name.lower()), None)
    source_root = Path("/kaggle/working/GenMusic")
    shutil.copytree(source_dataset_dir, source_root, dirs_exist_ok=True)

    print("--- STEP 3: Downloading DiffRhythm2 (for distillation teacher) ---")
    # Plain tarball download instead of git clone -- skips git history/protocol
    # overhead entirely, much faster on a fresh Kaggle kernel (GPU quota matters).
    diffrhythm2_tar = "/kaggle/working/diffrhythm2.tar.gz"
    urllib.request.urlretrieve("https://github.com/ASLP-lab/DiffRhythm2/archive/refs/heads/main.tar.gz", diffrhythm2_tar)
    with tarfile.open(diffrhythm2_tar) as tar:
        tar.extractall(str(source_root))
    os.remove(diffrhythm2_tar)

    # DiffRhythm2's lyric tokenizer uses phonemizer's eSpeak backend, which is not
    # installed in the default Kaggle image. Install only this missing system package.
    print("--- STEP 3.5: Installing espeak-ng for the lyric tokenizer ---")
    subprocess.run(["apt-get", "update", "-y"], check=False)
    subprocess.run(["apt-get", "install", "-y", "-q", "--no-install-recommends", "espeak-ng"], check=True)

    print("--- STEP 4: Installing dependencies ---")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
        "torch", "torchaudio", "librosa", "matplotlib", "openai-whisper", "demucs",
        "imageio-ffmpeg", "kaggle", "transformers", "vocos", "huggingface_hub", "safetensors"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(source_root / "DiffRhythm2-main/requirements.txt")], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "muq"], check=False)

    os.environ["PYTHONPATH"] = str(source_root) + os.pathsep + str(source_root / "DiffRhythm2-main") + os.pathsep + os.environ.get("PYTHONPATH", "")

    print("--- STEP 5: Running full experiment ---")
    subprocess.run([
        sys.executable, str(source_root / "scripts/run_full_experiment.py"),
        "--raw-dataset", str(raw_dataset),
        "--output-root", "/kaggle/working",
        "--max-files", "{max_files}",
        "--whisper-model", "{whisper_model}",
        "--baseline-epochs", "{baseline_epochs}",
        "--distill-epochs", "{distill_epochs}",
    ], cwd=str(source_root), env=os.environ, check=True)

    print("--- ALL PROCESSES COMPLETED SUCCESSFULLY ---")
    Path("/kaggle/working/success.txt").write_text("success", encoding="utf-8")

except Exception:
    tb = traceback.format_exc()
    print("Error occurred during experiment run:")
    print(tb)
    Path("/kaggle/working/error.txt").write_text(tb, encoding="utf-8")
'''


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-files", type=int, default=12)
    parser.add_argument("--whisper-model", default="tiny")
    parser.add_argument("--baseline-epochs", type=int, default=40)
    parser.add_argument("--distill-epochs", type=int, default=15)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    tokens = kaggle_auth_environment(load_kaggle_api_tokens())
    username = resolve_kaggle_username(tokens.get("KAGGLE_USERNAME"))
    cli = kaggle_cli_command()

    if not username or not kaggle_auth_available(tokens) or not cli:
        print("Error: missing Kaggle credentials.")
        return

    raw_dataset_ref = tokens.get("KAGGLE_RAW_DATASET_REF", "sonlest/vietnamese-music-dataset-version3-part6")
    raw_dataset_slug = raw_dataset_ref.split("/")[-1]

    run_id = f"full-exp-{int(time.time())}"
    job_dir = project_root / "outputs" / "kaggle_full_experiment" / run_id
    dataset_dir = job_dir / "dataset"
    kernel_dir = job_dir / "kernel"
    for d in (dataset_dir, kernel_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Initializing full-experiment Kaggle job: {run_id}")
    print("=" * 70)

    print("Zipping local source code...")
    write_source_zip(project_root, dataset_dir / "genmusic_vn_source.zip")

    source_dataset_slug = f"genmusic-source-{run_id}"
    source_dataset_ref = f"{username}/{source_dataset_slug}"
    (dataset_dir / "dataset-metadata.json").write_text(json.dumps({
        "title": f"GenMusic Source {run_id}", "id": source_dataset_ref, "licenses": [{"name": "other"}],
    }, indent=2))

    print(f"Uploading source code to Kaggle dataset '{source_dataset_ref}'...")
    try:
        subprocess.run(cli + ["datasets", "create", "-p", str(dataset_dir), "-r", "zip"], env={**os.environ, **tokens}, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[WARNING] dataset create returned an error (may be transient): {e}. Proceeding anyway...")

    print("Waiting for source dataset to be ready...")
    time.sleep(20)
    for _ in range(60):
        res = subprocess.run(cli + ["datasets", "status", source_dataset_ref], env={**os.environ, **tokens}, capture_output=True, text=True, check=False)
        if "ready" in res.stdout.lower():
            break
        time.sleep(10)

    kernel_slug = f"genmusic-fullexp-{int(time.time())}"
    kernel_ref = f"{username}/{kernel_slug}"

    (kernel_dir / "run_full_experiment_kernel.py").write_text(
        _kernel_script_content(
            raw_dataset_slug, max_files=args.max_files, whisper_model=args.whisper_model,
            baseline_epochs=args.baseline_epochs, distill_epochs=args.distill_epochs,
        ),
        encoding="utf-8",
    )
    (kernel_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": kernel_ref,
        "title": kernel_slug,
        "code_file": "run_full_experiment_kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [raw_dataset_ref, source_dataset_ref],
    }, indent=2))

    print(f"Pushing full-experiment kernel: {kernel_ref}...")
    time.sleep(10)
    for attempt in range(3):
        try:
            subprocess.run(cli + ["kernels", "push", "-p", str(kernel_dir)], env={**os.environ, **tokens}, check=True)
            break
        except subprocess.CalledProcessError as e:
            if attempt == 2:
                raise e
            print(f"push attempt {attempt+1} failed, retrying in 15s...")
            time.sleep(15)

    state = {
        "run_id": run_id, "kernel_ref": kernel_ref, "job_dir": str(job_dir),
        "download_dir": str(job_dir / "downloaded_output"), "status": "submitted",
        "history": [],
    }
    (job_dir / "job_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")

    print("\nSUBMITTED. Watch logs at:")
    print(f"-> https://www.kaggle.com/code/{kernel_ref}")
    print(f"\nState saved to: {job_dir / 'job_state.json'}")


if __name__ == "__main__":
    main()
