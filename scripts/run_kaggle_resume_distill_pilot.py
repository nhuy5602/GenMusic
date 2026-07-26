"""Bounded pilot: resume a completed train-distill checkpoint for a handful of
additional epochs (default 3), reusing its own already-precomputed latent
dataset (no re-encoding). Uses --amp + --num-workers for speed. Meant to
answer one question cheaply (within ~1-2 hours): does more training reduce
the near-total conditioning collapse measured on the base checkpoint, before
committing to a full from-scratch retrain (~7-9 hours)?
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


def _kernel_script_content(epochs: int, batch_size: int = 2) -> str:
    return f'''import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

try:
    print("--- STEP 1: Locating checkpoint and its precomputed latent dataset ---")
    input_dir = Path("/kaggle/input")
    checkpoint_candidates = list(input_dir.rglob("distilled_student.pt"))
    if not checkpoint_candidates:
        raise RuntimeError("Could not find distilled_student.pt in /kaggle/input.")
    checkpoint_path = checkpoint_candidates[0]
    print(f"Resuming from checkpoint: {{checkpoint_path.resolve()}}")

    records_file = next(input_dir.rglob("records.jsonl"), None)
    if not records_file:
        raise RuntimeError("Could not find records.jsonl (precomputed latent dataset) in /kaggle/input.")
    latent_dataset = str(records_file.parent.resolve())
    print(f"Using latent dataset: {{latent_dataset}}")

    print("--- STEP 2: Setting up source code ---")
    source_dataset_dir = next(
        (d for d in input_dir.rglob("*") if d.is_dir() and "genmusic-source-" in d.name.lower()),
        None
    )
    if not source_dataset_dir:
        raise RuntimeError("Could not find the source code dataset directory.")
    source_root = Path("/kaggle/working/GenMusic")
    shutil.copytree(source_dataset_dir, source_root, dirs_exist_ok=True)

    print("--- STEP 2.2: Downloading DiffRhythm2 official repository (needed for `bigvgan`) ---")
    diffrhythm2_tar = "/kaggle/working/diffrhythm2.tar.gz"
    urllib.request.urlretrieve("https://github.com/ASLP-lab/DiffRhythm2/archive/refs/heads/main.tar.gz", diffrhythm2_tar)
    with tarfile.open(diffrhythm2_tar) as tar:
        tar.extractall(str(source_root))

    print("--- STEP 2.5: Installing system packages (espeak-ng) ---")
    subprocess.run(["apt-get", "update", "-y"], check=False)
    subprocess.run(["apt-get", "install", "-y", "--fix-missing", "espeak-ng"], check=True)

    print("--- STEP 3: Installing python dependencies ---")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "librosa", "soundfile", "transformers", "vocos", "muq"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "text2phonemesequence"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(source_root / "DiffRhythm2-main/requirements.txt")], check=True)

    os.environ["PYTHONPATH"] = (
        str(source_root) + os.pathsep + str(source_root / "DiffRhythm2-main") + os.pathsep + os.environ.get("PYTHONPATH", "")
    )

    print("--- STEP 3.5: Checking CUDA compatibility ---")
    subprocess.run([sys.executable, "-c", "import torch; print(f'torch={{torch.__version__}} cuda={{torch.cuda.is_available()}}'); a=torch.randn(2,2,device=\\'cuda\\'); print((a@a))"], check=True)

    print("--- STEP 4: Resuming train-distill for {epochs} more epoch(s) ---")
    cli = str(source_root / "cli.py")
    student_checkpoint = "/kaggle/working/distilled_student_resumed.pt"
    subprocess.run([
        sys.executable, cli, "train-distill",
        "--dataset", latent_dataset,
        "--student-checkpoint", student_checkpoint,
        "--resume-checkpoint", str(checkpoint_path),
        "--epochs", "{epochs}",
        "--batch-size", "{batch_size}",
        "--dim", "256",
        "--depth", "4",
        "--heads", "4",
        "--ff-mult", "4",
        "--alpha-feature", "0.8",
        "--learning-rate", "1e-4",
        "--beta-repa", "0.0",
        "--lambda-vocal", "1.0",
        "--log-every-steps", "20",
        "--num-workers", "2",
        "--amp",
        "--device", "cuda",
    ], env=os.environ, check=True)

    print("RESUME PILOT COMPLETED SUCCESSFULLY!")
except Exception as e:
    import traceback
    print("ERROR OCCURRED DURING KERNEL EXECUTION:")
    traceback.print_exc()
    sys.exit(1)
'''


def run_kaggle_resume_distill_pilot(checkpoint_kernel_ref: str, epochs: int = 3, batch_size: int = 2) -> str:
    project_root = Path(__file__).resolve().parents[1]

    tokens = kaggle_auth_environment(load_kaggle_api_tokens())
    username = resolve_kaggle_username(tokens.get("KAGGLE_USERNAME"))
    cli = kaggle_cli_command()
    if not username or not kaggle_auth_available(tokens) or not cli:
        raise RuntimeError("Missing KAGGLE_USERNAME or Kaggle auth (KAGGLE_API_TOKEN=KGAT_... / legacy KAGGLE_KEY)")

    run_id = f"resumepilot-{int(time.time())}"
    job_dir = project_root / "outputs" / "kaggle_resume_pilot" / run_id
    dataset_dir = job_dir / "dataset"
    kernel_dir = job_dir / "kernel"

    for d in (dataset_dir, kernel_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Initializing Kaggle Resume-Distill Pilot Job: {run_id}")
    print("=" * 70)

    print("Zipping local source code...")
    write_source_zip(project_root, dataset_dir / "genmusic_vn_source.zip")

    source_dataset_slug = f"genmusic-source-{run_id}"
    source_dataset_ref = f"{username}/{source_dataset_slug}"

    dataset_title = f"Resume Pilot {run_id}"
    assert 6 <= len(dataset_title) <= 50, f"dataset title length {len(dataset_title)} out of Kaggle's 6-50 range: {dataset_title!r}"
    (dataset_dir / "dataset-metadata.json").write_text(json.dumps({
        "title": dataset_title,
        "id": source_dataset_ref,
        "licenses": [{"name": "other"}]
    }, indent=2))

    print(f"Uploading source code to Kaggle Dataset '{source_dataset_ref}'...")
    try:
        subprocess.run(cli + ["datasets", "create", "-p", str(dataset_dir), "-r", "zip"], env={**os.environ, **tokens}, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[WARNING] Kaggle dataset creation returned an error (often transient): {e}. Proceeding anyway...")

    print("Waiting for source dataset to be ready...")
    time.sleep(20)
    for _ in range(60):
        try:
            res = subprocess.run(cli + ["datasets", "status", source_dataset_ref], env={**os.environ, **tokens}, capture_output=True, text=True, check=False)
            if "ready" in res.stdout.lower():
                break
        except Exception:
            pass
        time.sleep(10)

    kernel_slug = f"genmusic-resumepilot-{int(time.time())}"
    kernel_ref = f"{username}/{kernel_slug}"

    (kernel_dir / "run_resume_pilot.py").write_text(_kernel_script_content(epochs, batch_size), encoding="utf-8")

    (kernel_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": kernel_ref,
        "title": kernel_slug,
        "code_file": "run_resume_pilot.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [source_dataset_ref],
        "kernel_sources": [checkpoint_kernel_ref],
        "competition_sources": []
    }, indent=2))

    print(f"Pushing Resume-Distill Pilot Kernel to Kaggle: {kernel_ref}...")

    time.sleep(20)
    for attempt in range(3):
        try:
            subprocess.run(cli + ["kernels", "push", "-p", str(kernel_dir)], env={**os.environ, **tokens}, check=True)
            print("\nRESUME-DISTILL PILOT JOB SUBMITTED SUCCESSFULLY!")
            print(f"-> https://www.kaggle.com/code/{kernel_ref}")
            return kernel_ref
        except subprocess.CalledProcessError as e:
            if attempt == 2:
                raise e
            print(f"Kaggle kernel push failed on attempt {attempt+1}. Retrying in 15 seconds...", flush=True)
            time.sleep(15)
    return kernel_ref


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-kernel-ref", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    run_kaggle_resume_distill_pilot(args.checkpoint_kernel_ref, args.epochs, args.batch_size)


if __name__ == "__main__":
    main()
