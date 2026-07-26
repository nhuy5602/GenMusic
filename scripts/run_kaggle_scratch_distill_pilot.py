"""Bounded pilot: train-distill FROM SCRATCH (fresh randomly-initialized
MicroDiT, no --resume-checkpoint) for a handful of epochs (default 1), reusing
an already-precomputed latent dataset (no re-encoding). Uses --amp +
--num-workers for speed.

Meant to isolate one question the resume-pilot cannot answer cleanly: the
resume-pilot resumes a checkpoint that was fully trained (15 epochs) against a
target later found to be corrupted by a reconstruct_full_mix bug (see
docs/main.tex, Kết luận) -- its weights may already be too entrenched toward
the wrong output distribution for a couple of extra (correct-target) epochs to
visibly move the needle on output collapse. Training fresh from scratch with
the FIXED code removes that confound: if collapse is still just as severe from
step zero, the bug fix alone isn't sufficient to explain/fix it; if a fresh
model shows more diverse output early on, that points at the entrenched-weights
explanation instead.

Source code is fetched by `git clone` + `git checkout <pinned SHA>` straight
from GitHub inside the kernel, instead of zipping the local tree and uploading
it as a fresh Kaggle Dataset every run (the old approach: slow to upload, and
left a redundant full source copy baked into every downloaded kernel output --
see docs/project_history.md). This means the kernel only ever sees committed,
*pushed* code; run_kaggle_scratch_distill_pilot() pushes the current HEAD
before launching and refuses to run with uncommitted local changes.

Distillation (the teacher DiffRhythm2 forward pass) is skipped entirely here,
not just down-weighted: the kernel never clones DiffRhythm2-main or installs
its requirements, so `_load_teacher()` fails its `import diffrhythm2...` and
falls back to teacher=None (pure ground-truth CFM loss, see
src/training/distill_training.py). That forward pass was the dominant per-step
cost (a ~1B-parameter model called every step even under torch.no_grad()), so
skipping it buys far more steps per unit of Kaggle GPU-hour -- useful while the
open question is "does cross-attention/AdaLN-Zero sharpen with more training
steps", which doesn't need the teacher signal to answer.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.integrations.kaggle_auto import (
    kaggle_auth_available,
    kaggle_auth_environment,
    kaggle_cli_command,
    load_kaggle_api_tokens,
    resolve_kaggle_username,
)

GIT_REMOTE_URL = "https://github.com/nhuy5602/GenMusic.git"


def _kernel_script_content(epochs: int, batch_size: int, commit_sha: str) -> str:
    return f'''import json
import os
import subprocess
import sys
from pathlib import Path

try:
    print("--- STEP 1: Locating the precomputed latent dataset ---")
    input_dir = Path("/kaggle/input")
    records_file = next(input_dir.rglob("records.jsonl"), None)
    if not records_file:
        raise RuntimeError("Could not find records.jsonl (precomputed latent dataset) in /kaggle/input.")
    latent_dataset = str(records_file.parent.resolve())
    print(f"Using latent dataset: {{latent_dataset}}")

    print("--- STEP 2: Cloning source code from GitHub (pinned commit {commit_sha}) ---")
    source_root = Path("/kaggle/working/GenMusic")
    subprocess.run(["git", "clone", "{GIT_REMOTE_URL}", str(source_root)], check=True)
    subprocess.run(["git", "checkout", "{commit_sha}"], cwd=str(source_root), check=True)

    print("--- STEP 2.5: Installing system packages (espeak-ng) ---")
    subprocess.run(["apt-get", "update", "-y"], check=False)
    subprocess.run(["apt-get", "install", "-y", "--fix-missing", "espeak-ng"], check=True)

    print("--- STEP 3: Installing python dependencies (no DiffRhythm2/bigvgan -- distillation is skipped) ---")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "librosa", "soundfile", "transformers", "vocos", "muq"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "text2phonemesequence"], check=True)

    os.environ["PYTHONPATH"] = str(source_root) + os.pathsep + os.environ.get("PYTHONPATH", "")

    print("--- STEP 3.5: Checking CUDA compatibility ---")
    subprocess.run([sys.executable, "-c", "import torch; print(f'torch={{torch.__version__}} cuda={{torch.cuda.is_available()}}'); a=torch.randn(2,2,device=\\'cuda\\'); print((a@a))"], check=True)

    print("--- STEP 4: train-distill FROM SCRATCH, no teacher, for {epochs} epoch(s) ---")
    cli = str(source_root / "cli.py")
    student_checkpoint = "/kaggle/working/distilled_student_scratch.pt"
    subprocess.run([
        sys.executable, cli, "train-distill",
        "--dataset", latent_dataset,
        "--student-checkpoint", student_checkpoint,
        "--epochs", "{epochs}",
        "--batch-size", "{batch_size}",
        "--dim", "256",
        "--depth", "4",
        "--heads", "4",
        "--ff-mult", "4",
        "--alpha-feature", "1.0",
        "--learning-rate", "1e-4",
        "--beta-repa", "0.0",
        "--lambda-vocal", "1.0",
        "--log-every-steps", "20",
        "--num-workers", "2",
        "--amp",
        "--device", "cuda",
    ], env=os.environ, check=True)

    print("SCRATCH PILOT COMPLETED SUCCESSFULLY!")
except Exception as e:
    import traceback
    print("ERROR OCCURRED DURING KERNEL EXECUTION:")
    traceback.print_exc()
    sys.exit(1)
'''


def run_kaggle_scratch_distill_pilot(dataset_kernel_ref: str, epochs: int = 1, batch_size: int = 2) -> str:
    project_root = Path(__file__).resolve().parents[1]

    tokens = kaggle_auth_environment(load_kaggle_api_tokens())
    username = resolve_kaggle_username(tokens.get("KAGGLE_USERNAME"))
    cli = kaggle_cli_command()
    if not username or not kaggle_auth_available(tokens) or not cli:
        raise RuntimeError("Missing KAGGLE_USERNAME or Kaggle auth (KAGGLE_API_TOKEN=KGAT_... / legacy KAGGLE_KEY)")

    dirty = subprocess.run(["git", "status", "--short"], cwd=project_root, capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise RuntimeError(
            "Working tree has uncommitted changes -- the Kaggle kernel clones from GitHub, so it would "
            "not see them. Commit first:\n" + dirty
        )
    commit_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=project_root, capture_output=True, text=True, check=True).stdout.strip()
    print(f"Pushing current HEAD ({commit_sha[:10]}) to origin so the kernel can clone it...")
    subprocess.run(["git", "push"], cwd=project_root, check=True)

    run_id = f"scratchpilot-{int(time.time())}"
    job_dir = project_root / "outputs" / "kaggle_scratch_pilot" / run_id
    kernel_dir = job_dir / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Initializing Kaggle Scratch-Distill Pilot Job: {run_id}")
    print("=" * 70)

    kernel_slug = f"genmusic-scratchpilot-{int(time.time())}"
    kernel_ref = f"{username}/{kernel_slug}"

    (kernel_dir / "run_scratch_pilot.py").write_text(_kernel_script_content(epochs, batch_size, commit_sha), encoding="utf-8")

    (kernel_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": kernel_ref,
        "title": kernel_slug,
        "code_file": "run_scratch_pilot.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [],
        "kernel_sources": [dataset_kernel_ref],
        "competition_sources": []
    }, indent=2))

    print(f"Pushing Scratch-Distill Pilot Kernel to Kaggle: {kernel_ref}...")

    for attempt in range(3):
        try:
            subprocess.run(cli + ["kernels", "push", "-p", str(kernel_dir)], env={**os.environ, **tokens}, check=True)
            print("\nSCRATCH-DISTILL PILOT JOB SUBMITTED SUCCESSFULLY!")
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
    parser.add_argument("--dataset-kernel-ref", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    run_kaggle_scratch_distill_pilot(args.dataset_kernel_ref, args.epochs, args.batch_size)


if __name__ == "__main__":
    main()
