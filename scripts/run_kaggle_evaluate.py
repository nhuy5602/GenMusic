import argparse
import json
import os
import sys
import time
import subprocess
from pathlib import Path

# Add project root to sys.path to allow imports from src package
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.integrations.kaggle_auto import (
    kaggle_auth_available,
    kaggle_auth_environment,
    kaggle_cli_command,
    load_kaggle_api_tokens,
    resolve_kaggle_username,
    write_source_zip,
)


def _kernel_script_content(max_records: str = "6") -> str:
    return f'''import os
import shutil
import subprocess
import sys
from pathlib import Path

try:
    print("--- STEP 1: Locating processed dataset and checkpoint ---")
    input_dir = Path("/kaggle/input")

    records_file = next(input_dir.rglob("records.jsonl"), None)
    if not records_file:
        raise RuntimeError(f"Could not find the processed dataset in /kaggle/input (looked in {{input_dir}}).")
    processed_dataset = records_file.parent
    print(f"Using processed dataset: {{processed_dataset.resolve()}}")

    checkpoint_candidates = list(input_dir.rglob("distilled_student.pt"))
    if not checkpoint_candidates:
        raise RuntimeError("Could not find distilled_student.pt in /kaggle/input (checkpoint source kernel not mounted?).")
    checkpoint_path = checkpoint_candidates[0]
    print(f"Using checkpoint: {{checkpoint_path.resolve()}}")

    print("--- STEP 2: Setting up source code ---")
    source_dataset_dir = next(
        (d for d in input_dir.rglob("*") if d.is_dir() and "genmusic-source-" in d.name.lower()),
        None
    )
    if not source_dataset_dir:
        raise RuntimeError("Could not find the source code dataset directory.")

    source_root = Path("/kaggle/working/GenMusic")
    shutil.copytree(source_dataset_dir, source_root, dirs_exist_ok=True)

    print("--- STEP 2.5: Installing system packages (espeak-ng) ---")
    subprocess.run(["apt-get", "update", "-y"], check=False)
    subprocess.run(["apt-get", "install", "-y", "--fix-missing", "espeak-ng"], check=True)

    print("--- STEP 3: Installing dependencies ---")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "librosa", "soundfile", "transformers", "vocos", "muq"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "text2phonemesequence"], check=True)

    os.environ["PYTHONPATH"] = str(source_root) + os.pathsep + os.environ.get("PYTHONPATH", "")
    # This kernel session's accelerator reported compute capability (6, 0) (Pascal/P100),
    # for which the preinstalled torch 2.10 (cu128) build has no compiled kernels --
    # torch.cuda.is_available() still returns True (driver-level check only), but any
    # actual kernel launch fails with "no kernel image is available for execution on
    # the device" (confirmed by two real failed runs). This eval only generates 6 short
    # 8s clips through a small (dim=256) student, cheap enough on CPU, so force CPU
    # instead of gambling on GPU assignment/torch-build compatibility.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    print("--- STEP 4: Running quality evaluation ---")
    subprocess.run([
        sys.executable, str(source_root / "scripts/evaluate_generation_quality.py"),
        str(checkpoint_path), str(processed_dataset), "/kaggle/working/eval_out",
        "{max_records}",
    ], env=os.environ, check=True)

    print("EVALUATION COMPLETED SUCCESSFULLY!")
    print("Report saved at: /kaggle/working/eval_out/quality_report.json")
except Exception as e:
    import traceback
    print("ERROR OCCURRED DURING KERNEL EXECUTION:")
    traceback.print_exc()
    sys.exit(1)
'''


def run_kaggle_evaluate(checkpoint_kernel_ref: str, processed_kernel_ref_override: str | None = None, max_records: int = 6) -> None:
    project_root = Path(__file__).resolve().parents[1]

    tokens = kaggle_auth_environment(load_kaggle_api_tokens())
    username = resolve_kaggle_username(tokens.get("KAGGLE_USERNAME"))
    cli = kaggle_cli_command()
    if not username or not kaggle_auth_available(tokens) or not cli:
        raise RuntimeError("Missing KAGGLE_USERNAME or Kaggle auth (KAGGLE_API_TOKEN=KGAT_... / legacy KAGGLE_KEY)")

    run_id = f"eval-{int(time.time())}"
    job_dir = project_root / "outputs" / "kaggle_evaluate" / run_id
    dataset_dir = job_dir / "dataset"
    kernel_dir = job_dir / "kernel"

    for d in (dataset_dir, kernel_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Initializing Kaggle Evaluation Job: {run_id}")
    print("=" * 70)

    print("Zipping local source code...")
    write_source_zip(project_root, dataset_dir / "genmusic_vn_source.zip")

    source_dataset_slug = f"genmusic-source-{run_id}"
    source_dataset_ref = f"{username}/{source_dataset_slug}"

    (dataset_dir / "dataset-metadata.json").write_text(json.dumps({
        "title": f"GenMusic Source Evaluate {run_id}",
        "id": source_dataset_ref,
        "licenses": [{"name": "other"}]
    }, indent=2))

    print(f"Uploading source code to Kaggle Dataset '{source_dataset_ref}'...")
    try:
        subprocess.run(cli + ["datasets", "create", "-p", str(dataset_dir), "-r", "zip"], env={**os.environ, **tokens}, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[WARNING] Kaggle dataset creation returned an error (often due to transient API gateway issues): {e}. Proceeding anyway...")

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

    kernel_slug = f"genmusic-evaluate-{int(time.time())}"
    kernel_ref = f"{username}/{kernel_slug}"

    processed_kernel_ref = processed_kernel_ref_override or tokens.get("KAGGLE_PROCESSED_KERNEL_REF")
    processed_dataset_ref = None if processed_kernel_ref else tokens.get(
        "KAGGLE_PROCESSED_DATASET_REF", f"{username}/vietnamese-music-processed-dataset"
    )

    (kernel_dir / "run_evaluate.py").write_text(_kernel_script_content(str(max_records)), encoding="utf-8")

    (kernel_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": kernel_ref,
        "title": kernel_slug,
        "code_file": "run_evaluate.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [source_dataset_ref] + ([processed_dataset_ref] if processed_dataset_ref else []),
        "kernel_sources": [checkpoint_kernel_ref] + ([processed_kernel_ref] if processed_kernel_ref else []),
        "competition_sources": []
    }, indent=2))

    print(f"Pushing Evaluation Kernel to Kaggle: {kernel_ref}...")

    time.sleep(20)
    for attempt in range(3):
        try:
            subprocess.run(cli + ["kernels", "push", "-p", str(kernel_dir)], env={**os.environ, **tokens}, check=True)
            print("\nEVALUATION JOB SUBMITTED SUCCESSFULLY!")
            print("Watch live logs on Kaggle Web UI:")
            print(f"-> https://www.kaggle.com/code/{kernel_ref}")
            break
        except subprocess.CalledProcessError as e:
            if attempt == 2:
                raise e
            print(f"Kaggle kernel push failed on attempt {attempt+1}. Retrying in 15 seconds...", flush=True)
            time.sleep(15)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-kernel-ref", type=str, required=True, help="Kernel ref (owner/slug) whose output contains distilled_student.pt.")
    parser.add_argument("--processed-kernel-ref", type=str, default=None, help="Override KAGGLE_PROCESSED_KERNEL_REF for this run.")
    parser.add_argument("--max-records", type=int, default=6)
    args = parser.parse_args()
    run_kaggle_evaluate(args.checkpoint_kernel_ref, args.processed_kernel_ref, args.max_records)


if __name__ == "__main__":
    main()
