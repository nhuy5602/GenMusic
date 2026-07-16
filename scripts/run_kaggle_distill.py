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

def _kernel_script_content(epochs: str = "25", batch_size: str = "8", dim: str = "256", depth: str = "4", heads: str = "4", ff_mult: str = "4") -> str:
    return f'''import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

try:
    print("--- STEP 1: Locating preprocessed dataset ---")
    input_dir = Path("/kaggle/input")

    # Find the processed dataset by its actual contents (records.jsonl), not by a
    # guessed folder name -- kernel_sources mounts under the source kernel's slug,
    # not any fixed "vietnamese-music-processed-dataset" name.
    records_file = next(input_dir.rglob("records.jsonl"), None)
    if not records_file:
        raise RuntimeError(f"Could not find the processed dataset in /kaggle/input (looked in {{input_dir}}).")
    processed_dataset = records_file.parent

    print(f"Using processed dataset: {{processed_dataset.resolve()}}")

    print("--- STEP 2: Setting up source code ---")
    source_dataset_dir = next(
        (d for d in input_dir.rglob("*") if d.is_dir() and "genmusic-source-" in d.name.lower()), 
        None
    )
    if not source_dataset_dir:
        raise RuntimeError("Could not find the source code dataset directory.")
        
    source_root = Path("/kaggle/working/GenMusic")
    shutil.copytree(source_dataset_dir, source_root, dirs_exist_ok=True)

    print("--- STEP 2.5: Downloading DiffRhythm2 official repository ---")
    diffrhythm2_tar = "/kaggle/working/diffrhythm2.tar.gz"
    urllib.request.urlretrieve("https://github.com/ASLP-lab/DiffRhythm2/archive/refs/heads/main.tar.gz", diffrhythm2_tar)
    with tarfile.open(diffrhythm2_tar) as tar:
        tar.extractall(str(source_root))
    os.remove(diffrhythm2_tar)

    print("--- STEP 2.8: Installing system packages (espeak-ng) ---")
    subprocess.run(["apt-get", "update", "-y"], check=False)  # ignore mirror sync errors
    subprocess.run(["apt-get", "install", "-y", "--fix-missing", "espeak-ng"], check=True)

    print("--- STEP 3: Installing dependencies ---")
    # First install requirements of DiffRhythm2
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(source_root / "DiffRhythm2-main/requirements.txt")], check=True)
    # Then install additional dependencies
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "muq"], check=True)

    # Add source code and DiffRhythm2-main to PYTHONPATH
    os.environ["PYTHONPATH"] = str(source_root) + os.pathsep + str(source_root / "DiffRhythm2-main") + os.pathsep + os.environ.get("PYTHONPATH", "")

    # Variable-length lyric batches produce many differently-sized CUDA
    # allocations; expandable_segments reduces the fragmentation that caused
    # a real run to OOM on epoch 3 despite epochs 1-2 fitting fine.
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    print("--- STEP 4: Running Knowledge Distillation training ---")
    # Execute the train-distill command. By default it downloads the ASLP-lab/DiffRhythm2 teacher from HF
    subprocess.run([
        sys.executable, str(source_root / "cli.py"), "train-distill",
        "--dataset", str(processed_dataset),
        "--student-checkpoint", "/kaggle/working/distilled_student.pt",
        "--epochs", "{epochs}",
        "--batch-size", "{batch_size}",
        "--learning-rate", "1e-4",
        "--alpha-feature", "0.5",
        "--dim", "{dim}",
        "--depth", "{depth}",
        "--heads", "{heads}",
        "--ff-mult", "{ff_mult}",
    ], env=os.environ, check=True)

    print("DISTILLATION TRAINING COMPLETED SUCCESSFULLY!")
    print("Output model checkpoint saved at: /kaggle/working/distilled_student.pt")
except Exception as e:
    import traceback
    print("ERROR OCCURRED DURING KERNEL EXECUTION:")
    traceback.print_exc()
    sys.exit(1)
'''

def run_kaggle_distillation(epochs: int = 25, batch_size: int = 8, processed_kernel_ref_override: str | None = None, dim: int = 256, depth: int = 4, heads: int = 4, ff_mult: int = 4) -> None:
    project_root = Path(__file__).resolve().parents[1]

    # 0. Load tokens and authenticate
    tokens = kaggle_auth_environment(load_kaggle_api_tokens())
    username = resolve_kaggle_username(tokens.get("KAGGLE_USERNAME"))
    cli = kaggle_cli_command()
    if not username or not kaggle_auth_available(tokens) or not cli:
        raise RuntimeError("Missing KAGGLE_USERNAME or Kaggle auth (KAGGLE_API_TOKEN=KGAT_... / legacy KAGGLE_KEY)")
    
    # Credentials are passed in-memory so project .env remains authoritative.

    run_id = f"distill-run-{int(time.time())}"
    job_dir = project_root / "outputs" / "kaggle_distill" / run_id
    dataset_dir = job_dir / "dataset"
    kernel_dir = job_dir / "kernel"
    
    for d in (dataset_dir, kernel_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("======================================================================")
    print(f"Initializing Kaggle Distillation Job: {run_id}")
    print("======================================================================")

    # 1. Zip source code
    print("Zipping local source code...")
    write_source_zip(project_root, dataset_dir / "genmusic_vn_source.zip")

    # 2. Upload source code zip as a Kaggle Dataset
    source_dataset_slug = f"genmusic-source-{run_id}"
    source_dataset_ref = f"{username}/{source_dataset_slug}"
    
    (dataset_dir / "dataset-metadata.json").write_text(json.dumps({
        "title": f"GenMusic Source Distill {run_id}",
        "id": source_dataset_ref,
        "licenses": [{"name": "other"}]
    }, indent=2))

    print(f"Uploading source code to Kaggle Dataset '{source_dataset_ref}'...")
    try:
        subprocess.run(cli + ["datasets", "create", "-p", str(dataset_dir), "-r", "zip"], env={**os.environ, **tokens}, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[WARNING] Kaggle dataset creation returned an error (often due to transient API gateway issues): {e}. Proceeding anyway...")

    # Wait until dataset is ready
    print("Waiting for source dataset to be ready...")
    time.sleep(20) # sleep initial 20s to allow Kaggle backend to process metadata
    for _ in range(60):
        try:
            res = subprocess.run(cli + ["datasets", "status", source_dataset_ref], env={**os.environ, **tokens}, capture_output=True, text=True, check=False)
            if "ready" in res.stdout.lower():
                break
        except Exception:
            pass
        time.sleep(10)

    # 3. Create Kernel script and metadata
    kernel_slug = f"genmusic-distill-{int(time.time())}"
    kernel_ref = f"{username}/{kernel_slug}"
    
    (kernel_dir / "run_distill.py").write_text(_kernel_script_content(str(epochs), str(batch_size), str(dim), str(depth), str(heads), str(ff_mult)), encoding="utf-8")

    # Processed data source: a preprocess-kernel output (kernel_sources, no credentials
    # needed) takes priority; falls back to a pre-existing published Dataset for
    # compatibility with datasets published before this fix.
    processed_kernel_ref = processed_kernel_ref_override or tokens.get("KAGGLE_PROCESSED_KERNEL_REF")
    processed_dataset_ref = None if processed_kernel_ref else tokens.get(
        "KAGGLE_PROCESSED_DATASET_REF", f"{username}/vietnamese-music-processed-dataset"
    )

    (kernel_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": kernel_ref,
        "title": kernel_slug,
        "code_file": "run_distill.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [source_dataset_ref] + ([processed_dataset_ref] if processed_dataset_ref else []),
        "kernel_sources": [processed_kernel_ref] if processed_kernel_ref else [],
        "competition_sources": []
    }, indent=2))

    # 4. Push Kernel to Kaggle
    print(f"Pushing Distillation Kernel to Kaggle: {kernel_ref}...")
    
    # Retry push up to 3 times with a delay to allow dataset metadata propagation on Kaggle servers
    time.sleep(20)
    for attempt in range(3):
        try:
            subprocess.run(cli + ["kernels", "push", "-p", str(kernel_dir)], env={**os.environ, **tokens}, check=True)
            print("\nDISTILLATION JOB SUBMITTED SUCCESSFULLY!")
            print("Watch live training logs on Kaggle Web UI:")
            print(f"-> https://www.kaggle.com/code/{kernel_ref}")
            break
        except subprocess.CalledProcessError as e:
            if attempt == 2:
                raise e
            print(f"Kaggle kernel push failed on attempt {attempt+1}. Retrying in 15 seconds...", flush=True)
            time.sleep(15)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--processed-kernel-ref", type=str, default=None, help="Override KAGGLE_PROCESSED_KERNEL_REF for this run.")
    parser.add_argument("--dim", type=int, default=256, help="Hidden dim of the MicroDiT student.")
    parser.add_argument("--depth", type=int, default=4, help="Number of transformer blocks.")
    parser.add_argument("--heads", type=int, default=4, help="Number of attention heads.")
    parser.add_argument("--ff-mult", type=int, default=4, help="Feed-forward multiplier.")
    args = parser.parse_args()
    run_kaggle_distillation(args.epochs, args.batch_size, args.processed_kernel_ref, args.dim, args.depth, args.heads, args.ff_mult)

if __name__ == "__main__":
    main()
