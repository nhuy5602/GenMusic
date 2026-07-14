import json
import os
import shutil
import sys
import time
import zipfile
import subprocess
from pathlib import Path

# Add project root to sys.path to allow imports from src package
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.integrations.kaggle_auto import load_kaggle_api_tokens, resolve_kaggle_username, kaggle_cli_command

def _write_source_zip(project_root: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    excluded = {".git", "outputs", "__pycache__", ".pytest_cache", ".venv", ".kaggle", "dataset", "datasets"}
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in project_root.rglob("*"):
            relative = path.relative_to(project_root)
            if not path.is_file() or any(part in excluded for part in relative.parts):
                continue
            if relative.name.startswith(".env") or relative.name == "kaggle.json":
                continue
            archive.write(path, relative.as_posix())

def _kernel_script_content() -> str:
    return f'''import os
import shutil
import subprocess
import sys
from pathlib import Path

try:
    print("--- STEP 1: Locating preprocessed dataset ---")
    input_dir = Path("/kaggle/input")
    
    # Find the processed dataset
    processed_dataset = next(
        (d for d in input_dir.rglob("*") if d.is_dir() and "vietnamese-music-processed-dataset" in d.name.lower()), 
        None
    )
    if not processed_dataset:
        # Check standard Kaggle path structure
        processed_dataset = input_dir / "vietnamese-music-processed-dataset"

    if not processed_dataset.exists():
        raise RuntimeError(f"Could not find the processed dataset in /kaggle/input (looked in {{input_dir}}).")

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

    print("--- STEP 2.5: Cloning DiffRhythm2 official repository ---")
    subprocess.run(["git", "clone", "https://github.com/ASLP-lab/DiffRhythm2.git", str(source_root / "DiffRhythm2-main")], check=True)

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

    print("--- STEP 4: Running Knowledge Distillation training ---")
    # Execute the train-distill command. By default it downloads the ASLP-lab/DiffRhythm2 teacher from HF
    subprocess.run([
        sys.executable, str(source_root / "cli.py"), "train-distill",
        "--dataset", str(processed_dataset),
        "--student-checkpoint", "/kaggle/working/distilled_student.pt",
        "--epochs", "25",
        "--batch-size", "8",
        "--learning-rate", "1e-4",
        "--alpha-feature", "0.5"
    ], env=os.environ, check=True)

    print("DISTILLATION TRAINING COMPLETED SUCCESSFULLY!")
    print("Output model checkpoint saved at: /kaggle/working/distilled_student.pt")
except Exception as e:
    import traceback
    print("ERROR OCCURRED DURING KERNEL EXECUTION:")
    traceback.print_exc()
    sys.exit(1)
'''

def run_kaggle_distillation() -> None:
    project_root = Path(__file__).resolve().parents[1]
    
    # 0. Load tokens and authenticate
    tokens = load_kaggle_api_tokens()
    username = resolve_kaggle_username(None)
    cli = kaggle_cli_command()
    
    # Force creation of kaggle config folder under USERPROFILE/KHome
    kaggle_home = Path.home() / ".kaggle"
    kaggle_home.mkdir(parents=True, exist_ok=True)
    
    try:
        api_token = tokens["KAGGLE_KEY"]
        kaggle_json = kaggle_home / "kaggle.json"
        kaggle_json.write_text(json.dumps({
            "username": tokens["KAGGLE_USERNAME"],
            "key": api_token
        }, indent=2), encoding="utf-8")
        
        access_token_file = kaggle_home / "access_token"
        access_token_file.write_text(api_token, encoding="utf-8")
    except Exception as e:
        print(f"Warning: Could not write kaggle credentials: {e}")

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
    _write_source_zip(project_root, dataset_dir / "genmusic_vn_source.zip")

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
    
    (kernel_dir / "run_distill.py").write_text(_kernel_script_content(), encoding="utf-8")
    
    # Processed data source: a preprocess-kernel output (kernel_sources, no credentials
    # needed) takes priority; falls back to a pre-existing published Dataset for
    # compatibility with datasets published before this fix.
    processed_kernel_ref = tokens.get("KAGGLE_PROCESSED_KERNEL_REF")
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

if __name__ == "__main__":
    run_kaggle_distillation()
