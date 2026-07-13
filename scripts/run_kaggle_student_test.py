import json
import os
import shutil
import sys
import time
import zipfile
import subprocess
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.integrations.kaggle_auto import load_kaggle_api_tokens, resolve_kaggle_username, kaggle_cli_command
from scripts.run_kaggle_distill import _write_source_zip

def _kernel_script_content(dataset_slug: str) -> str:
    return f'''import os
import shutil
import subprocess
import sys
from pathlib import Path

try:
    print("--- STEP 1: Setting up source code ---")
    input_dir = Path("/kaggle/input")
    source_dataset_dir = next(
        (d for d in input_dir.rglob("*") if d.is_dir() and "genmusic-source-" in d.name.lower()), 
        None
    )
    if not source_dataset_dir:
        raise RuntimeError("Could not find the source code dataset directory.")
        
    source_root = Path("/kaggle/working/GenMusic")
    shutil.copytree(source_dataset_dir, source_root, dirs_exist_ok=True)

    print("--- STEP 1.5: Cloning DiffRhythm2 official repository ---")
    subprocess.run(["git", "clone", "https://github.com/ASLP-lab/DiffRhythm2.git", str(source_root / "DiffRhythm2-main")], check=True)

    print("--- STEP 1.8: Installing system packages (espeak-ng) ---")
    # Skip apt-get update to avoid NVIDIA mirror sync failures; use cached package lists
    import subprocess as _sp
    _sp.run(["apt-get", "install", "-y", "-q", "--no-install-recommends", "espeak-ng"], check=False)

    print("--- STEP 2: Installing dependencies ---")
    # First install official requirements.txt of DiffRhythm2
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(source_root / "DiffRhythm2-main/requirements.txt")], check=True)
    # Then install additional test dependencies
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pedalboard", "muq"], check=True)

    # Add source code to path
    os.environ["PYTHONPATH"] = str(source_root) + os.pathsep + os.environ.get("PYTHONPATH", "")

    # Locate distilled student checkpoint from input if available
    print("--- STEP 2.5: Locating student checkpoint ---")
    checkpoint_candidates = list(input_dir.rglob("distilled_student.pt"))
    if checkpoint_candidates:
        print(f"Found student checkpoint in inputs: {{checkpoint_candidates[0]}}")
        # Copy to expected output working dir
        shutil.copy(checkpoint_candidates[0], "/kaggle/working/distilled_student.pt")
    else:
        print("Student checkpoint not found in inputs. Attempting to copy from sister output directory if mounted...")

    print("--- STEP 3: Running Student model inference test ---")
    subprocess.run([
        sys.executable, str(source_root / "scripts/test_student_inference.py")
    ], env=os.environ, check=True)

    print("STUDENT INFERENCE TEST COMPLETED SUCCESSFULLY!")
    print("Output song saved at: /kaggle/working/student_generated_song.mp3")
except Exception as e:
    import traceback
    print("ERROR OCCURRED DURING KERNEL EXECUTION:")
    traceback.print_exc()
    sys.exit(1)
'''

def run_kaggle_student_test():
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
    except Exception as e:
        print(f"Warning: Could not write kaggle credentials: {e}")
        
    run_id = f"st-{int(time.time())}"
    job_dir = project_root / "outputs" / "kaggle_student_test" / run_id
    dataset_dir = job_dir / "dataset"
    kernel_dir = job_dir / "kernel"
    
    for d in (dataset_dir, kernel_dir):
        d.mkdir(parents=True, exist_ok=True)
        
    print("=" * 70)
    print(f"Initializing Kaggle Student Test Job: {run_id}")
    print("=" * 70)
    
    # 1. Zip source code using standard helper
    print("Zipping local source code...")
    _write_source_zip(project_root, dataset_dir / "genmusic_vn_source.zip")
    
    # 2. Upload source code zip as a Kaggle Dataset
    source_dataset_slug = f"genmusic-source-{run_id}"
    source_dataset_ref = f"{username}/{source_dataset_slug}"
    
    (dataset_dir / "dataset-metadata.json").write_text(json.dumps({
        "title": f"GenMusic Source Student Test {run_id}",
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
    kernel_slug = f"genmusic-studenttest-{int(time.time())}"
    kernel_ref = f"{username}/{kernel_slug}"
    
    # Reference the successful distillation run outputs
    distill_output_kernel = "mvitanh/genmusic-distill-1783963847"
    
    kernel_meta = {
        "id": kernel_ref,
        "title": kernel_slug,
        "code_file": "__script__.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [
            source_dataset_ref,
            "mvitanh/vietnamese-music-processed-dataset"
        ],
        "kernel_sources": [
            distill_output_kernel
        ]
    }
    
    (kernel_dir / "kernel-metadata.json").write_text(json.dumps(kernel_meta, indent=2))
    (kernel_dir / "__script__.py").write_text(_kernel_script_content(source_dataset_slug), encoding="utf-8")
    
    print(f"Pushing Student Test Kernel to Kaggle: {kernel_ref}...")
    
    # Retry push up to 3 times with a delay to allow dataset metadata propagation on Kaggle servers
    time.sleep(20)
    for attempt in range(3):
        try:
            subprocess.run(cli + ["kernels", "push", "-p", str(kernel_dir)], env={**os.environ, **tokens}, check=True)
            print("\nSTUDENT TEST JOB SUBMITTED SUCCESSFULLY!")
            print("Watch live testing logs on Kaggle Web UI:")
            print(f"-> https://www.kaggle.com/code/{kernel_ref}")
            break
        except subprocess.CalledProcessError as e:
            if attempt == 2:
                raise e
            print(f"Kaggle kernel push failed on attempt {attempt+1}. Retrying in 15 seconds...", flush=True)
            time.sleep(15)

if __name__ == "__main__":
    run_kaggle_student_test()
