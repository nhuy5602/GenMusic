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

def _kernel_script_content(dataset_slug: str) -> str:
    return f'''import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
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

    print("--- STEP 1.5: Downloading DiffRhythm2 official repository ---")
    diffrhythm2_tar = "/kaggle/working/diffrhythm2.tar.gz"
    urllib.request.urlretrieve("https://github.com/ASLP-lab/DiffRhythm2/archive/refs/heads/main.tar.gz", diffrhythm2_tar)
    with tarfile.open(diffrhythm2_tar) as tar:
        tar.extractall(str(source_root))
    os.remove(diffrhythm2_tar)

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

    print("--- STEP 3: Running Teacher model inference test ---")
    subprocess.run([
        sys.executable, str(source_root / "scripts/test_teacher_inference.py")
    ], env=os.environ, check=True)

    print("TEACHER INFERENCE TEST COMPLETED SUCCESSFULLY!")
    print("Output song saved at: /kaggle/working/teacher_generated_song.mp3")
except Exception as e:
    import traceback
    print("ERROR OCCURRED DURING KERNEL EXECUTION:")
    traceback.print_exc()
    sys.exit(1)
'''

def run_kaggle_teacher_test() -> None:
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

    run_id = f"tt-{int(time.time())}"
    job_dir = project_root / "outputs" / "kaggle_teacher_test" / run_id
    dataset_dir = job_dir / "dataset"
    kernel_dir = job_dir / "kernel"
    
    for d in (dataset_dir, kernel_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("======================================================================")
    print(f"Initializing Kaggle Teacher Test Job: {run_id}")
    print("======================================================================")

    # 1. Zip source code
    print("Zipping local source code...")
    _write_source_zip(project_root, dataset_dir / "genmusic_vn_source.zip")

    # 2. Upload source code zip as a Kaggle Dataset
    source_dataset_slug = f"genmusic-source-{run_id}"
    source_dataset_ref = f"{username}/{source_dataset_slug}"
    
    (dataset_dir / "dataset-metadata.json").write_text(json.dumps({
        "title": f"GenMusic Source TTest {int(time.time())}",
        "id": source_dataset_ref,
        "licenses": [{"name": "other"}]
    }, indent=2))

    print(f"Uploading source code to Kaggle Dataset '{source_dataset_ref}'...")
    subprocess.run(cli + ["datasets", "create", "-p", str(dataset_dir), "-r", "zip"], env={**os.environ, **tokens}, check=True)

    # Wait until dataset is ready
    print("Waiting for source dataset to be ready...")
    for _ in range(60):
        res = subprocess.run(cli + ["datasets", "status", source_dataset_ref], env={**os.environ, **tokens}, capture_output=True, text=True)
        if "ready" in res.stdout.lower():
            break
        time.sleep(5)

    # 3. Create Kernel script and metadata
    kernel_slug = f"genmusic-teachertest-{int(time.time())}"
    kernel_ref = f"{username}/{kernel_slug}"
    
    raw_dataset_ref = tokens.get("KAGGLE_RAW_DATASET_REF", "sonlest/vietnamese-music-dataset-version3-part6")
    raw_dataset_slug = raw_dataset_ref.split("/")[-1]
    
    (kernel_dir / "run_teachertest.py").write_text(_kernel_script_content(raw_dataset_slug), encoding="utf-8")
    
    (kernel_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": kernel_ref,
        "title": kernel_slug,
        "code_file": "run_teachertest.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [
            source_dataset_ref,
            raw_dataset_ref
        ],
        "kernel_sources": [],
        "competition_sources": []
    }, indent=2))

    # 4. Push Kernel to Kaggle
    print(f"Pushing Teacher Test Kernel to Kaggle: {kernel_ref}...")
    subprocess.run(cli + ["kernels", "push", "-p", str(kernel_dir)], env={**os.environ, **tokens}, check=True)
    
    print("\nTEACHER TEST JOB SUBMITTED SUCCESSFULLY!")
    print("Watch live training logs on Kaggle Web UI:")
    print(f"-> https://www.kaggle.com/code/{username}/{kernel_slug}")

if __name__ == "__main__":
    run_kaggle_teacher_test()
