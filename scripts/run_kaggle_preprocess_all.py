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

def _kernel_script_content(raw_dataset_slug: str, output_dataset_ref: str, kaggle_username: str, kaggle_key: str) -> str:
    # Script runs on Kaggle GPU instance and processes ALL files, then uploads to a new Kaggle dataset
    return f'''import os
import json
import shutil
import subprocess
import sys
import zipfile
import traceback
from pathlib import Path

# Set up Kaggle credentials inside the kernel environment so it can upload
os.environ["KAGGLE_USERNAME"] = "{kaggle_username}"
os.environ["KAGGLE_KEY"] = "{kaggle_key}"
# Disable output buffering to force real-time log printing in Kaggle console
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

    print("--- STEP 3: Installing dependencies ---")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch", "torchaudio", "librosa", "matplotlib", "openai-whisper", "demucs", "imageio-ffmpeg", "kaggle", "transformers", "vocos"], check=True)

    # Add source to python path
    os.environ["PYTHONPATH"] = str(source_root) + os.pathsep + os.environ.get("PYTHONPATH", "")

    print("--- STEP 4: Preprocessing ALL audio tracks ---")
    preprocessed_dir = Path("/kaggle/working/processed_dataset")
    preprocessed_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run([
        sys.executable, str(source_root / "cli.py"), "preprocess-raw",
        "--input", str(raw_dataset),
        "--output", str(preprocessed_dir),
        "--whisper-model", "tiny",
        "--keep-separated-count", "100",
        "--max-files", "100"
    ], env=os.environ, check=True)

    print("--- STEP 5: Creating and Uploading Processed Dataset to Kaggle ---")
    metadata = {{
        "title": "Vietnamese Music Processed Dataset",
        "id": "{output_dataset_ref}",
        "licenses": [{{ "name": "other" }}]
    }}
    with open(preprocessed_dir / "dataset-metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Creating Kaggle Dataset: {output_dataset_ref}...")
    subprocess.run(["kaggle", "datasets", "create", "-p", str(preprocessed_dir), "-r", "zip"], check=True)

    print("--- ALL PROCESSES COMPLETED SUCCESSFULLY ---")
    Path("/kaggle/working/success.txt").write_text("success", encoding="utf-8")

except Exception as e:
    tb = traceback.format_exc()
    print("Error occurred during preprocessing:")
    print(tb)
    Path("/kaggle/working/error.txt").write_text(tb, encoding="utf-8")
'''

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # Resolved parent because it is located inside the scripts/ directory
    project_root = Path(__file__).resolve().parents[1]
    tokens = load_kaggle_api_tokens()
    username = resolve_kaggle_username(tokens.get("KAGGLE_USERNAME"))
    cli = kaggle_cli_command()

    if not username or not tokens.get("KAGGLE_KEY") or not cli:
        print("❌ Error: Missing Kaggle credentials.")
        return

    # Write ~/.kaggle credentials
    try:
        kaggle_home = Path.home() / ".kaggle"
        kaggle_home.mkdir(exist_ok=True)
        
        kaggle_json = kaggle_home / "kaggle.json"
        kaggle_json.write_text(json.dumps({"username": username, "key": tokens["KAGGLE_KEY"]}, indent=2), encoding="utf-8")
        
        access_token_file = kaggle_home / "access_token"
        access_token_file.write_text(tokens["KAGGLE_KEY"], encoding="utf-8")
    except Exception as e:
        print(f"⚠️ Warning: {e}")

    raw_dataset_ref = tokens.get("KAGGLE_RAW_DATASET_REF", "sonlest/vietnamese-music-dataset-version3-part6")
    raw_dataset_slug = raw_dataset_ref.split("/")[-1]
    
    output_dataset_ref = tokens.get("KAGGLE_PROCESSED_DATASET_REF", f"{username}/vietnamese-music-processed-dataset")

    run_id = f"preprocess-all-{int(time.time())}"
    job_dir = project_root / "outputs" / "kaggle_preprocess" / run_id
    dataset_dir = job_dir / "dataset"
    kernel_dir = job_dir / "kernel"
    
    for d in (dataset_dir, kernel_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("======================================================================")
    print(f"🚀 Initializing Preprocess Request: {run_id}")
    print(f"   Source Dataset: {raw_dataset_ref}")
    print(f"   Target Dataset: https://www.kaggle.com/datasets/{output_dataset_ref}")
    print("======================================================================")

    # 1. Zip source code
    _write_source_zip(project_root, dataset_dir / "genmusic_vn_source.zip")

    # 2. Upload source code zip as a Kaggle Dataset
    source_dataset_slug = f"genmusic-source-{run_id}"
    source_dataset_ref = f"{username}/{source_dataset_slug}"
    
    (dataset_dir / "dataset-metadata.json").write_text(json.dumps({
        "title": f"GenMusic Source {run_id}",
        "id": source_dataset_ref,
        "licenses": [{"name": "other"}]
    }, indent=2))

    print(f"📤 Uploading source code to Kaggle...")
    subprocess.run(cli + ["datasets", "create", "-p", str(dataset_dir), "-r", "zip"], env={**os.environ, **tokens}, check=True)

    # Wait until dataset is ready
    print("⏳ Waiting for source dataset to be ready...")
    for _ in range(60):
        res = subprocess.run(cli + ["datasets", "status", source_dataset_ref], env={**os.environ, **tokens}, capture_output=True, text=True)
        if "ready" in res.stdout.lower():
            break
        time.sleep(5)

    # 3. Create Kernel script (Keep slug/title short to fit Kaggle's 50-char limit)
    kernel_slug = f"genmusic-prep-{int(time.time())}"
    kernel_ref = f"{username}/{kernel_slug}"
    
    kernel_script = _kernel_script_content(raw_dataset_slug, output_dataset_ref, username, tokens["KAGGLE_KEY"])
    (kernel_dir / "run_preprocess.py").write_text(kernel_script, encoding="utf-8")
    (kernel_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": kernel_ref,
        "title": kernel_slug,
        "code_file": "run_preprocess.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [
            raw_dataset_ref,
            source_dataset_ref
        ]
    }, indent=2))

    # 4. Push Kernel to Kaggle
    print(f"🚀 Pushing Preprocess Kernel to Kaggle...")
    subprocess.run(cli + ["kernels", "push", "-p", str(kernel_dir)], env={**os.environ, **tokens}, check=True)

    print("\n✅ PREPROCESS REQUEST SUBMITTED SUCCESSFULLY!")
    print(f"Watch live logs on Kaggle Web UI:")
    print(f"➔ https://www.kaggle.com/code/{kernel_ref}")
    print(f"\nThe preprocessed dataset will be uploaded to your Kaggle profile at:")
    print(f"➔ https://www.kaggle.com/datasets/{output_dataset_ref}")

if __name__ == "__main__":
    main()
