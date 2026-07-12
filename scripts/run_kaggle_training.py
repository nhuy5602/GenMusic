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
    # This script will run on the Kaggle GPU instance and log errors to output files instead of crashing
    # We use pure ASCII characters to prevent Windows cp1252 encoding crashes during kaggle push
    return f'''import os
import shutil
import subprocess
import sys
import zipfile
import traceback
from pathlib import Path

# Write directory structure for debugging
try:
    input_dir = Path("/kaggle/input")
    structure = []
    for root, dirs, files in os.walk(str(input_dir)):
        structure.append(f"Folder: {{root}}\\n  Dirs: {{dirs}}\\n  Files: {{files[:20]}}\\n")
    Path("/kaggle/working/dir_structure.txt").write_text("\\n".join(structure), encoding="utf-8")
except Exception as de:
    Path("/kaggle/working/dir_error.txt").write_text(str(de), encoding="utf-8")

try:
    print("--- STEP 1: Locating input datasets ---")
    input_dir = Path("/kaggle/input")
    # Find the raw audio dataset using recursive rglob
    raw_dataset = next(
        (d for d in input_dir.rglob("*") if d.is_dir() and "{dataset_slug}" in d.name.lower()), 
        None
    )
    if not raw_dataset:
        # Fallback to any directory under input/datasets/sonlest/
        raw_dataset = next(
            (d for d in input_dir.rglob("*") if d.is_dir() and "vietnamese-music-dataset" in d.name.lower()), 
            None
        )

    if not raw_dataset:
        raise RuntimeError("Could not find the raw music dataset in /kaggle/input.")

    print(f"Using raw audio dataset: {{raw_dataset.resolve()}}")

    # Find and copy exactly 1 audio file (.mp3 or .wav) to test the pipeline
    audio_file = next(
        (f for f in raw_dataset.rglob("*") if f.is_file() and f.suffix.lower() in (".mp3", ".wav")), 
        None
    )
    if not audio_file:
        raise RuntimeError("No raw audio file found in the dataset.")

    print(f"Selected audio file for 1-file test run: {{audio_file.name}}")
    test_input_dir = Path("/kaggle/working/test_songs")
    test_input_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(audio_file, test_input_dir / audio_file.name)

    print("--- STEP 2: Setting up source code ---")
    # Locate the already unzipped source directory under input_dir
    source_dataset_dir = next(
        (d for d in input_dir.rglob("*") if d.is_dir() and "genmusic-source-" in d.name.lower()), 
        None
    )
    if not source_dataset_dir:
        raise RuntimeError("Could not find the source code dataset directory.")
        
    source_root = Path("/kaggle/working/GenMusic")
    shutil.copytree(source_dataset_dir, source_root, dirs_exist_ok=True)

    print("--- STEP 3: Installing dependencies ---")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch", "torchaudio", "librosa", "matplotlib", "openai-whisper", "demucs", "imageio-ffmpeg", "transformers", "vocos"], check=True)

    # Add source code to path
    os.environ["PYTHONPATH"] = str(source_root) + os.pathsep + os.environ.get("PYTHONPATH", "")

    print("--- STEP 4: Preprocessing the single audio track ---")
    preprocessed_dir = Path("/kaggle/working/preprocessed_dataset")
    subprocess.run([
        sys.executable, str(source_root / "cli.py"), "preprocess-raw",
        "--input", str(test_input_dir),
        "--output", str(preprocessed_dir),
        "--whisper-model", "tiny"
    ], check=True)

    print("--- STEP 5: Training model on the 1-file dataset (1 Epoch) ---")
    checkpoint_path = Path("/kaggle/working/my_trained_model.pt")
    subprocess.run([
        sys.executable, str(source_root / "cli.py"), "train-self",
        "--dataset", str(preprocessed_dir),
        "--checkpoint", str(checkpoint_path),
        "--epochs", "1",
        "--batch-size", "1"
    ], check=True)

    print("--- PIPELINE COMPLETED SUCCESSFULLY ---")
    print(f"Model saved to: {{checkpoint_path.resolve()}}")
    Path("/kaggle/working/success.txt").write_text("success", encoding="utf-8")

except Exception as e:
    tb = traceback.format_exc()
    print("Error occurred during training:")
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
        print("❌ Error: Missing Kaggle credentials. Please configure KAGGLE_USERNAME and KAGGLE_KEY in your .env file.")
        return

    # Set up Kaggle API Token for the new Kaggle CLI format (OAuth/access_token/KAGGLE_API_TOKEN)
    api_token = tokens.get("KAGGLE_KEY")
    tokens["KAGGLE_API_TOKEN"] = api_token
    
    try:
        kaggle_home = Path.home() / ".kaggle"
        kaggle_home.mkdir(exist_ok=True)
        # Write classic kaggle.json just in case
        kaggle_json = kaggle_home / "kaggle.json"
        kaggle_json.write_text(json.dumps({
            "username": tokens["KAGGLE_USERNAME"],
            "key": api_token
        }, indent=2), encoding="utf-8")
        
        # Write new access_token file
        access_token_file = kaggle_home / "access_token"
        access_token_file.write_text(api_token, encoding="utf-8")
        
        try:
            os.chmod(kaggle_json, 0o600)
            os.chmod(access_token_file, 0o600)
        except AttributeError:
            pass
    except Exception as e:
        print(f"⚠️ Warning: Could not write kaggle.json/access_token: {e}")

    raw_dataset_ref = tokens.get("KAGGLE_RAW_DATASET_REF", "sonlest/vietnamese-music-dataset-version3-part6")
    raw_dataset_slug = raw_dataset_ref.split("/")[-1]
    
    run_id = f"train-run-{int(time.time())}"
    job_dir = project_root / "outputs" / "kaggle_training" / run_id
    dataset_dir = job_dir / "dataset"
    kernel_dir = job_dir / "kernel"
    download_dir = job_dir / "downloaded_model"
    
    for d in (dataset_dir, kernel_dir, download_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("======================================================================")
    print(f"🚀 Initializing Kaggle Job: {run_id}")
    print(f"   Raw Dataset: {raw_dataset_ref}")
    print("======================================================================")

    # 1. Zip source code
    print("📦 Zipping local source code...")
    _write_source_zip(project_root, dataset_dir / "genmusic_vn_source.zip")

    # 2. Upload source code zip as a Kaggle Dataset
    source_dataset_slug = f"genmusic-source-{run_id}"
    source_dataset_ref = f"{username}/{source_dataset_slug}"
    
    (dataset_dir / "dataset-metadata.json").write_text(json.dumps({
        "title": f"GenMusic Source {run_id}",
        "id": source_dataset_ref,
        "licenses": [{"name": "other"}]
    }, indent=2))

    print(f"📤 Uploading source code to Kaggle Dataset '{source_dataset_ref}'...")
    subprocess.run(cli + ["datasets", "create", "-p", str(dataset_dir), "-r", "zip"], env={**os.environ, **tokens}, check=True)

    # Wait until dataset is ready on Kaggle
    print("⏳ Waiting for source dataset to be ready...")
    for _ in range(60):
        res = subprocess.run(cli + ["datasets", "status", source_dataset_ref], env={**os.environ, **tokens}, capture_output=True, text=True)
        if "ready" in res.stdout.lower():
            break
        time.sleep(5)

    # 3. Create Kernel script and metadata (Keep slug/title short to fit Kaggle's 50-char limit)
    kernel_slug = f"genmusic-train-{int(time.time())}"
    kernel_ref = f"{username}/{kernel_slug}"
    
    (kernel_dir / "run_training.py").write_text(_kernel_script_content(raw_dataset_slug), encoding="utf-8")
    (kernel_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": kernel_ref,
        "title": kernel_slug,
        "code_file": "run_training.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",       # Enable GPU training
        "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [
            raw_dataset_ref,
            source_dataset_ref
        ]
    }, indent=2))

    # 4. Push Kernel to Kaggle
    print(f"🚀 Pushing training Kernel '{kernel_ref}' to Kaggle (GPU: T4)...")
    subprocess.run(cli + ["kernels", "push", "-p", str(kernel_dir)], env={**os.environ, **tokens}, check=True)

    # 5. Poll Kernel status
    print("⏳ Monitoring training execution status on Kaggle...")
    completed = False
    for _ in range(240): # Poll for up to 40 minutes (Whisper/Demucs on 1 file + GPU train is fast, ~5-8 mins)
        res = subprocess.run(cli + ["kernels", "status", kernel_ref], env={**os.environ, **tokens}, capture_output=True, text=True)
        status_text = res.stdout.strip().lower()
        print(f"   Current Status: {status_text}")
        if "complete" in status_text:
            completed = True
            break
        if "failed" in status_text or "error" in status_text:
            print("❌ Kaggle Kernel training failed.")
            break
        time.sleep(15)

    if completed:
        print("📥 Training complete! Downloading trained model checkpoint...")
        subprocess.run(cli + ["kernels", "output", kernel_ref, "-p", str(download_dir), "-o"], env={**os.environ, **tokens}, check=True)
        checkpoint = download_dir / "my_trained_model.pt"
        if checkpoint.exists():
            # Copy checkpoint to outputs/
            final_checkpoint_path = project_root / "outputs" / "my_trained_model.pt"
            shutil.copy2(checkpoint, final_checkpoint_path)
            print(f"🎉 SUCCESS! Model checkpoint successfully downloaded to: {final_checkpoint_path.resolve()}")
        else:
            print("❌ Error: Checked completed but 'my_trained_model.pt' not found in kernel outputs.")
    else:
        print("❌ Error: Kernel run did not complete successfully.")

if __name__ == "__main__":
    main()
