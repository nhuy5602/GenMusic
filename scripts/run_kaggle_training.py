import argparse
import json
import os
import shutil
import sys
import time
import subprocess
from pathlib import Path

# Add project root to sys.path to allow imports from src package
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.integrations.kaggle_auto import load_kaggle_api_tokens, resolve_kaggle_username, kaggle_cli_command, write_source_zip

def _kernel_script_content(dataset_slug: str, epochs: str = "5", batch_size: str = "4") -> str:
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
    processed_dataset = next(input_dir.rglob("records.jsonl"), None)
    if not processed_dataset or not processed_dataset.is_file():
        raise RuntimeError("Could not find processed dataset records.jsonl in /kaggle/input.")
    processed_dataset = processed_dataset.parent
    print(f"Using processed dataset: {{processed_dataset.resolve()}}")

    print("--- STEP 2: Setting up source code ---")
    # Locate source code either as a mounted directory or as the source zip.
    source_dataset_dir = next(
        (d for d in input_dir.rglob("*") if d.is_dir() and "genmusic-source-" in d.name.lower()), 
        None
    )
    source_root = Path("/kaggle/working/GenMusic")
    source_zip = next(input_dir.rglob("genmusic_vn_source.zip"), None)
    if source_zip and source_zip.is_file():
        source_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(source_zip) as archive:
            archive.extractall(source_root)
    elif source_dataset_dir:
        shutil.copytree(source_dataset_dir, source_root, dirs_exist_ok=True)
    else:
        raise RuntimeError("Could not find the source code dataset directory or zip.")

    print("--- STEP 3: Installing dependencies ---")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "librosa", "imageio-ffmpeg"], check=True)

    print("--- STEP 4: Checking CUDA compatibility ---")
    torch_probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch; print('torch=%s cuda=%s available=%s' % (torch.__version__, torch.version.cuda, torch.cuda.is_available())); print(torch.randn((2, 2), device='cuda') @ torch.randn((2, 2), device='cuda')) if torch.cuda.is_available() else None",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    torch_probe_output = (torch_probe.stdout or "") + chr(10) + (torch_probe.stderr or "")
    print(torch_probe_output, flush=True)
    if torch_probe.returncode != 0:
        print("CUDA smoke test failed; installing P100-compatible Torch.", flush=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-cache-dir",
                "--force-reinstall",
                "--extra-index-url",
                "https://download.pytorch.org/whl/cu121",
                "torch==2.5.1+cu121",
                "torchaudio==2.5.1+cu121",
            ],
            check=True,
        )
        repaired_probe = subprocess.run(
            [
                sys.executable,
                "-c",
                "import torch; print('torch=%s cuda=%s available=%s' % (torch.__version__, torch.version.cuda, torch.cuda.is_available())); print(torch.randn((2, 2), device='cuda') @ torch.randn((2, 2), device='cuda')) if torch.cuda.is_available() else None",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        repaired_output = (repaired_probe.stdout or "") + chr(10) + (repaired_probe.stderr or "")
        print(repaired_output, flush=True)
        if repaired_probe.returncode != 0 or "available=True" not in repaired_output:
            raise RuntimeError("CUDA is unavailable after P100 Torch repair.")

    # Add source code to path
    os.environ["PYTHONPATH"] = str(source_root) + os.pathsep + os.environ.get("PYTHONPATH", "")

    print("--- STEP 5: Training model on ALL processed records ---")
    checkpoint_path = Path("/kaggle/working/my_trained_model.pt")
    subprocess.run([
        sys.executable, str(source_root / "cli.py"), "train-self",
        "--dataset", str(processed_dataset),
        "--checkpoint", str(checkpoint_path),
        "--epochs", "{epochs}",
        "--batch-size", "{batch_size}",
        "--device", "cuda"
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

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
    kaggle_env = {
        **os.environ,
        **tokens,
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
    }
    
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

    # KAGGLE_PROCESSED_KERNEL_REF: output of a preprocess kernel run with the fixed
    # run_kaggle_preprocess_all.py (attached via kernel_sources, no credentials needed).
    # KAGGLE_PROCESSED_DATASET_REF: a pre-existing published Dataset (attached via
    # dataset_sources instead) -- kept for compatibility with datasets published before
    # this fix, or shared manually outside this project's scripts.
    processed_kernel_ref = os.getenv("KAGGLE_PROCESSED_KERNEL_REF") or tokens.get("KAGGLE_PROCESSED_KERNEL_REF")
    processed_dataset_ref = os.getenv("KAGGLE_PROCESSED_DATASET_REF") or tokens.get(
        "KAGGLE_PROCESSED_DATASET_REF",
        "ngochuy5602/genmusic-vn-part3-vocal-vocos-smoke" if not processed_kernel_ref else None,
    )
    processed_dataset_slug = (processed_kernel_ref or processed_dataset_ref).split("/")[-1]
    epochs = args.epochs
    batch_size = args.batch_size
    
    run_id = f"train-run-{int(time.time())}"
    job_dir = project_root / "outputs" / "kaggle_training" / run_id
    dataset_dir = job_dir / "dataset"
    kernel_dir = job_dir / "kernel"
    download_dir = job_dir / "downloaded_model"
    
    for d in (dataset_dir, kernel_dir, download_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("======================================================================")
    print(f"🚀 Initializing Kaggle Job: {run_id}")
    print(f"   Processed data source: {processed_kernel_ref or processed_dataset_ref} ({'kernel' if processed_kernel_ref else 'dataset'})")
    print(f"   Training config: epochs={epochs}, batch_size={batch_size}")
    print("======================================================================")

    # 1. Zip source code
    print("📦 Zipping local source code...")
    write_source_zip(project_root, dataset_dir / "genmusic_vn_source.zip")

    # 2. Upload source code zip as a Kaggle Dataset
    source_dataset_slug = f"genmusic-source-{run_id}"
    source_dataset_ref = f"{username}/{source_dataset_slug}"
    
    (dataset_dir / "dataset-metadata.json").write_text(json.dumps({
        "title": f"GenMusic Source {run_id}",
        "id": source_dataset_ref,
        "licenses": [{"name": "other"}]
    }, indent=2))

    print(f"📤 Uploading source code to Kaggle Dataset '{source_dataset_ref}'...")
    subprocess.run(cli + ["datasets", "create", "-p", str(dataset_dir), "-r", "zip"], env=kaggle_env, check=True)

    # Wait until dataset is ready on Kaggle
    print("⏳ Waiting for source dataset to be ready...")
    for _ in range(60):
        res = subprocess.run(cli + ["datasets", "status", source_dataset_ref], env=kaggle_env, capture_output=True, text=True)
        if "ready" in res.stdout.lower():
            break
        time.sleep(5)

    # 3. Create Kernel script and metadata (Keep slug/title short to fit Kaggle's 50-char limit)
    kernel_slug = f"genmusic-train-{int(time.time())}"
    kernel_ref = f"{username}/{kernel_slug}"
    
    (kernel_dir / "run_training.py").write_text(
        _kernel_script_content(processed_dataset_slug, epochs, batch_size),
        encoding="utf-8",
    )
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
        "dataset_sources": [source_dataset_ref] + ([] if processed_kernel_ref else [processed_dataset_ref]),
        "kernel_sources": [processed_kernel_ref] if processed_kernel_ref else [],
    }, indent=2))

    # 4. Push Kernel to Kaggle
    print(f"🚀 Pushing training Kernel '{kernel_ref}' to Kaggle (GPU: T4)...")
    subprocess.run(cli + ["kernels", "push", "-p", str(kernel_dir)], env=kaggle_env, check=True)

    # 5. Poll Kernel status
    print("⏳ Monitoring training execution status on Kaggle...")
    completed = False
    for _ in range(240): # Poll for up to 40 minutes (Whisper/Demucs on 1 file + GPU train is fast, ~5-8 mins)
        res = subprocess.run(cli + ["kernels", "status", kernel_ref], env=kaggle_env, capture_output=True, text=True)
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
        subprocess.run(cli + ["kernels", "output", kernel_ref, "-p", str(download_dir), "-o"], env=kaggle_env, check=True)
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
