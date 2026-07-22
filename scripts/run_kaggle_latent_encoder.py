"""Kaggle launcher for `cli.py train-latent-encoder` -- pretrains the new
`LatentAudioEncoder` (src/models/latent_codec.py) against DiffRhythm2's real,
frozen, pretrained BigVGAN decoder (`decoder.bin`, downloaded from
ASLP-lab/DiffRhythm2 on HuggingFace) on this project's own processed dataset.

Mirrors scripts/run_kaggle_distill.py's structure (same DiffRhythm2 repo
clone, since `bigvgan` is only importable that way -- not a pip package).
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


def _kernel_script_content(epochs: str = "1", batch_size: str = "4", learning_rate: str = "2e-4", max_records: str | None = None) -> str:
    max_records_line = f'        "--max-records", "{max_records}",\n' if max_records is not None else ""
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

    print("--- STEP 2.5: Downloading DiffRhythm2 official repository (needed for the `bigvgan` package) ---")
    diffrhythm2_tar = "/kaggle/working/diffrhythm2.tar.gz"
    urllib.request.urlretrieve("https://github.com/ASLP-lab/DiffRhythm2/archive/refs/heads/main.tar.gz", diffrhythm2_tar)
    with tarfile.open(diffrhythm2_tar) as tar:
        tar.extractall(str(source_root))
    os.remove(diffrhythm2_tar)

    print("--- STEP 3: Installing dependencies ---")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(source_root / "DiffRhythm2-main/requirements.txt")], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vocos"], check=True)

    os.environ["PYTHONPATH"] = str(source_root) + os.pathsep + str(source_root / "DiffRhythm2-main") + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    print("--- STEP 3.5: Checking CUDA compatibility (Kaggle sometimes assigns a P100, sm_60, which the preinstalled torch build does not support) ---")
    cuda_probe_code = "import torch; print('torch=%s cuda=%s available=%s' % (torch.__version__, torch.version.cuda, torch.cuda.is_available())); print(torch.randn((2, 2), device='cuda') @ torch.randn((2, 2), device='cuda')) if torch.cuda.is_available() else None"
    torch_probe = subprocess.run([sys.executable, "-c", cuda_probe_code], capture_output=True, text=True, encoding="utf-8", errors="replace")
    torch_probe_output = (torch_probe.stdout or "") + chr(10) + (torch_probe.stderr or "")
    print(torch_probe_output, flush=True)
    if torch_probe.returncode != 0:
        print("CUDA smoke test failed; installing P100-compatible Torch.", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-cache-dir", "--force-reinstall", "--extra-index-url", "https://download.pytorch.org/whl/cu121", "torch==2.5.1+cu121", "torchaudio==2.5.1+cu121", "torchvision==0.20.1+cu121"], check=True)
        repaired_probe = subprocess.run([sys.executable, "-c", cuda_probe_code], capture_output=True, text=True, encoding="utf-8", errors="replace")
        repaired_output = (repaired_probe.stdout or "") + chr(10) + (repaired_probe.stderr or "")
        print(repaired_output, flush=True)
        if repaired_probe.returncode != 0 or "available=True" not in repaired_output:
            raise RuntimeError("CUDA is unavailable after P100 Torch repair.")

    print("--- STEP 4: Running latent encoder pretraining ---")
    checkpoint_path = Path("/kaggle/working/latent_encoder.pt")
    train_result = subprocess.run([
        sys.executable, str(source_root / "cli.py"), "train-latent-encoder",
        "--dataset", str(processed_dataset),
        "--checkpoint", str(checkpoint_path),
        "--epochs", "{epochs}",
        "--batch-size", "{batch_size}",
        "--learning-rate", "{learning_rate}",
        "--device", "cuda",
{max_records_line}    ], env=os.environ, capture_output=True, text=True)
    Path("/kaggle/working/train_latent_encoder_stdout.txt").write_text(train_result.stdout, encoding="utf-8")
    Path("/kaggle/working/train_latent_encoder_stderr.txt").write_text(train_result.stderr, encoding="utf-8")
    train_result.check_returncode()

    print("LATENT ENCODER PRETRAINING COMPLETED SUCCESSFULLY!")
    print("Checkpoint saved at: /kaggle/working/latent_encoder.pt")
    report_path = Path("/kaggle/working/latent_encoder_report.json")
    if report_path.is_file():
        print("--- latent_encoder_report.json ---")
        print(report_path.read_text(encoding="utf-8"))
    Path("/kaggle/working/success.txt").write_text("success", encoding="utf-8")
except Exception as e:
    import traceback
    tb = traceback.format_exc()
    print("ERROR OCCURRED DURING KERNEL EXECUTION:")
    print(tb)
    Path("/kaggle/working/error.txt").write_text(tb, encoding="utf-8")
    sys.exit(1)
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-records", type=int, default=None, help="Limit training to the first N usable records (for cheap smoke tests).")
    parser.add_argument("--processed-kernel-ref", type=str, default=None, help="Override KAGGLE_PROCESSED_KERNEL_REF for this run.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    tokens = kaggle_auth_environment(load_kaggle_api_tokens())
    username = resolve_kaggle_username(tokens.get("KAGGLE_USERNAME"))
    cli = kaggle_cli_command()
    if not username or not kaggle_auth_available(tokens) or not cli:
        raise RuntimeError("Missing KAGGLE_USERNAME or Kaggle auth (KAGGLE_API_TOKEN=KGAT_... / legacy KAGGLE_KEY)")

    run_id = f"latentenc-{int(time.time())}"
    job_dir = project_root / "outputs" / "kaggle_latent_encoder" / run_id
    dataset_dir = job_dir / "dataset"
    kernel_dir = job_dir / "kernel"
    for d in (dataset_dir, kernel_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Initializing Kaggle Latent Encoder Job: {run_id}")
    print("=" * 70)

    print("Zipping local source code...")
    write_source_zip(project_root, dataset_dir / "genmusic_vn_source.zip")

    source_dataset_slug = f"genmusic-source-{run_id}"
    source_dataset_ref = f"{username}/{source_dataset_slug}"
    (dataset_dir / "dataset-metadata.json").write_text(json.dumps({
        "title": f"GenMusic LatentEnc {run_id}",
        "id": source_dataset_ref,
        "licenses": [{"name": "other"}],
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

    kernel_slug = f"genmusic-latentenc-{int(time.time())}"
    kernel_ref = f"{username}/{kernel_slug}"
    (kernel_dir / "run_latent_encoder.py").write_text(
        _kernel_script_content(
            str(args.epochs), str(args.batch_size), str(args.learning_rate),
            str(args.max_records) if args.max_records is not None else None,
        ),
        encoding="utf-8",
    )

    processed_kernel_ref = args.processed_kernel_ref or tokens.get("KAGGLE_PROCESSED_KERNEL_REF")
    processed_dataset_ref = None if processed_kernel_ref else tokens.get(
        "KAGGLE_PROCESSED_DATASET_REF", f"{username}/vietnamese-music-processed-dataset"
    )

    (kernel_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": kernel_ref,
        "title": kernel_slug,
        "code_file": "run_latent_encoder.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [source_dataset_ref] + ([processed_dataset_ref] if processed_dataset_ref else []),
        "kernel_sources": [processed_kernel_ref] if processed_kernel_ref else [],
        "competition_sources": [],
    }, indent=2))

    print(f"Pushing Latent Encoder Kernel to Kaggle: {kernel_ref}...")
    time.sleep(20)
    for attempt in range(3):
        try:
            subprocess.run(cli + ["kernels", "push", "-p", str(kernel_dir)], env={**os.environ, **tokens}, check=True)
            print("\nJOB SUBMITTED SUCCESSFULLY!")
            print(f"-> https://www.kaggle.com/code/{kernel_ref}")
            print(f"KERNEL_REF={kernel_ref}")
            break
        except subprocess.CalledProcessError as e:
            if attempt == 2:
                raise e
            print(f"Kaggle kernel push failed on attempt {attempt+1}. Retrying in 15 seconds...", flush=True)
            time.sleep(15)


if __name__ == "__main__":
    main()
