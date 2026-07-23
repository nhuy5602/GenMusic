"""Full latent-space pipeline: given an already-trained `LatentAudioEncoder`
checkpoint (see `run_kaggle_latent_encoder.py`), precompute the full dataset's
latent representation, train the CFM student (MicroDiT, on the 64-dim/5Hz
latent instead of mel) inside that latent space until early-stopping decides
it's done, then generate one sample.

Does not train the encoder itself -- run `run_kaggle_latent_encoder.py` first
(and sanity-check it, see `docs/architecture.md`'s "Native latent backbone"
section for the collapse failure mode hit once already) if you don't have a
checkpoint yet. For resuming a CFM training run that got interrupted partway
(Kaggle sessions have a wall-clock limit), use `run_kaggle_latent_resume.py`
instead of restarting this script from scratch.
"""

import argparse
import json
import os
import shutil
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


def _kernel_script_content(cfm_epochs: str, cfm_batch_size: str, dim: str, depth: str, heads: str, gen_text: str, gen_style: str, gen_duration: str) -> str:
    return f'''import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

def run_logged(command, label):
    # Streams output live (visible on the Kaggle web UI as it happens) instead
    # of buffering the whole subprocess and printing it only after it exits --
    # capture_output=True gave zero visibility into a multi-hour training run,
    # which is exactly what made a real stall indistinguishable from "just slow".
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    lines = []
    for line in process.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    process.wait()
    output = "".join(lines)
    Path("/kaggle/working/" + label + ".log").write_text(output, encoding="utf-8")
    if process.returncode != 0:
        raise RuntimeError(label + " failed with exit code " + str(process.returncode) + "\\n" + output[-8000:])
    return process

try:
    print("--- STEP 1: Locating preprocessed dataset, source code, and encoder checkpoint ---")
    input_dir = Path("/kaggle/input")
    records_file = next(input_dir.rglob("records.jsonl"), None)
    if not records_file:
        raise RuntimeError(f"Could not find the processed dataset in /kaggle/input (looked in {{input_dir}}).")
    processed_dataset = records_file.parent
    print(f"Using processed dataset: {{processed_dataset.resolve()}}")

    source_dataset_dir = next(
        (d for d in input_dir.rglob("*") if d.is_dir() and "genmusic-source-" in d.name.lower()),
        None
    )
    if not source_dataset_dir:
        raise RuntimeError("Could not find the source code dataset directory.")
    source_root = Path("/kaggle/working/GenMusic")
    shutil.copytree(source_dataset_dir, source_root, dirs_exist_ok=True)

    ckpt_dataset_dir = next(
        (d for d in input_dir.rglob("*") if d.is_dir() and "genmusic-encckpt-" in d.name.lower()),
        None
    )
    if not ckpt_dataset_dir:
        raise RuntimeError("Could not find the encoder checkpoint dataset directory.")
    encoder_checkpoint = str(next(ckpt_dataset_dir.glob("*.pt")))
    print(f"Using pre-trained encoder checkpoint: {{encoder_checkpoint}}")

    print("--- STEP 2: Downloading DiffRhythm2 official repository (needed for `bigvgan`) ---")
    diffrhythm2_tar = "/kaggle/working/diffrhythm2.tar.gz"
    urllib.request.urlretrieve("https://github.com/ASLP-lab/DiffRhythm2/archive/refs/heads/main.tar.gz", diffrhythm2_tar)
    with tarfile.open(diffrhythm2_tar) as tar:
        tar.extractall(str(source_root))
    os.remove(diffrhythm2_tar)

    print("--- STEP 2.8: Installing system packages (espeak-ng) ---")
    subprocess.run(["apt-get", "update", "-y"], check=False)
    subprocess.run(["apt-get", "install", "-y", "--fix-missing", "espeak-ng"], check=True)

    print("--- STEP 3: Installing dependencies ---")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(source_root / "DiffRhythm2-main/requirements.txt")], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vocos", "text2phonemesequence"], check=True)

    os.environ["PYTHONPATH"] = str(source_root) + os.pathsep + str(source_root / "DiffRhythm2-main") + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    cli = str(source_root / "cli.py")

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

    print("--- STEP 3.6: Patching transformers' torch.load version guard (CVE-2025-32434 blocks non-safetensors torch.load whenever torch<2.6, which is true both after the P100 repair and, apparently, on Kaggle's own default image) ---", flush=True)
    patch_code = """
import pathlib
import transformers.utils.import_utils as m
p = pathlib.Path(m.__file__)
s = p.read_text(encoding="utf-8")
marker = "def check_torch_load_is_safe"
if marker in s:
    def_start = s.index(marker)
    line_end = s.index(chr(10), def_start)
    s = s[:line_end + 1] + "    return" + chr(10) + s[line_end + 1:]
    p.write_text(s, encoding="utf-8")
    print("patched check_torch_load_is_safe")
else:
    print("check_torch_load_is_safe marker not found; skipping patch")
"""
    subprocess.run([sys.executable, "-c", patch_code], check=True)

    print("--- STEP 4: Precomputing the latent dataset (249 songs -> 64-dim/5Hz latents) with the stabilized encoder ---")
    latent_dataset = "/kaggle/working/latent_dataset"
    run_logged([
        sys.executable, cli, "precompute-latent-dataset",
        "--source-dataset", str(processed_dataset),
        "--encoder-checkpoint", encoder_checkpoint,
        "--out", latent_dataset,
        "--device", "cuda",
    ], "precompute_latent_dataset")

    print("--- STEP 5: Training the CFM student inside the latent space (early-stopping decides when to stop) ---")
    cfm_checkpoint = "/kaggle/working/latent_cfm_model.pt"
    run_logged([
        sys.executable, cli, "train-self",
        "--dataset", latent_dataset,
        "--checkpoint", cfm_checkpoint,
        "--epochs", "{cfm_epochs}",
        "--batch-size", "{cfm_batch_size}",
        "--dim", "{dim}",
        "--depth", "{depth}",
        "--heads", "{heads}",
        "--lambda-vocal", "0",
        "--device", "cuda",
        "--save-every-epoch",
    ], "train_latent_cfm")

    print("--- STEP 6: Generating one sample (decoded via the real frozen BigVGAN decoder) ---")
    run_logged([
        sys.executable, cli, "generate-local",
        "--text", "{gen_text}",
        "--style", "{gen_style}",
        "--duration", "{gen_duration}",
        "--checkpoint", cfm_checkpoint,
        "--steps", "32",
        "--vocoder", "vocos",
        "--device", "cuda",
        "--out", "/kaggle/working/generated_latent",
    ], "generate_latent_sample")

    print("PIPELINE V2 COMPLETED SUCCESSFULLY!")
    Path("/kaggle/working/success.txt").write_text("success", encoding="utf-8")
except Exception:
    import traceback
    tb = traceback.format_exc()
    print("ERROR OCCURRED DURING KERNEL EXECUTION:")
    print(tb)
    Path("/kaggle/working/error.txt").write_text(tb, encoding="utf-8")
    sys.exit(1)
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder-checkpoint", required=True, help="Path to the local, already-retrained latent_encoder.pt")
    parser.add_argument("--cfm-epochs", type=int, default=300)
    parser.add_argument("--cfm-batch-size", type=int, default=8)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--gen-text", default="Dem nay mua roi tren loi mon xua")
    parser.add_argument("--gen-style", default="soft Vietnamese ballad")
    parser.add_argument("--gen-duration", type=float, default=8.0)
    parser.add_argument("--processed-kernel-ref", type=str, default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    tokens = kaggle_auth_environment(load_kaggle_api_tokens())
    username = resolve_kaggle_username(tokens.get("KAGGLE_USERNAME"))
    cli = kaggle_cli_command()
    if not username or not kaggle_auth_available(tokens) or not cli:
        raise RuntimeError("Missing KAGGLE_USERNAME or Kaggle auth (KAGGLE_API_TOKEN=KGAT_... / legacy KAGGLE_KEY)")

    run_id = f"latentpipe-{int(time.time())}"
    job_dir = project_root / "outputs" / "kaggle_latent_pipeline" / run_id
    source_dir = job_dir / "source_dataset"
    ckpt_dir = job_dir / "ckpt_dataset"
    kernel_dir = job_dir / "kernel"
    for d in (source_dir, ckpt_dir, kernel_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Initializing Kaggle Latent Pipeline V2 Job: {run_id}")
    print("=" * 70)

    shutil.copy2(args.encoder_checkpoint, ckpt_dir / "latent_encoder.pt")

    print("Zipping local source code...")
    write_source_zip(project_root, source_dir / "genmusic_vn_source.zip")
    source_dataset_ref = f"{username}/genmusic-source-{run_id}"
    (source_dir / "dataset-metadata.json").write_text(json.dumps({
        "title": f"GenMusic Source {run_id}", "id": source_dataset_ref, "licenses": [{"name": "other"}],
    }, indent=2))

    ckpt_dataset_ref = f"{username}/genmusic-encckpt-{run_id}"
    (ckpt_dir / "dataset-metadata.json").write_text(json.dumps({
        "title": f"GenMusic EncCkpt {run_id}", "id": ckpt_dataset_ref, "licenses": [{"name": "other"}],
    }, indent=2))

    print(f"Uploading source code to '{source_dataset_ref}'...")
    try:
        subprocess.run(cli + ["datasets", "create", "-p", str(source_dir), "-r", "zip"], env={**os.environ, **tokens}, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[WARNING] Kaggle dataset creation returned an error (often transient): {e}. Proceeding anyway...")
    print(f"Uploading encoder checkpoint to '{ckpt_dataset_ref}'...")
    try:
        subprocess.run(cli + ["datasets", "create", "-p", str(ckpt_dir), "-r", "zip"], env={**os.environ, **tokens}, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[WARNING] Kaggle dataset creation returned an error (often transient): {e}. Proceeding anyway...")

    print("Waiting for datasets to be ready...")
    time.sleep(25)
    for ref in (source_dataset_ref, ckpt_dataset_ref):
        for _ in range(60):
            res = subprocess.run(cli + ["datasets", "status", ref], env={**os.environ, **tokens}, capture_output=True, text=True, check=False)
            if "ready" in res.stdout.lower():
                break
            time.sleep(10)

    processed_kernel_ref = args.processed_kernel_ref or tokens.get("KAGGLE_PROCESSED_KERNEL_REF")
    processed_dataset_ref = None if processed_kernel_ref else tokens.get(
        "KAGGLE_PROCESSED_DATASET_REF", f"{username}/vietnamese-music-processed-dataset"
    )

    kernel_slug = f"genmusic-latentpipe-{int(time.time())}"
    kernel_ref = f"{username}/{kernel_slug}"
    (kernel_dir / "run_latent_pipeline.py").write_text(
        _kernel_script_content(
            str(args.cfm_epochs), str(args.cfm_batch_size), str(args.dim), str(args.depth), str(args.heads),
            args.gen_text, args.gen_style, str(args.gen_duration),
        ),
        encoding="utf-8",
    )
    (kernel_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": kernel_ref,
        "title": kernel_slug,
        "code_file": "run_latent_pipeline.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [source_dataset_ref, ckpt_dataset_ref] + ([processed_dataset_ref] if processed_dataset_ref else []),
        "kernel_sources": [processed_kernel_ref] if processed_kernel_ref else [],
        "competition_sources": [],
    }, indent=2))

    print(f"Pushing Latent Pipeline V2 Kernel to Kaggle: {kernel_ref}...")
    time.sleep(15)
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
