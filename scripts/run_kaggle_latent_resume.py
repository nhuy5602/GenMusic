"""Resumes a `run_kaggle_latent_pipeline.py` CFM training run that got cut off
partway (a Kaggle session hitting its wall-clock limit, or being killed
deliberately to bound GPU-quota risk, is a recurring event, not a one-off --
see docs/project_history.md §4.24 for a session this actually happened in).
Uploads an already-downloaded latent_dataset/ + checkpoint as a Kaggle
dataset and continues CFM training from there (`--save-every-epoch`
checkpoints from the original run make this possible), then generates one
sample. Skips re-running the encoder/precompute stages entirely.

Launch with a small, bounded `--cfm-epochs` (a handful of epochs past the
checkpoint's current epoch) rather than a large open-ended cap -- this makes
the run complete-or-fail visibly instead of sitting at Kaggle's status
"RUNNING" for hours with no way to tell a real stall from normal progress
(`kaggle kernels output` returns nothing for a still-running kernel).
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
    print("--- STEP 1: Locating source code and resume dataset ---")
    input_dir = Path("/kaggle/input")
    source_dataset_dir = next(
        (d for d in input_dir.rglob("*") if d.is_dir() and "genmusic-source-" in d.name.lower()),
        None
    )
    if not source_dataset_dir:
        raise RuntimeError("Could not find the source code dataset directory.")
    source_root = Path("/kaggle/working/GenMusic")
    shutil.copytree(source_dataset_dir, source_root, dirs_exist_ok=True)

    resume_dataset_dir = next(
        (d for d in input_dir.rglob("*") if d.is_dir() and "genmusic-resume-" in d.name.lower()),
        None
    )
    if not resume_dataset_dir:
        raise RuntimeError("Could not find the resume dataset directory (latent_dataset + checkpoint).")
    latent_dataset = str(resume_dataset_dir / "latent_dataset")
    checkpoint_source = resume_dataset_dir / "latent_cfm_model.pt"
    cfm_checkpoint = "/kaggle/working/latent_cfm_model.pt"
    shutil.copy2(checkpoint_source, cfm_checkpoint)

    print("--- STEP 2: Installing dependencies (Vocos + XPhoneBERT G2P; no DiffRhythm2 clone needed for resume) ---")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vocos", "text2phonemesequence"], check=True)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    cli = str(source_root / "cli.py")

    print("--- STEP 2.5: Checking CUDA compatibility (Kaggle sometimes assigns a P100, sm_60, which the preinstalled torch build does not support) ---")
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

    print("--- STEP 2.6: Patching transformers' torch.load version guard (CVE-2025-32434 blocks non-safetensors torch.load whenever torch<2.6, which is true both after the P100 repair and, apparently, on Kaggle's own default image) ---", flush=True)
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

    print("--- STEP 3: Resuming CFM training in latent space ---")
    run_logged([
        sys.executable, cli, "train-self",
        "--dataset", latent_dataset,
        "--checkpoint", cfm_checkpoint,
        "--resume",
        "--epochs", "{cfm_epochs}",
        "--batch-size", "{cfm_batch_size}",
        "--dim", "{dim}",
        "--depth", "{depth}",
        "--heads", "{heads}",
        "--lambda-vocal", "0",
        "--device", "cuda",
        "--save-every-epoch",
    ], "resume_train_latent_cfm")

    print("--- STEP 4: Generating one sample (decoded via the real frozen BigVGAN decoder) ---")
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

    print("PIPELINE COMPLETED SUCCESSFULLY!")
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
    parser.add_argument("--downloaded-dir", required=True, help="Path to the previously downloaded kernel output dir containing latent_dataset/ and latent_cfm_model.pt")
    parser.add_argument("--cfm-epochs", type=int, default=25)
    parser.add_argument("--cfm-batch-size", type=int, default=8)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--gen-text", default="Dem nay mua roi tren loi mon xua")
    parser.add_argument("--gen-style", default="soft Vietnamese ballad")
    parser.add_argument("--gen-duration", type=float, default=8.0)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    tokens = kaggle_auth_environment(load_kaggle_api_tokens())
    username = resolve_kaggle_username(tokens.get("KAGGLE_USERNAME"))
    cli = kaggle_cli_command()
    if not username or not kaggle_auth_available(tokens) or not cli:
        raise RuntimeError("Missing KAGGLE_USERNAME or Kaggle auth (KAGGLE_API_TOKEN=KGAT_... / legacy KAGGLE_KEY)")

    run_id = f"latentresume-{int(time.time())}"
    job_dir = project_root / "outputs" / "kaggle_latent_pipeline" / run_id
    source_dir = job_dir / "source_dataset"
    resume_dir = job_dir / "resume_dataset"
    kernel_dir = job_dir / "kernel"
    for d in (source_dir, resume_dir, kernel_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Initializing Kaggle Latent Resume Job: {run_id}")
    print("=" * 70)

    downloaded = Path(args.downloaded_dir)
    shutil.copytree(downloaded / "latent_dataset", resume_dir / "latent_dataset")
    shutil.copy2(downloaded / "latent_cfm_model.pt", resume_dir / "latent_cfm_model.pt")

    print("Zipping local source code...")
    write_source_zip(project_root, source_dir / "genmusic_vn_source.zip")
    source_dataset_ref = f"{username}/genmusic-source-{run_id}"
    (source_dir / "dataset-metadata.json").write_text(json.dumps({
        "title": f"GenMusic Source {run_id}", "id": source_dataset_ref, "licenses": [{"name": "other"}],
    }, indent=2))

    resume_dataset_ref = f"{username}/genmusic-resume-{run_id}"
    (resume_dir / "dataset-metadata.json").write_text(json.dumps({
        "title": f"GenMusic Resume {run_id}", "id": resume_dataset_ref, "licenses": [{"name": "other"}],
    }, indent=2))

    print(f"Uploading source code to '{source_dataset_ref}'...")
    subprocess.run(cli + ["datasets", "create", "-p", str(source_dir), "-r", "zip"], env={**os.environ, **tokens}, check=True)
    print(f"Uploading resume dataset (latent_dataset + checkpoint) to '{resume_dataset_ref}'...")
    subprocess.run(cli + ["datasets", "create", "-p", str(resume_dir), "-r", "zip"], env={**os.environ, **tokens}, check=True)

    print("Waiting for both datasets to be ready...")
    time.sleep(25)
    for ref in (source_dataset_ref, resume_dataset_ref):
        for _ in range(60):
            res = subprocess.run(cli + ["datasets", "status", ref], env={**os.environ, **tokens}, capture_output=True, text=True, check=False)
            if "ready" in res.stdout.lower():
                break
            time.sleep(10)

    kernel_slug = f"genmusic-latentresume-{int(time.time())}"
    kernel_ref = f"{username}/{kernel_slug}"
    (kernel_dir / "run_latent_resume.py").write_text(
        _kernel_script_content(str(args.cfm_epochs), str(args.cfm_batch_size), str(args.dim), str(args.depth), str(args.heads), args.gen_text, args.gen_style, str(args.gen_duration)),
        encoding="utf-8",
    )
    (kernel_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": kernel_ref,
        "title": kernel_slug,
        "code_file": "run_latent_resume.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [source_dataset_ref, resume_dataset_ref],
        "kernel_sources": [],
        "competition_sources": [],
    }, indent=2))

    print(f"Pushing Latent Resume Kernel to Kaggle: {kernel_ref}...")
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
