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

# (lyric, style, short name) triples -- deliberately distinct lyrics/styles so
# a listener can judge whether the model actually differentiates prompts, not
# just whether one fixed prompt sounds okay.
PROMPTS = [
    ("Dem nay mua roi tren loi mon xua, long anh nho em nhieu nguoi oi co biet chang",
     "Vietnamese pop ballad, soft piano, slow tempo", "ballad"),
    ("Nang xuan tren pho phuong rop hoa, tieng cuoi vang len khap noi gan xa",
     "upbeat Vietnamese dance pop, energetic drums, bright synths", "upbeat"),
    ("Bien xanh song vo bo cat trang, gio mang huong vi cua mua he sang",
     "acoustic guitar, chill lo-fi, relaxed tempo", "chill"),
]
GUIDANCE_SCALES = [1.0, 4.0]


def _kernel_script_content() -> str:
    prompts_literal = json.dumps(PROMPTS, ensure_ascii=False)
    scales_literal = json.dumps(GUIDANCE_SCALES)
    return f'''import base64
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

try:
    print("--- STEP 1: Locating checkpoint ---")
    input_dir = Path("/kaggle/input")
    checkpoint_candidates = list(input_dir.rglob("distilled_student.pt")) or list(input_dir.rglob("my_trained_model.pt"))
    if not checkpoint_candidates:
        raise RuntimeError("Could not find distilled_student.pt or my_trained_model.pt in /kaggle/input.")
    checkpoint_path = checkpoint_candidates[0]
    print(f"Using checkpoint: {{checkpoint_path.resolve()}}")

    print("--- STEP 2: Setting up source code ---")
    source_dataset_dir = next(
        (d for d in input_dir.rglob("*") if d.is_dir() and "genmusic-source-" in d.name.lower()),
        None
    )
    if not source_dataset_dir:
        raise RuntimeError("Could not find the source code dataset directory.")
    source_root = Path("/kaggle/working/GenMusic")
    shutil.copytree(source_dataset_dir, source_root, dirs_exist_ok=True)

    print("--- STEP 2.2: Downloading DiffRhythm2 official repository (needed for `bigvgan`) ---")
    diffrhythm2_tar = "/kaggle/working/diffrhythm2.tar.gz"
    urllib.request.urlretrieve("https://github.com/ASLP-lab/DiffRhythm2/archive/refs/heads/main.tar.gz", diffrhythm2_tar)
    with tarfile.open(diffrhythm2_tar) as tar:
        tar.extractall(str(source_root))

    print("--- STEP 2.5: Installing system packages (espeak-ng, ffmpeg) ---")
    subprocess.run(["apt-get", "update", "-y"], check=False)
    subprocess.run(["apt-get", "install", "-y", "--fix-missing", "espeak-ng", "ffmpeg"], check=True)

    print("--- STEP 3: Installing python dependencies ---")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "librosa", "soundfile", "transformers", "vocos", "muq"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "text2phonemesequence"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(source_root / "DiffRhythm2-main/requirements.txt")], check=True)

    os.environ["PYTHONPATH"] = (
        str(source_root) + os.pathsep + str(source_root / "DiffRhythm2-main") + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    sys.path.insert(0, str(source_root))
    sys.path.insert(0, str(source_root / "DiffRhythm2-main"))

    print("--- STEP 3.5: Checking CUDA ---")
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch={{torch.__version__}} device={{device}}")
    if device == "cuda":
        try:
            _ = (torch.randn(2, 2, device="cuda") @ torch.randn(2, 2, device="cuda"))
            print("CUDA smoke test ok")
        except Exception as cuda_err:
            print(f"CUDA smoke test failed ({{cuda_err}}), falling back to CPU")
            device = "cpu"

    print("--- STEP 4: Loading checkpoint and generating samples ---")
    from src.models.text_to_music_diffusion import load_checkpoint, generate_audio

    out_dir = Path("/kaggle/working/samples")
    out_dir.mkdir(parents=True, exist_ok=True)

    model, config, payload = load_checkpoint(str(checkpoint_path), device=device)

    prompts = {prompts_literal}
    guidance_scales = {scales_literal}

    manifest = []
    for text, style, short_name in prompts:
        for scale in guidance_scales:
            sample_name = f"{{short_name}}_gs{{scale}}"
            wav_path = out_dir / f"{{sample_name}}.wav"
            print(f"Generating {{sample_name}} ...")
            info = generate_audio(
                model, text, style, wav_path,
                duration_seconds=6.0, config=config, device=device,
                steps=24, guidance_scale=float(scale), seed=5602,
            )
            mp3_path = wav_path.with_suffix(".mp3")
            ffmpeg = shutil.which("ffmpeg")
            subprocess.run([ffmpeg, "-y", "-i", str(wav_path), "-b:a", "64k", str(mp3_path)],
                           check=True, capture_output=True)
            manifest.append({{"name": sample_name, "text": text, "style": style, "guidance_scale": scale}})
            with open(mp3_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
            print(f"===SAMPLE_START:{{sample_name}}===")
            print(b64)
            print(f"===SAMPLE_END:{{sample_name}}===")

    print("MANIFEST_JSON:" + json.dumps(manifest, ensure_ascii=False))
    print("GUIDANCE DEMO COMPLETED SUCCESSFULLY!")
except Exception as e:
    import traceback
    print("ERROR OCCURRED DURING KERNEL EXECUTION:")
    traceback.print_exc()
    sys.exit(1)
'''


def run_kaggle_guidance_demo(checkpoint_kernel_ref: str) -> str:
    project_root = Path(__file__).resolve().parents[1]

    tokens = kaggle_auth_environment(load_kaggle_api_tokens())
    username = resolve_kaggle_username(tokens.get("KAGGLE_USERNAME"))
    cli = kaggle_cli_command()
    if not username or not kaggle_auth_available(tokens) or not cli:
        raise RuntimeError("Missing KAGGLE_USERNAME or Kaggle auth (KAGGLE_API_TOKEN=KGAT_... / legacy KAGGLE_KEY)")

    run_id = f"guidance-demo-{int(time.time())}"
    job_dir = project_root / "outputs" / "kaggle_guidance_demo" / run_id
    dataset_dir = job_dir / "dataset"
    kernel_dir = job_dir / "kernel"

    for d in (dataset_dir, kernel_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Initializing Kaggle Guidance-Scale Demo Job: {run_id}")
    print("=" * 70)

    print("Zipping local source code...")
    write_source_zip(project_root, dataset_dir / "genmusic_vn_source.zip")

    source_dataset_slug = f"genmusic-source-{run_id}"
    source_dataset_ref = f"{username}/{source_dataset_slug}"

    # Kaggle dataset titles must be 6-50 chars -- a title over that limit fails
    # dataset creation, and the kernel then silently pushes WITHOUT the source
    # attached (dataset_sources entry gets dropped with only a warning), which
    # fails later at "locating source code" instead of at push time.
    dataset_title = f"GS Demo {run_id}"
    assert 6 <= len(dataset_title) <= 50, f"dataset title length {len(dataset_title)} out of Kaggle's 6-50 range: {dataset_title!r}"
    (dataset_dir / "dataset-metadata.json").write_text(json.dumps({
        "title": dataset_title,
        "id": source_dataset_ref,
        "licenses": [{"name": "other"}]
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

    kernel_slug = f"genmusic-guidance-demo-{int(time.time())}"
    kernel_ref = f"{username}/{kernel_slug}"

    (kernel_dir / "run_guidance_demo.py").write_text(_kernel_script_content(), encoding="utf-8")

    (kernel_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": kernel_ref,
        "title": kernel_slug,
        "code_file": "run_guidance_demo.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [source_dataset_ref],
        "kernel_sources": [checkpoint_kernel_ref],
        "competition_sources": []
    }, indent=2))

    print(f"Pushing Guidance Demo Kernel to Kaggle: {kernel_ref}...")

    time.sleep(20)
    for attempt in range(3):
        try:
            subprocess.run(cli + ["kernels", "push", "-p", str(kernel_dir)], env={**os.environ, **tokens}, check=True)
            print("\nGUIDANCE DEMO JOB SUBMITTED SUCCESSFULLY!")
            print(f"-> https://www.kaggle.com/code/{kernel_ref}")
            return kernel_ref
        except subprocess.CalledProcessError as e:
            if attempt == 2:
                raise e
            print(f"Kaggle kernel push failed on attempt {attempt+1}. Retrying in 15 seconds...", flush=True)
            time.sleep(15)
    return kernel_ref


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-kernel-ref", type=str, required=True)
    args = parser.parse_args()
    run_kaggle_guidance_demo(args.checkpoint_kernel_ref)


if __name__ == "__main__":
    main()
