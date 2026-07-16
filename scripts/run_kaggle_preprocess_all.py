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

def _kernel_script_content(
    raw_dataset_slug: str,
    max_files: str | None = None,
    whisper_model: str = "tiny",
) -> str:
    # Script runs on Kaggle GPU instance and processes the requested file set.
    # Deliberately never embeds any Kaggle credentials -- this source is pushed to
    # Kaggle and becomes visible/shareable, so it must stay safe to hand to anyone.
    # The processed dataset is left as this kernel's own retained output; downstream
    # kernels attach it via `kernel_sources` (see run_kaggle_training.py/run_kaggle_distill.py)
    # instead of the preprocess kernel re-authenticating to publish a separate Dataset.
    kernel_max_files = str(max_files or "")
    return f'''import os
import json
import shutil
import subprocess
import sys
import threading
import time
import zipfile
import traceback
from pathlib import Path

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
    source_root = Path("/kaggle/working/GenMusic")
    source_dataset_dir = next((d for d in input_dir.rglob("*") if d.is_dir() and "genmusic-source-" in d.name.lower()), None)
    source_zip = next((p for p in input_dir.rglob("genmusic_vn_source.zip") if p.is_file()), None)
    if source_zip:
        source_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(source_zip) as archive:
            archive.extractall(source_root)
    elif source_dataset_dir:
        shutil.copytree(source_dataset_dir, source_root, dirs_exist_ok=True)
    if not (source_root / "cli.py").exists():
        raise RuntimeError(f"GenMusic source code was not found under {{source_root}}.")

    def run_logged(command, label, timeout):
        print("--- RUNNING " + label + " ---", flush=True)
        started = time.monotonic()
        output_lines = []
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        def forward_output():
            if process.stdout is None:
                return
            for line in process.stdout:
                output_lines.append(line)
                print("[" + label + "] " + line.rstrip(), flush=True)

        forwarder = threading.Thread(target=forward_output, daemon=True)
        forwarder.start()
        timed_out = False
        next_heartbeat = 30
        while process.poll() is None:
            elapsed = int(time.monotonic() - started)
            if elapsed >= timeout:
                process.kill()
                timed_out = True
                break
            if elapsed >= next_heartbeat:
                print("[" + label + "] still running (" + str(elapsed) + "s)", flush=True)
                next_heartbeat += 30
            time.sleep(1)

        returncode = process.wait()
        forwarder.join(timeout=10)
        output = "".join(output_lines)
        Path("/kaggle/working/" + label + ".log").write_text(output, encoding="utf-8")
        if timed_out:
            message = "TIMEOUT after %ss\\n%s" % (timeout, output[-4000:])
            Path("/kaggle/working/" + label + ".log").write_text(message, encoding="utf-8")
            raise RuntimeError(label + " timed out; see /kaggle/working/" + label + ".log")
        if returncode != 0:
            raise RuntimeError(label + " failed with exit code " + str(returncode))
        return subprocess.CompletedProcess(command, returncode, output, "")

    print("--- STEP 3: Checking dependencies ---")
    dependency_probe = subprocess.run(
        [sys.executable, "-c", "import torch, torchaudio, librosa, whisper, demucs.separate, vocos, encodec, imageio_ffmpeg, muq"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    if dependency_probe.returncode != 0:
        run_logged(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-cache-dir",
                "--prefer-binary",
                "--no-deps",
                "--timeout",
                "60",
                "--retries",
                "1",
                "librosa",
                "openai-whisper",
                "demucs==4.0.1",
                # demucs's own runtime deps (--no-deps skips these too): demucs.separate
                # does `from dora.log import fatal`, which on import executes dora's
                # __init__.py -> explore/conf/grid/git_save/link/main/shep/xp submodules.
                # Traced the ACTUAL import chain (not the full dora requirements.txt,
                # most of which -- hydra-core, pytorch_lightning -- isn't touched by this
                # narrow usage) via github.com/facebookresearch/dora source:
                "dora-search",
                "treetable",
                "omegaconf",
                "antlr4-python3-runtime==4.9.*",
                "retrying",
                "submitit",
                "cloudpickle",
                "typing_extensions",
                "openunmix",  # htdemucs.py (the model actually used) needs openunmix.filtering
                "imageio-ffmpeg",
                "vocos",
                "muq",
                # muq's own runtime deps (--no-deps above skips these, so list them
                # explicitly -- confirmed via PyPI metadata for x-clip; torch/einops
                # are already covered above). Deliberately NOT adding torchvision here:
                # x-clip's metadata pins it to an exact torch version, and blindly
                # installing --no-deps risks a torch/torchvision ABI mismatch worse than
                # the missing-module error this is fixing, with no evidence (the actual
                # traceback) that torchvision is even missing on the Kaggle base image.
                "x-clip",
                "beartype",
                "ftfy",
                "wcwidth",
                "regex",
                "encodec",
                "einops",
                "huggingface-hub",
                "julius",
                "lameenc",
                "more-itertools",
                "numba",
                "pyyaml",
                "safetensors",
                "scipy",
                "sphn",
                "tiktoken",
                "tqdm",
            ],
            "install",
            900,
        )
        dependency_verify = subprocess.run(
            [sys.executable, "-c", "import torch, torchaudio, librosa, whisper, demucs.separate, vocos, encodec, imageio_ffmpeg, muq"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
        if dependency_verify.returncode != 0:
            Path("/kaggle/working/dependency_verify.log").write_text(
                (dependency_verify.stdout or "") + chr(10) + (dependency_verify.stderr or ""),
                encoding="utf-8",
            )
            raise RuntimeError("Dependency import failed after fast install; see dependency_verify.log")
    else:
        print("All preprocessing dependencies are already available.", flush=True)

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
        timeout=120,
    )
    torch_probe_output = (torch_probe.stdout or "") + chr(10) + (torch_probe.stderr or "")
    Path("/kaggle/working/torch_probe.log").write_text(torch_probe_output, encoding="utf-8")
    print(torch_probe_output, flush=True)
    if torch_probe.returncode != 0:
        print("CUDA smoke test failed; installing a P100-compatible Torch pair.", flush=True)
        run_logged(
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
            "torch_repair",
            1200,
        )
        repaired_probe = subprocess.run(
            [sys.executable, "-c", "import torch; print('torch=%s cuda=%s available=%s' % (torch.__version__, torch.version.cuda, torch.cuda.is_available())); print(torch.randn((2, 2), device='cuda') @ torch.randn((2, 2), device='cuda')) if torch.cuda.is_available() else None"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
        repaired_output = (repaired_probe.stdout or "") + chr(10) + (repaired_probe.stderr or "")
        Path("/kaggle/working/torch_probe_repaired.log").write_text(repaired_output, encoding="utf-8")
        print(repaired_output, flush=True)
        if repaired_probe.returncode != 0 or "available=True" not in repaired_output:
            raise RuntimeError("CUDA is still unavailable after the P100 Torch repair; see torch_probe_repaired.log")
        torch_probe_output = repaired_output
    elif "available=True" not in torch_probe_output:
        raise RuntimeError("Kaggle không có GPU CUDA khả dụng; dừng trước khi rơi xuống CPU.")

    # Add source to python path
    os.environ["PYTHONPATH"] = str(source_root) + os.pathsep + os.environ.get("PYTHONPATH", "")

    print("--- STEP 4: Preprocessing ALL audio tracks ---")
    preprocessed_dir = Path("/kaggle/working/processed_dataset")
    preprocessed_dir.mkdir(parents=True, exist_ok=True)

    preprocess_command = [
        sys.executable, str(source_root / "cli.py"), "preprocess-raw",
        "--input", str(raw_dataset),
        "--output", str(preprocessed_dir),
        "--whisper-model", "{whisper_model}",
        "--keep-separated-count", "0",
        "--demucs-device", "cuda",
        "--whisper-device", "cuda"
    ]
    max_files_value = "{kernel_max_files}"
    if max_files_value:
        print("Preprocessing limit: " + max_files_value + " file(s)", flush=True)
        preprocess_command.extend(["--max-files", max_files_value])
    # Scale the timeout with the actual file count -- a flat 1800s here was fine
    # for a 1-2 file smoke test but silently killed a real 250-file run at the
    # 30-minute mark while it was still healthily processing song 46/250.
    # Budget: fixed one-time model warm-up overhead + a generous per-file margin
    # (measured steady-state was ~32-39s/file; 90s/file leaves ample headroom).
    preprocess_timeout = max(1800, 300 + 90 * int(max_files_value)) if max_files_value else 43200
    preprocess_result = run_logged(preprocess_command, "preprocess", preprocess_timeout)
    records_path = preprocessed_dir / "records.jsonl"
    record_count = sum(1 for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()) if records_path.exists() else 0
    if preprocess_result.returncode != 0 and record_count == 0:
        raise RuntimeError(f"Preprocessing failed without usable records. See /kaggle/working/preprocess.log")
    print(f"Preprocessing produced {{record_count}} usable records.", flush=True)
    print("Output kept at /kaggle/working/processed_dataset -- this kernel's own retained "
          "output. Attach it to downstream kernels via kernel_sources (this kernel's ref), "
          "no re-upload or credentials needed.", flush=True)

    print("--- ALL PROCESSES COMPLETED SUCCESSFULLY ---")
    Path("/kaggle/working/success.txt").write_text("success", encoding="utf-8")

except Exception as e:
    tb = traceback.format_exc()
    print("Error occurred during preprocessing:")
    print(tb)
    Path("/kaggle/working/error.txt").write_text(tb, encoding="utf-8")
    # Propagate the failure so Kaggle reports ERROR instead of a misleading
    # COMPLETE kernel whose output contains no processed records.
    raise
'''

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-files", type=int, default=None, help="Limit how many raw files to preprocess.")
    parser.add_argument("--whisper-model", type=str, default="tiny", help="Whisper model size for lyric transcription (tiny/base/small/...).")
    args = parser.parse_args()

    # Resolved parent because it is located inside the scripts/ directory
    project_root = Path(__file__).resolve().parents[1]
    tokens = kaggle_auth_environment(load_kaggle_api_tokens())
    username = resolve_kaggle_username(tokens.get("KAGGLE_USERNAME"))
    cli = kaggle_cli_command()

    if not username or not kaggle_auth_available(tokens) or not cli:
        print("❌ Error: Missing Kaggle credentials.")
        return

    # Credentials are passed in-memory so project .env remains authoritative.

    raw_dataset_ref = os.getenv("KAGGLE_RAW_DATASET_REF") or tokens.get("KAGGLE_RAW_DATASET_REF", "sonlest/vietnamese-music-dataset-version3-part6")
    raw_dataset_slug = raw_dataset_ref.split("/")[-1]
    
    if args.max_files is not None and args.max_files < 1:
        raise ValueError("--max-files must be a positive integer")
    max_files = args.max_files

    run_id = f"preprocess-all-{int(time.time())}"
    job_dir = project_root / "outputs" / "kaggle_preprocess" / run_id
    dataset_dir = job_dir / "dataset"
    kernel_dir = job_dir / "kernel"
    
    for d in (dataset_dir, kernel_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("======================================================================")
    print(f"🚀 Initializing Preprocess Request: {run_id}")
    print(f"   Source Dataset: {raw_dataset_ref}")
    print("======================================================================")

    # 1. Zip source code
    write_source_zip(project_root, dataset_dir / "genmusic_vn_source.zip")

    # 2. Upload source code zip as a Kaggle Dataset
    source_dataset_slug = f"genmusic-source-{run_id}"
    source_dataset_ref = f"{username}/{source_dataset_slug}"
    
    (dataset_dir / "dataset-metadata.json").write_text(json.dumps({
        "title": f"GenMusic Source {run_id}",
        "id": source_dataset_ref,
        "licenses": [{"name": "other"}]
    }, indent=2))

    print("📤 Uploading source code to Kaggle...")
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
    
    kernel_script = _kernel_script_content(raw_dataset_slug, max_files, args.whisper_model)
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
    print("🚀 Pushing Preprocess Kernel to Kaggle...")
    push_result = subprocess.run(
        cli + ["kernels", "push", "-p", str(kernel_dir)],
        env={**os.environ, **tokens},
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    push_output = (push_result.stdout or "") + "\n" + (push_result.stderr or "")
    print(push_output, end="")
    if push_result.returncode != 0 or "kernel push error" in push_output.lower() or "maximum batch gpu session count" in push_output.lower():
        raise RuntimeError("Kaggle không tạo được kernel; kiểm tra quota GPU hoặc các job đang chạy.")

    print("\n✅ PREPROCESS REQUEST SUBMITTED SUCCESSFULLY!")
    print("Watch live logs on Kaggle Web UI:")
    print(f"➔ https://www.kaggle.com/code/{kernel_ref}")
    print("\nOnce it completes, the processed dataset is this kernel's own retained output")
    print("(no separate Dataset upload, no credentials embedded in the shared code).")
    print("Point downstream kernels (train/distill) at it with:")
    print(f'  KAGGLE_PROCESSED_KERNEL_REF={kernel_ref}')

if __name__ == "__main__":
    main()
