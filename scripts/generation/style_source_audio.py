"""Reconstructs (vocal+backing, ~= original pre-separation mix) the first 10s
of the 3 specific songs used as style_anchor in the CFM diversity tests, so
they can actually be listened to. The true original (pre-Demucs) file itself
isn't retained anywhere past preprocessing -- this sums the separated stems
back together as the closest available approximation (small Demucs artifacts
aside, this is audibly very close to the source mix).
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

RECORD_IDS = ["-6s_eRHYqVM", "l_GigiDg3Wc", "ZWY5WXssw_w"]
NAMES = ["ballad", "upbeat", "chill"]


def _kernel_script_content() -> str:
    ids_literal = json.dumps(RECORD_IDS)
    names_literal = json.dumps(NAMES)
    return f'''import base64
import json
import shutil
import subprocess
from pathlib import Path

try:
    print("--- STEP 1: Locating raw-audio-mode processed dataset ---")
    input_dir = Path("/kaggle/input")
    records_file = next(input_dir.rglob("records.jsonl"), None)
    if not records_file:
        raise RuntimeError("Could not find records.jsonl in /kaggle/input.")
    dataset_dir = records_file.parent
    print(f"Using dataset: {{dataset_dir.resolve()}}")

    print("--- STEP 1.5: Installing minimal dependencies (ffmpeg, soundfile) ---")
    subprocess.run(["apt-get", "update", "-y"], check=False)
    subprocess.run(["apt-get", "install", "-y", "--fix-missing", "ffmpeg"], check=True)
    import sys as _sys
    subprocess.run([_sys.executable, "-m", "pip", "install", "-q", "soundfile"], check=True)

    import torch
    import soundfile as sf

    records_by_id = {{}}
    for line in records_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        records_by_id[r.get("id")] = r

    target_ids = {ids_literal}
    names = {names_literal}
    sample_rate = 24000
    duration_samples = sample_rate * 10

    out_dir = Path("/kaggle/working/style_source_clips")
    out_dir.mkdir(parents=True, exist_ok=True)

    for record_id, name in zip(target_ids, names):
        record = records_by_id.get(record_id)
        if record is None:
            print(f"WARNING: record {{record_id}} not found in this dataset, skipping.")
            continue
        vocal_path = dataset_dir / record["vocal_wav_path"] if not Path(record["vocal_wav_path"]).is_absolute() else Path(record["vocal_wav_path"])
        backing_path = dataset_dir / record["backing_wav_path"] if not Path(record["backing_wav_path"]).is_absolute() else Path(record["backing_wav_path"])
        vocal = torch.load(vocal_path, map_location="cpu", weights_only=True)
        backing = torch.load(backing_path, map_location="cpu", weights_only=True)
        length = min(vocal.shape[-1], backing.shape[-1], duration_samples)
        mix = (vocal[:length] + backing[:length]).numpy()

        wav_path = out_dir / f"{{name}}_source.wav"
        sf.write(str(wav_path), mix, sample_rate)
        mp3_path = wav_path.with_suffix(".mp3")
        ffmpeg = shutil.which("ffmpeg")
        subprocess.run([ffmpeg, "-y", "-i", str(wav_path), "-b:a", "96k", str(mp3_path)], check=True, capture_output=True)
        with open(mp3_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        print(f"===SAMPLE_START:{{name}}_source===")
        print(b64)
        print(f"===SAMPLE_END:{{name}}_source===")
        print(f"Wrote {{name}}_source.mp3 from record {{record_id}}, {{length/sample_rate:.2f}}s")

    print("STYLE SOURCE AUDIO EXTRACTION COMPLETED SUCCESSFULLY!")
except Exception as e:
    import traceback
    print("ERROR OCCURRED DURING KERNEL EXECUTION:")
    traceback.print_exc()
    raise
'''


def run_kaggle_style_source_audio(dataset_kernel_ref: str) -> str:
    project_root = Path(__file__).resolve().parents[1]

    tokens = kaggle_auth_environment(load_kaggle_api_tokens())
    username = resolve_kaggle_username(tokens.get("KAGGLE_USERNAME"))
    cli = kaggle_cli_command()
    if not username or not kaggle_auth_available(tokens) or not cli:
        raise RuntimeError("Missing KAGGLE_USERNAME or Kaggle auth (KAGGLE_API_TOKEN=KGAT_... / legacy KAGGLE_KEY)")

    run_id = f"stylesrc-{int(time.time())}"
    job_dir = project_root / "outputs" / "kaggle_style_source" / run_id
    dataset_dir = job_dir / "dataset"
    kernel_dir = job_dir / "kernel"

    for d in (dataset_dir, kernel_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Initializing Kaggle Style-Source-Audio Job: {run_id}")
    print("=" * 70)

    print("Zipping local source code...")
    write_source_zip(project_root, dataset_dir / "genmusic_vn_source.zip")

    source_dataset_slug = f"genmusic-source-{run_id}"
    source_dataset_ref = f"{username}/{source_dataset_slug}"

    dataset_title = f"Style Src {run_id}"
    assert 6 <= len(dataset_title) <= 50, f"dataset title length {len(dataset_title)} out of range: {dataset_title!r}"
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

    kernel_slug = f"genmusic-stylesrc-{int(time.time())}"
    kernel_ref = f"{username}/{kernel_slug}"

    (kernel_dir / "run_style_source.py").write_text(_kernel_script_content(), encoding="utf-8")

    (kernel_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": kernel_ref,
        "title": kernel_slug,
        "code_file": "run_style_source.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [source_dataset_ref],
        "kernel_sources": [dataset_kernel_ref],
        "competition_sources": []
    }, indent=2))

    print(f"Pushing Style-Source-Audio Kernel to Kaggle: {kernel_ref}...")

    time.sleep(20)
    for attempt in range(3):
        try:
            subprocess.run(cli + ["kernels", "push", "-p", str(kernel_dir)], env={**os.environ, **tokens}, check=True)
            print("\nSTYLE-SOURCE-AUDIO JOB SUBMITTED SUCCESSFULLY!")
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
    parser.add_argument("--dataset-kernel-ref", type=str, required=True)
    args = parser.parse_args()
    run_kaggle_style_source_audio(args.dataset_kernel_ref)


if __name__ == "__main__":
    main()
