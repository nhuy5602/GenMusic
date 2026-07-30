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

# Three genuinely distinct lyrics, paired with three DIFFERENT real style_anchor
# embeddings pulled from the precomputed dataset (not text descriptions -- the
# model was only ever trained on audio-derived MuQ-MuLan embeddings, and
# generate_audio has no text->style path, so a "style" string alone is a no-op).
LYRICS = [
    "Dem nay mua roi tren loi mon xua, long anh nho em nhieu nguoi oi co biet chang",
    "Nang xuan tren pho phuong rop hoa, tieng cuoi vang len khap noi gan xa",
    "Bien xanh song vo bo cat trang, gio mang huong vi cua mua he sang",
]
SHORT_NAMES = ["ballad", "upbeat", "chill"]


def _kernel_script_content(guidance_scale: float = 1.0) -> str:
    lyrics_literal = json.dumps(LYRICS, ensure_ascii=False)
    names_literal = json.dumps([f"{n}_gs{guidance_scale}" for n in SHORT_NAMES])
    guidance_scale_literal = repr(float(guidance_scale))
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
    print("--- STEP 1: Locating checkpoint and precomputed latent dataset ---")
    input_dir = Path("/kaggle/input")
    checkpoint_candidates = (
        list(input_dir.rglob("distilled_student_resumed.pt"))
        or list(input_dir.rglob("distilled_student_scratch.pt"))
        or list(input_dir.rglob("distilled_student.pt"))
        or list(input_dir.rglob("my_trained_model.pt"))
    )
    if not checkpoint_candidates:
        raise RuntimeError("Could not find distilled_student_resumed.pt, distilled_student_scratch.pt, distilled_student.pt, or my_trained_model.pt in /kaggle/input.")
    checkpoint_path = checkpoint_candidates[0]
    print(f"Using checkpoint: {{checkpoint_path.resolve()}}")

    records_file = next(input_dir.rglob("records.jsonl"), None)
    if not records_file:
        raise RuntimeError("Could not find records.jsonl (precomputed latent dataset) in /kaggle/input.")
    dataset_dir = records_file.parent
    print(f"Using dataset: {{dataset_dir.resolve()}}")

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

    print("--- STEP 4: Picking 3 distinct real style_anchor embeddings from the dataset ---")
    from src.models.text_to_music_diffusion import load_checkpoint, render_mel_to_wav, denormalize_mel
    from src.training.self_diffusion import _load_mel
    from src.models.cfm_flow import sample_cfm

    records = [json.loads(line) for line in records_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    style_records = [r for r in records if r.get("style_embed_path")]
    print(f"{{len(style_records)}} / {{len(records)}} records have a style_embed_path")
    if len(style_records) < 3:
        raise RuntimeError(f"Need at least 3 records with style_embed_path, found {{len(style_records)}}.")

    # Spread the picks across the dataset instead of taking the first 3 in a
    # row, so the 3 style anchors are more likely to actually differ.
    picks = [style_records[i] for i in [0, len(style_records)//2, len(style_records)-1]]
    style_anchors = []
    for r in picks:
        anchor = _load_mel(dataset_dir / r["style_embed_path"]).float().view(-1)
        style_anchors.append(anchor)
        print(f"Picked style anchor from record {{r['id']}}: shape={{tuple(anchor.shape)}}, norm={{anchor.norm().item():.4f}}")

    print("--- STEP 5: Loading checkpoint and sampling raw latents (pre-decode) ---")
    model, config, payload = load_checkpoint(str(checkpoint_path), device=device)

    lyrics = {lyrics_literal}
    names = {names_literal}
    frames = int(6.0 * config.sample_rate / config.hop_length)

    latents = []
    out_dir = Path("/kaggle/working/samples")
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, text, anchor in zip(names, lyrics, style_anchors):
        style_prompt = anchor.unsqueeze(0).to(device)
        mel = sample_cfm(
            model, [text], frames, config=config, device=device,
            steps=24, seed=5602, style_prompt=style_prompt, guidance_scale={guidance_scale_literal},
        )  # (1, n_mels, frames)
        latent = mel.squeeze(0).detach().float().cpu()
        latents.append(latent)
        print(f"{{name}}: latent shape={{tuple(latent.shape)}} mean={{latent.mean().item():.6f}} std={{latent.std().item():.6f}} "
              f"min={{latent.min().item():.6f}} max={{latent.max().item():.6f}}")

        denorm = denormalize_mel(latent, config)
        wav_path = out_dir / f"{{name}}.wav"
        render_mel_to_wav(denorm, wav_path, config, vocoder_type="vocos")
        mp3_path = wav_path.with_suffix(".mp3")
        ffmpeg = shutil.which("ffmpeg")
        subprocess.run([ffmpeg, "-y", "-i", str(wav_path), "-b:a", "64k", str(mp3_path)], check=True, capture_output=True)
        with open(mp3_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        print(f"===SAMPLE_START:{{name}}===")
        print(b64)
        print(f"===SAMPLE_END:{{name}}===")

    print("--- STEP 6: Pairwise distance between the 3 raw latents ---")
    import itertools
    distances = {{}}
    for (i, a), (j, b) in itertools.combinations(enumerate(latents), 2):
        flat_a, flat_b = a.flatten(), b.flatten()
        l2 = torch.norm(flat_a - flat_b).item()
        cos = torch.nn.functional.cosine_similarity(flat_a.unsqueeze(0), flat_b.unsqueeze(0)).item()
        rel_l2 = l2 / max(1e-8, torch.norm(flat_a).item())
        key = f"{{names[i]}}_vs_{{names[j]}}"
        distances[key] = {{"l2": l2, "relative_l2": rel_l2, "cosine_similarity": cos}}
        print(f"{{key}}: L2={{l2:.4f}} (relative={{rel_l2:.4f}}), cosine_similarity={{cos:.6f}}")

    # Anchor: distance between two independent noise draws of the same shape,
    # as a reference for "how far apart do two truly different things get".
    noise_a = torch.randn_like(latents[0]).flatten()
    noise_b = torch.randn_like(latents[0]).flatten()
    noise_l2 = torch.norm(noise_a - noise_b).item()
    print(f"reference noise_vs_noise: L2={{noise_l2:.4f}}")

    print("DISTANCES_JSON:" + json.dumps({{"pairwise": distances, "noise_reference_l2": noise_l2}}, ensure_ascii=False))
    print("LATENT DIVERSITY CHECK COMPLETED SUCCESSFULLY!")
except Exception as e:
    import traceback
    print("ERROR OCCURRED DURING KERNEL EXECUTION:")
    traceback.print_exc()
    sys.exit(1)
'''


def run_kaggle_latent_diversity_check(checkpoint_kernel_ref: str, guidance_scale: float = 1.0, dataset_kernel_ref: str | None = None) -> str:
    project_root = Path(__file__).resolve().parents[1]

    tokens = kaggle_auth_environment(load_kaggle_api_tokens())
    username = resolve_kaggle_username(tokens.get("KAGGLE_USERNAME"))
    cli = kaggle_cli_command()
    if not username or not kaggle_auth_available(tokens) or not cli:
        raise RuntimeError("Missing KAGGLE_USERNAME or Kaggle auth (KAGGLE_API_TOKEN=KGAT_... / legacy KAGGLE_KEY)")

    run_id = f"latentdiv-{int(time.time())}"
    job_dir = project_root / "outputs" / "kaggle_latent_diversity" / run_id
    dataset_dir = job_dir / "dataset"
    kernel_dir = job_dir / "kernel"

    for d in (dataset_dir, kernel_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Initializing Kaggle Latent Diversity Check Job: {run_id}")
    print("=" * 70)

    print("Zipping local source code...")
    write_source_zip(project_root, dataset_dir / "genmusic_vn_source.zip")

    source_dataset_slug = f"genmusic-source-{run_id}"
    source_dataset_ref = f"{username}/{source_dataset_slug}"

    dataset_title = f"Latent Div {run_id}"
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

    kernel_slug = f"genmusic-latentdiv-{int(time.time())}"
    kernel_ref = f"{username}/{kernel_slug}"

    (kernel_dir / "run_latent_div.py").write_text(_kernel_script_content(guidance_scale), encoding="utf-8")

    (kernel_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": kernel_ref,
        "title": kernel_slug,
        "code_file": "run_latent_div.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [source_dataset_ref],
        "kernel_sources": [checkpoint_kernel_ref] + ([dataset_kernel_ref] if dataset_kernel_ref else []),
        "competition_sources": []
    }, indent=2))

    print(f"Pushing Latent Diversity Kernel to Kaggle: {kernel_ref}...")

    time.sleep(20)
    for attempt in range(3):
        try:
            subprocess.run(cli + ["kernels", "push", "-p", str(kernel_dir)], env={**os.environ, **tokens}, check=True)
            print("\nLATENT DIVERSITY JOB SUBMITTED SUCCESSFULLY!")
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
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--dataset-kernel-ref", type=str, default=None, help="Second kernel to mount for records.jsonl, if the checkpoint kernel didn't also output the dataset (e.g. a resume-pilot run that only saved the checkpoint).")
    args = parser.parse_args()
    run_kaggle_latent_diversity_check(args.checkpoint_kernel_ref, args.guidance_scale, args.dataset_kernel_ref)


if __name__ == "__main__":
    main()
