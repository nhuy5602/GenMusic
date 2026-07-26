"""Reference test: does the frozen DiffRhythm2 TEACHER itself produce diverse
latents for different (lyric, style) conditions, using the exact same 3 real
style anchors + 3 lyrics as the student diversity check? This is the missing
baseline -- if the teacher (a real, properly-trained ~1.1B model) also
collapses under this exact test harness, that would implicate the test
methodology itself rather than the student. If the teacher clearly
differentiates, that confirms the student's collapse is a real training
failure, not an artifact of how the test is set up.

Only generates a SINGLE native block (block_size frames, ~2s at the teacher's
5Hz latent rate) with no preceding "clean" block history, i.e. equivalent to
generating the very first block of a song from scratch -- this avoids having
to reimplement DiffRhythm2's full block-autoregressive multi-block generation
loop, which is out of scope for a quick reference check.
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

LYRICS = [
    "Dem nay mua roi tren loi mon xua, long anh nho em nhieu nguoi oi co biet chang",
    "Nang xuan tren pho phuong rop hoa, tieng cuoi vang len khap noi gan xa",
    "Bien xanh song vo bo cat trang, gio mang huong vi cua mua he sang",
]
SHORT_NAMES = ["ballad", "upbeat", "chill"]


def _kernel_script_content() -> str:
    lyrics_literal = json.dumps(LYRICS, ensure_ascii=False)
    names_literal = json.dumps(SHORT_NAMES)
    return f'''import itertools
import json
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

try:
    print("--- STEP 1: Locating source code and precomputed latent dataset (for style anchors) ---")
    input_dir = Path("/kaggle/input")
    records_file = next(input_dir.rglob("records.jsonl"), None)
    if not records_file:
        raise RuntimeError("Could not find records.jsonl in /kaggle/input.")
    dataset_dir = records_file.parent
    print(f"Using dataset: {{dataset_dir.resolve()}}")

    source_dataset_dir = next(
        (d for d in input_dir.rglob("*") if d.is_dir() and "genmusic-source-" in d.name.lower()),
        None
    )
    if not source_dataset_dir:
        raise RuntimeError("Could not find the source code dataset directory.")
    source_root = Path("/kaggle/working/GenMusic")
    shutil.copytree(source_dataset_dir, source_root, dirs_exist_ok=True)

    print("--- STEP 1.2: Downloading DiffRhythm2 official repository ---")
    diffrhythm2_tar = "/kaggle/working/diffrhythm2.tar.gz"
    urllib.request.urlretrieve("https://github.com/ASLP-lab/DiffRhythm2/archive/refs/heads/main.tar.gz", diffrhythm2_tar)
    with tarfile.open(diffrhythm2_tar) as tar:
        tar.extractall(str(source_root))

    print("--- STEP 1.5: Installing python dependencies ---")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "text2phonemesequence"], check=True)
    subprocess.run(["apt-get", "update", "-y"], check=False)
    subprocess.run(["apt-get", "install", "-y", "--fix-missing", "espeak-ng", "ffmpeg"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "soundfile"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(source_root / "DiffRhythm2-main/requirements.txt")], check=True)

    import os
    os.environ["PYTHONPATH"] = (
        str(source_root) + os.pathsep + str(source_root / "DiffRhythm2-main") + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    sys.path.insert(0, str(source_root))
    sys.path.insert(0, str(source_root / "DiffRhythm2-main"))

    print("--- STEP 2: Checking CUDA ---")
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch={{torch.__version__}} device={{device}}")

    print("--- STEP 3: Loading frozen DiffRhythm2 teacher + lyric tokenizer ---")
    from src.training.distill_training import _load_teacher, _load_lyric_tokenizer, _tokenize_lyrics_batch
    from src.training.self_diffusion import _load_mel

    teacher, teacher_config, teacher_status = _load_teacher("ASLP-lab/DiffRhythm2", None, device)
    print(f"Teacher load status: {{teacher_status}}")
    if teacher is None:
        raise RuntimeError(f"Could not load teacher: {{teacher_status}}")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    parse_lyrics_fn, tokenizer_status = _load_lyric_tokenizer()
    print(f"Lyric tokenizer status: {{tokenizer_status}}")
    if parse_lyrics_fn is None:
        raise RuntimeError(f"Could not load lyric tokenizer: {{tokenizer_status}}")

    block_size = teacher_config.get("block_size", 10) if teacher_config else 10
    T = block_size  # single native block, no preceding clean history
    print(f"Using block_size={{block_size}} as the single-block generation length")

    print("--- STEP 4: Picking the SAME 3 real style anchors used in the student test ---")
    records = [json.loads(line) for line in records_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    style_records = [r for r in records if r.get("style_embed_path")]
    picks = [style_records[i] for i in [0, len(style_records)//2, len(style_records)-1]]
    style_anchors = []
    for r in picks:
        anchor = _load_mel(dataset_dir / r["style_embed_path"]).float().view(-1)
        style_anchors.append(anchor.to(device))
        print(f"Picked style anchor from record {{r['id']}}: shape={{tuple(anchor.shape)}}")

    def build_text_noisy_mask(batch_size, text_len, T, token_valid, device):
        total_len = text_len + T
        mask = torch.zeros((total_len, total_len), dtype=torch.bool, device=device)
        is_text = torch.zeros(total_len, dtype=torch.bool, device=device)
        is_text[:text_len] = True
        is_noisy = ~is_text
        col_is_text = is_text.unsqueeze(0)
        row_is_noisy = is_noisy.unsqueeze(1)
        row_is_text = is_text.unsqueeze(1)
        col_is_noisy = is_noisy.unsqueeze(0)
        mask = mask | col_is_text
        mask = mask | (row_is_noisy & col_is_noisy)
        mask = mask & ~(row_is_text & col_is_noisy)
        mask_4d = mask.unsqueeze(0).unsqueeze(1).repeat(batch_size, 1, 1, 1)
        audio_valid = torch.ones((batch_size, T), dtype=torch.bool, device=device)
        full_valid = torch.cat([token_valid, audio_valid], dim=1)
        mask_4d = mask_4d & full_valid.unsqueeze(1).unsqueeze(2)
        return mask_4d

    def teacher_sample(text, style_prompt, steps=24, seed=5602):
        torch.manual_seed(seed)
        xt = torch.randn(1, T, 64, device=device)
        token_ids, token_valid = _tokenize_lyrics_batch(parse_lyrics_fn, [text], device)
        text_len = token_ids.shape[1]
        text_emb = teacher.text_embed(token_ids)
        text_time = torch.full((1, text_len), -1.0, device=device, dtype=xt.dtype)
        text_position_ids = torch.arange(text_len, device=device).unsqueeze(0)
        dt = 1.0 / steps
        style_batched = style_prompt.unsqueeze(0) if style_prompt.dim() == 1 else style_prompt
        with torch.no_grad():
            for step in range(steps):
                t_val = step / steps
                t = torch.full((1,), t_val, device=device, dtype=xt.dtype)
                noisy_latent = teacher.latent_embed(xt)
                noisy_time = t[:, None].repeat(1, T)
                noisy_position_ids = torch.arange(T, device=device).unsqueeze(0)
                x = torch.cat([text_emb, noisy_latent], dim=1)
                time_cat = torch.cat([text_time, noisy_time], dim=1)
                position_ids = torch.cat([text_position_ids, noisy_position_ids], dim=1)
                attn_mask = build_text_noisy_mask(1, text_len, T, token_valid, device)
                outputs = teacher(
                    x=x, time=time_cat, position_ids=position_ids, style_prompt=style_batched,
                    attn_mask=attn_mask, use_cache=False, past_key_value=None,
                )
                pred = outputs[0] if isinstance(outputs, tuple) else outputs
                v = pred[:, text_len:]
                xt = xt + v * dt
        return xt.squeeze(0).detach().float().cpu()  # (T, 64)

    print("--- STEP 5: Sampling teacher latents for 3 (lyric, style) conditions ---")
    from src.models.latent_codec import load_frozen_decoder
    import base64
    import soundfile as sf

    decoder_handle = load_frozen_decoder(device)
    out_dir = Path("/kaggle/working/teacher_samples")
    out_dir.mkdir(parents=True, exist_ok=True)

    lyrics = {lyrics_literal}
    names = {names_literal}
    latents = []
    for name, text, anchor in zip(names, lyrics, style_anchors):
        latent = teacher_sample(text, anchor)
        latents.append(latent)
        print(f"{{name}}: teacher latent shape={{tuple(latent.shape)}} mean={{latent.mean().item():.6f}} "
              f"std={{latent.std().item():.6f}} min={{latent.min().item():.6f}} max={{latent.max().item():.6f}}")

        with torch.no_grad():
            latent_for_decode = latent.transpose(0, 1).unsqueeze(0).to(device)  # (1, 64, T)
            chunk_size = min(20, max(1, latent_for_decode.shape[2]))
            audio_tensor = decoder_handle.decoder.decode_audio(
                latent_for_decode, overlap=min(5, chunk_size - 1), chunk_size=chunk_size,
            )
        audio = audio_tensor.squeeze(0).squeeze(0).cpu().numpy()
        wav_path = out_dir / f"teacher_{{name}}.wav"
        sf.write(str(wav_path), audio, decoder_handle.sampling_rate)
        mp3_path = wav_path.with_suffix(".mp3")
        ffmpeg = shutil.which("ffmpeg")
        subprocess.run([ffmpeg, "-y", "-i", str(wav_path), "-b:a", "96k", str(mp3_path)], check=True, capture_output=True)
        with open(mp3_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        print(f"===SAMPLE_START:teacher_{{name}}===")
        print(b64)
        print(f"===SAMPLE_END:teacher_{{name}}===")

    print("--- STEP 6: Pairwise distance between the 3 teacher latents ---")
    distances = {{}}
    for (i, a), (j, b) in itertools.combinations(enumerate(latents), 2):
        flat_a, flat_b = a.flatten(), b.flatten()
        l2 = torch.norm(flat_a - flat_b).item()
        cos = torch.nn.functional.cosine_similarity(flat_a.unsqueeze(0), flat_b.unsqueeze(0)).item()
        rel_l2 = l2 / max(1e-8, torch.norm(flat_a).item())
        key = f"{{names[i]}}_vs_{{names[j]}}"
        distances[key] = {{"l2": l2, "relative_l2": rel_l2, "cosine_similarity": cos}}
        print(f"{{key}}: L2={{l2:.4f}} (relative={{rel_l2:.4f}}), cosine_similarity={{cos:.6f}}")

    noise_a = torch.randn_like(latents[0]).flatten()
    noise_b = torch.randn_like(latents[0]).flatten()
    noise_l2 = torch.norm(noise_a - noise_b).item()
    print(f"reference noise_vs_noise: L2={{noise_l2:.4f}}")

    print("TEACHER_DISTANCES_JSON:" + json.dumps({{"pairwise": distances, "noise_reference_l2": noise_l2}}, ensure_ascii=False))
    print("TEACHER DIVERSITY CHECK COMPLETED SUCCESSFULLY!")
except Exception as e:
    import traceback
    print("ERROR OCCURRED DURING KERNEL EXECUTION:")
    traceback.print_exc()
    raise
'''


def run_kaggle_teacher_diversity_check(dataset_kernel_ref: str) -> str:
    project_root = Path(__file__).resolve().parents[1]

    tokens = kaggle_auth_environment(load_kaggle_api_tokens())
    username = resolve_kaggle_username(tokens.get("KAGGLE_USERNAME"))
    cli = kaggle_cli_command()
    if not username or not kaggle_auth_available(tokens) or not cli:
        raise RuntimeError("Missing KAGGLE_USERNAME or Kaggle auth (KAGGLE_API_TOKEN=KGAT_... / legacy KAGGLE_KEY)")

    run_id = f"teacherdiv-{int(time.time())}"
    job_dir = project_root / "outputs" / "kaggle_teacher_diversity" / run_id
    dataset_dir = job_dir / "dataset"
    kernel_dir = job_dir / "kernel"

    for d in (dataset_dir, kernel_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Initializing Kaggle Teacher Diversity Check Job: {run_id}")
    print("=" * 70)

    print("Zipping local source code...")
    write_source_zip(project_root, dataset_dir / "genmusic_vn_source.zip")

    source_dataset_slug = f"genmusic-source-{run_id}"
    source_dataset_ref = f"{username}/{source_dataset_slug}"

    dataset_title = f"Teacher Div {run_id}"
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

    kernel_slug = f"genmusic-teacherdiv-{int(time.time())}"
    kernel_ref = f"{username}/{kernel_slug}"

    (kernel_dir / "run_teacher_div.py").write_text(_kernel_script_content(), encoding="utf-8")

    (kernel_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": kernel_ref,
        "title": kernel_slug,
        "code_file": "run_teacher_div.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [source_dataset_ref],
        "kernel_sources": [dataset_kernel_ref],
        "competition_sources": []
    }, indent=2))

    print(f"Pushing Teacher Diversity Kernel to Kaggle: {kernel_ref}...")

    time.sleep(20)
    for attempt in range(3):
        try:
            subprocess.run(cli + ["kernels", "push", "-p", str(kernel_dir)], env={**os.environ, **tokens}, check=True)
            print("\nTEACHER DIVERSITY JOB SUBMITTED SUCCESSFULLY!")
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
    run_kaggle_teacher_diversity_check(args.dataset_kernel_ref)


if __name__ == "__main__":
    main()
