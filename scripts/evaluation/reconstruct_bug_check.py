"""Cheap, CPU-only, no-GPU verification of a suspected bug: distill_training.py's
train_epoch() calls reconstruct_full_mix(vocal, backing, config) UNCONDITIONALLY,
even when config.latent_mode=True. In latent mode, records only have
vocal_mel_path (holding the full-mix LATENT, not mel) -- there is no
backing_mel_path, so MusicDiffusionDataset.__getitem__ falls back to
backing_mel = torch.zeros_like(vocal_mel). reconstruct_full_mix then applies
log(exp(vocal)+exp(backing)), a formula meant for log-mel ENERGY, directly to
raw VAE latent channel values.

Closed form (mel_mean cancels out): x1_actual = softplus(mel_std * v) / mel_std,
for v = the true latent value. This script loads the real precomputed latent
dataset, computes the real mel_mean/mel_std via the exact same
estimate_vocal_mel_stats() used by train-distill, and reports how much this
distorts the true target relative to leaving it alone.
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


def _kernel_script_content() -> str:
    return r'''import json
import shutil
import sys
from pathlib import Path

try:
    print("--- STEP 1: Locating latent dataset ---")
    input_dir = Path("/kaggle/input")
    records_file = next(input_dir.rglob("records.jsonl"), None)
    if not records_file:
        raise RuntimeError("Could not find records.jsonl in /kaggle/input.")
    dataset_dir = records_file.parent
    print(f"Using dataset: {dataset_dir.resolve()}")

    print("--- STEP 2: Setting up source code ---")
    source_dataset_dir = next(
        (d for d in input_dir.rglob("*") if d.is_dir() and "genmusic-source-" in d.name.lower()),
        None
    )
    if not source_dataset_dir:
        raise RuntimeError("Could not find the source code dataset directory.")
    source_root = Path("/kaggle/working/GenMusic")
    shutil.copytree(source_dataset_dir, source_root, dirs_exist_ok=True)
    sys.path.insert(0, str(source_root))

    import torch
    from dataclasses import replace
    from src.training.self_diffusion import (
        _filter_training_records, _read_records, _with_absolute_paths, estimate_vocal_mel_stats,
    )
    from src.models.text_to_music_diffusion import MusicDiffusionConfig, reconstruct_full_mix

    config = MusicDiffusionConfig(**json.loads((dataset_dir / "config.json").read_text(encoding="utf-8")))
    print(f"config.latent_mode={config.latent_mode} latent_dim={config.latent_dim}")

    records = [_with_absolute_paths(dataset_dir, r) for r in _filter_training_records(_read_records(dataset_dir))]
    print(f"{len(records)} usable records")

    print("--- STEP 3: Computing REAL mel_mean/mel_std via estimate_vocal_mel_stats (exact train-distill recipe) ---")
    mel_mean, mel_std = estimate_vocal_mel_stats(dataset_dir, records)
    print(f"mel_mean={mel_mean:.6f} mel_std={mel_std:.6f}")
    config_calibrated = replace(config, mel_mean=mel_mean, mel_std=mel_std)

    print("--- STEP 4: Comparing true latent vs. the actual x1 target train_epoch() produces ---")
    from src.training.self_diffusion import _load_mel
    sample_records = records[:min(40, len(records))]
    true_vals = []
    distorted_vals = []
    for r in sample_records:
        latent = _load_mel(Path(r["vocal_mel_path"])).float()  # (channels, T) -- the TRUE encoder latent
        vocal_x1 = latent.transpose(0, 1).unsqueeze(0)  # (1, T, channels), matching train_epoch's layout
        backing_zeros = torch.zeros_like(vocal_x1)  # exact fallback MusicDiffusionDataset produces
        x1_actual = reconstruct_full_mix(vocal_x1, backing_zeros, config_calibrated)
        true_vals.append(vocal_x1.flatten())
        distorted_vals.append(x1_actual.flatten())

    true_all = torch.cat(true_vals)
    distorted_all = torch.cat(distorted_vals)

    diff = distorted_all - true_all
    corr = torch.corrcoef(torch.stack([true_all, distorted_all]))[0, 1].item()

    summary = {
        "mel_mean": mel_mean,
        "mel_std": mel_std,
        "true_latent": {
            "mean": true_all.mean().item(), "std": true_all.std().item(),
            "min": true_all.min().item(), "max": true_all.max().item(),
            "frac_negative": (true_all < 0).float().mean().item(),
        },
        "distorted_x1_actually_trained_on": {
            "mean": distorted_all.mean().item(), "std": distorted_all.std().item(),
            "min": distorted_all.min().item(), "max": distorted_all.max().item(),
            "frac_negative": (distorted_all < 0).float().mean().item(),
        },
        "diff_true_vs_distorted": {
            "mean_abs_diff": diff.abs().mean().item(),
            "max_abs_diff": diff.abs().max().item(),
            "correlation": corr,
        },
        "num_elements_compared": true_all.numel(),
        "num_records_sampled": len(sample_records),
    }
    print("RECONSTRUCT_BUG_JSON:" + json.dumps(summary))
    print("RECONSTRUCT BUG CHECK COMPLETED SUCCESSFULLY!")
except Exception as e:
    import traceback
    print("ERROR OCCURRED DURING KERNEL EXECUTION:")
    traceback.print_exc()
    raise
'''


def run_kaggle_reconstruct_bug_check(dataset_kernel_ref: str) -> str:
    project_root = Path(__file__).resolve().parents[1]

    tokens = kaggle_auth_environment(load_kaggle_api_tokens())
    username = resolve_kaggle_username(tokens.get("KAGGLE_USERNAME"))
    cli = kaggle_cli_command()
    if not username or not kaggle_auth_available(tokens) or not cli:
        raise RuntimeError("Missing KAGGLE_USERNAME or Kaggle auth (KAGGLE_API_TOKEN=KGAT_... / legacy KAGGLE_KEY)")

    run_id = f"reconbugcheck-{int(time.time())}"
    job_dir = project_root / "outputs" / "kaggle_reconstruct_bug_check" / run_id
    dataset_dir = job_dir / "dataset"
    kernel_dir = job_dir / "kernel"

    for d in (dataset_dir, kernel_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Initializing Kaggle Reconstruct-Bug Check Job: {run_id}")
    print("=" * 70)

    print("Zipping local source code...")
    write_source_zip(project_root, dataset_dir / "genmusic_vn_source.zip")

    source_dataset_slug = f"genmusic-source-{run_id}"
    source_dataset_ref = f"{username}/{source_dataset_slug}"

    dataset_title = f"Recon Bug {run_id}"
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

    kernel_slug = f"genmusic-reconbug-{int(time.time())}"
    kernel_ref = f"{username}/{kernel_slug}"

    (kernel_dir / "run_recon_bug_check.py").write_text(_kernel_script_content(), encoding="utf-8")

    (kernel_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": kernel_ref,
        "title": kernel_slug,
        "code_file": "run_recon_bug_check.py",
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

    print(f"Pushing Reconstruct-Bug Check Kernel to Kaggle: {kernel_ref}...")

    time.sleep(20)
    for attempt in range(3):
        try:
            subprocess.run(cli + ["kernels", "push", "-p", str(kernel_dir)], env={**os.environ, **tokens}, check=True)
            print("\nRECONSTRUCT-BUG CHECK JOB SUBMITTED SUCCESSFULLY!")
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
    run_kaggle_reconstruct_bug_check(args.dataset_kernel_ref)


if __name__ == "__main__":
    main()
