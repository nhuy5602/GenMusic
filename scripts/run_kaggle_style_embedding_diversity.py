"""Cheap, model-free check: how diverse are the raw MuQ-MuLan style_anchor
embeddings themselves across the dataset (computed from each song's first 10s
only, see compute_style_embedding's docstring)? If Vietnamese songs' intros
tend to sound similar, the style CONDITIONING itself could already be
low-diversity before the model ever sees it -- this would confound the
CFM-output collapse measurements with a data artifact, not a model failure.
No GPU/DiffRhythm2/heavy deps needed -- just loads precomputed .pt tensors.
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


def _kernel_script_content() -> str:
    return r'''import itertools
import json
from pathlib import Path

try:
    print("--- STEP 1: Locating dataset ---")
    input_dir = Path("/kaggle/input")
    records_file = next(input_dir.rglob("records.jsonl"), None)
    if not records_file:
        raise RuntimeError("Could not find records.jsonl in /kaggle/input.")
    dataset_dir = records_file.parent
    print(f"Using dataset: {dataset_dir.resolve()}")

    import torch

    records = [json.loads(line) for line in records_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    style_records = [r for r in records if r.get("style_embed_path")]
    print(f"{len(style_records)} / {len(records)} records have a style_embed_path")

    def load_anchor(r):
        return torch.load(dataset_dir / r["style_embed_path"], map_location="cpu", weights_only=True).float().view(-1)

    print("--- STEP 2: Pairwise similarity across ALL available style embeddings ---")
    all_anchors = [(r["id"], load_anchor(r)) for r in style_records]
    n = len(all_anchors)
    sims = []
    l2s = []
    for (id_a, a), (id_b, b) in itertools.combinations(all_anchors, 2):
        cos = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
        l2 = torch.norm(a - b).item()
        sims.append(cos)
        l2s.append(l2)

    sims_t = torch.tensor(sims)
    l2s_t = torch.tensor(l2s)
    print(f"Pairs compared: {len(sims)} (from {n} songs)")
    print(f"Cosine similarity across ALL song pairs: mean={sims_t.mean().item():.4f} "
          f"std={sims_t.std().item():.4f} min={sims_t.min().item():.4f} max={sims_t.max().item():.4f}")
    print(f"L2 distance across ALL song pairs: mean={l2s_t.mean().item():.4f} "
          f"std={l2s_t.std().item():.4f} min={l2s_t.min().item():.4f} max={l2s_t.max().item():.4f}")

    # A random-direction reference in the SAME 512-dim space, to know what
    # "genuinely different" looks like at this dimensionality/norm scale.
    dim = all_anchors[0][1].numel()
    mean_norm = torch.stack([a for _, a in all_anchors]).norm(dim=1).mean().item()
    torch.manual_seed(0)
    ref_a = torch.nn.functional.normalize(torch.randn(dim), dim=0) * mean_norm
    ref_b = torch.nn.functional.normalize(torch.randn(dim), dim=0) * mean_norm
    ref_cos = torch.nn.functional.cosine_similarity(ref_a.unsqueeze(0), ref_b.unsqueeze(0)).item()
    ref_l2 = torch.norm(ref_a - ref_b).item()
    print(f"Reference (2 random unit directions, same norm): cosine={ref_cos:.4f} L2={ref_l2:.4f}")

    print("--- STEP 3: The exact 3 style anchors used in the CFM diversity tests ---")
    picks = [style_records[i] for i in [0, len(style_records)//2, len(style_records)-1]]
    three = [(r["id"], load_anchor(r)) for r in picks]
    for (id_a, a), (id_b, b) in itertools.combinations(three, 2):
        cos = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
        l2 = torch.norm(a - b).item()
        print(f"{id_a} vs {id_b}: cosine={cos:.4f} L2={l2:.4f}")

    summary = {
        "num_songs": n,
        "all_pairs_cosine_mean": sims_t.mean().item(),
        "all_pairs_cosine_std": sims_t.std().item(),
        "all_pairs_cosine_min": sims_t.min().item(),
        "all_pairs_cosine_max": sims_t.max().item(),
        "all_pairs_l2_mean": l2s_t.mean().item(),
        "all_pairs_l2_std": l2s_t.std().item(),
        "random_reference_cosine": ref_cos,
        "random_reference_l2": ref_l2,
    }
    print("STYLE_DIVERSITY_JSON:" + json.dumps(summary))
    print("STYLE EMBEDDING DIVERSITY CHECK COMPLETED SUCCESSFULLY!")
except Exception as e:
    import traceback
    print("ERROR OCCURRED DURING KERNEL EXECUTION:")
    traceback.print_exc()
    raise
'''


def run_kaggle_style_embedding_diversity(dataset_kernel_ref: str) -> str:
    project_root = Path(__file__).resolve().parents[1]

    tokens = kaggle_auth_environment(load_kaggle_api_tokens())
    username = resolve_kaggle_username(tokens.get("KAGGLE_USERNAME"))
    cli = kaggle_cli_command()
    if not username or not kaggle_auth_available(tokens) or not cli:
        raise RuntimeError("Missing KAGGLE_USERNAME or Kaggle auth (KAGGLE_API_TOKEN=KGAT_... / legacy KAGGLE_KEY)")

    run_id = f"styledivcheck-{int(time.time())}"
    job_dir = project_root / "outputs" / "kaggle_style_diversity" / run_id
    dataset_dir = job_dir / "dataset"
    kernel_dir = job_dir / "kernel"

    for d in (dataset_dir, kernel_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Initializing Kaggle Style-Embedding Diversity Check: {run_id}")
    print("=" * 70)

    print("Zipping local source code...")
    write_source_zip(project_root, dataset_dir / "genmusic_vn_source.zip")

    source_dataset_slug = f"genmusic-source-{run_id}"
    source_dataset_ref = f"{username}/{source_dataset_slug}"

    dataset_title = f"Style Div {run_id}"
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

    kernel_slug = f"genmusic-styledivcheck-{int(time.time())}"
    kernel_ref = f"{username}/{kernel_slug}"

    (kernel_dir / "run_style_div.py").write_text(_kernel_script_content(), encoding="utf-8")

    (kernel_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": kernel_ref,
        "title": kernel_slug,
        "code_file": "run_style_div.py",
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

    print(f"Pushing Style-Diversity Kernel to Kaggle: {kernel_ref}...")

    time.sleep(20)
    for attempt in range(3):
        try:
            subprocess.run(cli + ["kernels", "push", "-p", str(kernel_dir)], env={**os.environ, **tokens}, check=True)
            print("\nSTYLE-DIVERSITY JOB SUBMITTED SUCCESSFULLY!")
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
    run_kaggle_style_embedding_diversity(args.dataset_kernel_ref)


if __name__ == "__main__":
    main()
