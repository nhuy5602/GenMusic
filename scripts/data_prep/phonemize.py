"""Kaggle kernel script: reads records.jsonl from a pre-processed raw-audio
dataset (vocal/backing .pt already separated), adds G2P phoneme fields, and
writes an updated records.jsonl alongside symlinks to the waveform files.

This is a lightweight "Stage 2" that runs on top of quynhvu03's Stage 1
outputs (Demucs + Whisper already done), avoiding the expensive 6-8h GPU
separation/transcription step.

Input layout (from kernel_sources, e.g. quynhvu03/genmusic-data-prep-p1-...):
  /kaggle/input/<kernel-slug>/processed_dataset_raw_audio/
      config.json
      records.jsonl          ← has 'text', but no 'phoneme_text' yet
      waveforms/
          <id>_vocal.pt
          <id>_backing.pt
          <id>_style.pt

Output layout (/kaggle/working/processed_dataset_raw_audio/):
  config.json                ← copied verbatim
  records.jsonl              ← same records + phoneme_text / phoneme_words_list
  waveforms/                 ← symlinks (or copies if symlink fails) to originals
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import traceback
from pathlib import Path

# ── helpers ──────────────────────────────────────────────────────────────────

def _install(packages: list[str]) -> None:
    import subprocess
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check"] + packages,
    )


def _find_input_dir(kernel_slug: str) -> Path:
    """Locate the processed_dataset_raw_audio dir from the kernel source input."""
    # Debug: list all /kaggle/input to see how Kaggle mounts kernel_sources
    kaggle_input = Path("/kaggle/input")
    print("[DEBUG] /kaggle/input contents:", flush=True)
    if kaggle_input.exists():
        for item in sorted(kaggle_input.iterdir()):
            print(f"  {item}", flush=True)
            if item.is_dir():
                for sub in sorted(item.iterdir())[:8]:
                    print(f"    {sub}", flush=True)
    else:
        print("  /kaggle/input does not exist!", flush=True)

    # Search entire /kaggle/input for records.jsonl
    print("[DEBUG] Searching for records.jsonl...", flush=True)
    for p in sorted(kaggle_input.rglob("records.jsonl")):
        print(f"  Found: {p}", flush=True)
        return p.parent

    raise FileNotFoundError(
        f"Cannot locate records.jsonl anywhere under /kaggle/input/\n"
        f"kernel_slug={kernel_slug!r}"
    )


# ── main ─────────────────────────────────────────────────────────────────────

def main(kernel_slug: str) -> None:
    print(f"=== Phonemize Records Stage 2 ===", flush=True)
    print(f"Source kernel slug: {kernel_slug}", flush=True)

    # 1. No installation needed — transformers is pre-installed on Kaggle.
    # We use transformers directly instead of text2phonemesequence to avoid
    # the snapshot_download + hf_transfer hang that blocked all CPU runs.
    print("\n[1/4] Loading G2P model via transformers...", flush=True)
    import torch
    from transformers import T5ForConditionalGeneration, AutoTokenizer

    MODEL_ID = "charsiu/g2p_multilingual_byT5_small_100"
    LANGUAGE = "vie-c"
    print(f"Loading tokenizer from {MODEL_ID}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print("Tokenizer loaded. Loading model weights...", flush=True)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_ID)
    model.eval()
    # Force CPU — Kaggle T4 reports cuda.is_available()=True but the installed
    # PyTorch build has no kernel image for T4 (sm_75), causing CUDA errors on all records.
    device = torch.device("cpu")
    model = model.to(device)
    print(f"G2P model loaded on {device}.", flush=True)

    def g2p_infer(text: str) -> str:
        """Run G2P inference using ByT5 directly."""
        if not text.strip():
            return ""
        input_text = f"<{LANGUAGE}> {text}"
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_length=400)
        return tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # 2. Locate input
    print("\n[2/4] Locating input dataset...", flush=True)
    input_dir = _find_input_dir(kernel_slug)
    print(f"Found input at: {input_dir}", flush=True)

    records_in = input_dir / "records.jsonl"
    config_in  = input_dir / "config.json"
    waveforms_in = input_dir / "waveforms"

    # 3. Set up output directory
    out_dir = Path("/kaggle/working/processed_dataset_raw_audio")
    out_waveforms = out_dir / "waveforms"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_waveforms.mkdir(exist_ok=True)

    if config_in.exists():
        shutil.copy2(config_in, out_dir / "config.json")
        print("Copied config.json", flush=True)

    # 4. Symlink or copy waveform .pt files (avoid duplicating gigabytes)
    print("\n[3/4] Linking waveform files...", flush=True)
    n_linked = 0
    if waveforms_in.exists():
        for pt_file in sorted(waveforms_in.glob("*.pt")):
            dst = out_waveforms / pt_file.name
            if dst.exists():
                continue
            try:
                dst.symlink_to(pt_file.resolve())
            except OSError:
                shutil.copy2(pt_file, dst)
            n_linked += 1
    print(f"Linked/copied {n_linked} waveform files.", flush=True)

    # 5. Phonemize records — ONE call per record (full text only).
    # Per-word phonemization would need 100+ G2P calls/record on CPU ByT5,
    # pushing total runtime way past Kaggle's 9h limit.
    print("\n[4/4] Adding phoneme fields to records...", flush=True)
    import time as _time
    records_out_path = out_dir / "records.jsonl"
    n_total = n_ok = n_skip = n_err = 0
    t_start = _time.time()

    with open(records_in, encoding="utf-8") as fin, \
         open(records_out_path, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_total += 1

            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                print(f"  [WARN] Skipping malformed JSON line {n_total}", flush=True)
                n_skip += 1
                continue

            # If already phonemized with real content, pass through
            if rec.get("phoneme_text"):
                n_skip += 1
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            # ONE G2P call per record on the full transcript text
            try:
                text = str(rec.get("text", ""))
                rec["phoneme_text"] = g2p_infer(text)
            except Exception as e:
                print(f"  [WARN] G2P failed rec {rec.get('id','?')}: {e}", flush=True)
                rec["phoneme_text"] = ""
                n_err += 1

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_ok += 1

            # Progress every 10 records with timing + ETA
            if n_total % 10 == 0:
                elapsed = _time.time() - t_start
                rate = n_total / elapsed if elapsed > 0 else 0
                eta = (219 - n_total) / rate if rate > 0 else 0
                print(f"  [{n_total}/~219] ok={n_ok} err={n_err} | "
                      f"{elapsed:.0f}s elapsed | ETA ~{eta:.0f}s", flush=True)

    print(f"\nDone: {n_total} total | {n_ok} phonemized | {n_skip} skipped | {n_err} errors", flush=True)
    (Path("/kaggle/working") / "success.txt").write_text(
        f"phonemize ok: {n_ok}/{n_total}", encoding="utf-8"
    )
    print("✅ Stage 2 phonemize complete.", flush=True)


if __name__ == "__main__":
    # kernel_slug is injected at push time via string substitution
    KERNEL_SLUG = "{{KERNEL_SLUG}}"
    try:
        main(KERNEL_SLUG)
    except Exception:
        print("\nError occurred during phonemization:", flush=True)
        traceback.print_exc()
        sys.exit(1)
