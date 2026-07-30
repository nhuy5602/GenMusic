"""Integrated Data Preparation: Cleaning + G2P Phonemization + XPhoneBERT Tokenization.

Outputs ready-to-train `phoneme_ids` in records.jsonl for zero-overhead training steps.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.lyric_quality import clean_vietnamese_lyric


def process_dataset_to_phoneme_ids(
    input_dir: Path,
    output_dir: Path,
    *,
    g2p_model_id: str = "charsiu/g2p_multilingual_byT5_small_100",
    xphonebert_model_id: str = "vinai/xphonebert-base",
    language: str = "vie-c",
    max_length: int = 128,
) -> dict[str, int]:
    """Clean lyrics, phonemize with G2P ByT5, and tokenize directly into phoneme_ids."""
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records_in = input_dir / "records.jsonl"
    if not records_in.exists():
        raise FileNotFoundError(f"Missing records.jsonl at {records_in}")

    # Copy metadata configs and wave/mel directories
    for config_name in ("config.json", "dataset-metadata.json"):
        if (input_dir / config_name).exists():
            shutil.copy2(input_dir / config_name, output_dir / config_name)

    for media_dir in ("waveforms", "mels"):
        if (input_dir / media_dir).exists():
            dst_media = output_dir / media_dir
            dst_media.mkdir(exist_ok=True)
            for f in (input_dir / media_dir).glob("*.pt"):
                dst_file = dst_media / f.name
                if not dst_file.exists():
                    try:
                        dst_file.symlink_to(f.resolve())
                    except OSError:
                        shutil.copy2(f, dst_file)

    # 1. Load G2P Model (ByT5)
    print(f" Loading G2P Model: {g2p_model_id}...", flush=True)
    g2p_tokenizer = AutoTokenizer.from_pretrained(g2p_model_id)
    g2p_model = T5ForConditionalGeneration.from_pretrained(g2p_model_id)
    g2p_model.eval()
    device = torch.device("cpu")
    g2p_model = g2p_model.to(device)

    # 2. Load XPhoneBERT Tokenizer
    print(f" Loading XPhoneBERT Tokenizer: {xphonebert_model_id}...", flush=True)
    xphonebert_tokenizer = AutoTokenizer.from_pretrained(xphonebert_model_id)

    def text_to_phonemes(text: str) -> str:
        if not text.strip():
            return ""
        inp = g2p_tokenizer(f"<{language}> {text}", return_tensors="pt").to(device)
        with torch.no_grad():
            out = g2p_model.generate(**inp, max_length=400)
        return g2p_tokenizer.decode(out[0], skip_special_tokens=True)

    records_out = output_dir / "records.jsonl"
    stats = {"total": 0, "cleaned": 0, "rejected": 0}
    t0 = time.time()

    with records_in.open("r", encoding="utf-8") as fin, records_out.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            stats["total"] += 1
            rec = json.loads(line)

            raw_text = str(rec.get("text") or rec.get("lyrics") or "")

            # Step A: Cleaning
            cleaned_text = clean_vietnamese_lyric(raw_text)
            if not cleaned_text:
                stats["rejected"] += 1
                continue

            rec["cleaned_text"] = cleaned_text

            # Step B: Phonemization
            if not rec.get("phoneme_text"):
                rec["phoneme_text"] = text_to_phonemes(cleaned_text)

            # Step C: Tokenization to Integer IDs
            encoding = xphonebert_tokenizer(
                rec["phoneme_text"],
                padding=False,
                truncation=True,
                max_length=max_length,
            )
            rec["phoneme_ids"] = encoding["input_ids"]
            rec["attention_mask"] = encoding["attention_mask"]

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats["cleaned"] += 1

            if stats["total"] % 50 == 0:
                print(f"   [{stats['total']}] processed, cleaned={stats['cleaned']}, rejected={stats['rejected']}", flush=True)

    print(f"🎉 Complete! Processed {stats['total']} records in {time.time()-t0:.1f}s.")
    print(f"   - Cleaned & Tokenized: {stats['cleaned']}")
    print(f"   - Rejected noise/mojibake: {stats['rejected']}")
    print(f"   - Output saved to: {records_out.resolve()}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean, Phonemize, and Pre-tokenize Dataset to Phoneme IDs")
    parser.add_argument("--input-dir", type=Path, required=True, help="Path to input dataset directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Path to output processed dataset directory")
    parser.add_argument("--max-length", type=int, default=128, help="Max sequence length for XPhoneBERT tokenizer")
    args = parser.parse_args()

    process_dataset_to_phoneme_ids(args.input_dir, args.output_dir, max_length=args.max_length)


if __name__ == "__main__":
    main()
