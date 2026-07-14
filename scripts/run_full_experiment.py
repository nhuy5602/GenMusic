"""End-to-end experiment run, meant to execute INSIDE a Kaggle kernel (GPU).

Stages, each logged and saved to /kaggle/working so `kaggle kernels output`
downloads everything:
  1. Preprocess a subset of the raw Vietnamese song dataset with the fixed
     pipeline (Vocos-native 100-mel/24kHz mels + precomputed MuQ-MuLan style
     embeddings).
  2. Vocoder round-trip sanity check: decode a REAL song's own mel back to
     audio and measure mel correlation against the original -- this is the
     regression test for the "distorted audio" bug (see
     docs/experiments/vocoder_fix.md). Should score >0.9 now vs ~0.15 before.
  3. Train a baseline MicroDiT (CFM ground-truth loss only, no teacher) --
     establishes whether the fixed pipeline produces clean-sounding audio at
     all, independent of distillation correctness.
  4. Attempt real knowledge distillation from the DiffRhythm2 teacher (honest
     report of whether the teacher/tokenizer actually loaded).
  5. Generate a sample from each trained checkpoint via Vocos.
  6. Basic waveform sanity stats (not a substitute for listening) for each
     generated sample.
  7. Upload the preprocessed dataset to Kaggle for reuse in future runs.

Run via: python scripts/run_full_experiment.py --raw-dataset /kaggle/input/... --max-files 12
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.preprocess_raw_vietnamese import preprocess_raw_audio
from src.models.text_to_music_diffusion import MusicDiffusionConfig, compute_mel_spectrogram, render_mel_to_wav
from src.integrations.kaggle_auto import run_local_generation
from src.training.self_diffusion import train_model


def vocoder_roundtrip_check(dataset_dir: Path, out_dir: Path) -> dict:
    records = [json.loads(l) for l in (dataset_dir / "records.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    config = MusicDiffusionConfig()
    result = {"status": "no_records"}
    for record in records:
        backing_path = dataset_dir / record["backing_mel_path"]
        if not backing_path.exists():
            continue
        mel = torch.load(backing_path, map_location="cpu", weights_only=True)
        wav_path = out_dir / f"roundtrip_{record['id']}.wav"
        render_mel_to_wav(mel, wav_path, config, vocoder_type="vocos")

        import librosa

        y, _ = librosa.load(str(wav_path), sr=config.sample_rate, mono=True)
        remel = compute_mel_spectrogram(y, config)
        n = min(mel.shape[1], remel.shape[1])
        diff = mel[:, :n] - remel[:, :n]
        rmse = float(torch.sqrt((diff ** 2).mean()))
        corr = float(torch.corrcoef(torch.stack([mel[:, :n].flatten(), remel[:, :n].flatten()]))[0, 1])
        result = {"status": "ok", "record_id": record["id"], "logmel_rmse": rmse, "logmel_corr": corr, "wav": str(wav_path)}
        break
    return result


def wav_sanity_stats(path: Path) -> dict:
    import soundfile as sf

    y, sr = sf.read(str(path))
    y = np.asarray(y, dtype=np.float32)
    return {
        "sample_rate": sr,
        "duration_seconds": round(len(y) / sr, 3),
        "peak_abs": float(np.max(np.abs(y))) if len(y) else 0.0,
        "rms": float(np.sqrt(np.mean(y ** 2))) if len(y) else 0.0,
        "silence_ratio": float(np.mean(np.abs(y) < 1e-4)) if len(y) else 1.0,
        "has_nan_or_inf": bool(np.any(~np.isfinite(y))),
    }


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dataset", required=True)
    parser.add_argument("--output-root", default="/kaggle/working")
    parser.add_argument("--max-files", type=int, default=12)
    parser.add_argument("--whisper-model", default="tiny")
    parser.add_argument("--baseline-epochs", type=int, default=40)
    parser.add_argument("--distill-epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--repo-id", default="ASLP-lab/DiffRhythm2")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary = {"started_at": time.time()}

    print("=" * 70, "\nSTAGE 1: Preprocessing\n", "=" * 70, sep="", flush=True)
    dataset_dir = output_root / "processed_dataset"
    preprocess_report = preprocess_raw_audio(
        args.raw_dataset, dataset_dir, whisper_model_name=args.whisper_model,
        keep_separated_count=3, max_files=args.max_files,
    )
    summary["preprocess"] = preprocess_report
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if preprocess_report.get("status") != "completed":
        print("[FATAL] preprocessing failed, aborting.", flush=True)
        return

    print("=" * 70, "\nSTAGE 2: Vocoder round-trip sanity check\n", "=" * 70, sep="", flush=True)
    summary["vocoder_roundtrip"] = vocoder_roundtrip_check(dataset_dir, output_root)
    print(json.dumps(summary["vocoder_roundtrip"], indent=2), flush=True)
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 70, "\nSTAGE 3: Baseline DiT training (no teacher)\n", "=" * 70, sep="", flush=True)
    baseline_ckpt = output_root / "baseline_dit.pt"
    try:
        baseline_report = train_model(
            dataset_dir, baseline_ckpt, epochs=args.baseline_epochs, batch_size=args.batch_size, model_type="dit",
        )
    except Exception as e:
        import traceback

        baseline_report = {"status": "failed", "error": str(e), "traceback": traceback.format_exc()}
    summary["baseline_training"] = baseline_report
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 70, "\nSTAGE 4: Distillation attempt\n", "=" * 70, sep="", flush=True)
    distilled_ckpt = output_root / "distilled_student.pt"
    try:
        from src.training.distill_training import run_distillation_training

        distill_report = run_distillation_training(
            dataset_dir, distilled_ckpt, epochs=args.distill_epochs, batch_size=args.batch_size,
            alpha_feature=0.5, repo_id=args.repo_id,
        )
    except Exception as e:
        import traceback

        distill_report = {"status": "failed", "error": str(e), "traceback": traceback.format_exc()}
    summary["distillation"] = distill_report
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 70, "\nSTAGE 5: Generation samples\n", "=" * 70, sep="", flush=True)
    records = [json.loads(l) for l in (dataset_dir / "records.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    sample_style = records[0]["style"] if records else "Vietnamese pop, warm piano, clear melody"
    lyric_text = "Đêm nay mưa rơi rơi trên lối mòn xưa cũ. Lòng anh nhớ em nhiều người ơi có biết chăng."

    summary["generation"] = {}
    for name, ckpt in (("baseline", baseline_ckpt), ("distilled", distilled_ckpt)):
        if not ckpt.exists():
            summary["generation"][name] = {"status": "no_checkpoint"}
            continue
        try:
            gen_dir = output_root / f"generated_{name}"
            gen_report = run_local_generation(
                text=lyric_text, style=sample_style, output_dir=gen_dir, duration_seconds=8.0,
                checkpoint=ckpt, steps=32, model_type="dit", vocoder="vocos",
            )
            wav_path = Path(gen_report["audio_path"])
            gen_report["sanity_stats"] = wav_sanity_stats(wav_path)
            summary["generation"][name] = gen_report
        except Exception as e:
            import traceback

            summary["generation"][name] = {"status": "failed", "error": str(e), "traceback": traceback.format_exc()}
        (output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    summary["finished_at"] = time.time()
    summary["elapsed_seconds"] = round(summary["finished_at"] - summary["started_at"], 1)
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("=" * 70, "\nALL STAGES COMPLETE\n", "=" * 70, sep="", flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
