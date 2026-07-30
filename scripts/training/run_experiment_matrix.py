"""Runs a comparison matrix of training configs against ONE preprocessed dataset,
inside a single Kaggle kernel (preprocessing happens once, then each config trains
independently) -- meant to answer: does real distillation actually help this small
student converge faster / reach lower ground-truth CFM loss than training it from
scratch, at equal epoch budget? See docs/project_history.md
for the write-up this feeds.

Each config's per-epoch loss_gt (ground-truth CFM loss -- directly comparable whether
or not a teacher was used, see [[distill_training.KnowledgeDistillationTrainer]]) is
recorded, plus a generated sample + sanity stats, into one summary.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.preprocess_raw_vietnamese import preprocess_raw_audio
from src.integrations.kaggle_auto import run_local_generation
from src.training.self_diffusion import train_model


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


CONFIGS = [
    {"name": "baseline_no_distill", "mode": "baseline", "dim": 256, "depth": 4, "heads": 4, "alpha_feature": None},
    {"name": "distill_alpha0.5", "mode": "distill", "dim": 256, "depth": 4, "heads": 4, "alpha_feature": 0.5},
    {"name": "distill_alpha0.2", "mode": "distill", "dim": 256, "depth": 4, "heads": 4, "alpha_feature": 0.2},
    {"name": "distill_alpha0.8", "mode": "distill", "dim": 256, "depth": 4, "heads": 4, "alpha_feature": 0.8},
    {"name": "baseline_small_dim128depth2", "mode": "baseline", "dim": 128, "depth": 2, "heads": 2, "alpha_feature": None},
    {"name": "distill_small_dim128depth2", "mode": "distill", "dim": 128, "depth": 2, "heads": 2, "alpha_feature": 0.5},
]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dataset", required=True)
    parser.add_argument("--output-root", default="/kaggle/working")
    parser.add_argument("--max-files", type=int, default=40)
    parser.add_argument("--whisper-model", default="tiny")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--repo-id", default="ASLP-lab/DiffRhythm2")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary = {"started_at": time.time(), "configs": {}}

    print("=" * 70, "\nPreprocessing (shared across all configs)\n", "=" * 70, sep="", flush=True)
    dataset_dir = output_root / "processed_dataset"
    preprocess_report = preprocess_raw_audio(
        args.raw_dataset, dataset_dir, whisper_model_name=args.whisper_model,
        keep_separated_count=1, max_files=args.max_files,
    )
    summary["preprocess"] = {k: v for k, v in preprocess_report.items() if k != "failures"}
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if preprocess_report.get("status") != "completed":
        print("[FATAL] preprocessing failed, aborting.", flush=True)
        return

    records = [json.loads(l) for l in (dataset_dir / "records.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    sample_style = records[0]["style"] if records else "Vietnamese pop, warm piano, clear melody"
    lyric_text = "Đêm nay mưa rơi rơi trên lối mòn xưa cũ. Lòng anh nhớ em nhiều người ơi có biết chăng."

    for cfg in CONFIGS:
        name = cfg["name"]
        print("=" * 70, f"\nCONFIG: {name}\n", "=" * 70, sep="", flush=True)
        ckpt_path = output_root / f"{name}.pt"
        try:
            if cfg["mode"] == "baseline":
                report = train_model(
                    dataset_dir, ckpt_path, epochs=args.epochs, batch_size=args.batch_size,
                    dim=cfg["dim"], depth=cfg["depth"], heads=cfg["heads"],
                )
            else:
                from src.training.distill_training import run_distillation_training

                report = run_distillation_training(
                    dataset_dir, ckpt_path, epochs=args.epochs, batch_size=args.batch_size,
                    alpha_feature=cfg["alpha_feature"], repo_id=args.repo_id,
                    dim=cfg["dim"], depth=cfg["depth"], heads=cfg["heads"],
                )
        except Exception as e:
            import traceback

            report = {"status": "failed", "error": str(e), "traceback": traceback.format_exc()}
        summary["configs"][name] = {"training": report}
        (output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

        if ckpt_path.exists():
            try:
                gen_dir = output_root / f"generated_{name}"
                gen_report = run_local_generation(
                    text=lyric_text, style=sample_style, output_dir=gen_dir, duration_seconds=8.0,
                    checkpoint=ckpt_path, steps=32, vocoder="vocos",
                    reference_dataset=dataset_dir, reference_id=records[0]["id"] if records else None,
                )
                wav_path = Path(gen_report["audio_path"])
                gen_report["sanity_stats"] = wav_sanity_stats(wav_path)
                summary["configs"][name]["generation"] = gen_report
            except Exception as e:
                import traceback

                summary["configs"][name]["generation"] = {"status": "failed", "error": str(e), "traceback": traceback.format_exc()}
        (output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    summary["finished_at"] = time.time()
    summary["elapsed_seconds"] = round(summary["finished_at"] - summary["started_at"], 1)
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("=" * 70, "\nALL CONFIGS COMPLETE\n", "=" * 70, sep="", flush=True)


if __name__ == "__main__":
    main()
