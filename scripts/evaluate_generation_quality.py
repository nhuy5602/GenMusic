"""Objective (no human listening) audio quality check for a generated-music checkpoint.

Compares model-generated audio against the *real* vocal track of the same
song, rendered through the identical Vocos vocoder -- this isolates model
quality from vocoder artifacts, since both sides go through the same decode
path. Metrics:

- spectral_flatness (librosa): ~0 = tonal/harmonic (music-like), ~1 = white
  noise. This is the direct proxy for "toan nhieu" (pure noise) complaints.
- clip_ratio: fraction of samples saturated at the waveform ceiling.
- silence_ratio: fraction of near-zero samples (dead air).
- rms: overall loudness, sanity-checked against a plausible range.

A synthesized white-noise clip is included as a fixed sanity anchor so the
flatness numbers have a concrete "this is what noise looks like" reference.
"""
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import torch
import librosa

from src.models.text_to_music_diffusion import load_checkpoint, generate_audio, render_mel_to_wav
from src.training.self_diffusion import _load_mel

FIXED_TEXT = "Dem nay mua roi tren loi mon xua, long anh nho em nhieu nguoi oi co biet chang"
FIXED_STYLE = "Vietnamese pop, warm piano, clear melody"


def wav_metrics(path: Path) -> dict:
    y, sr = librosa.load(str(path), sr=None, mono=True)
    if len(y) == 0:
        return {"error": "empty audio"}
    flatness = librosa.feature.spectral_flatness(y=y)
    return {
        "spectral_flatness": float(np.mean(flatness)),
        "rms": float(np.sqrt(np.mean(y.astype(np.float64) ** 2))),
        "clip_ratio": float(np.mean(np.abs(y) > 0.98)),
        "silence_ratio": float(np.mean(np.abs(y) < 1e-4)),
        "duration_seconds": float(len(y) / sr),
    }


def main() -> None:
    checkpoint_path = sys.argv[1]
    dataset_dir = Path(sys.argv[2])
    out_dir = Path(sys.argv[3])
    max_records = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading checkpoint {checkpoint_path} on {device}...")
    model, config, payload = load_checkpoint(checkpoint_path, device=device)

    records = [json.loads(line) for line in (dataset_dir / "records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [r for r in records if (dataset_dir / r["backing_mel_path"]).exists()][:max_records]
    print(f"Evaluating {len(records)} sample record(s).")

    # Fixed sanity anchor: what does the metric read on literal white noise?
    noise = np.random.default_rng(0).uniform(-1.0, 1.0, size=24_000 * 4).astype(np.float32)
    noise_path = out_dir / "white_noise_anchor.wav"
    import soundfile as sf
    sf.write(str(noise_path), noise, 24_000)
    noise_metrics = wav_metrics(noise_path)
    print("white_noise_anchor:", noise_metrics)

    results = {"white_noise_anchor": noise_metrics, "samples": []}
    for record in records:
        record_id = record["id"]
        backing_mel = _load_mel(dataset_dir / record["backing_mel_path"])
        style_anchor = _load_mel(dataset_dir / record["style_embed_path"]).float().view(-1)
        real_vocal_mel = _load_mel(dataset_dir / record["vocal_mel_path"])

        gen_path = out_dir / f"{record_id}_generated.wav"
        generate_audio(
            model, FIXED_TEXT, FIXED_STYLE, gen_path,
            duration_seconds=8.0, config=config, device=device, steps=6, seed=5602,
            backing_mel=backing_mel, style_anchor=style_anchor,
        )
        real_path = out_dir / f"{record_id}_real.wav"
        render_mel_to_wav(real_vocal_mel, real_path, config, vocoder_type="vocos")

        entry = {
            "id": record_id,
            "generated": wav_metrics(gen_path),
            "real_vocal_same_vocoder": wav_metrics(real_path),
        }
        results["samples"].append(entry)
        print(record_id, "gen:", entry["generated"], "real:", entry["real_vocal_same_vocoder"])

    flatness_gen = [s["generated"]["spectral_flatness"] for s in results["samples"]]
    flatness_real = [s["real_vocal_same_vocoder"]["spectral_flatness"] for s in results["samples"]]
    results["summary"] = {
        "mean_flatness_generated": float(np.mean(flatness_gen)) if flatness_gen else None,
        "mean_flatness_real": float(np.mean(flatness_real)) if flatness_real else None,
        "white_noise_flatness": noise_metrics["spectral_flatness"],
        "mean_clip_ratio_generated": float(np.mean([s["generated"]["clip_ratio"] for s in results["samples"]])) if results["samples"] else None,
        "mean_silence_ratio_generated": float(np.mean([s["generated"]["silence_ratio"] for s in results["samples"]])) if results["samples"] else None,
    }
    (out_dir / "quality_report.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nSummary:", json.dumps(results["summary"], indent=2))


if __name__ == "__main__":
    main()
