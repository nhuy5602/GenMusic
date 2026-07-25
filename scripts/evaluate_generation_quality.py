"""Objective (no human listening) audio quality check for a generated-music checkpoint.

Compares model-generated audio against the *real full mix* (vocal +
accompaniment, see reconstruct_full_mix) of the same song, rendered through
the identical Vocos vocoder -- this isolates model quality from vocoder
artifacts, since both sides go through the same decode path. The reference is
the full song, not an isolated a cappella vocal, because that is now the
model's actual training target (see reconstruct_full_mix's docstring).
Metrics:

- spectral_flatness (librosa): ~0 = tonal/harmonic (music-like), ~1 = white
  noise. This is a per-instant proxy for "toan nhieu" (pure noise) complaints
  -- but NOT sufficient on its own (see voiced_ratio below).
- clip_ratio: fraction of samples saturated at the waveform ceiling.
- silence_ratio: fraction of near-zero samples (dead air).
- rms: overall loudness, sanity-checked against a plausible range.
- voiced_ratio / mean_voiced_prob (librosa.pyin): fraction of frames with a
  detectable, *stable* pitch and the tracker's confidence. This is the metric
  that actually distinguishes "sounds like singing" from "sounds like noise"
  -- flatness only checks whether a single frame's spectrum is peaky, not
  whether that peak holds together as a coherent note across frames. A run
  with good mel std/flatness but near-zero voiced_ratio still sounds like
  noise to a human ear (confirmed the hard way: see
  docs/project_history.md's correction after a real listening report).
- pitch_std_semitones: std of the pyin f0 track (converted to semitones)
  across voiced frames only. voiced_ratio alone cannot tell a real moving
  melody apart from a monotone held note/drone -- both score high, since
  pyin only checks "is there a stable pitch this frame", not "does the pitch
  change over time". A checkpoint can have near-real voiced_ratio yet a
  semitone std an order of magnitude below the real vocal reference, which
  is itself evidence of a subtler form of regression-to-the-mean that
  voiced_ratio does not catch (see docs/project_history.md report section on
  the model-size/epoch ablation).

A synthesized white-noise clip is included as a fixed sanity anchor so the
flatness numbers have a concrete "this is what noise looks like" reference.

- lyric_wer (optional, --with-wer): transcribes generated audio with
  PhoWhisper (vinai/phowhisper-small) and computes Word Error Rate against
  the fixed target lyric. Complements the metrics above with an axis they
  cannot see at all -- intelligible Vietnamese *content*, not just
  spectral/pitch plausibility. A decisive discriminator in practice: audio
  with no real singing produces incoherent, repetitive ASR hallucination
  (e.g. one real check on a pre-VAE-fix sample returned a transcript
  repeating "chu nghia xa hoi" dozens of times, WER > 20) rather than a
  merely-imperfect transcript of the real words.
"""
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import torch
import librosa

from src.models.text_to_music_diffusion import load_checkpoint, generate_audio, render_mel_to_wav, reconstruct_full_mix
from src.training.self_diffusion import _load_mel

FIXED_TEXT = "Dem nay mua roi tren loi mon xua, long anh nho em nhieu nguoi oi co biet chang"
FIXED_STYLE = "Vietnamese pop, warm piano, clear melody"


def wav_metrics(path: Path) -> dict:
    y, sr = librosa.load(str(path), sr=None, mono=True)
    if len(y) == 0:
        return {"error": "empty audio"}
    flatness = librosa.feature.spectral_flatness(y=y)
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
    )
    voiced_f0 = f0[voiced_flag & ~np.isnan(f0)]
    pitch_std_semitones = float(np.std(12.0 * np.log2(voiced_f0 / 440.0))) if len(voiced_f0) >= 2 else None
    return {
        "spectral_flatness": float(np.mean(flatness)),
        "rms": float(np.sqrt(np.mean(y.astype(np.float64) ** 2))),
        "clip_ratio": float(np.mean(np.abs(y) > 0.98)),
        "silence_ratio": float(np.mean(np.abs(y) < 1e-4)),
        "voiced_ratio": float(np.mean(voiced_flag)),
        "mean_voiced_prob": float(np.mean(voiced_prob)),
        "pitch_std_semitones": pitch_std_semitones,
        "duration_seconds": float(len(y) / sr),
    }


_ASR_PIPELINE = None


def lyric_wer(path: Path, target_text: str) -> dict:
    """Transcribe `path` with PhoWhisper and score Word Error Rate against
    `target_text`. Loads the ASR pipeline once (module-level cache) since it
    is expensive relative to a single transcription."""
    global _ASR_PIPELINE
    import soundfile as sf
    import librosa as _librosa
    from jiwer import wer as _wer

    if _ASR_PIPELINE is None:
        from transformers import pipeline

        _ASR_PIPELINE = pipeline("automatic-speech-recognition", model="vinai/phowhisper-small")

    audio, sr = sf.read(str(path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16_000:
        audio = _librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16_000)
        sr = 16_000
    predicted = _ASR_PIPELINE({"array": audio.astype(np.float32), "sampling_rate": sr})["text"]
    return {"predicted_text": predicted, "target_text": target_text, "wer": float(_wer(target_text.lower(), predicted.lower()))}


def main() -> None:
    checkpoint_path = sys.argv[1]
    dataset_dir = Path(sys.argv[2])
    out_dir = Path(sys.argv[3])
    max_records = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    with_wer = "--with-wer" in sys.argv
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
        real_full_mix_mel = reconstruct_full_mix(real_vocal_mel, backing_mel, config)

        gen_path = out_dir / f"{record_id}_generated.wav"
        generate_audio(
            model, FIXED_TEXT, FIXED_STYLE, gen_path,
            duration_seconds=8.0, config=config, device=device, steps=6, seed=5602,
            backing_mel=backing_mel, style_anchor=style_anchor,
        )
        real_path = out_dir / f"{record_id}_real.wav"
        render_mel_to_wav(real_full_mix_mel, real_path, config, vocoder_type="vocos")

        entry = {
            "id": record_id,
            "generated": wav_metrics(gen_path),
            "real_full_mix_same_vocoder": wav_metrics(real_path),
        }
        if with_wer:
            entry["lyric_wer"] = lyric_wer(gen_path, FIXED_TEXT)
        results["samples"].append(entry)
        print(record_id, "gen:", entry["generated"], "real:", entry["real_full_mix_same_vocoder"])

    flatness_gen = [s["generated"]["spectral_flatness"] for s in results["samples"]]
    flatness_real = [s["real_full_mix_same_vocoder"]["spectral_flatness"] for s in results["samples"]]
    voiced_gen = [s["generated"]["voiced_ratio"] for s in results["samples"]]
    voiced_real = [s["real_full_mix_same_vocoder"]["voiced_ratio"] for s in results["samples"]]
    pitch_std_gen = [s["generated"]["pitch_std_semitones"] for s in results["samples"] if s["generated"]["pitch_std_semitones"] is not None]
    pitch_std_real = [s["real_full_mix_same_vocoder"]["pitch_std_semitones"] for s in results["samples"] if s["real_full_mix_same_vocoder"]["pitch_std_semitones"] is not None]
    results["summary"] = {
        "mean_flatness_generated": float(np.mean(flatness_gen)) if flatness_gen else None,
        "mean_flatness_real": float(np.mean(flatness_real)) if flatness_real else None,
        "white_noise_flatness": noise_metrics["spectral_flatness"],
        "mean_voiced_ratio_generated": float(np.mean(voiced_gen)) if voiced_gen else None,
        "mean_voiced_ratio_real": float(np.mean(voiced_real)) if voiced_real else None,
        "mean_pitch_std_semitones_generated": float(np.mean(pitch_std_gen)) if pitch_std_gen else None,
        "mean_pitch_std_semitones_real": float(np.mean(pitch_std_real)) if pitch_std_real else None,
        "mean_clip_ratio_generated": float(np.mean([s["generated"]["clip_ratio"] for s in results["samples"]])) if results["samples"] else None,
        "mean_silence_ratio_generated": float(np.mean([s["generated"]["silence_ratio"] for s in results["samples"]])) if results["samples"] else None,
    }
    if with_wer:
        wer_values = [s["lyric_wer"]["wer"] for s in results["samples"]]
        results["summary"]["mean_lyric_wer"] = float(np.mean(wer_values)) if wer_values else None
    (out_dir / "quality_report.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nSummary:", json.dumps(results["summary"], indent=2))


if __name__ == "__main__":
    main()
