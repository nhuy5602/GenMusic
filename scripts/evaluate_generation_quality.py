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
  docs/PROJECT_REPORT.md's correction after a real listening report).
- pitch_std_semitones: std of the pyin f0 track (converted to semitones)
  across voiced frames only. voiced_ratio alone cannot tell a real moving
  melody apart from a monotone held note/drone -- both score high, since
  pyin only checks "is there a stable pitch this frame", not "does the pitch
  change over time". A checkpoint can have near-real voiced_ratio yet a
  semitone std an order of magnitude below the real vocal reference, which
  is itself evidence of a subtler form of regression-to-the-mean that
  voiced_ratio does not catch (see docs/PROJECT_REPORT.md report section on
  the model-size/epoch ablation).

A synthesized white-noise clip is included as a fixed sanity anchor so the
flatness numbers have a concrete "this is what noise looks like" reference.
"""
import argparse
import json
import os
import re
import shutil
import sys
import unicodedata
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import torch
import librosa

from src.models.text_to_music_diffusion import load_checkpoint, generate_audio, render_mel_to_wav, reconstruct_full_mix
from src.models.pronunciation_prior import trim_wav_silence
from src.training.self_diffusion import _load_mel, lyric_text_for_window, usable_lyric_spans

FIXED_TEXT = "Dem nay mua roi tren loi mon xua, long anh nho em nhieu nguoi oi co biet chang"
FIXED_STYLE = "Vietnamese pop, warm piano, clear melody"


def _normalized_units(text: str, *, words: bool) -> list[str]:
    folded = unicodedata.normalize("NFD", str(text).casefold().replace("đ", "d"))
    folded = "".join(character for character in folded if unicodedata.category(character) != "Mn")
    folded = re.sub(r"[^a-z0-9]+", " ", folded).strip()
    return folded.split() if words else list(folded.replace(" ", ""))


def _edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, reference_item in enumerate(reference, start=1):
        current = [row]
        for column, hypothesis_item in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (reference_item != hypothesis_item),
                )
            )
        previous = current
    return previous[-1]


def transcription_metrics(reference: str, hypothesis: str) -> dict:
    reference_words = _normalized_units(reference, words=True)
    hypothesis_words = _normalized_units(hypothesis, words=True)
    reference_chars = _normalized_units(reference, words=False)
    hypothesis_chars = _normalized_units(hypothesis, words=False)
    wer = _edit_distance(reference_words, hypothesis_words) / max(1, len(reference_words))
    cer = _edit_distance(reference_chars, hypothesis_chars) / max(1, len(reference_chars))
    return {
        "reference": reference,
        "hypothesis": hypothesis,
        "wer": float(wer),
        "cer": float(cer),
        "word_accuracy": float(max(0.0, 1.0 - wer)),
        "reference_word_count": len(reference_words),
        "hypothesis_word_count": len(hypothesis_words),
    }


def evenly_spaced_records(records: list[dict], count: int) -> list[dict]:
    """Select deterministic samples across every attached dataset part."""
    limit = max(0, int(count))
    if limit == 0 or not records:
        return []
    if len(records) <= limit:
        return list(records)
    if limit == 1:
        return [records[0]]
    # Combined Kaggle records are grouped by source part. Taking records[:N]
    # made all ASR plots represent only part 1, so spread demo samples over the
    # complete corpus without introducing random report-to-report variation.
    indices = [
        round(index * (len(records) - 1) / (limit - 1))
        for index in range(limit)
    ]
    return [records[index] for index in indices]


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


def generation_candidate_rank(candidate: dict, real_metrics: dict | None = None) -> tuple[float, ...]:
    """Rank CFG candidates by intelligibility, then similarity to real vocals.

    ASR remains the primary signal. When every hypothesis is empty, however,
    voiced confidence alone is actively misleading: a monotone drone can have
    a perfect voiced ratio. In that tie case, prefer pitch movement, spectral
    flatness and voiced coverage close to the real vocal decoded by Vocos.
    """
    asr = candidate.get("asr") or {}
    metrics = candidate.get("metrics") or {}
    reference = real_metrics or {}

    def positive(value, fallback: float) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return fallback
        return numeric if np.isfinite(numeric) and numeric > 0.0 else fallback

    pitch = positive(metrics.get("pitch_std_semitones"), 1e-4)
    reference_pitch = positive(reference.get("pitch_std_semitones"), 2.5)
    flatness = positive(metrics.get("spectral_flatness"), 1e-6)
    reference_flatness = positive(reference.get("spectral_flatness"), 0.02)
    voiced = float(metrics.get("voiced_ratio") or 0.0)
    reference_voiced = float(reference.get("voiced_ratio") or 0.75)
    acoustic_distance = (
        abs(float(np.log(pitch / reference_pitch)))
        + 0.5 * abs(float(np.log(flatness / reference_flatness)))
        + abs(voiced - reference_voiced)
        + 10.0 * float(metrics.get("clip_ratio") or 0.0)
    )
    return (
        float(asr.get("word_accuracy", 0.0)),
        -float(asr.get("cer", 1.0)),
        -acoustic_distance,
        float(metrics.get("mean_voiced_prob", 0.0)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("dataset_dir")
    parser.add_argument("out_dir")
    parser.add_argument("max_records", nargs="?", type=int, default=8)
    parser.add_argument("--whisper-model", default="small")
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument(
        "--guidance-scales",
        default="1.0,2.0,3.0,4.0",
        help="Comma-separated lyric CFG values; the report keeps the best ASR candidate per sample.",
    )
    parser.add_argument(
        "--pronunciation-prior-strengths",
        default="0.0",
        help=(
            "Comma-separated Vietnamese TTS anchor strengths in [0,1]. "
            "Zero keeps the original pure-noise sampler."
        ),
    )
    parser.add_argument(
        "--use-style-anchor",
        action="store_true",
        help="Evaluate reference-conditioned generation. Default evaluates the user path with no MuQ anchor.",
    )
    args = parser.parse_args()
    checkpoint_path = args.checkpoint
    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)
    max_records = args.max_records
    guidance_scales = [
        float(value.strip())
        for value in str(args.guidance_scales).split(",")
        if value.strip()
    ]
    if not guidance_scales or any(value <= 0.0 for value in guidance_scales):
        raise ValueError("--guidance-scales must contain positive numbers")
    prior_strengths = [
        float(value.strip())
        for value in str(args.pronunciation_prior_strengths).split(",")
        if value.strip()
    ]
    if not prior_strengths or any(value < 0.0 or value > 1.0 for value in prior_strengths):
        raise ValueError("--pronunciation-prior-strengths must contain values in [0,1]")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading checkpoint {checkpoint_path} on {device}...")
    model, config, payload = load_checkpoint(checkpoint_path, device=device)
    whisper_model = None
    if args.whisper_model.casefold() not in {"", "none", "off"}:
        import whisper

        print(f"Loading Whisper {args.whisper_model} for Vietnamese WER/CER...", flush=True)
        offline_whisper = os.getenv("GENMUSIC_WHISPER_MODEL_PATH")
        whisper_name_or_path = (
            offline_whisper
            if offline_whisper and Path(offline_whisper).is_file()
            else args.whisper_model
        )
        whisper_model = whisper.load_model(whisper_name_or_path, device=device)

    records = [json.loads(line) for line in (dataset_dir / "records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [
        record
        for record in records
        if (dataset_dir / record["backing_mel_path"]).exists()
        and usable_lyric_spans(record)
    ]
    records = evenly_spaced_records(records, max_records)
    print(f"Evaluating {len(records)} sample record(s).")

    # Fixed sanity anchor: what does the metric read on literal white noise?
    noise = np.random.default_rng(0).uniform(-1.0, 1.0, size=24_000 * 4).astype(np.float32)
    noise_path = out_dir / "white_noise_anchor.wav"
    import soundfile as sf
    sf.write(str(noise_path), noise, 24_000)
    noise_metrics = wav_metrics(noise_path)
    print("white_noise_anchor:", noise_metrics)

    results = {
        "white_noise_anchor": noise_metrics,
        "conditioning_mode": "reference_style" if args.use_style_anchor else "no_style_anchor",
        "guidance_scales": guidance_scales,
        "pronunciation_prior_strengths": prior_strengths,
        "duration_seconds": float(args.duration),
        "diffusion_steps": int(args.steps),
        "samples": [],
    }
    for record in records:
        record_id = record["id"]
        backing_mel = _load_mel(dataset_dir / record["backing_mel_path"])
        style_anchor = (
            _load_mel(dataset_dir / record["style_embed_path"]).float().view(-1)
            if args.use_style_anchor
            else None
        )
        real_vocal_mel = _load_mel(dataset_dir / record["vocal_mel_path"])

        segments = [segment for segment in (record.get("segments") or []) if str(segment.get("text", "")).strip()]
        quality_spans = usable_lyric_spans(record)
        start_seconds = quality_spans[0][0] if quality_spans else 0.0
        reference_text = lyric_text_for_window(
            str(record.get("text", "")),
            segments,
            start_seconds,
            start_seconds + args.duration,
        )
        if not reference_text:
            reference_text = " ".join(str(record.get("text", "")).split()[:20])
            start_seconds = 0.0
        start_frame = int(start_seconds * config.sample_rate / config.hop_length)
        duration_frames = max(1, int(args.duration * config.sample_rate / config.hop_length))
        real_vocal_mel = real_vocal_mel[:, start_frame:start_frame + duration_frames]
        backing_mel = backing_mel[:, start_frame:start_frame + duration_frames]
        real_full_mix_mel = reconstruct_full_mix(
            real_vocal_mel.transpose(0, 1),
            backing_mel.transpose(0, 1),
            config,
        ).transpose(0, 1)

        real_vocal_path = out_dir / f"{record_id}_real_vocal.wav"
        real_mix_path = out_dir / f"{record_id}_real_mix.wav"
        render_mel_to_wav(real_vocal_mel, real_vocal_path, config, vocoder_type="vocos")
        render_mel_to_wav(real_full_mix_mel, real_mix_path, config, vocoder_type="vocos")
        real_vocal_metrics = wav_metrics(real_vocal_path)
        real_mix_metrics = wav_metrics(real_mix_path)
        real_asr = None
        if whisper_model is not None:
            real_transcript = whisper_model.transcribe(
                str(real_vocal_path), language="vi", fp16=device == "cuda"
            ).get("text", "").strip()
            real_asr = transcription_metrics(reference_text, real_transcript)

        # Search CFG at evaluation time instead of guessing one global value.
        # Every candidate uses the same seed, so differences come from lyric
        # guidance rather than random initialization.
        candidates = []
        for prior_strength in prior_strengths:
            for guidance_scale in guidance_scales:
                scale_tag = str(guidance_scale).replace(".", "p")
                prior_tag = str(prior_strength).replace(".", "p")
                candidate_path = out_dir / (
                    f"{record_id}_prior{prior_tag}_cfg{scale_tag}_generated.wav"
                )
                generate_audio(
                    model, reference_text, FIXED_STYLE, candidate_path,
                    duration_seconds=args.duration, config=config, device=device,
                    steps=args.steps, guidance_scale=guidance_scale, seed=5602,
                    backing_mel=backing_mel,
                    style_anchor=style_anchor,
                    enforce_minimum_duration=False,
                    pronunciation_prior_strength=prior_strength,
                )
                candidate = {
                    "guidance_scale": guidance_scale,
                    "pronunciation_prior_strength": prior_strength,
                    "path": candidate_path.name,
                    "metrics": wav_metrics(candidate_path),
                }
                if prior_strength > 0.0:
                    trimmed_path = candidate_path.with_name(
                        candidate_path.stem + "_trimmed.wav"
                    )
                    trim_wav_silence(candidate_path, trimmed_path)
                    candidate["fixed_window_metrics"] = candidate["metrics"]
                    candidate["trimmed_path"] = trimmed_path.name
                    # Ranking and the user-facing artifact should measure the
                    # audible phrase, not two seconds of intentional tensor pad.
                    candidate["metrics"] = wav_metrics(trimmed_path)
                if whisper_model is not None:
                    transcript = whisper_model.transcribe(
                        str(candidate_path), language="vi", fp16=device == "cuda"
                    ).get("text", "").strip()
                    candidate["asr"] = transcription_metrics(reference_text, transcript)
                candidates.append(candidate)

        best_candidate = max(
            candidates,
            key=lambda candidate: generation_candidate_rank(candidate, real_mix_metrics),
        )
        gen_path = out_dir / f"{record_id}_generated.wav"
        selected_candidate_path = out_dir / best_candidate.get(
            "trimmed_path", best_candidate["path"]
        )
        shutil.copy2(selected_candidate_path, gen_path)

        entry = {
            "id": record_id,
            "reference_text": reference_text,
            "start_seconds": start_seconds,
            "selected_guidance_scale": best_candidate["guidance_scale"],
            "selected_pronunciation_prior_strength": best_candidate[
                "pronunciation_prior_strength"
            ],
            "guidance_candidates": candidates,
            "generated": best_candidate["metrics"],
            "generated_fixed_window": best_candidate.get("fixed_window_metrics"),
            "real_vocal_same_vocoder": real_vocal_metrics,
            "real_full_mix_same_vocoder": real_mix_metrics,
        }
        if whisper_model is not None:
            entry["generated_asr"] = best_candidate["asr"]
            entry["real_vocal_asr"] = real_asr
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
    generated_asr = [sample["generated_asr"] for sample in results["samples"] if "generated_asr" in sample]
    real_asr = [sample["real_vocal_asr"] for sample in results["samples"] if "real_vocal_asr" in sample]
    if generated_asr:
        mean_word_accuracy = float(np.mean([item["word_accuracy"] for item in generated_asr]))
        mean_cer = float(np.mean([item["cer"] for item in generated_asr]))
        real_word_accuracy = float(np.mean([item["word_accuracy"] for item in real_asr])) if real_asr else 0.0
        passing_samples = sum(item["word_accuracy"] >= 0.30 for item in generated_asr)
        voiced_ratio = results["summary"]["mean_voiced_ratio_generated"] or 0.0
        pitch_std = results["summary"]["mean_pitch_std_semitones_generated"] or 0.0
        results["summary"].update({
            "mean_word_accuracy_generated": mean_word_accuracy,
            "mean_cer_generated": mean_cer,
            "mean_word_accuracy_real": real_word_accuracy,
            "asr_passing_samples": passing_samples,
            # Stop only when several independent signals agree: Whisper finds
            # a substantial fraction of the requested words, and the waveform
            # also has stable but non-monotone pitch rather than noise/drone.
            "intelligibility_pass": bool(
                mean_word_accuracy >= 0.40
                and mean_word_accuracy >= 0.50 * max(0.01, real_word_accuracy)
                and mean_cer <= 0.70
                and passing_samples >= max(1, len(generated_asr) // 2)
                and voiced_ratio >= 0.35
                and pitch_std >= 1.0
            ),
        })
    (out_dir / "quality_report.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nSummary:", json.dumps(results["summary"], indent=2))


if __name__ == "__main__":
    main()
