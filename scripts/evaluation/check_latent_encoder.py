"""Sanity-check a `LatentAudioEncoder` checkpoint before trusting any
downstream CFM training on its latents: encodes a few real records'
GROUND-TRUTH audio (no CFM student involved), decodes the resulting latent
back through the real frozen BigVGAN decoder, and reports
`pitch_std_semitones` -- see docs/architecture.md's "Native latent backbone
and encoder" section for the collapse failure mode this catches (near-zero
pitch_std_semitones despite plausible spectral_flatness) and
docs/project_history.md §4.24 for the original before/after numbers.

Works against either a mel dataset or a `--raw-audio` dataset (config.json's
raw_audio_mode: true) -- accepts one or more dataset dirs, combined like
`train-latent-encoder`. Needs the real frozen decoder (`bigvgan`), so only
runs where the DiffRhythm2 repo is on PYTHONPATH -- i.e. on Kaggle, same
constraint as train-latent-encoder/precompute-latent-dataset.
"""
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "scripts"))

import torch

from evaluate_generation_quality import wav_metrics
from src.models.latent_codec import LatentAudioEncoder
from src.models.text_to_music_diffusion import (
    MusicDiffusionConfig, denormalize_mel, reconstruct_full_mix, render_mel_to_wav,
)
from src.training.self_diffusion import _filter_training_records, _load_mel, _read_records, _with_absolute_paths


def _ground_truth_waveform(record: dict, raw_audio_mode: bool, config: MusicDiffusionConfig, vocos):
    if raw_audio_mode:
        vocal = torch.load(Path(record["vocal_wav_path"]), map_location="cpu", weights_only=True)
        backing = torch.load(Path(record["backing_wav_path"]), map_location="cpu", weights_only=True)
        length = min(vocal.shape[-1], backing.shape[-1])
        return (vocal[:length] + backing[:length]).unsqueeze(0)
    vocal_mel = _load_mel(Path(record["vocal_mel_path"])).unsqueeze(0)
    backing_mel = _load_mel(Path(record["backing_mel_path"])).unsqueeze(0)
    full_mix_normalized = reconstruct_full_mix(vocal_mel.transpose(1, 2), backing_mel.transpose(1, 2), config)
    log_mel = denormalize_mel(full_mix_normalized, config).transpose(1, 2)
    return vocos.decode(log_mel).clone()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, nargs="+", help="One or more preprocessed dataset dirs (mel or --raw-audio).")
    parser.add_argument("--encoder-checkpoint", required=True)
    parser.add_argument("--out", default="outputs/latent_encoder_quality_check")
    parser.add_argument("--max-records", type=int, default=5)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_dirs = [Path(d) for d in args.dataset]
    root = dataset_dirs[0]
    config = MusicDiffusionConfig(**json.loads((root / "config.json").read_text(encoding="utf-8")))
    raw_audio_mode = config.raw_audio_mode

    records = []
    for d in dataset_dirs:
        records.extend(_with_absolute_paths(d, r) for r in _filter_training_records(_read_records(d)))
    records = records[: args.max_records]
    if not records:
        raise ValueError("No usable records found across the given dataset(s).")

    encoder = LatentAudioEncoder().to(device)
    payload = torch.load(args.encoder_checkpoint, map_location=device, weights_only=False)
    encoder.load_state_dict(payload["encoder"])
    encoder.eval()

    vocos = None
    if not raw_audio_mode:
        from vocos import Vocos
        vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device).eval()

    latent_config = replace(MusicDiffusionConfig(), latent_mode=True)

    import soundfile as sf

    results = []
    with torch.no_grad():
        for record in records:
            waveform = _ground_truth_waveform(record, raw_audio_mode, config, vocos).to(device)
            latent = encoder(waveform).squeeze(0).cpu()  # (channels, T), matches render_mel_to_wav's latent_mode input

            decoded_path = out_dir / f"{record['id']}_decoded_ground_truth_latent.wav"
            render_mel_to_wav(latent, decoded_path, latent_config, vocoder_type="vocos")

            real_path = out_dir / f"{record['id']}_real.wav"
            sf.write(str(real_path), waveform.squeeze(0).cpu().numpy(), config.sample_rate)

            entry = {
                "id": record["id"],
                "decoded_ground_truth_latent": wav_metrics(decoded_path),
                "real": wav_metrics(real_path),
            }
            results.append(entry)
            print(record["id"], "decoded:", entry["decoded_ground_truth_latent"], "real:", entry["real"], flush=True)

    pitch_std_decoded = [r["decoded_ground_truth_latent"]["pitch_std_semitones"] for r in results if r["decoded_ground_truth_latent"]["pitch_std_semitones"] is not None]
    pitch_std_real = [r["real"]["pitch_std_semitones"] for r in results if r["real"]["pitch_std_semitones"] is not None]
    summary = {
        "record_count": len(results),
        "raw_audio_mode": raw_audio_mode,
        "encoder_checkpoint": str(Path(args.encoder_checkpoint).resolve()),
        "mean_pitch_std_semitones_decoded": float(sum(pitch_std_decoded) / len(pitch_std_decoded)) if pitch_std_decoded else None,
        "mean_pitch_std_semitones_real": float(sum(pitch_std_real) / len(pitch_std_real)) if pitch_std_real else None,
    }
    report = {"summary": summary, "samples": results}
    (out_dir / "latent_encoder_quality_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nSummary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
