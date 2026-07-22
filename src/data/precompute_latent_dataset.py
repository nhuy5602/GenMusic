"""Converts an existing preprocessed mel dataset (records.jsonl + config.json,
100-dim mel @ 93.75 Hz) into a new dataset in DiffRhythm2's real latent space
(64-dim @ 5 Hz), using a `LatentAudioEncoder` already pretrained against the
real frozen BigVGAN decoder (see src/training/latent_encoder_training.py).

Output dataset has the SAME records.jsonl/config.json shape as any other
dataset in this project (see validate_dataset in self_diffusion.py), so the
existing train_model()/MusicDiffusionDataset/cfm_loss pipeline can train on it
unchanged, just with MusicDiffusionConfig.latent_mode=True (see that field's
docstring: skips reconstruct_full_mix since the full-mix combination already
happened here, in mel space, before encoding).

Each record's full-mix mel (vocal + backing, combined the same way training
always has) is decoded to real 24kHz audio via the frozen Vocos vocoder, then
encoded through the trained LatentAudioEncoder -- NOT reconstruct_full_mix'd
in latent space, since that formula only holds for log-mel channels.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any


def precompute_latent_dataset(
    source_dataset_dir: str | Path,
    encoder_checkpoint: str | Path,
    output_dir: str | Path,
    *,
    device: str | None = None,
    max_records: int | None = None,
    crop_seconds: float = 4.096,
) -> dict[str, Any]:
    import torch
    from vocos import Vocos

    from ..models.latent_codec import DECODER_FPS, DECODER_IN_CHANNELS, LatentAudioEncoder
    from ..models.text_to_music_diffusion import MusicDiffusionConfig, denormalize_mel, reconstruct_full_mix
    from ..training.self_diffusion import _filter_training_records, _load_mel, _read_records, _with_absolute_paths

    source_root = Path(source_dataset_dir)
    output_root = Path(output_dir)
    (output_root / "mels").mkdir(parents=True, exist_ok=True)

    source_config = MusicDiffusionConfig(**json.loads((source_root / "config.json").read_text(encoding="utf-8")))
    records = [_with_absolute_paths(source_root, record) for record in _filter_training_records(_read_records(source_root))]
    if max_records is not None:
        records = records[:max_records]
    if not records:
        raise ValueError("Source dataset has no usable records after filtering.")

    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    encoder = LatentAudioEncoder().to(selected_device)
    payload = torch.load(encoder_checkpoint, map_location=selected_device, weights_only=False)
    encoder.load_state_dict(payload["encoder"])
    encoder.eval()

    vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(selected_device).eval()

    output_records = []
    frames_per_chunk = max(1, round(crop_seconds * DECODER_FPS))
    with torch.no_grad():
        for record in records:
            vocal_mel = _load_mel(Path(record["vocal_mel_path"])).unsqueeze(0).to(selected_device)
            backing_mel = _load_mel(Path(record["backing_mel_path"])).unsqueeze(0).to(selected_device)
            # (1, n_mels, T) -> (1, T, n_mels) to match reconstruct_full_mix's expected layout
            full_mix_normalized = reconstruct_full_mix(vocal_mel.transpose(1, 2), backing_mel.transpose(1, 2), source_config)
            log_mel = denormalize_mel(full_mix_normalized, source_config).transpose(1, 2)  # (1, n_mels, T), Vocos convention
            waveform_24k = vocos.decode(log_mel).clone()  # (1, samples)

            latent = encoder(waveform_24k).squeeze(0).transpose(0, 1).cpu()  # (T_latent, DECODER_IN_CHANNELS)

            record_id = record.get("id") or f"record_{len(output_records):05d}"
            latent_path = output_root / "mels" / f"{record_id}_latent.pt"
            torch.save(latent.transpose(0, 1), latent_path)  # save as (channels, T), matching _load_mel's expected layout

            style_source = record.get("style_embed_path")
            style_dest_rel = None
            if style_source and Path(style_source).is_file():
                style_dest_rel = f"mels/{record_id}_style.pt"
                torch.save(torch.load(style_source, map_location="cpu", weights_only=True), output_root / style_dest_rel)

            output_record = {
                "id": record_id,
                "text": record.get("text", ""),
                "style": record.get("style", ""),
                "frames": int(latent.shape[0]),
                "vocal_mel_path": f"mels/{record_id}_latent.pt",
            }
            if style_dest_rel is not None:
                output_record["style_embed_path"] = style_dest_rel
            output_records.append(output_record)

    output_config = replace(
        MusicDiffusionConfig(),
        n_mels=DECODER_IN_CHANNELS,
        sample_rate=DECODER_FPS,  # repurposed: "frames per second" of this representation
        hop_length=1,
        frames_per_chunk=frames_per_chunk,
        chunk_seconds=crop_seconds,
        latent_mode=True,
    )
    (output_root / "config.json").write_text(json.dumps(asdict(output_config), ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "records.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in output_records), encoding="utf-8"
    )

    report = {
        "status": "complete",
        "backend": "genmusic-vn-latent-dataset",
        "source_dataset": str(source_root.resolve()),
        "output_dataset": str(output_root.resolve()),
        "record_count": len(output_records),
        "frames_per_chunk": frames_per_chunk,
        "encoder_checkpoint": str(Path(encoder_checkpoint).resolve()),
    }
    (output_root / "precompute_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
