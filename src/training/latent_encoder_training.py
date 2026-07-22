"""Pretrains `LatentAudioEncoder` (src/models/latent_codec.py) against
DiffRhythm2's real, frozen, pretrained BigVGAN decoder, using a plain
reconstruction loss on this project's own 250-song corpus.

This is a separate, first stage: only once this encoder reconstructs real
audio reasonably well does it make sense to train a CFM student *inside* the
resulting compressed latent space (see the conversation notes in
src/models/latent_codec.py's module docstring for why -- this project's
student has always generated raw, uncompressed mel directly, at ~19x the
frame rate DiffRhythm2's own DiT was actually designed around).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..models.text_to_music_diffusion import MusicDiffusionConfig, denormalize_mel, reconstruct_full_mix
from .self_diffusion import _filter_training_records, _load_mel, _read_records, _with_absolute_paths


def _torch():
    import torch
    from torch.utils.data import DataLoader

    return torch, DataLoader


class _ReconstructionDataset:
    """Fixed-length crops of the full-mix mel (vocal + backing), for
    reconstruction pretraining only -- no text/style/CFM-specific fields."""

    def __init__(self, records: list[dict[str, Any]], config: MusicDiffusionConfig, crop_frames: int):
        self.records = records
        self.config = config
        self.crop_frames = crop_frames

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        torch, _ = _torch()
        record = self.records[index]
        vocal_mel = _load_mel(Path(record["vocal_mel_path"]))
        backing_mel = _load_mel(Path(record["backing_mel_path"]))
        frames = self.crop_frames
        available = min(vocal_mel.shape[-1], backing_mel.shape[-1])
        if available <= frames:
            vocal_crop = torch.nn.functional.pad(vocal_mel, (0, frames - available + 1))[:, : frames + 1][:, :frames]
            backing_crop = torch.nn.functional.pad(backing_mel, (0, frames - available + 1))[:, : frames + 1][:, :frames]
        else:
            start = torch.randint(0, available - frames, (1,)).item()
            vocal_crop = vocal_mel[:, start : start + frames]
            backing_crop = backing_mel[:, start : start + frames]
        # (n_mels, frames) -> (frames, n_mels), matching normalize_mel's expected layout
        return {"vocal_mel": vocal_crop.transpose(0, 1), "backing_mel": backing_crop.transpose(0, 1)}


def train_latent_encoder(
    dataset_dir: str | Path,
    checkpoint_path: str | Path,
    *,
    epochs: int = 1,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    device: str | None = None,
    max_records: int | None = None,
    log_every_steps: int = 10,
    repo_id: str = "ASLP-lab/DiffRhythm2",
    crop_seconds: float = 1.0,
    warmup_steps: int = 200,
    grad_clip_norm: float = 1.0,
) -> dict[str, Any]:
    """crop_seconds is deliberately short (1s default, not this project's usual
    ~4s CFM crop): backprop through the full BigVGAN decoder (1536 initial
    channels, 6 transposed-conv upsample stages) on a multi-second segment
    exhausted a T4's 16GB VRAM even at batch_size=2 (confirmed via a real OOM
    on Kaggle) -- standard practice for vocoder training is short random crops
    for exactly this reason, and the reconstruction loss is local/spectral so
    it doesn't need long temporal context anyway.

    warmup_steps/grad_clip_norm exist because a first attempt (flat 2e-4 LR,
    no clipping, 15 epochs/945 steps) produced a non-monotonic, oscillating
    loss curve (2.78 -> 2.44 -> 2.20 -> 2.28 -> ... -> 1.87, bouncing the whole
    way) and, when decoded through the real frozen decoder, ground-truth
    latents came back with near-zero pitch variation (~0.9 semitones std vs.
    a white-noise reference's ~0.5 and real vocals' several semitones) --
    a collapsed, degenerate solution, not literal white noise. A fresh
    encoder pushing gradients through a frozen decoder it was never
    co-trained with is a fragile optimization; linear warmup + cosine decay +
    grad-norm clipping is the standard fix for exactly this failure mode."""
    torch, DataLoaderClass = _torch()
    from ..models.latent_codec import LatentAudioEncoder, load_frozen_decoder, multi_scale_mel_loss

    root = Path(dataset_dir)
    config = MusicDiffusionConfig(**json.loads((root / "config.json").read_text(encoding="utf-8")))
    crop_frames = max(1, round(crop_seconds * config.sample_rate / config.hop_length))
    records = [_with_absolute_paths(root, record) for record in _filter_training_records(_read_records(root))]
    if max_records is not None:
        records = records[:max_records]
    if not records:
        raise ValueError("Dataset has no usable records after vocal/transcript quality filtering.")

    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    encoder = LatentAudioEncoder().to(selected_device)
    decoder_handle = load_frozen_decoder(selected_device, repo_id=repo_id)

    from vocos import Vocos

    vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(selected_device).eval()
    for param in vocos.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(encoder.parameters(), lr=learning_rate)
    dataset = _ReconstructionDataset(records, config, crop_frames)

    def collate_fn(batch):
        return {
            "vocal_mel": torch.stack([item["vocal_mel"] for item in batch]),
            "backing_mel": torch.stack([item["backing_mel"] for item in batch]),
        }

    dataloader = DataLoaderClass(dataset, batch_size=max(1, int(batch_size)), shuffle=True, collate_fn=collate_fn)

    total_steps = max(1, len(dataloader) * max(1, int(epochs)))
    effective_warmup = min(max(0, int(warmup_steps)), max(1, total_steps - 1))

    def _lr_at(step: int) -> float:
        if effective_warmup > 0 and step < effective_warmup:
            return learning_rate * (step + 1) / effective_warmup
        progress = (step - effective_warmup) / max(1, total_steps - effective_warmup)
        progress = min(1.0, max(0.0, progress))
        import math

        return learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))

    started = time.perf_counter()
    losses: list[float] = []
    loss_curve: list[dict[str, float]] = []
    global_step = 0

    for epoch in range(max(1, int(epochs))):
        epoch_losses: list[float] = []
        for batch in dataloader:
            vocal_mel = batch["vocal_mel"].to(selected_device)
            backing_mel = batch["backing_mel"].to(selected_device)
            full_mix_normalized = reconstruct_full_mix(vocal_mel, backing_mel, config)

            with torch.no_grad():
                # (B, T, n_mels) -> (B, n_mels, T), Vocos convention.
                log_mel = denormalize_mel(full_mix_normalized, config).transpose(1, 2)
                # Vocos.decode() runs under torch.inference_mode() internally, which
                # produces tensors that can NEVER re-enter an autograd graph even
                # after this block exits (stricter than plain no_grad) -- clone to a
                # normal tensor before the encoder (a real, gradient-tracked module)
                # consumes it below.
                target_24k = vocos.decode(log_mel).clone()  # (B, samples) at config.sample_rate (24kHz)
                target_48k = torch.nn.functional.interpolate(
                    target_24k.unsqueeze(1), scale_factor=decoder_handle.sampling_rate / config.sample_rate,
                    mode="linear", align_corners=False,
                )  # (B, 1, samples*2) -- cheap resample, good enough as a reconstruction TARGET

            for group in optimizer.param_groups:
                group["lr"] = _lr_at(global_step)

            optimizer.zero_grad(set_to_none=True)
            latent = encoder(target_24k)
            chunk_size = min(20, max(1, latent.shape[2]))
            reconstructed_48k = decoder_handle.decoder.decode_audio(latent, overlap=min(2, chunk_size - 1), chunk_size=chunk_size)
            loss = multi_scale_mel_loss(reconstructed_48k, target_48k, sample_rate=decoder_handle.sampling_rate)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

            loss_value = float(loss.detach().cpu())
            losses.append(loss_value)
            epoch_losses.append(loss_value)
            global_step += 1
            if log_every_steps > 0 and global_step % log_every_steps == 0:
                print(f"epoch={epoch + 1} step={global_step} lr={_lr_at(global_step):.7f} loss={loss_value:.6f}", flush=True)

        avg_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        loss_curve.append({"epoch": epoch + 1, "loss": avg_loss})
        print(f"epoch={epoch + 1}/{epochs} avg_loss={avg_loss:.6f}", flush=True)

    destination = Path(checkpoint_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"encoder": encoder.state_dict(), "config": {"repo_id": repo_id}}, destination)

    final_loss = sum(losses[-min(10, len(losses)):]) / max(1, min(10, len(losses))) if losses else 0.0
    report = {
        "status": "complete",
        "backend": "genmusic-vn-latent-encoder",
        "dataset": str(root.resolve()),
        "checkpoint": str(destination.resolve()),
        "device": selected_device,
        "record_count": len(records),
        "epochs": epochs,
        "batch_size": batch_size,
        "step_count": len(losses),
        "final_loss": round(final_loss, 6),
        "loss_curve": loss_curve,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "decoder_repo_id": repo_id,
        "decoder_sampling_rate": decoder_handle.sampling_rate,
        "decoder_fps": decoder_handle.fps,
    }
    (destination.parent / "latent_encoder_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
