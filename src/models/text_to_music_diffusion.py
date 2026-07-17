"""MicroDiT/CFM text-to-music model: config, mel/waveform I/O, checkpointing.

The denoising network itself lives in `dit_transformer.py` (MicroDiT) and
`cfm_flow.py` (Conditional Flow Matching loss/sampling); this module holds the
shared config, mel-spectrogram <-> waveform conversion, and checkpoint I/O.
"""

from __future__ import annotations

import math
import os
import wave
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MusicDiffusionConfig:
    # Mel parameters intentionally match Vocos's "charactr/vocos-mel-24khz" native
    # feature extractor exactly (sample_rate, n_fft, hop_length, n_mels, log scale).
    # This lets predicted mels be decoded directly with no resampling/interpolation
    # hack, which was previously destroying almost all signal (see docs/experiments).
    sample_rate: int = 24_000
    n_mels: int = 100
    n_fft: int = 1024
    hop_length: int = 256
    # 384 frames is 4.096 seconds at 24 kHz / hop 256. Keeping train and
    # generation windows at the same scale avoids asking the transformer to
    # extrapolate from the old 1.37-second crops to four-second phrases.
    frames_per_chunk: int = 384
    chunk_seconds: float = 4.096
    # Training estimates these scalar statistics from vocal targets and stores
    # them in the checkpoint. Old checkpoints default to identity normalization.
    mel_mean: float = 0.0
    mel_std: float = 1.0
    mel_clip: float = 6.0


def _config_from_dict(data: dict) -> "MusicDiffusionConfig":
    """Build a config from a saved dict, ignoring unknown keys.

    Checkpoints/configs saved by older code (the removed Conv1D architecture)
    may carry fields no longer in this dataclass -- drop them rather than
    failing to load an otherwise-fine checkpoint.
    """
    known = {field.name for field in fields(MusicDiffusionConfig)}
    return MusicDiffusionConfig(**{key: value for key, value in data.items() if key in known})


def _torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover - dependency boundary
        raise RuntimeError("Cần cài torch để chạy model sinh nhạc tự code.") from exc
    return torch, nn


VOCOS_MEL_CLIP = 1e-7


def normalize_mel(mel, config: "MusicDiffusionConfig"):
    """Map raw Vocos log-mels into the approximately N(0, 1) flow domain."""
    scale = max(1e-4, float(config.mel_std))
    normalized = (mel - float(config.mel_mean)) / scale
    return normalized.clamp(-float(config.mel_clip), float(config.mel_clip))


def denormalize_mel(mel, config: "MusicDiffusionConfig"):
    """Restore model-space mels to the exact log-mel scale expected by Vocos."""
    clipped = mel.clamp(-float(config.mel_clip), float(config.mel_clip))
    return clipped * max(1e-4, float(config.mel_std)) + float(config.mel_mean)


def compute_mel_spectrogram(waveform, config: "MusicDiffusionConfig"):
    """Log-mel matching Vocos's own MelSpectrogramFeatures exactly (magnitude mel,
    power=1, natural log with 1e-7 clip, no upper clip). Any mel produced this way
    can be handed straight to ``Vocos.decode`` with no resampling/interpolation.

    ``waveform`` may be a 1D numpy array / torch tensor of samples at
    ``config.sample_rate``. Returns a ``(n_mels, frames)`` float32 torch tensor.
    """
    torch, _ = _torch()
    import torchaudio

    if not isinstance(waveform, torch.Tensor):
        waveform = torch.as_tensor(waveform, dtype=torch.float32)
    waveform = waveform.float()
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        center=True,
        power=1.0,
    )
    mel = mel_transform(waveform)
    log_mel = torch.log(torch.clip(mel, min=VOCOS_MEL_CLIP))
    return log_mel.squeeze(0)


def build_lyric_timing(text: str, duration_seconds: float) -> list[dict[str, Any]]:
    """Allocate time by lyric line so long inputs are not compressed into one prompt."""
    lines = [line.strip() for line in text.splitlines() if line.strip()] or [text.strip()]
    weights = [max(1, len(line.split())) for line in lines]
    pause = min(0.18, max(0.0, float(duration_seconds) * 0.04))
    total_pause = pause * max(0, len(lines) - 1)
    available = max(0.1, float(duration_seconds) - total_pause)
    total_weight = sum(weights)
    timing = []
    cursor = 0.0
    for index, (line, weight) in enumerate(zip(lines, weights)):
        line_duration = available * weight / total_weight
        start = cursor
        end = start + line_duration
        timing.append({"line": line, "line_index": index, "start_seconds": round(start, 4), "end_seconds": round(end, 4), "duration_seconds": round(line_duration, 4), "word_count": weight})
        cursor = end + (pause if index < len(lines) - 1 else 0.0)
    return timing


def estimate_minimum_lyric_duration(text: str, *, words_per_second: float = 2.2, line_pause_seconds: float = 0.25) -> float:
    """Estimate a non-rushed minimum duration from lyric word count."""
    lines = [line.strip() for line in text.splitlines() if line.strip()] or [text.strip()]
    word_count = sum(max(1, len(line.split())) for line in lines)
    pauses = max(0, len(lines) - 1) * max(0.0, float(line_pause_seconds))
    return round(max(1.0, word_count / max(0.1, float(words_per_second)) + pauses), 3)


def render_mel_to_wav(mel, destination: str | Path, config: MusicDiffusionConfig, vocoder_type: str = "vocos") -> Path:
    """Render a log-mel tensor/array (n_mels, frames) into a WAV file.

    ``vocoder_type="vocos"`` (default, recommended) decodes with the pretrained
    neural vocoder "charactr/vocos-mel-24khz". It requires ``config`` to match
    Vocos's native mel format exactly (sample_rate=24000, n_fft=1024,
    hop_length=256, n_mels=100 -- the ``MusicDiffusionConfig`` defaults), so the
    mel is handed to Vocos unmodified with no lossy resampling. If the config
    does not match, or Vocos is unavailable, this falls back to a proper
    multi-iteration Griffin-Lim mel inversion (``vocoder_type="griffinlim"``),
    never to the old fabricated-phase iSTFT hack (removed: it produced audio with
    ~0.15 correlation to the true spectrogram, i.e. near-noise -- see
    docs/experiments/vocoder_fix.md).
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch, _ = _torch()
    values = mel.detach().float().cpu() if hasattr(mel, "detach") else torch.as_tensor(np.asarray(mel, dtype=np.float32))

    vocos_native = (
        config.sample_rate == 24_000 and config.n_fft == 1024 and config.hop_length == 256 and config.n_mels == 100
    )
    if vocoder_type == "vocos":
        if not vocos_native:
            print(
                f"⚠️ Warning: config ({config.sample_rate}Hz, n_fft={config.n_fft}, hop={config.hop_length}, "
                f"n_mels={config.n_mels}) does not match Vocos's native mel format; falling back to Griffin-Lim "
                "instead of resampling (resampling previously caused severe distortion)."
            )
        else:
            try:
                from vocos import Vocos

                vocos_model = Vocos.from_pretrained("charactr/vocos-mel-24khz")
                with torch.no_grad():
                    audio_tensor = vocos_model.decode(values.unsqueeze(0))
                audio = audio_tensor.squeeze(0).cpu().numpy()
                return _write_wav(audio, destination, config.sample_rate)
            except Exception as e:
                print(f"⚠️ Warning: Vocos decoding failed ({e}). Falling back to Griffin-Lim...")

    try:
        import librosa
    except ImportError as exc:  # pragma: no cover - dependency boundary
        raise RuntimeError("Cần librosa để đổi mel thành WAV.") from exc
    # Our mels use Vocos's convention: log(magnitude_mel), power=1. Recover
    # magnitude and run real iterative Griffin-Lim phase estimation (NOT a
    # fabricated linear phase ramp) via librosa's mel pseudo-inverse + Griffin-Lim.
    magnitude_mel = np.exp(values.numpy().astype(np.float32))
    audio = librosa.feature.inverse.mel_to_audio(
        magnitude_mel, sr=config.sample_rate, n_fft=config.n_fft, hop_length=config.hop_length, power=1.0, n_iter=64,
    )
    return _write_wav(audio, destination, config.sample_rate)


def _write_wav(audio: np.ndarray, destination: Path, sample_rate: int) -> Path:
    audio = audio / max(1e-6, float(np.max(np.abs(audio)))) * 0.8
    pcm = (audio.clip(-1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(destination), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(pcm.tobytes())
    return destination.resolve()


def save_checkpoint(
    model,
    path: str | Path,
    config: MusicDiffusionConfig,
    *,
    optimizer=None,
    scheduler=None,
    ema_state: dict[str, Any] | None = None,
    epoch: int = 0,
    loss: float | None = None,
    arch: dict[str, int] | None = None,
    training_state: dict[str, Any] | None = None,
) -> Path:
    torch, _ = _torch()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclude the frozen pretrained RoBERTa text encoder (~1.1GB of never-updated
    # weights) from the saved checkpoint -- it is always re-downloaded fresh from
    # HuggingFace by PretrainedRobertaEncoder.__init__ on load, so saving it here
    # only bloats every checkpoint by ~1.1GB for no benefit.
    model_state = {k: v for k, v in model.state_dict().items() if ".roberta." not in k}
    payload = {"config": asdict(config), "model": model_state, "epoch": epoch, "loss": loss, "arch": arch or {}}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if ema_state is not None:
        # Keep raw trainable weights in `model` for exact resume, and store EMA
        # separately so inference can use the smoother weights by default.
        payload["ema"] = {
            name: value.detach().cpu()
            for name, value in ema_state.items()
        }
    if training_state is not None:
        payload["training_state"] = training_state

    # A remote worker can terminate while storage is being written. Save beside
    # the previous checkpoint and replace it only after torch.save succeeds, so
    # an interrupted write never destroys the last resumable checkpoint.
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination.resolve()


def load_checkpoint(
    path: str | Path,
    *,
    device="cpu",
    roberta_model: str | None = None,
    use_ema: bool = True,
) -> tuple[Any, MusicDiffusionConfig, dict[str, Any]]:
    torch, _ = _torch()
    from .dit_transformer import MicroDiT

    payload = torch.load(path, map_location=device, weights_only=False)
    config = _config_from_dict(payload["config"])

    # Reconstruct with the exact architecture size the checkpoint was trained
    # with (defaults match the old hardcoded values, for checkpoints saved
    # before "arch" was tracked) -- a size mismatch here would silently fail
    # to load most weights via load_state_dict(strict=False). Same for the
    # frozen text encoder: it's not saved in the checkpoint (see save_checkpoint),
    # so reloading with the wrong model_name silently feeds it text tokenized/
    # embedded a completely different way than what the trainable projection
    # layer actually learned against. Explicit roberta_model always wins;
    # otherwise trust what the checkpoint says it was trained with, falling
    # back to the pre-XPhoneBERT default only for checkpoints saved before
    # "roberta_model" was tracked in arch.
    arch = payload.get("arch") or {}
    resolved_roberta_model = roberta_model or arch.get("roberta_model", "xlm-roberta-base")
    model = MicroDiT(
        config, roberta_model=resolved_roberta_model,
        dim=arch.get("dim", 256), depth=arch.get("depth", 4),
        heads=arch.get("heads", 4), ff_mult=arch.get("ff_mult", 4),
        style_dim=arch.get("style_dim", 512),
    ).to(device)

    # strict=False: checkpoints deliberately omit the frozen RoBERTa encoder
    # weights (see save_checkpoint) -- those are already loaded fresh from
    # HuggingFace by the model constructor above.
    model.load_state_dict(payload["model"], strict=False)
    if use_ema and payload.get("ema"):
        model.load_state_dict(payload["ema"], strict=False)
    return model, config, payload


def generate_audio(
    model,
    text: str,
    style: str,
    destination: str | Path,
    *,
    duration_seconds: float,
    config: MusicDiffusionConfig,
    device="cpu",
    steps: int = 6,
    guidance_scale: float = 1.0,
    seed: int = 5602,
    mel_output: str | Path | None = None,
    vocoder_type: str = "vocos",
    backing_mel=None,
    style_anchor=None,
    style_prompt=None,
) -> dict[str, Any]:
    """Generate audio with the same backing/style inputs used during training.

    ``backing_mel`` and ``style_anchor`` can come from
    ``src.training.self_diffusion.load_reference_conditioning``. Without them,
    generation falls back to zero backing conditioning and the model's text
    style path. ``style_prompt`` is retained as a compatibility alias for local
    callers that already passed a MuQ-MuLan anchor directly.
    """
    torch, _ = _torch()
    model.to(device)

    if style_anchor is not None and style_prompt is not None:
        raise ValueError("Pass only one of style_anchor or style_prompt")
    style_condition = style_anchor if style_anchor is not None else style_prompt
    normalized_style = None
    if style_condition is not None:
        normalized_style = torch.as_tensor(style_condition, dtype=torch.float32, device=device)
        if normalized_style.dim() == 1:
            normalized_style = normalized_style.unsqueeze(0)
        if normalized_style.dim() != 2 or normalized_style.shape[0] != 1:
            raise ValueError(
                "generate_audio currently generates one item and requires a style anchor shape "
                "(style_dim,) or (1, style_dim)"
            )

    duration_seconds = max(float(duration_seconds), estimate_minimum_lyric_duration(text))
    rendered = []
    lyric_timing = build_lyric_timing(text, duration_seconds)
    section_number = 0
    # Derive the generation span from the exact training tensor length.
    # We no longer condition on backing track frames.
    for section in lyric_timing:
        chunk_duration = float(section["duration_seconds"])
        chunk_text = section["line"]
        chunk_frames = max(8, int(chunk_duration * config.sample_rate / config.hop_length))
        from .cfm_flow import sample_cfm

        mel = sample_cfm(
            model,
            [chunk_text],
            chunk_frames,
            config=config,
            device=device,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed + section_number,
            style_prompt=normalized_style,
        )
        rendered.append(mel.squeeze(0))
        section_number += 1
    mel = torch.cat(rendered, dim=1)
    target_frames = max(1, int(float(duration_seconds) * config.sample_rate / config.hop_length))
    mel = mel[:, :target_frames]
    mel = denormalize_mel(mel, config)
    if mel_output:
        mel_path = Path(mel_output)
        mel_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"mel": mel.detach().cpu(), "text": text, "style": style}, mel_path)
    audio_path = render_mel_to_wav(mel, destination, config, vocoder_type=vocoder_type)
    return {
        "status": "complete",
        "backend": "genmusic-vn-self-diffusion",
        "audio_path": str(audio_path),
        "mel_path": str(Path(mel_output).resolve()) if mel_output else None,
        "duration_seconds": float(duration_seconds),
        "diffusion_steps": steps,
        "guidance_scale": float(guidance_scale),
        "seed": seed,
        "lyric_timing": lyric_timing,
        "backing_conditioned": False,
        "muq_style_conditioned": normalized_style is not None,
    }


def structured_random_mel(config: MusicDiffusionConfig, frames: int, *, seed: int):
    torch, _ = _torch()
    generator = torch.Generator().manual_seed(seed)
    time = torch.linspace(0.0, 1.0, frames).unsqueeze(0)
    frequency = torch.linspace(0.0, 1.0, config.n_mels).unsqueeze(1)
    harmonic = torch.sin(2 * math.pi * (frequency * 2.5 + time * (1.0 + seed % 5)))
    noise = torch.randn((config.n_mels, frames), generator=generator) * 0.18
    return (harmonic * 0.8 + noise - 0.5).float()
