"""MicroDiT/CFM text-to-music model: config, mel/waveform I/O, checkpointing.

The denoising network itself lives in `dit_transformer.py` (MicroDiT) and
`cfm_flow.py` (Conditional Flow Matching loss/sampling); this module holds the
shared config, mel-spectrogram <-> waveform conversion, and checkpoint I/O.
"""

from __future__ import annotations

import math
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
    frames_per_chunk: int = 128
    chunk_seconds: float = 4.0


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


def save_checkpoint(model, path: str | Path, config: MusicDiffusionConfig, *, optimizer=None, epoch: int = 0, loss: float | None = None, arch: dict[str, int] | None = None) -> Path:
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
    torch.save(payload, destination)
    return destination.resolve()


def load_checkpoint(path: str | Path, *, device="cpu", roberta_model: str = "xlm-roberta-base") -> tuple[Any, MusicDiffusionConfig, dict[str, Any]]:
    torch, _ = _torch()
    from .dit_transformer import MicroDiT

    payload = torch.load(path, map_location=device, weights_only=False)
    config = _config_from_dict(payload["config"])

    # Reconstruct with the exact architecture size the checkpoint was trained
    # with (defaults match the old hardcoded values, for checkpoints saved
    # before "arch" was tracked) -- a size mismatch here would silently fail
    # to load most weights via load_state_dict(strict=False).
    arch = payload.get("arch") or {}
    model = MicroDiT(
        config, roberta_model=roberta_model,
        dim=arch.get("dim", 256), depth=arch.get("depth", 4),
        heads=arch.get("heads", 4), ff_mult=arch.get("ff_mult", 4),
        style_dim=arch.get("style_dim", 512),
    ).to(device)

    # strict=False: checkpoints deliberately omit the frozen RoBERTa encoder
    # weights (see save_checkpoint) -- those are already loaded fresh from
    # HuggingFace by the model constructor above.
    model.load_state_dict(payload["model"], strict=False)
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

    normalized_backing = None
    if backing_mel is not None:
        normalized_backing = torch.as_tensor(backing_mel, dtype=torch.float32, device=device)
        if normalized_backing.dim() == 2:
            if normalized_backing.shape[0] == config.n_mels:
                normalized_backing = normalized_backing.unsqueeze(0)
            elif normalized_backing.shape[1] == config.n_mels:
                normalized_backing = normalized_backing.transpose(0, 1).unsqueeze(0)
            else:
                raise ValueError(
                    f"backing_mel must contain an n_mels={config.n_mels} axis, got {tuple(normalized_backing.shape)}"
                )
        elif normalized_backing.dim() == 3:
            if normalized_backing.shape[0] != 1:
                raise ValueError("generate_audio currently generates one item and requires backing batch size 1")
            if normalized_backing.shape[1] != config.n_mels and normalized_backing.shape[2] == config.n_mels:
                normalized_backing = normalized_backing.transpose(1, 2)
            elif normalized_backing.shape[1] != config.n_mels:
                raise ValueError(
                    f"backing_mel must contain an n_mels={config.n_mels} axis, got {tuple(normalized_backing.shape)}"
                )
        else:
            raise ValueError(f"backing_mel must have 2 or 3 dimensions, got {tuple(normalized_backing.shape)}")

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
    backing_frame_cursor = 0
    for section in lyric_timing:
        section_count = max(1, math.ceil(section["duration_seconds"] / config.chunk_seconds))
        chunk_duration = section["duration_seconds"] / section_count
        for chunk_index in range(section_count):
            chunk_text = (
                f"{style}. lyric line {section['line_index'] + 1} of {len(lyric_timing)}: {section['line']}. "
                f"sing naturally across {chunk_duration:.2f} seconds; keep space between lyric lines."
            )
            chunk_frames = max(8, int(chunk_duration * config.sample_rate / config.hop_length))
            from .cfm_flow import sample_cfm

            # Training crops the real backing mel at the same temporal offset as
            # the target vocal. Preserve that alignment while generating chunks
            # instead of repeatedly conditioning every chunk on frame zero. Wrap
            # a short reference so later chunks remain conditioned rather than
            # silently becoming zero-padded.
            chunk_backing = None
            if normalized_backing is not None:
                total_frames = normalized_backing.shape[2]
                if total_frames == 0:
                    raise ValueError("backing_mel must contain at least one frame")
                indices = torch.arange(
                    backing_frame_cursor,
                    backing_frame_cursor + chunk_frames,
                    device=device,
                ) % total_frames
                chunk_backing = normalized_backing.index_select(2, indices)
            mel = sample_cfm(
                model,
                [chunk_text],
                chunk_frames,
                config=config,
                device=device,
                steps=steps,
                seed=seed + section_number,
                backing_mel=chunk_backing,
                style_prompt=normalized_style,
            )
            rendered.append(mel.squeeze(0))
            backing_frame_cursor += chunk_frames
            section_number += 1
    mel = torch.cat(rendered, dim=1)
    target_frames = max(1, int(float(duration_seconds) * config.sample_rate / config.hop_length))
    mel = mel[:, :target_frames]
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
        "seed": seed,
        "lyric_timing": lyric_timing,
        "backing_conditioned": normalized_backing is not None,
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
