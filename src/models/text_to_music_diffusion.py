"""Self-contained conditional diffusion model for text-to-music experiments.

This is intentionally small enough to train on a Kaggle GPU smoke run while
keeping the essential generative path: text conditioning, noise schedule,
denoising network, checkpointing and mel-to-waveform rendering.
"""

from __future__ import annotations

import json
import math
import random
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class MusicDiffusionConfig:
    sample_rate: int = 16_000
    n_mels: int = 64
    n_fft: int = 512
    hop_length: int = 256
    frames_per_chunk: int = 128
    text_vocab_size: int = 256
    text_max_length: int = 256
    text_dim: int = 96
    conditioner_layers: int = 2
    hidden_dim: int = 96
    residual_layers: int = 4
    diffusion_steps: int = 32
    chunk_seconds: float = 4.0


def encode_text(text: str, *, max_length: int = 256, vocab_size: int = 256) -> list[int]:
    character_space = max(1, vocab_size - 3)
    values = [2 if char == "\n" else 3 + (ord(char) % character_space) for char in text[:max_length]]
    return values or [0]


def _torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover - dependency boundary
        raise RuntimeError("Cần cài torch để chạy model sinh nhạc tự code.") from exc
    return torch, nn


def _sinusoidal_embedding(timesteps, dimension: int):
    torch, _ = _torch()
    half = max(1, dimension // 2)
    frequencies = torch.exp(
        -math.log(10_000.0) * torch.arange(half, device=timesteps.device).float() / max(1, half - 1)
    )
    angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
    return embedding[:, :dimension]


class TextConditioner:
    def __init__(self, config: MusicDiffusionConfig):
        torch, nn = _torch()
        self.config = config
        self.embedding = nn.Embedding(config.text_vocab_size, config.text_dim)
        self.position = nn.Embedding(config.text_max_length, config.text_dim)
        attention_layer = nn.TransformerEncoderLayer(
            d_model=config.text_dim,
            nhead=4,
            dim_feedforward=config.text_dim * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(attention_layer, num_layers=config.conditioner_layers)
        self.projection = nn.Sequential(
            nn.Linear(config.text_dim, config.text_dim),
            nn.SiLU(),
            nn.Linear(config.text_dim, config.text_dim),
        )

    def modules(self):
        return [self.embedding, self.position, self.encoder, self.projection]

    def __call__(self, tokens):
        torch, _ = _torch()
        mask = (tokens != 0).float().unsqueeze(-1)
        width = tokens.shape[1]
        positions = torch.arange(width, device=tokens.device).clamp_max(self.config.text_max_length - 1)
        values = self.embedding(tokens) + self.position(positions).unsqueeze(0)
        values = self.encoder(values, src_key_padding_mask=~mask.squeeze(-1).bool())
        pooled = (values * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        return self.projection(pooled)


class ResidualDenoiser:
    """A compact conditional Conv1D denoiser used by the diffusion sampler."""

    def __init__(self, config: MusicDiffusionConfig):
        torch, nn = _torch()
        self.config = config
        self.input = nn.Conv1d(config.n_mels, config.hidden_dim, 3, padding=1)
        self.time = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.conditioner = TextConditioner(config)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, config.hidden_dim),
                    nn.SiLU(),
                    nn.Conv1d(config.hidden_dim, config.hidden_dim, 3, padding=1),
                    nn.GroupNorm(8, config.hidden_dim),
                    nn.SiLU(),
                    nn.Conv1d(config.hidden_dim, config.hidden_dim, 3, padding=1),
                )
                for _ in range(config.residual_layers)
            ]
        )
        self.output = nn.Sequential(
            nn.GroupNorm(8, config.hidden_dim),
            nn.SiLU(),
            nn.Conv1d(config.hidden_dim, config.n_mels, 3, padding=1),
        )

    def modules(self):
        return [self.input, self.time, *self.conditioner.modules(), self.blocks, self.output]

    def parameters(self):
        for module in self.modules():
            yield from module.parameters()

    def state_dict(self):
        state = {}
        for module_name, module in (
            ("input", self.input),
            ("time", self.time),
            ("conditioner.embedding", self.conditioner.embedding),
            ("conditioner.position", self.conditioner.position),
            ("conditioner.encoder", self.conditioner.encoder),
            ("conditioner.projection", self.conditioner.projection),
            ("blocks", self.blocks),
            ("output", self.output),
        ):
            state.update({f"{module_name}.{key}": value for key, value in module.state_dict().items()})
        return state

    def load_state_dict(self, state):
        groups = {
            "input": self.input,
            "time": self.time,
            "conditioner.embedding": self.conditioner.embedding,
            "conditioner.position": self.conditioner.position,
            "conditioner.encoder": self.conditioner.encoder,
            "conditioner.projection": self.conditioner.projection,
            "blocks": self.blocks,
            "output": self.output,
        }
        for name, module in groups.items():
            prefix = name + "."
            values = {key.removeprefix(prefix): value for key, value in state.items() if key.startswith(prefix)}
            if values:
                module.load_state_dict(values, strict=False)

    def train(self, mode: bool = True):
        for module in self.modules():
            module.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def to(self, device):
        for module in self.modules():
            module.to(device)
        return self

    def __call__(self, noisy, timesteps, tokens):
        torch, _ = _torch()
        time_condition = self.time(_sinusoidal_embedding(timesteps, self.config.hidden_dim)).unsqueeze(-1)
        text_condition = self.conditioner(tokens).unsqueeze(-1)
        hidden = self.input(noisy) + time_condition + text_condition
        for block in self.blocks:
            hidden = hidden + block(hidden)
        return self.output(hidden)


def make_model(config: MusicDiffusionConfig | None = None) -> ResidualDenoiser:
    return ResidualDenoiser(config or MusicDiffusionConfig())


def diffusion_schedule(config: MusicDiffusionConfig, device):
    torch, _ = _torch()
    beta = torch.linspace(1e-4, 0.02, config.diffusion_steps, device=device)
    alpha = 1.0 - beta
    cumulative = torch.cumprod(alpha, dim=0)
    return beta, alpha, cumulative


def text_batch(texts: Iterable[str], config: MusicDiffusionConfig, device):
    torch, _ = _torch()
    values = [encode_text(text, vocab_size=config.text_vocab_size) for text in texts]
    width = max(len(value) for value in values)
    result = torch.zeros((len(values), width), dtype=torch.long, device=device)
    for index, value in enumerate(values):
        result[index, : len(value)] = torch.tensor(value, dtype=torch.long, device=device)
    return result


def diffusion_loss(model: ResidualDenoiser, clean_mel, texts: list[str], config: MusicDiffusionConfig):
    torch, _ = _torch()
    device = clean_mel.device
    beta, _, cumulative = diffusion_schedule(config, device)
    timesteps = torch.randint(0, config.diffusion_steps, (clean_mel.shape[0],), device=device)
    noise = torch.randn_like(clean_mel)
    alpha_bar = cumulative[timesteps].view(-1, 1, 1)
    noisy = alpha_bar.sqrt() * clean_mel + (1.0 - alpha_bar).sqrt() * noise
    prediction = model(noisy, timesteps.float() / config.diffusion_steps, text_batch(texts, config, device))
    return torch.nn.functional.mse_loss(prediction, noise)


def sample_mel(model: ResidualDenoiser, text: str, frames: int, *, config: MusicDiffusionConfig, device, steps: int | None = None, seed: int | None = None):
    torch, _ = _torch()
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    steps = max(1, min(config.diffusion_steps, int(steps or config.diffusion_steps)))
    beta, alpha, cumulative = diffusion_schedule(config, device)
    indices = torch.linspace(config.diffusion_steps - 1, 0, steps, device=device).long()
    tokens = text_batch([text], config, device)
    sample = torch.randn((1, config.n_mels, frames), device=device)
    with torch.no_grad():
        for index in indices:
            timestep = index.float().view(1)
            prediction = model(sample, timestep / config.diffusion_steps, tokens)
            current_alpha = alpha[index]
            current_beta = beta[index]
            current_cumulative = cumulative[index]
            sample = (sample - current_beta / (1.0 - current_cumulative).sqrt() * prediction) / current_alpha.sqrt()
            if int(index) > 0:
                sample = sample + current_beta.sqrt() * 0.15 * torch.randn_like(sample)
    return sample.clamp(-5.0, 3.0)


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


def render_mel_to_wav(mel, destination: str | Path, config: MusicDiffusionConfig, vocoder_type: str = "istft") -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    if vocoder_type == "vocos":
        try:
            from vocos import Vocos
            import torch
            import torch.nn.functional as F
            
            # Load pretrained model
            vocos_model = Vocos.from_pretrained("charactr/vocos-mel-24khz")
            
            # Get raw mel as torch tensor
            if not isinstance(mel, torch.Tensor):
                mel_tensor = torch.tensor(mel, dtype=torch.float32)
            else:
                mel_tensor = mel.detach().cpu().float()
                
            # Shape is [n_mels, time_steps]
            n_mels, time_steps = mel_tensor.shape
            
            # Calculate target time steps (resample to 24000 Hz frame rate)
            # Vocos expects 24000 Hz sample rate, 256 hop length.
            # Original has config.sample_rate and config.hop_length.
            original_fps = config.sample_rate / config.hop_length
            vocos_fps = 24000 / 256
            time_scale = vocos_fps / original_fps
            target_time_steps = max(8, int(time_steps * time_scale))
            
            # Interpolate to shape [1, 100, target_time_steps]
            mel_tensor = mel_tensor.unsqueeze(0).unsqueeze(0) # [1, 1, n_mels, time_steps]
            mel_tensor = F.interpolate(
                mel_tensor, 
                size=(100, target_time_steps), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0) # [1, 100, target_time_steps]
            
            # Vocos expects log-mel in range matching its training distribution.
            # Our model output is in similar scale, so we pass it directly.
            with torch.no_grad():
                audio_tensor = vocos_model.decode(mel_tensor) # [1, audio_len]
                
            audio = audio_tensor.squeeze(0).cpu().numpy()
            audio = audio / max(1e-6, float(np.max(np.abs(audio)))) * 0.8
            pcm = (audio.clip(-1.0, 1.0) * 32767.0).astype(np.int16)
            
            with wave.open(str(destination), "wb") as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(24000) # Output sample rate of Vocos
                stream.writeframes(pcm.tobytes())
            return destination.resolve()
        except Exception as e:
            print(f"⚠️ Warning: Vocos decoding failed ({e}). Falling back to iSTFT...")

    try:
        import librosa
    except ImportError as exc:  # pragma: no cover - dependency boundary
        raise RuntimeError("Cần librosa để đổi mel thành WAV.") from exc
    values = mel.detach().float().cpu().numpy() if hasattr(mel, "detach") else np.asarray(mel, dtype=np.float32)
    values = np.exp(np.clip(values, -5.0, 3.0)).astype(np.float32)
    mel_filter = librosa.filters.mel(sr=config.sample_rate, n_fft=config.n_fft, n_mels=config.n_mels, dtype=np.float32)
    linear_power = np.maximum(0.0, np.linalg.pinv(mel_filter) @ values)
    magnitude = np.sqrt(linear_power + 1e-7)
    frequencies = np.linspace(0.0, config.sample_rate / 2.0, magnitude.shape[0], dtype=np.float32)[:, None]
    frame_times = np.arange(magnitude.shape[1], dtype=np.float32)[None, :] * config.hop_length / config.sample_rate
    phase = 2.0 * np.pi * frequencies * frame_times
    phase += np.linspace(0.0, np.pi, magnitude.shape[0], dtype=np.float32)[:, None]
    audio = librosa.istft(magnitude * np.exp(1j * phase), hop_length=config.hop_length, n_fft=config.n_fft)
    audio = audio / max(1e-6, float(np.max(np.abs(audio)))) * 0.8
    pcm = (audio.clip(-1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(destination), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(config.sample_rate)
        stream.writeframes(pcm.tobytes())
    return destination.resolve()


def save_checkpoint(model: ResidualDenoiser, path: str | Path, config: MusicDiffusionConfig, *, optimizer=None, epoch: int = 0, loss: float | None = None) -> Path:
    torch, _ = _torch()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {"config": asdict(config), "model": model.state_dict(), "epoch": epoch, "loss": loss}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, destination)
    return destination.resolve()


def load_checkpoint(path: str | Path, *, device="cpu", model_type: str | None = None, roberta_model: str = "xlm-roberta-base") -> tuple[nn.Module, MusicDiffusionConfig, dict[str, Any]]:
    torch, _ = _torch()
    payload = torch.load(path, map_location=device, weights_only=False)
    config = MusicDiffusionConfig(**payload["config"])
    
    # Detect model type based on saved state dict keys
    state_keys = payload["model"].keys()
    is_dit = any("transformer_blocks" in k or "text_encoder" in k for k in state_keys)
    
    if model_type == "dit" or (model_type is None and is_dit):
        from .dit_transformer import MicroDiT
        model = MicroDiT(config, roberta_model=roberta_model).to(device)
    else:
        model = make_model(config).to(device)
        
    model.load_state_dict(payload["model"])
    return model, config, payload


def generate_audio(model: ResidualDenoiser, text: str, style: str, destination: str | Path, *, duration_seconds: float, config: MusicDiffusionConfig, device="cpu", steps: int = 6, seed: int = 5602, mel_output: str | Path | None = None, vocoder_type: str = "istft") -> dict[str, Any]:
    torch, _ = _torch()
    model.to(device)
    duration_seconds = max(float(duration_seconds), estimate_minimum_lyric_duration(text))
    rendered = []
    lyric_timing = build_lyric_timing(text, duration_seconds)
    section_number = 0
    for section in lyric_timing:
        section_count = max(1, math.ceil(section["duration_seconds"] / config.chunk_seconds))
        chunk_duration = section["duration_seconds"] / section_count
        for chunk_index in range(section_count):
            chunk_text = (
                f"{style}. lyric line {section['line_index'] + 1} of {len(lyric_timing)}: {section['line']}. "
                f"sing naturally across {chunk_duration:.2f} seconds; keep space between lyric lines."
            )
            chunk_frames = max(8, int(chunk_duration * config.sample_rate / config.hop_length))
            is_dit = model.__class__.__name__ == "MicroDiT"
            if is_dit:
                from .cfm_flow import sample_cfm
                mel = sample_cfm(model, [chunk_text], chunk_frames, config=config, device=device, steps=steps, seed=seed + section_number)
            else:
                mel = sample_mel(model, chunk_text, chunk_frames, config=config, device=device, steps=steps, seed=seed + section_number)
            rendered.append(mel.squeeze(0))
            section_number += 1
    mel = torch.cat(rendered, dim=1)
    target_frames = max(1, int(float(duration_seconds) * config.sample_rate / config.hop_length))
    mel = mel[:, :target_frames]
    if mel_output:
        mel_path = Path(mel_output)
        mel_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"mel": mel.detach().cpu(), "text": text, "style": style}, mel_path)
    audio_path = render_mel_to_wav(mel, destination, config, vocoder_type=vocoder_type)
    return {"status": "complete", "backend": "genmusic-vn-self-diffusion", "audio_path": str(audio_path), "mel_path": str(Path(mel_output).resolve()) if mel_output else None, "duration_seconds": float(duration_seconds), "diffusion_steps": steps, "seed": seed, "lyric_timing": lyric_timing}


def structured_random_mel(config: MusicDiffusionConfig, frames: int, *, seed: int):
    torch, _ = _torch()
    generator = torch.Generator().manual_seed(seed)
    time = torch.linspace(0.0, 1.0, frames).unsqueeze(0)
    frequency = torch.linspace(0.0, 1.0, config.n_mels).unsqueeze(1)
    harmonic = torch.sin(2 * math.pi * (frequency * 2.5 + time * (1.0 + seed % 5)))
    noise = torch.randn((config.n_mels, frames), generator=generator) * 0.18
    return (harmonic * 0.8 + noise - 0.5).float()
