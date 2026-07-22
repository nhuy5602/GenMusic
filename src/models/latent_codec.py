"""A new, trainable audio encoder paired with DiffRhythm2's real, frozen,
pretrained decoder (`decoder.bin`/`decoder.json` on the ASLP-lab/DiffRhythm2
HuggingFace repo -- a BigVGAN-family generator, see `bigvgan/model.py` in the
official GitHub repo).

Why this exists: DiffRhythm2's own teacher operates on a 64-dim, 5 Hz Music
VAE latent (Stable-Audio-2-style encoder + a transformer block + BigVGAN
decoder, per the paper), while this project's student has always generated a
raw, uncompressed 100-dim mel spectrogram at 93.75 Hz -- a ~19x higher token
rate for the same audio duration than the architecture DiffRhythm2's own DiT
was actually designed/trained around (see the conversation this session:
docs/project_history.md §4.20-4.21 already found this exact mismatch, but only
fixed it on the teacher-*query* side during distillation, never gave the
*student* its own compressed latent space -- deferred as "too expensive to
train a full Music VAE from scratch").

Training a full paper-faithful VAE (Stable-Audio-2 encoder + adversarial
multi-period/multi-scale/CQT discriminators + multi-scale mel/STFT loss) from
scratch on this project's 250-song budget is a real, large undertaking with
genuine GAN-training-instability risk -- explicitly rejected this session in
favor of a cheaper path: DiffRhythm2's *decoder* is already published,
pretrained, and proven (it is the exact decoder the real teacher uses) --
reuse it FROZEN, and train only a new, much smaller ENCODER against it with a
plain reconstruction loss (no discriminators, no GAN training). This is
strictly cheaper and lower-risk than training the whole codec, at the cost of
some uncertainty: the frozen decoder was never co-trained with this new
encoder, so there's a real (but bounded, cheaply testable) risk it doesn't
converge to clean reconstructions.

Sample-rate note: the decoder outputs 48 kHz audio (9600x upsample from its
5 Hz latent, confirmed via decoder.json: upsample_rates
[10,10,8,3,2,2] -> product 9600). This new encoder takes 24 kHz input (this
project's native rate) and downsamples by 4800x (paper's own stated
*encode*-side ratio) -- one fewer 2x stage than the decoder's upsample
schedule, since the input starts at half the decoder's output rate.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

DECODER_REPO_ID = "ASLP-lab/DiffRhythm2"
DECODER_IN_CHANNELS = 64
DECODER_FPS = 5
# Paper: "achieving compression ratios of 4800x during encoding and 9600x
# during decoding" -- encoder (24kHz in) and decoder (48kHz out) each see
# half of the total 24kHz->48kHz span, so their own compression ratios
# relative to their own native rate differ by exactly 2x.
ENCODER_INPUT_SAMPLE_RATE = 24_000
ENCODER_DOWNSAMPLE_STRIDES = (10, 10, 8, 3, 2)  # product = 4800


def _downsample_factor() -> int:
    factor = 1
    for stride in ENCODER_DOWNSAMPLE_STRIDES:
        factor *= stride
    return factor


class _ResidualUnit(nn.Module):
    """Dilated conv residual block, standard for audio codec encoders (same
    family of building block as the decoder's own AMPBlock, just without the
    Snake activation's extra learned parameters -- this module is trained
    from scratch and does not need to match the decoder's exact activation)."""

    def __init__(self, channels: int, dilation: int):
        super().__init__()
        padding = dilation * (7 - 1) // 2
        self.block = nn.Sequential(
            nn.GroupNorm(min(8, channels), channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size=7, dilation=dilation, padding=padding),
            nn.GroupNorm(min(8, channels), channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class _DownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.residuals = nn.Sequential(
            _ResidualUnit(in_channels, dilation=1),
            _ResidualUnit(in_channels, dilation=3),
            _ResidualUnit(in_channels, dilation=9),
        )
        kernel_size = 2 * stride
        padding = (kernel_size - stride + 1) // 2
        self.downsample = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.residuals(x)
        return self.downsample(x)


class LatentAudioEncoder(nn.Module):
    """Mono 24kHz waveform -> (B, DECODER_IN_CHANNELS, T) latent at 5Hz.

    Trained from scratch (see module docstring) against DiffRhythm2's real,
    frozen decoder -- NOT an attempt to reproduce Stable Audio 2's actual
    encoder architecture, just a functionally-compatible replacement small
    enough to train on this project's own 250-song budget.
    """

    def __init__(self, base_channels: int = 32, max_channels: int = 512):
        super().__init__()
        self.conv_in = nn.Conv1d(1, base_channels, kernel_size=7, padding=3)
        channels = base_channels
        blocks = []
        for stride in ENCODER_DOWNSAMPLE_STRIDES:
            next_channels = min(max_channels, channels * 2)
            blocks.append(_DownsampleBlock(channels, next_channels, stride))
            channels = next_channels
        self.blocks = nn.ModuleList(blocks)
        self.conv_out = nn.Conv1d(channels, DECODER_IN_CHANNELS, kernel_size=7, padding=3)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform: (B, samples) or (B, 1, samples) mono audio at
        ENCODER_INPUT_SAMPLE_RATE. Returns (B, DECODER_IN_CHANNELS, T)."""
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)
        x = self.conv_in(waveform)
        for block in self.blocks:
            x = block(x)
        return self.conv_out(x)


@dataclass
class FrozenDecoderHandle:
    decoder: nn.Module  # bigvgan.model.Generator, kept frozen (eval, requires_grad=False)
    sampling_rate: int
    fps: int


def load_frozen_decoder(device: str, repo_id: str = DECODER_REPO_ID) -> FrozenDecoderHandle:
    """Downloads and loads the real, pretrained DiffRhythm2 BigVGAN decoder.

    Requires the DiffRhythm2 GitHub repo on PYTHONPATH (same requirement as
    `_load_teacher` in distill_training.py, and the same reason: `bigvgan` is
    not a pip package, only available by cloning the official repo -- see
    scripts/run_kaggle_distill.py). Raises with a clear message if
    unavailable rather than silently falling back to nothing, matching this
    project's established "no silent fake teacher/decoder" convention.

    IMPORTANT: the returned decoder's parameters have requires_grad=False
    (frozen weights) but forward() is NOT wrapped in torch.no_grad() by
    callers that need to train an encoder against it -- wrapping this call in
    torch.no_grad() would silently zero out the encoder's gradient entirely
    (this project already hit exactly this bug once, see the mel-dim adapter
    gradient bug in docs/project_history.md; do not repeat it).
    """
    try:
        from bigvgan.model import Generator
    except ImportError as exc:
        raise RuntimeError(
            f"bigvgan package not importable ({exc}). The DiffRhythm2 repo must be cloned and "
            "added to PYTHONPATH (see scripts/run_kaggle_distill.py) -- this only works on Kaggle."
        ) from exc

    from huggingface_hub import hf_hub_download

    decoder_ckpt_path = hf_hub_download(repo_id=repo_id, filename="decoder.bin", local_dir="./ckpt")
    decoder_config_path = hf_hub_download(repo_id=repo_id, filename="decoder.json", local_dir="./ckpt")
    decoder = Generator(decoder_config_path, decoder_ckpt_path)
    for param in decoder.parameters():
        param.requires_grad = False
    decoder.eval()
    decoder.to(device)
    import json

    with open(decoder_config_path) as f:
        config = json.load(f)
    return FrozenDecoderHandle(decoder=decoder, sampling_rate=config["sampling_rate"], fps=config["fps"])


def multi_scale_mel_loss(
    generated: torch.Tensor, target: torch.Tensor, sample_rate: int,
    n_ffts: tuple[int, ...] = (512, 1024, 2048), n_mels_per_scale: tuple[int, ...] = (40, 80, 80),
) -> torch.Tensor:
    """Multi-scale mel-spectrogram L1 loss (standard vocoder reconstruction
    loss family, same spirit as BigVGAN's own training loss) between two
    (B, 1, samples) or (B, samples) waveforms at the same sample rate.

    Kept dependency-light (torchaudio, already a project dependency via
    preprocessing) rather than reimplementing the real BigVGAN training
    loss's exact multi-resolution STFT term -- sufficient to tell the encoder
    "did this reconstruct the real spectral content", not intended to be a
    bit-exact replication of the official Music VAE's training objective."""
    import torchaudio

    if generated.dim() == 3:
        generated = generated.squeeze(1)
    if target.dim() == 3:
        target = target.squeeze(1)
    min_len = min(generated.shape[-1], target.shape[-1])
    generated = generated[..., :min_len]
    target = target[..., :min_len]

    total = generated.new_zeros(())
    for n_fft, n_mels in zip(n_ffts, n_mels_per_scale):
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=n_fft // 4, n_mels=n_mels,
        ).to(generated.device)
        gen_mel = torch.log(mel(generated).clamp_min(1e-5))
        tgt_mel = torch.log(mel(target).clamp_min(1e-5))
        total = total + F.l1_loss(gen_mel, tgt_mel)
    return total / len(n_ffts)
