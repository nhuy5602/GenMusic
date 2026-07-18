"""Vietnamese pronunciation prior used to anchor CFM generation.

The self-diffusion dataset was labelled by Whisper-tiny.  Those labels are
useful for locating vocal spans, but they are too noisy to teach a model from
scratch how every Vietnamese phoneme should sound.  A small frozen Vietnamese
VITS model supplies an intelligible waveform for the requested words.  CFM can
then start near that waveform instead of starting from pure Gaussian noise and
use the learned singing-vocal distribution as a refinement step.

This module is inference-only: it never changes the training target or hides a
fallback inside the model.  A prior strength of zero keeps the original
sampling path exactly unchanged.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


DEFAULT_VIETNAMESE_TTS_MODEL = "facebook/mms-tts-vie"


def trim_wav_silence(
    source: str | Path,
    destination: str | Path,
    *,
    top_db: float = 32.0,
    pad_seconds: float = 0.08,
) -> Path:
    """Remove leading/trailing dead air while preserving short consonants.

    Fixed-duration quality windows intentionally retain padding for tensor
    parity.  User-facing audio should not: silent padding makes spectral
    flatness look noise-like and leaves several seconds of dead air after a
    short lyric.  Keep a small pad so plosives at either boundary are not cut.
    """
    import librosa
    import numpy as np
    import soundfile as sf

    source_path = Path(source)
    destination_path = Path(destination)
    waveform, sample_rate = librosa.load(str(source_path), sr=None, mono=True)
    if waveform.size == 0:
        raise ValueError(f"Cannot trim empty audio: {source_path}")
    _, interval = librosa.effects.trim(
        waveform,
        top_db=float(top_db),
        frame_length=1024,
        hop_length=256,
    )
    pad = max(0, round(float(pad_seconds) * sample_rate))
    start = max(0, int(interval[0]) - pad)
    end = min(len(waveform), int(interval[1]) + pad)
    trimmed = waveform[start:end]
    if trimmed.size == 0:
        trimmed = waveform
    peak = float(np.max(np.abs(trimmed)))
    if peak > 1e-6:
        trimmed = trimmed / peak * 0.9
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(destination_path), trimmed, sample_rate)
    return destination_path.resolve()


@lru_cache(maxsize=4)
def _load_vietnamese_tts(model_name: str, device: str):
    """Load one frozen MMS/VITS checkpoint per model/device pair."""
    from transformers import AutoTokenizer, VitsModel

    resolved = os.getenv("GENMUSIC_VIETNAMESE_TTS_PATH") or model_name
    print(f"Loading Vietnamese pronunciation prior: {resolved}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(resolved)
    model = VitsModel.from_pretrained(resolved).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return tokenizer, model


def synthesize_pronunciation_mel(
    text: str,
    *,
    frames: int,
    config,
    device,
    seed: int = 5602,
    model_name: str = DEFAULT_VIETNAMESE_TTS_MODEL,
):
    """Return a normalized Vocos mel containing clearly spoken Vietnamese.

    The TTS waveform is padded to the requested window.  Only an overlong
    waveform is time-compressed, so a short phrase is not unnaturally stretched
    merely to fill a four-second evaluation crop.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    import torchaudio

    from .text_to_music_diffusion import compute_mel_spectrogram, normalize_mel

    target_frames = max(1, int(frames))
    target_samples = max(1, target_frames * int(config.hop_length))
    device_string = str(device)
    tokenizer, model = _load_vietnamese_tts(model_name, device_string)
    inputs = tokenizer(str(text), return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    cuda_devices = [torch.cuda.current_device()] if device_string.startswith("cuda") else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(int(seed))
        if cuda_devices:
            torch.cuda.manual_seed_all(int(seed))
        with torch.no_grad():
            waveform = model(**inputs).waveform[0].float().cpu()

    source_rate = int(model.config.sampling_rate)
    if source_rate != int(config.sample_rate):
        waveform = torchaudio.functional.resample(
            waveform,
            source_rate,
            int(config.sample_rate),
        )

    # Do not cut off final syllables.  Compress only when the TTS phrase is
    # longer than the requested generation window; otherwise pad with silence.
    if waveform.numel() > target_samples:
        waveform = F.interpolate(
            waveform.view(1, 1, -1),
            size=target_samples,
            mode="linear",
            align_corners=False,
        ).view(-1)
    elif waveform.numel() < target_samples:
        waveform = F.pad(waveform, (0, target_samples - waveform.numel()))

    # compute_mel_spectrogram adds a centered STFT frame.  Fit explicitly so
    # every caller receives the exact tensor length expected by the CFM model.
    mel = compute_mel_spectrogram(waveform.numpy().astype(np.float32), config)
    if mel.shape[1] < target_frames:
        mel = F.pad(mel, (0, target_frames - mel.shape[1]), value=float(mel.min()))
    mel = mel[:, :target_frames]
    return normalize_mel(mel, config).to(device)
