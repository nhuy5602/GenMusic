"""Official DiffRhythm V1 Oobleck VAE decoding.

The clean V1 latent corpus stores posterior means from the released
``ASLP-lab/DiffRhythm-vae`` TorchScript model. Those 64-channel latents are
not compatible with the DiffRhythm2 BigVGAN decoder used by older project
checkpoints, so the codec choice must be explicit.
"""

from __future__ import annotations

from functools import lru_cache

V1_VAE_REPO = "ASLP-lab/DiffRhythm-vae"
V1_VAE_FILENAME = "vae_model.pt"
V1_LATENT_CHANNELS = 64
V1_SAMPLE_RATE = 44_100
V1_DOWNSAMPLE_RATIO = 2_048


@lru_cache(maxsize=2)
def load_v1_oobleck_vae(
    device: str,
    *,
    repo_id: str = V1_VAE_REPO,
    filename: str = V1_VAE_FILENAME,
):
    """Load and freeze the released full V1 Music VAE."""
    import torch
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=repo_id, filename=filename)
    vae = torch.jit.load(path, map_location="cpu").eval()
    methods = set(vae._c._method_names())
    required = {"encode_export", "decode_export"}
    missing = sorted(required - methods)
    if missing:
        raise RuntimeError(
            "DiffRhythm V1 VAE is missing required methods "
            f"{missing}; "
            f"available methods: {sorted(methods)}"
        )
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    return vae.to(device)


def decode_v1_oobleck_latent(latent, *, device: str):
    """Decode ``(batch, 64, frames)`` V1 posterior-mean latents to stereo."""
    import torch

    values = torch.as_tensor(latent, dtype=torch.float32, device=device)
    if values.ndim != 3 or values.shape[1] != V1_LATENT_CHANNELS:
        raise ValueError(
            "Expected DiffRhythm V1 latent shape (batch, 64, frames), "
            f"got {tuple(values.shape)}"
        )
    vae = load_v1_oobleck_vae(device)
    with torch.inference_mode():
        audio = vae.decode_export(values)
    if audio.ndim != 3 or audio.shape[1] not in (1, 2):
        raise RuntimeError(
            "DiffRhythm V1 decoder returned an invalid waveform shape: "
            f"{tuple(audio.shape)}"
        )
    return audio.float()


def encode_v1_oobleck_audio(audio, *, device: str):
    """Encode stereo 44.1-kHz audio to deterministic posterior-mean latents."""
    import torch

    values = torch.as_tensor(audio, dtype=torch.float32, device=device)
    if values.ndim != 3 or values.shape[1] != 2:
        raise ValueError(
            "Expected DiffRhythm V1 audio shape (batch, 2, samples), "
            f"got {tuple(values.shape)}"
        )
    if values.shape[-1] % V1_DOWNSAMPLE_RATIO:
        raise ValueError(
            "V1 input samples must be divisible by "
            f"{V1_DOWNSAMPLE_RATIO}, got {values.shape[-1]}"
        )
    vae = load_v1_oobleck_vae(device)
    with torch.inference_mode():
        encoded = vae.encode_export(values)
    if encoded.ndim != 3 or encoded.shape[1] != V1_LATENT_CHANNELS * 2:
        raise RuntimeError(
            "DiffRhythm V1 encoder returned an invalid posterior shape: "
            f"{tuple(encoded.shape)}"
        )
    mean, scale = encoded.chunk(2, dim=1)
    if not torch.isfinite(mean).all() or not torch.isfinite(scale).all():
        raise RuntimeError("DiffRhythm V1 posterior contains non-finite values")
    return mean.float()


def roundtrip_v1_oobleck_mono_unit(
    audio,
    *,
    input_sample_rate: int,
    device: str,
    context_frames: int = 2,
):
    """Round-trip one short mono unit while preserving length and edges."""
    import torch
    import torch.nn.functional as F
    import torchaudio.functional as AF

    source = torch.as_tensor(audio, dtype=torch.float32).flatten()
    if source.numel() < 256 or not bool(torch.isfinite(source).all()):
        raise ValueError("V1 word unit must be finite and non-empty.")
    if int(input_sample_rate) <= 0:
        raise ValueError("input_sample_rate must be positive.")
    original_length = int(source.numel())
    encoded_rate_audio = AF.resample(
        source,
        int(input_sample_rate),
        V1_SAMPLE_RATE,
    )
    context = max(0, int(context_frames)) * V1_DOWNSAMPLE_RATIO
    padded = F.pad(encoded_rate_audio, (context, context))
    target_length = (
        (padded.numel() + V1_DOWNSAMPLE_RATIO - 1)
        // V1_DOWNSAMPLE_RATIO
        * V1_DOWNSAMPLE_RATIO
    )
    padded = F.pad(padded, (0, target_length - padded.numel()))
    stereo = padded.view(1, 1, -1).expand(1, 2, -1).contiguous()
    latent = encode_v1_oobleck_audio(stereo, device=device)
    decoded = decode_v1_oobleck_latent(latent, device=device).mean(dim=1)[0]
    decoded = decoded[context : context + encoded_rate_audio.numel()].cpu()
    restored = AF.resample(
        decoded,
        V1_SAMPLE_RATE,
        int(input_sample_rate),
    )
    if restored.numel() < original_length:
        restored = F.pad(restored, (0, original_length - restored.numel()))
    restored = restored[:original_length]
    source_rms = source.square().mean().sqrt().clamp_min(1e-6)
    restored_rms = restored.square().mean().sqrt().clamp_min(1e-6)
    restored = restored * (source_rms / restored_rms).clamp(0.70, 1.30)
    peak = restored.abs().amax().clamp_min(1e-6)
    restored = restored / torch.maximum(
        peak / 0.98,
        torch.ones_like(peak),
    )
    boundary = min(original_length // 2, max(8, original_length // 25))
    ramp = torch.linspace(0.0, 1.0, boundary)
    blend = torch.ones_like(restored)
    blend[:boundary] = ramp
    blend[-boundary:] = ramp.flip(0)
    restored = restored * blend + source * (1.0 - blend)
    return restored, latent.cpu()
