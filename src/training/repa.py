"""REPA-style (representation-alignment) auxiliary loss for the student.

Mirrors DiffRhythm2 teacher's own "Stochastic Block REPA" (see
docs/project_history.md, arXiv:2510.22950): align an intermediate transformer
hidden state with a frozen, externally pretrained self-supervised audio
encoder's features, instead of supervising only the final CFM output. The
motivation is the same as the teacher's: predicting/matching in representation
space rewards learning abstract structure (e.g. how pitch/melody evolves)
without spending student capacity on exact spectral reconstruction detail --
directly targeting this project's recurring failure mode of the student
collapsing to a near-monotone or condition-invariant output (see
docs/project_history.md's attention-distillation and root-cause sections).

Target features come from MuQ (`OpenMuQ/MuQ-large-msd-iter`), a frozen,
pretrained self-supervised MUSIC encoder (Wav2Vec2-Conformer, 1024-dim,
24 layers) -- NOT MuQ-MuLan (that's a single global contrastive embedding,
already used elsewhere for style conditioning; this needs frame-level
features). Since training only has the ground-truth MEL (not the raw
waveform) per crop, the mel is first decoded back to waveform via the
frozen Vocos vocoder already used elsewhere in this project -- this whole
path is no_grad (target features are always stop-gradient, standard REPA).
"""
from __future__ import annotations

import random

_muq_model = None
_vocos_model = None


def _torch():
    import torch
    import torch.nn.functional as F
    return torch, F


def _load_muq(device: str):
    global _muq_model
    if _muq_model is None:
        from muq import MuQ

        _muq_model = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")
        _muq_model = _muq_model.to(device).eval()
        for param in _muq_model.parameters():
            param.requires_grad = False
    return _muq_model


def _load_vocos(device: str):
    global _vocos_model
    if _vocos_model is None:
        from vocos import Vocos

        _vocos_model = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device).eval()
        for param in _vocos_model.parameters():
            param.requires_grad = False
    return _vocos_model


def repa_targets_available() -> bool:
    """Cheap check without actually downloading anything -- lets callers decide
    whether to pass repa_layer_idx to the student's forward at all this step."""
    try:
        import muq  # noqa: F401
        import vocos  # noqa: F401
    except ImportError:
        return False
    return True


def compute_repa_target(x1_normalized, config, device: str, num_hidden_layers: int = 24, layer_pool: tuple[int, int] = (8, 16)):
    """Decode ground-truth mel (x1, model-space normalized) to waveform via Vocos,
    run it through frozen MuQ, and return a (B, T, muq_dim) target feature
    sequence resampled to match x1's frame count T -- or None if either frozen
    model isn't available (caller should then skip the REPA loss for this step
    rather than crash a whole training run over an optional auxiliary signal).

    `layer_pool` restricts which of MuQ's hidden_states indices get sampled
    from (excludes the raw embedding layer and the very last layer) --
    mirrors the teacher's "Stochastic Block" idea: a different, randomly
    chosen layer supervises the student's fixed repa_layer_idx each step,
    instead of always matching the exact same fixed pair.
    """
    torch, F = _torch()
    from ..models.text_to_music_diffusion import denormalize_mel

    try:
        muq = _load_muq(device)
        vocos = _load_vocos(device)
    except Exception as e:
        print(f"[repa] frozen encoders unavailable ({e}); skipping REPA loss this step.", flush=True)
        return None

    batch, seq_len, n_mels = x1_normalized.shape
    with torch.no_grad():
        raw_log_mel = denormalize_mel(x1_normalized, config).transpose(1, 2)  # (B, n_mels, T) -- Vocos convention
        waveform = vocos.decode(raw_log_mel)  # (B, samples) at config.sample_rate (24kHz, matches MuQ's requirement)
        muq_out = muq(waveform, output_hidden_states=True)
        hidden_states = muq_out.hidden_states  # tuple of (B, F_muq, D_muq)
        lo, hi = layer_pool
        lo = max(0, min(lo, len(hidden_states) - 1))
        hi = max(lo, min(hi, len(hidden_states) - 1))
        chosen_layer = random.randint(lo, hi)
        target = hidden_states[chosen_layer]  # (B, F_muq, D_muq)
        target_resampled = F.interpolate(
            target.transpose(1, 2), size=seq_len, mode="linear", align_corners=False
        ).transpose(1, 2)  # (B, T, D_muq)
    return target_resampled.detach()


def repa_loss(student_projected, target_features):
    """Negative mean cosine similarity between student's projected hidden state
    and the frozen target features, averaged over batch and time -- standard
    REPA loss form (maximize alignment, target is always stop-gradient)."""
    torch, F = _torch()
    if student_projected is None or target_features is None:
        return None
    cos = F.cosine_similarity(student_projected, target_features, dim=-1)  # (B, T)
    return -cos.mean()
