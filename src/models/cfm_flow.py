import torch
import torch.nn.functional as F
from .text_to_music_diffusion import MusicDiffusionConfig


def _prepare_style_condition(
    style_prompt: torch.Tensor | None,
    *,
    batch_size: int,
    style_dim: int,
    device,
) -> torch.Tensor:
    """Normalize a MuQ-MuLan anchor to one embedding vector per generated item."""
    if style_prompt is None:
        # Training applies style dropout by replacing anchors with zero vectors.
        # Use the same representation when generation has no reference anchor.
        return torch.zeros((batch_size, style_dim), dtype=torch.float32, device=device)
    style = torch.as_tensor(style_prompt, dtype=torch.float32, device=device)
    if style.dim() == 1:
        style = style.unsqueeze(0)
    if style.dim() != 2:
        raise ValueError(f"style_prompt must have 1 or 2 dimensions, got {tuple(style.shape)}")
    if style.shape[0] == 1 and batch_size > 1:
        style = style.expand(batch_size, -1)
    elif style.shape[0] != batch_size:
        raise ValueError(f"style_prompt batch {style.shape[0]} does not match text batch {batch_size}")
    return style

def cfm_loss(
    model,
    clean_mel: torch.Tensor,
    style_anchor: torch.Tensor,
    texts: list[str],
    config: MusicDiffusionConfig,
    *,
    condition_dropout_prob: float = 0.1,
) -> torch.Tensor:
    """Computes the Conditional Flow Matching (CFM) velocity prediction loss."""
    device = clean_mel.device
    batch_size = clean_mel.shape[0]
    
    # 1. Sample t uniformly in [0, 1]
    t = torch.rand(batch_size, device=device)
    t_unsqueezed = t.view(-1, 1, 1) # Alignment for mel channels/frames
    
    # 2. Sample Gaussian noise x0
    x0 = torch.randn_like(clean_mel)
    x1 = clean_mel
    
    # 3. Compute linear interpolation xt
    xt = (1.0 - t_unsqueezed) * x0 + t_unsqueezed * x1
    
    # 4. Target velocity field vt = x1 - x0
    target_velocity = x1 - x0
    
    # 5. Classifier-free condition dropout teaches the model all inference modes:
    # real reference conditions, missing style, and an empty lyric prompt.
    normalized_style = style_anchor
    model_texts = list(texts)
    dropout = max(0.0, min(1.0, float(condition_dropout_prob)))
    if dropout > 0.0:
        style_drop = torch.rand(batch_size, device=device) < dropout
        text_drop = torch.rand(batch_size, device=device) < dropout
        normalized_style = normalized_style.masked_fill(style_drop[:, None], 0.0)
        text_drop_flags = text_drop.detach().cpu().tolist()
        model_texts = ["" if text_drop_flags[index] else text for index, text in enumerate(model_texts)]
    
    # 6. Predict velocity field using MicroDiT (no cond passed)
    predicted_velocity = model(
        x=xt,
        texts=model_texts,
        timestep=t,
        style_prompt=normalized_style
    )

    # Vocal-active frames carry the consonants/formants needed for intelligible
    # words, while long silent spans otherwise dominate an unweighted mean.
    frame_energy = clean_mel.mean(dim=-1)
    activity_threshold = torch.quantile(frame_energy.detach(), 0.55, dim=1, keepdim=True)
    activity = torch.sigmoid((frame_energy - activity_threshold) * 2.0)
    frame_weights = (1.0 + 2.0 * activity).unsqueeze(-1)
    frame_weights = frame_weights / frame_weights.mean(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    velocity_loss = ((predicted_velocity - target_velocity).square() * frame_weights).mean()

    # Reconstruct x1 from the predicted velocity and explicitly preserve its
    # time/frequency contours. These inexpensive auxiliary terms sharpen vocal
    # onsets and formant movement without changing the CFM sampling equation.
    predicted_clean = xt + (1.0 - t_unsqueezed) * predicted_velocity
    reconstruction_loss = ((predicted_clean - clean_mel).abs() * frame_weights).mean()
    time_delta_loss = F.l1_loss(torch.diff(predicted_clean, dim=1), torch.diff(clean_mel, dim=1))
    frequency_delta_loss = F.l1_loss(torch.diff(predicted_clean, dim=2), torch.diff(clean_mel, dim=2))
    return velocity_loss + 0.15 * reconstruction_loss + 0.05 * (time_delta_loss + frequency_delta_loss)


@torch.no_grad()
def sample_cfm(model, texts: list[str], frames: int, config: MusicDiffusionConfig, device, steps: int = 32, seed: int | None = None, style_prompt: torch.Tensor | None = None, guidance_scale: float = 1.0) -> torch.Tensor:
    """Sample a vocal mel, optionally using the style inputs from training."""
    model.eval()
    
    # Set seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        
        # Ensure numpy seed is aligned if needed
        import numpy as np
        np.random.seed(seed)
        
    batch_size = len(texts)
    
    # 1. Start with Gaussian noise x0 at t = 0
    xt = torch.randn((batch_size, frames, config.n_mels), device=device)
    
    normalized_style = _prepare_style_condition(
        style_prompt,
        batch_size=batch_size,
        style_dim=int(getattr(model, "style_dim", 512)),
        device=device,
    )
    
    dt = 1.0 / steps
    
    # 2. Euler integration loop from t = 0 to t = 1
    for step in range(steps):
        t_val = step / steps
        t = torch.full((batch_size,), t_val, device=device, dtype=torch.float32)
        
        # Predict velocity field (no cond passed)
        v_pred = model(
            x=xt,
            texts=texts,
            timestep=t,
            style_prompt=normalized_style
        )
        if guidance_scale != 1.0:
            unconditional = model(
                x=xt,
                texts=[""] * batch_size,
                timestep=t,
                style_prompt=normalized_style,
            )
            v_pred = unconditional + float(guidance_scale) * (v_pred - unconditional)
        
        # Euler update step
        xt = xt + v_pred * dt
        
    # Return the generated mel spectrogram (batch, n_mels, frames)
    # Match the output shape expected by the vocoders: (batch_size, n_mels, seq_len)
    return xt.transpose(1, 2)
