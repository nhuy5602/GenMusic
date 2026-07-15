import torch
import torch.nn.functional as F
from .text_to_music_diffusion import MusicDiffusionConfig


def _prepare_backing_condition(
    backing_mel: torch.Tensor | None,
    *,
    batch_size: int,
    frames: int,
    config: MusicDiffusionConfig,
    device,
) -> torch.Tensor:
    """Normalize backing mels to the model's (batch, frames, n_mels) layout."""
    if backing_mel is None:
        return torch.zeros((batch_size, frames, config.n_mels), device=device)

    backing = torch.as_tensor(backing_mel, dtype=torch.float32, device=device)
    if backing.dim() == 2:
        backing = backing.unsqueeze(0)
    if backing.dim() != 3:
        raise ValueError(f"backing_mel must have 2 or 3 dimensions, got {tuple(backing.shape)}")
    if backing.shape[-1] == config.n_mels:
        cond = backing
    elif backing.shape[1] == config.n_mels:
        cond = backing.transpose(1, 2)
    else:
        raise ValueError(
            f"backing_mel must contain an n_mels={config.n_mels} axis, got {tuple(backing.shape)}"
        )
    if cond.shape[0] == 1 and batch_size > 1:
        cond = cond.expand(batch_size, -1, -1)
    elif cond.shape[0] != batch_size:
        raise ValueError(f"backing_mel batch {cond.shape[0]} does not match text batch {batch_size}")
    if cond.shape[1] > frames:
        cond = cond[:, :frames]
    elif cond.shape[1] < frames:
        cond = F.pad(cond, (0, 0, 0, frames - cond.shape[1]))
    return cond


def _prepare_style_condition(style_prompt: torch.Tensor | None, *, batch_size: int, device) -> torch.Tensor | None:
    """Normalize a MuQ-MuLan anchor to one embedding vector per generated item."""
    if style_prompt is None:
        return None
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

def cfm_loss(model, clean_mel: torch.Tensor, backing_mel: torch.Tensor, style_anchor: torch.Tensor, texts: list[str], config: MusicDiffusionConfig) -> torch.Tensor:
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
    
    # 5. Cond is the backing track Mel spectrogram (acting as musical context)
    cond = backing_mel
    
    # 6. Predict velocity field using MicroDiT
    predicted_velocity = model(
        x=xt,
        cond=cond,
        texts=texts,
        timestep=t,
        style_prompt=style_anchor
    )
    
    # Compute MSE loss
    return F.mse_loss(predicted_velocity, target_velocity)


@torch.no_grad()
def sample_cfm(model, texts: list[str], frames: int, config: MusicDiffusionConfig, device, steps: int = 32, seed: int | None = None, backing_mel: torch.Tensor | None = None, style_prompt: torch.Tensor | None = None) -> torch.Tensor:
    """Sample a vocal mel, optionally using the same backing/style inputs as training."""
    model.eval()
    
    # Set seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        
    batch_size = len(texts)
    
    # 1. Start with Gaussian noise x0 at t = 0
    xt = torch.randn((batch_size, frames, config.n_mels), device=device)
    
    # Real preprocessed backing/style conditions should be supplied for models
    # trained on separated stems. The zero/None fallback remains for old smoke
    # checkpoints and deliberately unconditional calls.
    cond = _prepare_backing_condition(
        backing_mel, batch_size=batch_size, frames=frames, config=config, device=device
    )
    normalized_style = _prepare_style_condition(style_prompt, batch_size=batch_size, device=device)
    
    dt = 1.0 / steps
    
    # 2. Euler integration loop from t = 0 to t = 1
    for step in range(steps):
        t_val = step / steps
        t = torch.full((batch_size,), t_val, device=device, dtype=torch.float32)
        
        # Predict velocity field
        v_pred = model(
            x=xt,
            cond=cond,
            texts=texts,
            timestep=t,
            style_prompt=normalized_style
        )
        
        # Euler update step
        xt = xt + v_pred * dt
        
    # Return the generated mel spectrogram (batch, n_mels, frames)
    # Match the output shape expected by the vocoders: (batch_size, n_mels, seq_len)
    return xt.transpose(1, 2)
