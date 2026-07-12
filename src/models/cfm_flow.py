import torch
import torch.nn.functional as F
from .text_to_music_diffusion import MusicDiffusionConfig

def cfm_loss(model, clean_mel: torch.Tensor, backing_mel: torch.Tensor, texts: list[str], config: MusicDiffusionConfig) -> torch.Tensor:
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
        timestep=t
    )
    
    # Compute MSE loss
    return F.mse_loss(predicted_velocity, target_velocity)


@torch.no_grad()
def sample_cfm(model, texts: list[str], frames: int, config: MusicDiffusionConfig, device, steps: int = 32, seed: int | None = None, backing_mel: torch.Tensor | None = None) -> torch.Tensor:
    """Samples a mel spectrogram from Gaussian noise using Euler ODE integration."""
    model.eval()
    
    # Set seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        
    batch_size = len(texts)
    
    # 1. Start with Gaussian noise x0 at t = 0
    xt = torch.randn((batch_size, frames, config.n_mels), device=device)
    
    # Use backing mel as condition, or fallback to zeros if not provided
    if backing_mel is not None:
        cond = backing_mel.transpose(1, 2) if backing_mel.shape[1] == config.n_mels else backing_mel
        # Pad or slice cond to match frames dimension of x0
        if cond.shape[1] > frames:
            cond = cond[:, :frames]
        elif cond.shape[1] < frames:
            cond = F.pad(cond, (0, 0, 0, frames - cond.shape[1]))
    else:
        cond = torch.zeros_like(xt)
    
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
            timestep=t
        )
        
        # Euler update step
        xt = xt + v_pred * dt
        
    # Return the generated mel spectrogram (batch, n_mels, frames)
    # Match the output shape expected by the vocoders: (batch_size, n_mels, seq_len)
    return xt.transpose(1, 2)
