import torch
import torch.nn.functional as F
from .text_to_music_diffusion import MusicDiffusionConfig, reconstruct_full_mix


def build_mismatched_texts(texts: list[str]) -> tuple[list[str], list[bool]]:
    """Rotate each non-empty lyric to a different lyric in the same batch.

    Empty-text comparison only proves that the model reacts to *some* text. It
    does not prove that "em yeu anh" produces different phonemes from "mua roi".
    This helper supplies content-negative prompts while marking samples for
    which a genuinely different non-empty prompt exists.
    """
    normalized = [str(text).strip() for text in texts]
    mismatched = ["" for _ in normalized]
    valid = [False for _ in normalized]
    for index, text in enumerate(normalized):
        if not text:
            continue
        folded = text.casefold()
        for offset in range(1, len(normalized)):
            candidate = normalized[(index + offset) % len(normalized)]
            if candidate and candidate.casefold() != folded:
                mismatched[index] = candidate
                valid[index] = True
                break
    return mismatched, valid


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
    vocal_mel: torch.Tensor,
    backing_mel: torch.Tensor,
    style_anchor: torch.Tensor,
    texts: list[str],
    config: MusicDiffusionConfig,
    *,
    lambda_vocal: float = 1.0,
    condition_dropout_prob: float = 0.1,
    style_dropout_prob: float | None = None,
    text_dropout_prob: float | None = None,
    text_contrastive_weight: float = 0.0,
    text_contrastive_margin: float = 0.03,
    text_contrastive_prob: float = 0.5,
    text_sensitivity_weight: float = 0.0,
    text_sensitivity_target: float = 0.20,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Computes the Conditional Flow Matching (CFM) velocity prediction loss.

    Primary target is the full song (vocal + accompaniment, see
    reconstruct_full_mix's docstring) -- matches DiffRhythm2's own scope and
    this project's, not an isolated a cappella vocal track. An auxiliary
    vocal-only prediction loss ("Mixed Pro", MicroDiT.vocal_proj_out) keeps the
    model explicitly tracking the harder vocal component instead of only
    learning the easier, louder joint mix; set lambda_vocal=0 to disable it.

    text_contrastive_weight/text_sensitivity_weight (both 0.0 by default, i.e.
    disabled) add a lyric-content-specific supervision signal on top of the
    main CFM loss: the main loss can be minimized while the model reacts only
    to "some text present" rather than to *which* phonemes were requested.
    build_mismatched_texts swaps each lyric for a different one in the same
    batch (same mel target); the contrastive term penalizes the model for not
    predicting a *worse* velocity for the wrong lyric than for the correct one,
    and the sensitivity term penalizes the two predictions for being too
    similar in absolute terms. style_dropout_prob/text_dropout_prob let CFG
    dropout rates differ per condition (real usage rarely supplies a style
    reference but almost always supplies lyrics) -- both fall back to
    condition_dropout_prob when left as None.

    Returns (total_loss, loss_gt, loss_vocal_aux) so callers can log the
    components separately (loss_vocal_aux is None when lambda_vocal <= 0).
    """
    device = vocal_mel.device
    batch_size = vocal_mel.shape[0]

    x1 = reconstruct_full_mix(vocal_mel, backing_mel, config)

    # 1. Sample t uniformly in [0, 1]
    t = torch.rand(batch_size, device=device)
    t_unsqueezed = t.view(-1, 1, 1) # Alignment for mel channels/frames

    # 2. Sample Gaussian noise x0
    x0 = torch.randn_like(x1)

    # 3. Compute linear interpolation xt
    xt = (1.0 - t_unsqueezed) * x0 + t_unsqueezed * x1

    # 4. Target velocity field vt = x1 - x0
    target_velocity = x1 - x0

    # 5. Classifier-free condition dropout teaches the model all inference modes:
    # real reference conditions, missing style, and an empty lyric prompt.
    normalized_style = style_anchor
    model_texts = list(texts)
    default_dropout = max(0.0, min(1.0, float(condition_dropout_prob)))
    style_dropout = default_dropout if style_dropout_prob is None else max(0.0, min(1.0, float(style_dropout_prob)))
    text_dropout = default_dropout if text_dropout_prob is None else max(0.0, min(1.0, float(text_dropout_prob)))
    if style_dropout > 0.0 or text_dropout > 0.0:
        # Most user generation has no reference MuQ anchor, while text is
        # always supplied. Train the zero-style path substantially more often
        # without also erasing the Vietnamese lyric at the same high rate.
        style_drop = torch.rand(batch_size, device=device) < style_dropout
        text_drop = torch.rand(batch_size, device=device) < text_dropout
        normalized_style = normalized_style.masked_fill(style_drop[:, None], 0.0)
        text_drop_flags = text_drop.detach().cpu().tolist()
        model_texts = ["" if text_drop_flags[index] else text for index, text in enumerate(model_texts)]

    # 6. Predict velocity field using MicroDiT (no cond passed)
    want_vocal_aux = lambda_vocal > 0.0
    if want_vocal_aux:
        predicted_velocity, vocal_aux = model(
            x=xt, texts=model_texts, timestep=t, style_prompt=normalized_style, return_vocal_aux=True,
        )
    else:
        predicted_velocity = model(x=xt, texts=model_texts, timestep=t, style_prompt=normalized_style)

    # Vocal-active frames carry the consonants/formants needed for intelligible
    # words, while long silent spans otherwise dominate an unweighted mean.
    frame_energy = x1.mean(dim=-1)
    activity_threshold = torch.quantile(frame_energy.detach(), 0.55, dim=1, keepdim=True)
    activity = torch.sigmoid((frame_energy - activity_threshold) * 2.0)
    frame_weights = (1.0 + 2.0 * activity).unsqueeze(-1)
    frame_weights = frame_weights / frame_weights.mean(dim=(1, 2), keepdim=True).clamp_min(1e-6)

    # Keep all loss arithmetic in FP32 even when the denoiser forward runs under
    # autocast FP16 (train_model/run_distillation_training always enable AMP on
    # CUDA). Squaring two FP16 velocity predictions can overflow past FP16's
    # ~65504 max (256**2 alone is already inf) and turn a recoverable large
    # residual into inf/inf -> NaN, silently corrupting the checkpoint from
    # then on.
    predicted_velocity_fp32 = predicted_velocity.float()
    target_velocity_fp32 = target_velocity.float()
    frame_weights_fp32 = frame_weights.float()
    velocity_loss = ((predicted_velocity_fp32 - target_velocity_fp32).square() * frame_weights_fp32).mean()

    # Reconstruct x1 from the predicted velocity and explicitly preserve its
    # time/frequency contours. These inexpensive auxiliary terms sharpen vocal
    # onsets and formant movement without changing the CFM sampling equation.
    x1_fp32 = x1.float()
    predicted_clean = xt.float() + (1.0 - t_unsqueezed.float()) * predicted_velocity_fp32
    reconstruction_loss = ((predicted_clean - x1_fp32).abs() * frame_weights_fp32).mean()
    time_delta_loss = F.l1_loss(torch.diff(predicted_clean, dim=1), torch.diff(x1_fp32, dim=1))
    frequency_delta_loss = F.l1_loss(torch.diff(predicted_clean, dim=2), torch.diff(x1_fp32, dim=2))
    loss_gt = velocity_loss + 0.15 * reconstruction_loss + 0.05 * (time_delta_loss + frequency_delta_loss)

    loss_vocal_aux = None
    total_loss = loss_gt
    if want_vocal_aux:
        vocal_mel_fp32 = vocal_mel.float()
        vocal_target_velocity = vocal_mel_fp32 - x0.float()
        vocal_frame_energy = vocal_mel_fp32.mean(dim=-1)
        vocal_activity_threshold = torch.quantile(vocal_frame_energy.detach(), 0.55, dim=1, keepdim=True)
        vocal_activity = torch.sigmoid((vocal_frame_energy - vocal_activity_threshold) * 2.0)
        vocal_frame_weights = (1.0 + 2.0 * vocal_activity).unsqueeze(-1)
        vocal_frame_weights = vocal_frame_weights / vocal_frame_weights.mean(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        loss_vocal_aux = ((vocal_aux.float() - vocal_target_velocity).square() * vocal_frame_weights).mean()
        total_loss = total_loss + lambda_vocal * loss_vocal_aux

    # Conditional flow matching can minimize its marginal audio loss while
    # reacting only to "text present" rather than to the requested phonemes.
    # Compare each correct lyric against a different lyric from the same batch.
    # Text dropout already trains the empty classifier-free branch; this extra
    # forward is reserved for the missing content-specific supervision.
    contrastive_weight = max(0.0, float(text_contrastive_weight))
    sensitivity_weight = max(0.0, float(text_sensitivity_weight))
    contrastive_probability = max(0.0, min(1.0, float(text_contrastive_prob)))
    mismatched_texts, content_mask_flags = build_mismatched_texts(model_texts)
    content_mask = torch.tensor(content_mask_flags, dtype=torch.bool, device=device)
    if (
        (contrastive_weight > 0.0 or sensitivity_weight > 0.0)
        and bool(content_mask.any())
        and torch.rand((), device=device) < contrastive_probability
    ):
        mismatched_velocity = model(
            x=xt, texts=mismatched_texts, timestep=t, style_prompt=normalized_style,
        )
        mismatched_velocity_fp32 = mismatched_velocity.float()
        matched_error = (
            (predicted_velocity_fp32 - target_velocity_fp32).square() * frame_weights_fp32
        ).mean(dim=(1, 2))[content_mask]
        mismatched_error = (
            (mismatched_velocity_fp32 - target_velocity_fp32).square() * frame_weights_fp32
        ).mean(dim=(1, 2))[content_mask]
        contrastive_loss = F.relu(
            max(0.0, float(text_contrastive_margin)) + matched_error - mismatched_error
        ).mean()
        total_loss = total_loss + contrastive_weight * contrastive_loss

        # Error ranking alone has a weak gradient when two different lyrics
        # produce the same velocity. This response floor measures lyric A
        # versus lyric B, not lyric versus empty, so a generic "text on"
        # signal can no longer satisfy the gate.
        response_rms = (
            (predicted_velocity_fp32 - mismatched_velocity_fp32).square().mean(dim=(1, 2))
            # When the model ignores lyrics, matched and mismatched outputs can
            # be exactly equal. sqrt'(0) is singular and would otherwise
            # produce non-finite gradients even though the forward loss is finite.
            .clamp_min(1e-12).sqrt()[content_mask]
        )
        response_scale = 0.5 * (
            predicted_velocity_fp32.square().mean(dim=(1, 2)).clamp_min(1e-12).sqrt()
            + target_velocity_fp32.square().mean(dim=(1, 2)).clamp_min(1e-12).sqrt()
        ).detach().clamp_min(1e-6)[content_mask]
        relative_response = response_rms / response_scale
        response_shortfall = F.relu(max(0.0, float(text_sensitivity_target)) - relative_response)
        # A plain squared hinge becomes nearly inert just below the target. A
        # one-sided Huber penalty retains a useful gradient near the boundary
        # while staying bounded and smooth.
        huber_beta = 0.05
        sensitivity_loss = torch.where(
            response_shortfall < huber_beta,
            0.5 * response_shortfall.square() / huber_beta,
            response_shortfall - 0.5 * huber_beta,
        ).mean()
        total_loss = total_loss + sensitivity_weight * sensitivity_loss

    return total_loss, loss_gt, loss_vocal_aux


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
