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


def vocal_structure_loss(
    predicted_vocal: torch.Tensor,
    target_vocal: torch.Tensor,
    config: MusicDiffusionConfig,
) -> torch.Tensor:
    """Match differentiable vocal energy, onset and spectral-shape contours."""
    scale = max(1e-4, float(config.mel_std))
    clip = float(config.mel_clip)
    predicted_log_mel = predicted_vocal.float().clamp(-clip, clip) * scale + float(config.mel_mean)
    target_log_mel = target_vocal.float().clamp(-clip, clip) * scale + float(config.mel_mean)

    predicted_energy = torch.logsumexp(predicted_log_mel, dim=-1)
    target_energy = torch.logsumexp(target_log_mel, dim=-1)
    energy_loss = F.smooth_l1_loss(predicted_energy, target_energy, beta=0.5)
    onset_loss = F.smooth_l1_loss(
        torch.diff(predicted_energy, dim=1),
        torch.diff(target_energy, dim=1),
        beta=0.25,
    )
    # Normalizing over mel bins removes overall loudness and concentrates this
    # term on formant/harmonic shape needed for voiced Vietnamese syllables.
    spectral_shape_loss = F.smooth_l1_loss(
        F.log_softmax(predicted_log_mel, dim=-1),
        F.log_softmax(target_log_mel, dim=-1),
        beta=0.25,
    )
    return energy_loss + 0.25 * onset_loss + 0.25 * spectral_shape_loss


def joint_stem_channel_weights(
    n_mels: int,
    lambda_vocal: float,
    *,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """Weight vocal channels explicitly in the joint-stem CFM target.

    Backing energy is usually denser than a vocal stem, so an unweighted mean
    lets the denoiser improve accompaniment while leaving the sparse vocal
    channels near the normalized mean (silence). ``lambda_vocal`` now controls
    both the auxiliary vocal objective and this direct target weighting. The
    default value of ``1`` preserves the previous loss exactly.
    """
    weights = torch.ones(2 * int(n_mels), device=device, dtype=dtype)
    weights[int(n_mels):] = max(1.0, float(lambda_vocal))
    return weights

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
    native_ctc_weight: float = 0.0,
    native_ctc_teacher_weight: float = 0.0,
    native_frame_text_weight: float = 0.0,
    native_frame_text_teacher_weight: float = 0.0,
    frame_text_segments: list[list[dict]] | None = None,
    frame_text_crop_starts_seconds: list[float] | None = None,
    frame_text_crop_ends_seconds: list[float] | None = None,
    native_vocal_prior_weight: float = 0.0,
    vocal_structure_weight: float = 0.0,
    native_prosody_weight: float = 0.0,
    return_details: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Compute full-mix CFM plus vocal and lyric-content auxiliary losses."""
    device = vocal_mel.device
    batch_size = vocal_mel.shape[0]
    joint_stems = bool(getattr(model, "joint_stem_generation", False))
    real_full_mix = reconstruct_full_mix(vocal_mel, backing_mel, config)
    # Joint-stem checkpoints keep accompaniment and vocal in separate channels
    # throughout diffusion. This prevents the easier backing target from
    # masking the sparse phoneme signal while still using one shared DiT.
    x1 = (
        torch.cat((backing_mel, vocal_mel), dim=-1)
        if joint_stems
        else real_full_mix
    )
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
    want_native_ctc = (
        (native_ctc_weight > 0.0 or native_ctc_teacher_weight > 0.0)
        and bool(getattr(model, "native_generation", False))
    )
    want_native_frame_text = (
        (
            native_frame_text_weight > 0.0
            or native_frame_text_teacher_weight > 0.0
        )
        and bool(getattr(model, "native_generation", False))
    )
    want_native_vocal_prior = (
        (native_vocal_prior_weight > 0.0 or native_frame_text_weight > 0.0)
        and bool(getattr(model, "native_generation", False))
        and getattr(model, "native_vocal_prior", None) is not None
    )
    want_native_prosody = (
        native_prosody_weight > 0.0
        and bool(getattr(model, "native_generation", False))
        and getattr(model, "native_prosody", None) is not None
    )
    want_vocal_aux = (
        lambda_vocal > 0.0 or want_native_ctc or want_native_frame_text
    ) and not joint_stems
    native_vocal_prior = None
    native_prosody = None
    model_kwargs = {
        "x": xt,
        "texts": model_texts,
        "timestep": t,
        "style_prompt": normalized_style,
    }
    # Keep the compact call compatible with test doubles and legacy adapters
    # that predate the optional native auxiliary heads.
    if want_vocal_aux:
        model_kwargs["return_vocal_aux"] = True
    if want_native_vocal_prior:
        model_kwargs["return_native_vocal_prior"] = True
    if want_native_prosody:
        model_kwargs["return_native_prosody"] = True
    result = model(**model_kwargs)
    if isinstance(result, tuple):
        cursor = 0
        predicted_velocity = result[cursor]
        cursor += 1
        vocal_aux = result[cursor] if want_vocal_aux else None
        cursor += int(want_vocal_aux)
        native_vocal_prior = result[cursor] if want_native_vocal_prior else None
        cursor += int(want_native_vocal_prior)
        native_prosody = result[cursor] if want_native_prosody else None
    else:
        predicted_velocity = result
        vocal_aux = None

    # Vocal-active frames carry the consonants/formants needed for intelligible
    # words, while long silent spans otherwise dominate an unweighted mean.
    frame_energy = x1.mean(dim=-1)
    activity_threshold = torch.quantile(frame_energy.detach(), 0.55, dim=1, keepdim=True)
    activity = torch.sigmoid((frame_energy - activity_threshold) * 2.0)
    frame_weights = (1.0 + 2.0 * activity).unsqueeze(-1)
    frame_weights = frame_weights / frame_weights.mean(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    # Keep all loss arithmetic in FP32 even when the denoiser forward uses
    # autocast FP16. Squaring two FP16 velocity predictions can overflow above
    # 255 and turn a recoverable large response into inf/inf -> NaN.
    predicted_velocity_fp32 = predicted_velocity.float()
    target_velocity_fp32 = target_velocity.float()
    frame_weights_fp32 = frame_weights.float()
    channel_weights = None
    if joint_stems:
        channel_weights = joint_stem_channel_weights(
            int(config.n_mels),
            lambda_vocal,
            device=device,
            dtype=predicted_velocity_fp32.dtype,
        ).view(1, 1, -1)
    velocity_loss = (
        (predicted_velocity_fp32 - target_velocity_fp32).square()
        * frame_weights_fp32
        * (channel_weights if channel_weights is not None else 1.0)
    ).mean()

    # Reconstruct x1 from the predicted velocity and explicitly preserve its
    # time/frequency contours. These inexpensive auxiliary terms sharpen vocal
    # onsets and formant movement without changing the CFM sampling equation.
    predicted_clean = xt.float() + (1.0 - t_unsqueezed.float()) * predicted_velocity_fp32
    x1_fp32 = x1.float()
    reconstruction_loss = (
        (predicted_clean - x1_fp32).abs() * frame_weights_fp32
        * (channel_weights if channel_weights is not None else 1.0)
    ).mean()
    time_delta_loss = F.l1_loss(
        torch.diff(predicted_clean, dim=1),
        torch.diff(x1_fp32, dim=1),
    )
    frequency_delta_loss = F.l1_loss(
        torch.diff(predicted_clean, dim=2),
        torch.diff(x1_fp32, dim=2),
    )
    loss_gt = velocity_loss + 0.15 * reconstruction_loss + 0.05 * (time_delta_loss + frequency_delta_loss)
    total_loss = loss_gt

    loss_vocal_aux = None
    if joint_stems:
        mel_count = int(config.n_mels)
        vocal_predicted_velocity = predicted_velocity_fp32[..., mel_count:]
        vocal_target_velocity = target_velocity_fp32[..., mel_count:]
        vocal_frame_energy = vocal_mel.float().mean(dim=-1)
        vocal_activity_threshold = torch.quantile(
            vocal_frame_energy.detach(), 0.55, dim=1, keepdim=True
        )
        vocal_activity = torch.sigmoid(
            (vocal_frame_energy - vocal_activity_threshold) * 2.0
        )
        vocal_frame_weights = (1.0 + 2.0 * vocal_activity).unsqueeze(-1)
        vocal_frame_weights = vocal_frame_weights / vocal_frame_weights.mean(
            dim=(1, 2), keepdim=True
        ).clamp_min(1e-6)
        loss_vocal_aux = (
            (vocal_predicted_velocity - vocal_target_velocity).square()
            * vocal_frame_weights
        ).mean()
        total_loss = total_loss + max(0.0, float(lambda_vocal)) * loss_vocal_aux
    elif want_vocal_aux:
        vocal_target_velocity = vocal_mel.float() - x0.float()
        vocal_frame_energy = vocal_mel.float().mean(dim=-1)
        vocal_activity_threshold = torch.quantile(
            vocal_frame_energy.detach(), 0.55, dim=1, keepdim=True
        )
        vocal_activity = torch.sigmoid(
            (vocal_frame_energy - vocal_activity_threshold) * 2.0
        )
        vocal_frame_weights = (1.0 + 2.0 * vocal_activity).unsqueeze(-1)
        vocal_frame_weights = vocal_frame_weights / vocal_frame_weights.mean(
            dim=(1, 2), keepdim=True
        ).clamp_min(1e-6)
        loss_vocal_aux = (
            (vocal_aux.float() - vocal_target_velocity).square()
            * vocal_frame_weights
        ).mean()
        total_loss = total_loss + max(0.0, float(lambda_vocal)) * loss_vocal_aux

    native_vocal_prior_loss = None
    if native_vocal_prior is not None:
        prior_reconstruction = F.smooth_l1_loss(
            native_vocal_prior.float(),
            vocal_mel.float(),
            beta=0.25,
        )
        prior_time_delta = F.smooth_l1_loss(
            torch.diff(native_vocal_prior.float(), dim=1),
            torch.diff(vocal_mel.float(), dim=1),
            beta=0.1,
        )
        prior_structure = vocal_structure_loss(
            native_vocal_prior,
            vocal_mel,
            config,
        )
        native_vocal_prior_loss = (
            prior_reconstruction + 0.20 * prior_time_delta + 0.25 * prior_structure
        )
        total_loss = total_loss + (
            max(0.0, float(native_vocal_prior_weight)) * native_vocal_prior_loss
        )

    vocal_structure = None
    structure_weight = max(0.0, float(vocal_structure_weight))
    if structure_weight > 0.0:
        predicted_vocal_clean = (
            predicted_clean[..., int(config.n_mels):]
            if joint_stems
            else predicted_clean
        )
        vocal_structure = vocal_structure_loss(
            predicted_vocal_clean,
            vocal_mel,
            config,
        )
        total_loss = total_loss + structure_weight * vocal_structure

    native_prosody_loss = None
    if native_prosody is not None:
        from .native_text import ctc_guided_duration_targets

        # Derive detached acoustic targets from the real vocal mel. The mel
        # centroid is a stable pitch proxy at this resolution; energy above a
        # per-crop quantile supplies a soft voicing target.
        real_log_mel = vocal_mel.float().clamp(-float(config.mel_clip), float(config.mel_clip))
        mel_weights = real_log_mel.softmax(dim=-1)
        mel_bins = torch.linspace(0.0, 1.0, int(config.n_mels), device=device).view(1, 1, -1)
        target_pitch = (mel_weights * mel_bins).sum(dim=-1).detach()
        target_energy_raw = torch.logsumexp(real_log_mel, dim=-1)
        threshold = torch.quantile(target_energy_raw.detach(), 0.55, dim=1, keepdim=True)
        target_voicing = torch.sigmoid((target_energy_raw.detach() - threshold) * 2.0)
        target_energy = torch.tanh(
            (target_energy_raw - target_energy_raw.detach().mean(dim=1, keepdim=True))
            / target_energy_raw.detach().std(dim=1, keepdim=True).clamp_min(0.1)
        ).detach()
        with torch.no_grad():
            duration_target, duration_mask = ctc_guided_duration_targets(
                model.audio_text_logits(vocal_mel.float()),
                model_texts,
                token_width=native_prosody["duration_proportions"].shape[-1],
            )
        predicted_durations = native_prosody["duration_proportions"].float()
        duration_terms = []
        for index in range(batch_size):
            if bool(duration_mask[index].any()):
                duration_terms.append(
                    F.smooth_l1_loss(
                        predicted_durations[index][duration_mask[index]],
                        duration_target[index][duration_mask[index]],
                        beta=0.05,
                    )
                )
        duration_loss = torch.stack(duration_terms).mean() if duration_terms else predicted_durations.sum() * 0.0
        prosody_loss = (
            duration_loss
            + F.smooth_l1_loss(native_prosody["pitch"].float(), target_pitch, beta=0.05)
            + F.binary_cross_entropy_with_logits(
                native_prosody["voicing_logits"].float(), target_voicing
            )
            + 0.5 * F.smooth_l1_loss(native_prosody["energy"].float(), target_energy, beta=0.1)
        )
        native_prosody_loss = prosody_loss
        total_loss = total_loss + max(0.0, float(native_prosody_weight)) * native_prosody_loss

    native_ctc_pred = None
    native_ctc_teacher = None
    native_frame_text_pred = None
    native_frame_text_teacher = None
    native_frame_text_prior = None
    if want_native_ctc or want_native_frame_text:
        from .native_text import (
            build_timestamped_frame_text_targets,
            native_ctc_loss,
            native_frame_text_loss,
        )

        # The recognizer learns Vietnamese spelling from real separated vocals.
        # It never receives lyric embeddings, so it cannot solve this objective
        # by copying the prompt.
        vocal_teacher_logits = model.audio_text_logits(vocal_mel.float())
        exact_teacher_targets = None
        exact_model_targets = None
        if (
            want_native_frame_text
            and frame_text_segments is not None
            and frame_text_crop_starts_seconds is not None
            and frame_text_crop_ends_seconds is not None
            and any(frame_text_segments)
        ):
            exact_teacher_targets = build_timestamped_frame_text_targets(
                frame_text_segments,
                crop_starts_seconds=frame_text_crop_starts_seconds,
                crop_ends_seconds=frame_text_crop_ends_seconds,
                time_steps=vocal_teacher_logits.shape[1],
                device=vocal_teacher_logits.device,
            )
            exact_model_targets = exact_teacher_targets.clone()
            for index, model_text in enumerate(model_texts):
                if not str(model_text).strip():
                    exact_model_targets[index].fill_(-100)

        # Real vocal and real full-mix passes ground the recognizer before it
        # scores the denoiser's predicted full mix.
        if want_native_ctc:
            mix_teacher_logits = model.audio_text_logits(
                real_full_mix.float().detach()
            )
            native_ctc_teacher = 0.5 * (
                native_ctc_loss(vocal_teacher_logits, texts)
                + native_ctc_loss(mix_teacher_logits, texts)
            )
        if native_ctc_weight > 0.0 or native_frame_text_weight > 0.0:
            # Joint-stem inference renders this vocal half explicitly before it
            # is combined with backing. Legacy full-mix checkpoints still score
            # their single reconstructed output.
            predicted_ctc_mel = (
                predicted_clean[..., int(config.n_mels):]
                if joint_stems
                else predicted_clean
            )
            predicted_logits = model.audio_text_logits(predicted_ctc_mel)
            # Preserve classifier-free training: samples whose lyric was
            # deliberately dropped must stay unconditional, not be forced to
            # reconstruct the hidden original sentence through this loss.
            if native_ctc_weight > 0.0:
                native_ctc_pred = native_ctc_loss(
                    predicted_logits, model_texts
                )
            if native_frame_text_weight > 0.0:
                native_frame_text_pred = native_frame_text_loss(
                    predicted_logits,
                    model_texts,
                    targets=exact_model_targets,
                )
        if want_native_frame_text:
            # Unlike CTC, every approximate acoustic frame receives an ordered
            # grapheme target. Real-vocal supervision anchors the recognizer;
            # scoring the predicted vocal and native prior then sends a direct
            # phoneme-specific gradient into both generation paths.
            native_frame_text_teacher = native_frame_text_loss(
                vocal_teacher_logits,
                texts,
                targets=exact_teacher_targets,
            )
            if native_vocal_prior is not None:
                native_frame_text_prior = native_frame_text_loss(
                    model.audio_text_logits(native_vocal_prior.float()),
                    model_texts,
                    targets=exact_model_targets,
                )
        if native_ctc_teacher is not None:
            total_loss = total_loss + (
                max(0.0, float(native_ctc_teacher_weight))
                * native_ctc_teacher
            )
        if native_ctc_pred is not None:
            total_loss = total_loss + (
                max(0.0, float(native_ctc_weight)) * native_ctc_pred
            )
        if native_frame_text_teacher is not None:
            total_loss = total_loss + (
                max(0.0, float(native_frame_text_teacher_weight))
                * native_frame_text_teacher
            )
        if native_frame_text_pred is not None:
            total_loss = total_loss + (
                max(0.0, float(native_frame_text_weight))
                * native_frame_text_pred
            )
        if native_frame_text_prior is not None:
            total_loss = total_loss + (
                0.5
                * max(0.0, float(native_frame_text_weight))
                * native_frame_text_prior
            )

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
    native_vocal_prior_contrastive = None
    if (
        (contrastive_weight > 0.0 or sensitivity_weight > 0.0)
        and bool(content_mask.any())
        and torch.rand((), device=device) < contrastive_probability
    ):
        mismatched_result = model(
            x=xt,
            texts=mismatched_texts,
            timestep=t,
            style_prompt=normalized_style,
            # The native prior is the shortest trainable path from ordered
            # Vietnamese graphemes to the vocal stem. Return it for the wrong
            # lyric as well so a generic average vocal cannot satisfy the
            # matched reconstruction loss.
            **(
                {"return_native_vocal_prior": True}
                if want_native_vocal_prior else {}
            ),
        )
        if isinstance(mismatched_result, tuple):
            mismatched_velocity = mismatched_result[0]
            mismatched_native_vocal_prior = (
                mismatched_result[1] if want_native_vocal_prior else None
            )
        else:
            mismatched_velocity = mismatched_result
            mismatched_native_vocal_prior = None
        mismatched_velocity_fp32 = mismatched_velocity.float()
        if joint_stems:
            mel_count = int(config.n_mels)
            matched_content = predicted_velocity_fp32[..., mel_count:]
            mismatched_content = mismatched_velocity_fp32[..., mel_count:]
            target_content = target_velocity_fp32[..., mel_count:]
            content_weights = vocal_frame_weights.float()
        else:
            matched_content = predicted_velocity_fp32
            mismatched_content = mismatched_velocity_fp32
            target_content = target_velocity_fp32
            content_weights = frame_weights_fp32
        matched_error = (
            (matched_content - target_content).square()
            * content_weights
        ).mean(dim=(1, 2))[content_mask]
        mismatched_error = (
            (mismatched_content - target_content).square()
            * content_weights
        ).mean(dim=(1, 2))[content_mask]
        contrastive_loss = F.relu(
            max(0.0, float(text_contrastive_margin))
            + matched_error
            - mismatched_error
        ).mean()
        total_loss = total_loss + contrastive_weight * contrastive_loss

        if (
            native_vocal_prior is not None
            and mismatched_native_vocal_prior is not None
        ):
            matched_prior_error = (
                (native_vocal_prior.float() - vocal_mel.float()).square()
                * vocal_frame_weights.float()
            ).mean(dim=(1, 2))[content_mask]
            mismatched_prior_error = (
                (
                    mismatched_native_vocal_prior.float()
                    - vocal_mel.float()
                ).square()
                * vocal_frame_weights.float()
            ).mean(dim=(1, 2))[content_mask]
            native_vocal_prior_contrastive = F.relu(
                max(0.0, float(text_contrastive_margin))
                + matched_prior_error
                - mismatched_prior_error
            ).mean()
            total_loss = total_loss + (
                contrastive_weight * native_vocal_prior_contrastive
            )

        # Error ranking alone has a weak gradient when two different lyrics
        # produce the same velocity. This response floor now measures lyric A
        # versus lyric B, not lyric versus empty, so a generic "text on" signal
        # can no longer satisfy the gate.
        response_rms = (
            (matched_content - mismatched_content)
            .square()
            .mean(dim=(1, 2))
            # When the model ignores lyrics, matched and mismatched outputs can
            # be exactly equal. sqrt'(0) is singular and previously produced
            # non-finite gradients even though the forward loss was finite.
            .clamp_min(1e-12)
            .sqrt()[content_mask]
        )
        response_scale = 0.5 * (
            matched_content.square().mean(dim=(1, 2)).clamp_min(1e-12).sqrt()
            + target_content.square().mean(dim=(1, 2)).clamp_min(1e-12).sqrt()
        ).detach().clamp_min(1e-6)[content_mask]
        relative_response = response_rms / response_scale
        response_shortfall = F.relu(
            max(0.0, float(text_sensitivity_target)) - relative_response
        )
        # A plain squared hinge became nearly inert just below the target
        # (0.168 vs 0.20 in a real run). A one-sided Huber penalty retains a
        # useful gradient near the boundary while staying bounded and smooth.
        huber_beta = 0.05
        sensitivity_loss = torch.where(
            response_shortfall < huber_beta,
            0.5 * response_shortfall.square() / huber_beta,
            response_shortfall - 0.5 * huber_beta,
        ).mean()
        total_loss = total_loss + sensitivity_weight * sensitivity_loss
    if return_details:
        return total_loss, loss_gt, loss_vocal_aux, {
            "native_ctc_pred": native_ctc_pred,
            "native_ctc_teacher": native_ctc_teacher,
            "native_frame_text_pred": native_frame_text_pred,
            "native_frame_text_teacher": native_frame_text_teacher,
            "native_frame_text_prior": native_frame_text_prior,
            "native_vocal_prior": native_vocal_prior_loss,
            "native_vocal_prior_contrastive": native_vocal_prior_contrastive,
            "vocal_structure": vocal_structure,
            "native_prosody": native_prosody_loss,
        }
    return total_loss, loss_gt, loss_vocal_aux


@torch.no_grad()
def sample_cfm(
    model,
    texts: list[str],
    frames: int,
    config: MusicDiffusionConfig,
    device,
    steps: int = 32,
    seed: int | None = None,
    style_prompt: torch.Tensor | None = None,
    guidance_scale: float = 1.0,
    initial_mel: torch.Tensor | None = None,
    pronunciation_prior_strength: float = 0.0,
    native_prior_start_strength: float = 0.0,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Sample a full mix or a ``(vocal, backing)`` pair from one denoiser."""
    model.eval()
    
    # Set seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        
        # Ensure numpy seed is aligned if needed
        import numpy as np
        np.random.seed(seed)
        
    batch_size = len(texts)
    
    # 1. Normally start with Gaussian noise x0 at t=0.  When a Vietnamese
    # pronunciation prior is supplied, start at an intermediate point on the
    # same CFM path.  Values near 1 preserve more intelligible TTS articulation;
    # lower values let the learned singing-vocal distribution refine more.
    joint_stems = bool(getattr(model, "joint_stem_generation", False))
    model_mels = config.n_mels * (2 if joint_stems else 1)
    noise = torch.randn((batch_size, frames, model_mels), device=device)
    prior_strength = max(0.0, min(1.0, float(pronunciation_prior_strength)))
    native_start = max(0.0, min(1.0, float(native_prior_start_strength)))
    if joint_stems and (initial_mel is not None or prior_strength > 0.0):
        raise ValueError("Joint-stem native generation does not accept a pronunciation prior")
    if initial_mel is None or prior_strength <= 0.0:
        xt = noise
        start_time = 0.0
    else:
        prior = torch.as_tensor(initial_mel, dtype=noise.dtype, device=device)
        if prior.dim() == 2:
            prior = prior.unsqueeze(0)
        if prior.shape == (batch_size, config.n_mels, frames):
            prior = prior.transpose(1, 2)
        expected_shape = (batch_size, frames, config.n_mels)
        if tuple(prior.shape) != expected_shape:
            raise ValueError(
                f"initial_mel must have shape {expected_shape} or "
                f"{(batch_size, config.n_mels, frames)}, got {tuple(prior.shape)}"
            )
        xt = (1.0 - prior_strength) * noise + prior_strength * prior
        start_time = prior_strength
    
    normalized_style = _prepare_style_condition(
        style_prompt,
        batch_size=batch_size,
        style_dim=int(getattr(model, "style_dim", 512)),
        device=device,
    )

    # A native joint checkpoint already contains a learned text-to-mel branch.
    # Starting only the vocal half from a small interpolation with that branch
    # keeps the CFM trajectory unchanged for backing while giving a weakly
    # trained denoiser a formant/energy scaffold. This is an internal model
    # prediction, never a pretrained TTS/pronunciation model. The default is
    # zero so legacy sampling remains exactly reproducible.
    if joint_stems and native_start > 0.0:
        probe_t = torch.zeros((batch_size,), device=device, dtype=torch.float32)
        probe = model(
            x=noise,
            texts=texts,
            timestep=probe_t,
            style_prompt=normalized_style,
            return_native_vocal_prior=True,
        )
        native_prior = probe[-1] if isinstance(probe, tuple) else None
        if native_prior is None:
            raise ValueError(
                "native_prior_start_strength requires a native vocal prior"
            )
        mel_count = int(config.n_mels)
        xt[..., mel_count:] = (
            (1.0 - native_start) * noise[..., mel_count:]
            + native_start * native_prior.float().to(noise.dtype)
        )
        start_time = native_start
    elif joint_stems:
        start_time = 0.0
    
    step_count = max(1, int(steps))
    dt = (1.0 - start_time) / step_count
    integration_steps = 0 if start_time >= 1.0 else step_count
    
    # 2. Euler integration loop from t = 0 to t = 1
    for step in range(integration_steps):
        t_val = start_time + step * dt
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
        
    if joint_stems:
        backing, vocal = xt.split(int(config.n_mels), dim=-1)
        return vocal.transpose(1, 2), backing.transpose(1, 2)
    # Match the output shape expected by the vocoders: (batch_size, n_mels, seq_len).
    return xt.transpose(1, 2)
