import torch
import torch.nn.functional as F

from .open_vocabulary_lyrics import (
    split_lyric_words,
    split_vietnamese_grapheme_units,
)
from .text_to_music_diffusion import MusicDiffusionConfig, reconstruct_full_mix


def build_mismatched_texts(texts: list[str]) -> tuple[list[str], list[bool]]:
    """Rotate each non-empty lyric to a different lyric in the same batch.

    Empty-text comparison only proves that the model reacts to *some* text. It
    does not prove that "em yeu anh" produces different phonemes from "mua roi".
    This helper supplies content-negative prompts while marking samples for
    which a genuinely different non-empty prompt exists.
    """
    mismatched, valid, _ = build_mismatched_text_conditioning(texts)
    return mismatched, valid


def build_mismatched_text_conditioning(
    texts: list[str],
) -> tuple[list[str], list[bool], list[int]]:
    """Return content-negative lyrics and their source batch indices."""
    normalized = [str(text).strip() for text in texts]
    mismatched = ["" for _ in normalized]
    valid = [False for _ in normalized]
    source_indices = [0 for _ in normalized]
    for index, text in enumerate(normalized):
        if not text:
            continue
        folded = text.casefold()
        for offset in range(1, len(normalized)):
            candidate = normalized[(index + offset) % len(normalized)]
            if candidate and candidate.casefold() != folded:
                mismatched[index] = candidate
                valid[index] = True
                source_indices[index] = (index + offset) % len(normalized)
                break
    return mismatched, valid, source_indices


def _call_model(
    model,
    *,
    x: torch.Tensor,
    texts: list[str],
    timestep: torch.Tensor,
    style_prompt: torch.Tensor,
    lyric_frame_ids: torch.Tensor | None,
    return_vocal_aux: bool = False,
):
    kwargs = {
        "x": x,
        "texts": texts,
        "timestep": timestep,
        "style_prompt": style_prompt,
    }
    if lyric_frame_ids is not None:
        kwargs["lyric_frame_ids"] = lyric_frame_ids
    if return_vocal_aux:
        kwargs["return_vocal_aux"] = True
    return model(**kwargs)


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


def sample_cfm_timesteps(
    batch_size: int,
    device,
    *,
    early_fraction: float = 0.0,
    early_max: float = 0.35,
) -> torch.Tensor:
    """Sample CFM times with optional early-transport emphasis and replay."""
    count = max(1, int(batch_size))
    timesteps = torch.rand(count, device=device)
    fraction = max(0.0, min(1.0, float(early_fraction)))
    maximum = max(1e-4, min(1.0, float(early_max)))
    if fraction <= 0.0:
        return timesteps
    early_mask = torch.rand(count, device=device) < fraction
    early_values = torch.rand(count, device=device) * maximum
    return torch.where(early_mask, early_values, timesteps)


def corrupt_seed_source_spans(
    source: torch.Tensor,
    lyric_frame_ids: torch.Tensor,
    *,
    probability: float = 0.0,
    word_fraction: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace lyric spans with target-free spans from another batch item.

    A retrieval seed usually contains the right word for training vocabulary,
    so a refiner can minimize its acoustic loss by copying the donor and never
    learning how text should repair a wrong/fuzzy pronunciation.  This bounded
    augmentation rolls *source seeds only* across the batch and replaces whole
    exact-word spans.  The paired clean target is never used to construct the
    corruption, keeping the inference contract target-free.

    Returns the corrupted source and a ``(batch, frames, 1)`` mask marking the
    replaced spans so a masked replay objective can still expose them.
    """
    if source.ndim != 3:
        raise ValueError(
            "source must have shape (batch, frames, channels), got "
            f"{tuple(source.shape)}"
        )
    frame_ids = torch.as_tensor(
        lyric_frame_ids,
        dtype=torch.long,
        device=source.device,
    )
    if frame_ids.shape != source.shape[:2]:
        raise ValueError(
            "lyric_frame_ids must match source batch/frames; got "
            f"{tuple(frame_ids.shape)} for {tuple(source.shape)}"
        )
    corruption_mask = torch.zeros(
        (*source.shape[:2], 1),
        dtype=source.dtype,
        device=source.device,
    )
    chance = max(0.0, min(1.0, float(probability)))
    fraction = max(0.0, min(1.0, float(word_fraction)))
    if chance <= 0.0 or fraction <= 0.0 or source.shape[0] < 2:
        return source, corruption_mask

    corrupted = source.clone()
    donor_source = torch.roll(source.detach(), shifts=1, dims=0)
    for batch_index in range(source.shape[0]):
        if float(torch.rand((), device=source.device)) >= chance:
            continue
        word_ids = torch.unique(frame_ids[batch_index])
        word_ids = word_ids[word_ids > 0]
        if word_ids.numel() == 0:
            continue
        selected_count = max(
            1,
            min(
                int(word_ids.numel()),
                int(torch.ceil(torch.tensor(word_ids.numel() * fraction)).item()),
            ),
        )
        selection = torch.randperm(
            int(word_ids.numel()),
            device=source.device,
        )[:selected_count]
        selected_ids = word_ids.index_select(0, selection)
        selected_frames = torch.isin(
            frame_ids[batch_index],
            selected_ids,
        )
        corrupted[batch_index, selected_frames] = donor_source[
            batch_index,
            selected_frames,
        ]
        corruption_mask[batch_index, selected_frames, 0] = 1.0
    return corrupted, corruption_mask


def lyric_semantic_alignment_objective(
    model,
    target_audio: torch.Tensor,
    texts: list[str],
    lyric_frame_ids: torch.Tensor,
    *,
    temperature: float = 0.08,
    include_words: set[str] | None = None,
) -> dict[str, torch.Tensor | int]:
    """Align compositional word embeddings with their exact latent spans.

    A changed lyric can produce a large velocity response while still yielding
    the wrong Vietnamese sentence. This objective supplies the missing
    direction: every exact word span is a positive pair between a
    corpus-independent grapheme embedding and a locally centred acoustic
    feature. Other words in the batch are negatives, with repeated words
    treated as additional positives instead of false negatives.

    The target is the clean V1 full-mix latent. Concatenating its locally
    centred value with a temporal derivative suppresses stationary backing
    energy and emphasizes the vocal changes available inside exact timestamp
    spans. No closed word classifier or pretrained TTS component is used.
    """
    required = (
        "open_vocabulary_lyric",
        "lyric_semantic_audio_projection",
        "lyric_semantic_word_projection",
    )
    missing = [name for name in required if not hasattr(model, name)]
    if missing:
        raise ValueError(
            "Lyric semantic alignment requires open-vocabulary model heads; "
            f"missing {missing}."
        )
    if target_audio.dim() != 3:
        raise ValueError(
            "target_audio must have shape (batch, frames, channels), got "
            f"{tuple(target_audio.shape)}"
        )
    batch_size, frame_count, _ = target_audio.shape
    if len(texts) != batch_size:
        raise ValueError(
            f"text batch {len(texts)} does not match audio batch {batch_size}"
        )
    frame_ids = torch.as_tensor(
        lyric_frame_ids,
        dtype=torch.long,
        device=target_audio.device,
    )
    if frame_ids.shape != (batch_size, frame_count):
        raise ValueError(
            "lyric_frame_ids must have shape "
            f"{(batch_size, frame_count)}, got {tuple(frame_ids.shape)}"
        )

    projection_parameter = next(
        model.lyric_semantic_audio_projection.parameters()
    )
    acoustic = target_audio.to(dtype=projection_parameter.dtype)
    centred = acoustic - acoustic.mean(dim=1, keepdim=True)
    delta = torch.diff(
        acoustic,
        dim=1,
        prepend=acoustic[:, :1],
    )
    acoustic_frames = model.lyric_semantic_audio_projection(
        torch.cat([centred, delta], dim=-1)
    )
    word_memory, _, _, _ = model.open_vocabulary_lyric(
        texts,
        frames=frame_count,
        frame_word_ids=frame_ids,
        device=target_audio.device,
    )

    acoustic_occurrences: list[torch.Tensor] = []
    word_occurrences: list[torch.Tensor] = []
    labels: list[str] = []
    # Consecutive silence/padding frames can have an exact zero delta.
    # sqrt'(0) is infinite; when target_audio is the denoiser's predicted
    # clean tensor that singularity propagates NaN gradients through every
    # optimizer step even though the forward InfoNCE loss stays finite.
    derivative_activity = (
        delta.float()
        .square()
        .mean(dim=-1)
        .clamp_min(1e-12)
        .sqrt()
    )
    for batch_index, text in enumerate(texts):
        words = split_lyric_words(text)
        for word_index, word in enumerate(words, start=1):
            span_mask = frame_ids[batch_index] == word_index
            if not bool(span_mask.any()):
                continue
            span_features = acoustic_frames[batch_index, span_mask]
            span_activity = derivative_activity[batch_index, span_mask]
            # Retain vowel bodies with a nonzero floor while giving onset and
            # transition frames more influence than steady accompaniment.
            weights = 0.25 + span_activity / span_activity.mean().clamp_min(1e-6)
            pooled_acoustic = (
                span_features * weights.to(span_features.dtype).unsqueeze(-1)
            ).sum(dim=0) / weights.sum().to(span_features.dtype).clamp_min(1e-6)
            acoustic_occurrences.append(pooled_acoustic)
            word_occurrences.append(word_memory[batch_index, word_index - 1])
            labels.append(word)

    if not acoustic_occurrences:
        zero = acoustic_frames.sum() * 0.0
        return {
            "loss": zero,
            "accuracy": zero.detach(),
            "positive_cosine": zero.detach(),
            "margin": zero.detach(),
            "occurrences": 0,
            "distinct_words": 0,
        }

    audio_features = F.normalize(
        torch.stack(acoustic_occurrences).float(),
        dim=-1,
    )
    word_features = F.normalize(
        model.lyric_semantic_word_projection(
            torch.stack(word_occurrences)
        ).float(),
        dim=-1,
    )
    positive_cosine_all = (audio_features * word_features).sum(dim=-1)
    positive_loss = (1.0 - positive_cosine_all).mean()

    count = len(labels)
    score_mask = torch.tensor(
        [
            include_words is None or label in include_words
            for label in labels
        ],
        dtype=torch.bool,
        device=target_audio.device,
    )
    if not bool(score_mask.any()):
        zero = audio_features.sum() * 0.0
        return {
            "loss": zero,
            "accuracy": zero.detach(),
            "positive_cosine": zero.detach(),
            "margin": zero.detach(),
            "occurrences": 0,
            "distinct_words": 0,
        }
    label_matches = torch.tensor(
        [
            [left == right for right in labels]
            for left in labels
        ],
        dtype=torch.bool,
        device=target_audio.device,
    )
    logits = audio_features @ word_features.transpose(0, 1)
    logits = logits / max(1e-4, float(temperature))
    # Multi-positive InfoNCE: repeated occurrences of the same normalized word
    # are positives, not accidental negatives.
    negative_log_likelihood_audio = (
        torch.logsumexp(logits, dim=1)
        - torch.logsumexp(logits.masked_fill(~label_matches, -torch.inf), dim=1)
    ).mean()
    negative_log_likelihood_word = (
        torch.logsumexp(logits, dim=0)
        - torch.logsumexp(logits.masked_fill(~label_matches, -torch.inf), dim=0)
    ).mean()
    contrastive_loss = 0.5 * (
        negative_log_likelihood_audio + negative_log_likelihood_word
    )
    loss = contrastive_loss + 0.25 * positive_loss

    predicted_indices = logits.argmax(dim=1)
    accuracy = label_matches[
        torch.arange(count, device=target_audio.device),
        predicted_indices,
    ].float()[score_mask].mean()
    negative_logits = logits.masked_fill(label_matches, -torch.inf)
    hardest_negative = negative_logits.max(dim=1).values
    positive_logits = logits.masked_fill(~label_matches, -torch.inf).max(dim=1).values
    has_negative = (~label_matches).any(dim=1)
    margin = torch.where(
        has_negative,
        positive_logits - hardest_negative,
        torch.zeros_like(positive_logits),
    )[score_mask].mean()
    scored_labels = [
        label
        for label in labels
        if include_words is None or label in include_words
    ]
    return {
        "loss": loss,
        "accuracy": accuracy.detach(),
        "positive_cosine": positive_cosine_all[score_mask].mean().detach(),
        "margin": margin.detach(),
        "occurrences": len(scored_labels),
        "distinct_words": len(set(scored_labels)),
    }


def _contextualize_phrase_sequence(
    model,
    values: torch.Tensor,
    lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized phrase and per-word states without padding leakage."""
    packed = torch.nn.utils.rnn.pack_padded_sequence(
        values,
        lengths.detach().cpu(),
        batch_first=True,
        enforce_sorted=False,
    )
    packed_context, hidden = model.lyric_phrase_sequence_context(packed)
    contextual, _ = torch.nn.utils.rnn.pad_packed_sequence(
        packed_context,
        batch_first=True,
        total_length=values.shape[1],
    )
    pooled = torch.cat([hidden[-2], hidden[-1]], dim=-1)
    phrase = F.normalize(
        model.lyric_phrase_output_norm(pooled).float(),
        dim=-1,
    )
    words = F.normalize(
        model.lyric_phrase_output_norm(contextual).float(),
        dim=-1,
    )
    return phrase, words


def lyric_phrase_semantic_alignment_objective(
    model,
    target_audio: torch.Tensor,
    texts: list[str],
    lyric_frame_ids: torch.Tensor,
    *,
    temperature: float = 0.07,
    minimum_words: int = 4,
    maximum_words: int = 8,
    hard_negative_margin: float = 0.08,
    include_words: set[str] | None = None,
) -> dict[str, torch.Tensor | int | list[str]]:
    """Align exact 4--8 word sung phrases with ordered open-vocabulary text.

    Isolated sung words are not a reliable semantic target: neighbouring
    co-articulation, melody, and timing carry much of their identity.  This
    objective first contextualizes the exact vocal-latent frames, pools them
    into timestamped words, and then applies one shared order encoder to the
    acoustic and compositional text word sequences.  Rotated-order and
    two-word contextual replacements make phrase identity impossible to solve
    with a bag-of-words shortcut while respecting the empirical fact that
    isolated sung words are not reliably recognizable. Repeated phrase labels
    are multi-positive pairs rather than false negatives.
    """
    required = (
        "open_vocabulary_lyric",
        "lyric_phrase_audio_projection",
        "lyric_phrase_audio_context",
        "lyric_phrase_text_projection",
        "lyric_phrase_sequence_context",
        "lyric_phrase_output_norm",
    )
    missing = [name for name in required if not hasattr(model, name)]
    if missing:
        raise ValueError(
            "Lyric phrase alignment requires phrase semantic heads; "
            f"missing {missing}."
        )
    if target_audio.ndim != 3:
        raise ValueError(
            "target_audio must have shape (batch,frames,channels), got "
            f"{tuple(target_audio.shape)}"
        )
    batch_size, frame_count, _ = target_audio.shape
    if len(texts) != batch_size:
        raise ValueError(
            f"text batch {len(texts)} does not match audio batch {batch_size}"
        )
    frame_ids = torch.as_tensor(
        lyric_frame_ids,
        dtype=torch.long,
        device=target_audio.device,
    )
    if frame_ids.shape != (batch_size, frame_count):
        raise ValueError(
            "lyric_frame_ids must have shape "
            f"{(batch_size, frame_count)}, got {tuple(frame_ids.shape)}"
        )
    minimum = max(2, int(minimum_words))
    maximum = max(minimum, int(maximum_words))

    projection_parameter = next(
        model.lyric_phrase_audio_projection.parameters()
    )
    acoustic = target_audio.to(dtype=projection_parameter.dtype)
    centred = acoustic - acoustic.mean(dim=1, keepdim=True)
    delta = torch.diff(acoustic, dim=1, prepend=acoustic[:, :1])
    projected_audio = model.lyric_phrase_audio_projection(
        torch.cat([centred, delta], dim=-1)
    )
    derivative_activity = (
        delta.float().square().mean(dim=-1).clamp_min(1e-12).sqrt()
    )
    word_memory, _, _, _ = model.open_vocabulary_lyric(
        texts,
        frames=frame_count,
        frame_word_ids=frame_ids,
        device=target_audio.device,
    )

    selections: list[tuple[int, list[int], list[str]]] = []
    frame_lengths: list[int] = []
    for batch_index, text in enumerate(texts):
        words = split_lyric_words(text)
        present = [
            word_index
            for word_index in range(1, len(words) + 1)
            if bool((frame_ids[batch_index] == word_index).any())
        ]
        # Keep one contiguous context window. A crop boundary may remove a
        # word completely; splitting at ID gaps prevents bridging that hole.
        runs: list[list[int]] = []
        for word_index in present:
            if not runs or word_index != runs[-1][-1] + 1:
                runs.append([word_index])
            else:
                runs[-1].append(word_index)
        runs = [run for run in runs if len(run) >= minimum]
        if not runs:
            continue
        selected = max(runs, key=lambda run: (len(run), -run[0]))
        if len(selected) > maximum:
            start = (len(selected) - maximum) // 2
            selected = selected[start : start + maximum]
        selected_words = [words[index - 1] for index in selected]
        selections.append((batch_index, selected, selected_words))
        active = torch.nonzero(
            torch.isin(
                frame_ids[batch_index],
                torch.tensor(selected, device=frame_ids.device),
            ),
            as_tuple=False,
        ).flatten()
        frame_lengths.append(int(active[-1]) + 1)

    if len(selections) < 2:
        zero = projected_audio.sum() * 0.0
        return {
            "loss": zero,
            "retrieval_top1_accuracy": zero.detach(),
            "retrieval_top5_accuracy": zero.detach(),
            "hard_negative_accuracy": zero.detach(),
            "novel_retrieval_top1_accuracy": zero.detach(),
            "novel_hard_negative_accuracy": zero.detach(),
            "positive_cosine": zero.detach(),
            "rotated_margin": zero.detach(),
            "replacement_margin": zero.detach(),
            "word_alignment_accuracy": zero.detach(),
            "word_alignment_margin": zero.detach(),
            "word_retrieval_accuracy": zero.detach(),
            "replacement_span_words": 0,
            "phrases": 0,
            "novel_phrases": 0,
            "phrase_labels": [],
        }

    selected_audio = torch.stack(
        [projected_audio[index] for index, _, _ in selections]
    )
    lengths_tensor = torch.tensor(
        frame_lengths,
        dtype=torch.long,
        device=target_audio.device,
    )
    packed_audio = torch.nn.utils.rnn.pack_padded_sequence(
        selected_audio,
        lengths_tensor.detach().cpu(),
        batch_first=True,
        enforce_sorted=False,
    )
    packed_context, _ = model.lyric_phrase_audio_context(packed_audio)
    contextual_audio, _ = torch.nn.utils.rnn.pad_packed_sequence(
        packed_context,
        batch_first=True,
        total_length=frame_count,
    )

    maximum_selected = max(len(selected) for _, selected, _ in selections)
    semantic_dim = contextual_audio.shape[-1]
    audio_words = contextual_audio.new_zeros(
        (len(selections), maximum_selected, semantic_dim)
    )
    text_words = contextual_audio.new_zeros(
        (len(selections), maximum_selected, semantic_dim)
    )
    word_mask = torch.zeros(
        (len(selections), maximum_selected),
        dtype=torch.bool,
        device=target_audio.device,
    )
    phrase_labels: list[str] = []
    phrase_word_labels: list[list[str]] = []
    for row, (batch_index, selected, selected_words) in enumerate(selections):
        for column, word_index in enumerate(selected):
            positions = torch.nonzero(
                frame_ids[batch_index] == word_index,
                as_tuple=False,
            ).flatten()
            span = contextual_audio[row, positions]
            activity = derivative_activity[batch_index, positions]
            weights = 0.25 + activity / activity.mean().clamp_min(1e-6)
            audio_words[row, column] = (
                span * weights.to(span.dtype).unsqueeze(-1)
            ).sum(dim=0) / weights.sum().to(span.dtype).clamp_min(1e-6)
            text_words[row, column] = model.lyric_phrase_text_projection(
                word_memory[batch_index, word_index - 1]
            )
            word_mask[row, column] = True
        phrase_labels.append(" ".join(selected_words))
        phrase_word_labels.append(selected_words)

    phrase_lengths = word_mask.sum(dim=1).long()
    audio_phrase, contextual_audio_words = _contextualize_phrase_sequence(
        model,
        audio_words,
        phrase_lengths,
    )
    text_phrase, contextual_text_words = _contextualize_phrase_sequence(
        model,
        text_words,
        phrase_lengths,
    )

    rotated_words = text_words.clone()
    replacement_words = text_words.clone()
    candidate_locations = [
        (row, column, phrase_word_labels[row][column])
        for row, length_value in enumerate(phrase_lengths.tolist())
        for column in range(length_value)
    ]
    replacement_span_sizes: list[int] = []
    for row, length_value in enumerate(phrase_lengths.tolist()):
        rotated_words[row, :length_value] = text_words[
            row, :length_value
        ].roll(-1, dims=0)
        novel_positions = [
            position
            for position, word in enumerate(phrase_word_labels[row])
            if include_words is not None and word in include_words
        ]
        anchor = novel_positions[0] if novel_positions else length_value // 2
        span_start = max(0, min(anchor, length_value - 2))
        replace_positions = list(range(span_start, span_start + 2))
        replacement_span_sizes.append(len(replace_positions))
        for position in replace_positions:
            target_label = phrase_word_labels[row][position]
            choices = [
                location
                for location in candidate_locations
                if location[0] != row and location[2] != target_label
            ]
            if choices:
                choice = choices[(row + position) % len(choices)]
                replacement_words[row, position] = text_words[
                    choice[0], choice[1]
                ]
            else:
                replacement_words[row, position] = text_words[
                    row, (position + 1) % length_value
                ]
    rotated_phrase, contextual_rotated_words = _contextualize_phrase_sequence(
        model,
        rotated_words,
        phrase_lengths,
    )
    replacement_phrase, contextual_replacement_words = (
        _contextualize_phrase_sequence(
            model,
            replacement_words,
            phrase_lengths,
        )
    )

    label_matches = torch.tensor(
        [
            [left == right for right in phrase_labels]
            for left in phrase_labels
        ],
        dtype=torch.bool,
        device=target_audio.device,
    )
    logits = audio_phrase @ text_phrase.transpose(0, 1)
    scaled_logits = logits / max(1e-4, float(temperature))
    audio_nll = (
        torch.logsumexp(scaled_logits, dim=1)
        - torch.logsumexp(
            scaled_logits.masked_fill(~label_matches, -torch.inf),
            dim=1,
        )
    ).mean()
    text_nll = (
        torch.logsumexp(scaled_logits, dim=0)
        - torch.logsumexp(
            scaled_logits.masked_fill(~label_matches, -torch.inf),
            dim=0,
        )
    ).mean()
    retrieval_loss = 0.5 * (audio_nll + text_nll)
    positive = (audio_phrase * text_phrase).sum(dim=-1)
    rotated_score = (audio_phrase * rotated_phrase).sum(dim=-1)
    replacement_score = (audio_phrase * replacement_phrase).sum(dim=-1)
    rotated_loss = F.relu(
        float(hard_negative_margin) - positive + rotated_score
    ).mean()
    replacement_loss = F.relu(
        float(hard_negative_margin) - positive + replacement_score
    ).mean()
    hard_loss = rotated_loss + 2.0 * replacement_loss

    word_positive = (contextual_audio_words * contextual_text_words).sum(
        dim=-1
    )
    word_rotated = (
        contextual_audio_words * contextual_rotated_words
    ).sum(dim=-1)
    word_replaced = (
        contextual_audio_words * contextual_replacement_words
    ).sum(dim=-1)
    word_negative = torch.maximum(word_rotated, word_replaced)
    word_margin_loss = F.relu(
        0.04 - word_positive + word_negative
    )[word_mask].mean()
    flat_audio_words = contextual_audio_words[word_mask]
    flat_text_words = contextual_text_words[word_mask]
    flat_word_labels = [
        word
        for words in phrase_word_labels
        for word in words
    ]
    word_matches = torch.tensor(
        [
            [left == right for right in flat_word_labels]
            for left in flat_word_labels
        ],
        dtype=torch.bool,
        device=target_audio.device,
    )
    word_logits = flat_audio_words @ flat_text_words.transpose(0, 1)
    scaled_word_logits = word_logits / max(1e-4, float(temperature))
    word_audio_nll = (
        torch.logsumexp(scaled_word_logits, dim=1)
        - torch.logsumexp(
            scaled_word_logits.masked_fill(~word_matches, -torch.inf),
            dim=1,
        )
    ).mean()
    word_text_nll = (
        torch.logsumexp(scaled_word_logits, dim=0)
        - torch.logsumexp(
            scaled_word_logits.masked_fill(~word_matches, -torch.inf),
            dim=0,
        )
    ).mean()
    word_retrieval_loss = 0.5 * (word_audio_nll + word_text_nll)
    word_ranking = word_logits.argmax(dim=-1)
    word_retrieval_accuracy = word_matches.gather(
        1,
        word_ranking.unsqueeze(1),
    ).float().mean()
    loss = (
        retrieval_loss
        + hard_loss
        + 0.35 * word_retrieval_loss
        + 0.5 * word_margin_loss
    )

    ranking = logits.argsort(dim=-1, descending=True)
    top1 = label_matches.gather(1, ranking[:, :1]).any(dim=1)
    top5 = label_matches.gather(
        1,
        ranking[:, : min(5, ranking.shape[1])],
    ).any(dim=1)
    rotated_ok = positive > rotated_score
    replacement_ok = positive > replacement_score
    novel_mask = torch.tensor(
        [
            include_words is not None
            and any(word in include_words for word in words)
            for words in phrase_word_labels
        ],
        dtype=torch.bool,
        device=target_audio.device,
    )
    score_mask = (
        novel_mask
        if include_words is not None
        else torch.ones_like(novel_mask)
    )
    scored = int(score_mask.sum())
    if scored:
        novel_top1 = top1[score_mask].float().mean()
        novel_hard = torch.cat(
            [rotated_ok[score_mask], replacement_ok[score_mask]]
        ).float().mean()
    else:
        novel_top1 = positive.new_zeros(())
        novel_hard = positive.new_zeros(())
    return {
        "loss": loss,
        "retrieval_top1_accuracy": top1.float().mean().detach(),
        "retrieval_top5_accuracy": top5.float().mean().detach(),
        "hard_negative_accuracy": torch.cat(
            [rotated_ok, replacement_ok]
        ).float().mean().detach(),
        "novel_retrieval_top1_accuracy": novel_top1.detach(),
        "novel_hard_negative_accuracy": novel_hard.detach(),
        "positive_cosine": positive.mean().detach(),
        "rotated_margin": (positive - rotated_score).mean().detach(),
        "replacement_margin": (positive - replacement_score).mean().detach(),
        "word_alignment_accuracy": (
            (word_positive > word_negative)[word_mask].float().mean().detach()
        ),
        "word_alignment_margin": (
            (word_positive - word_negative)[word_mask].mean().detach()
        ),
        "word_retrieval_accuracy": word_retrieval_accuracy.detach(),
        "replacement_span_words": (
            sum(replacement_span_sizes) / max(1, len(replacement_span_sizes))
        ),
        "phrases": len(selections),
        "novel_phrases": scored,
        "phrase_labels": phrase_labels,
    }


def lyric_unit_alignment_objective(
    model,
    target_audio: torch.Tensor,
    texts: list[str],
    lyric_frame_ids: torch.Tensor,
    *,
    temperature: float = 0.08,
    include_words: set[str] | None = None,
) -> dict[str, torch.Tensor | int]:
    """Align reusable grapheme/phonetic units inside exact word spans.

    Word-level contrastive learning can memorize complete seen words. This
    stricter auxiliary partitions each exact word span monotonically across
    Vietnamese grapheme units (including common digraph onsets/codas), so the
    same unit target is shared by many words and can compose a truly unseen
    word at inference.
    """
    required = (
        "open_vocabulary_lyric",
        "lyric_semantic_audio_projection",
        "lyric_semantic_unit_projection",
    )
    missing = [name for name in required if not hasattr(model, name)]
    if missing:
        raise ValueError(
            "Lyric unit alignment requires open-vocabulary model heads; "
            f"missing {missing}."
        )
    if target_audio.dim() != 3:
        raise ValueError(
            "target_audio must have shape (batch, frames, channels), got "
            f"{tuple(target_audio.shape)}"
        )
    batch_size, frame_count, _ = target_audio.shape
    if len(texts) != batch_size:
        raise ValueError(
            f"text batch {len(texts)} does not match audio batch {batch_size}"
        )
    frame_ids = torch.as_tensor(
        lyric_frame_ids,
        dtype=torch.long,
        device=target_audio.device,
    )
    if frame_ids.shape != (batch_size, frame_count):
        raise ValueError(
            "lyric_frame_ids must have shape "
            f"{(batch_size, frame_count)}, got {tuple(frame_ids.shape)}"
        )

    projection_parameter = next(
        model.lyric_semantic_audio_projection.parameters()
    )
    acoustic = target_audio.to(dtype=projection_parameter.dtype)
    centred = acoustic - acoustic.mean(dim=1, keepdim=True)
    delta = torch.diff(
        acoustic,
        dim=1,
        prepend=acoustic[:, :1],
    )
    acoustic_frames = model.lyric_semantic_audio_projection(
        torch.cat([centred, delta], dim=-1)
    )
    derivative_activity = (
        delta.float()
        .square()
        .mean(dim=-1)
        .clamp_min(1e-12)
        .sqrt()
    )

    words_by_text = [split_lyric_words(text) for text in texts]
    units_by_text: list[list[str]] = []
    unit_layouts: list[list[tuple[str, list[str]]]] = []
    for words in words_by_text:
        layout = [
            (word, split_vietnamese_grapheme_units(word))
            for word in words
        ]
        unit_layouts.append(layout)
        units_by_text.append(
            [unit for _word, units in layout for unit in units]
        )
    unit_memory, _ = model.open_vocabulary_lyric.encode_units(
        units_by_text,
        device=target_audio.device,
    )

    acoustic_occurrences: list[torch.Tensor] = []
    unit_occurrences: list[torch.Tensor] = []
    unit_labels: list[str] = []
    parent_words: list[str] = []
    for batch_index, layout in enumerate(unit_layouts):
        memory_offset = 0
        for word_index, (word, units) in enumerate(layout, start=1):
            word_positions = torch.nonzero(
                frame_ids[batch_index] == word_index,
                as_tuple=False,
            ).flatten()
            if word_positions.numel() == 0:
                memory_offset += len(units)
                continue
            # Allocate in monotonic order. When a very short word has fewer
            # latent frames than spelling units, adjacent units share a frame
            # instead of being dropped from supervision.
            boundaries = torch.linspace(
                0,
                int(word_positions.numel()),
                len(units) + 1,
                device=word_positions.device,
            ).round().long()
            boundaries[0] = 0
            boundaries[-1] = int(word_positions.numel())
            for unit_index, unit in enumerate(units):
                start = min(
                    int(word_positions.numel()) - 1,
                    int(boundaries[unit_index]),
                )
                end = max(
                    start + 1,
                    min(
                        int(word_positions.numel()),
                        int(boundaries[unit_index + 1]),
                    ),
                )
                positions = word_positions[start:end]
                span_features = acoustic_frames[batch_index, positions]
                span_activity = derivative_activity[batch_index, positions]
                weights = (
                    0.25
                    + span_activity
                    / span_activity.mean().clamp_min(1e-6)
                )
                pooled_acoustic = (
                    span_features
                    * weights.to(span_features.dtype).unsqueeze(-1)
                ).sum(dim=0) / weights.sum().to(
                    span_features.dtype
                ).clamp_min(1e-6)
                acoustic_occurrences.append(pooled_acoustic)
                unit_occurrences.append(
                    unit_memory[
                        batch_index,
                        memory_offset + unit_index,
                    ]
                )
                unit_labels.append(unit)
                parent_words.append(word)
            memory_offset += len(units)

    if not acoustic_occurrences:
        zero = acoustic_frames.sum() * 0.0
        return {
            "loss": zero,
            "accuracy": zero.detach(),
            "positive_cosine": zero.detach(),
            "margin": zero.detach(),
            "occurrences": 0,
            "distinct_units": 0,
        }

    audio_features = F.normalize(
        torch.stack(acoustic_occurrences).float(),
        dim=-1,
    )
    unit_features = F.normalize(
        model.lyric_semantic_unit_projection(
            torch.stack(unit_occurrences)
        ).float(),
        dim=-1,
    )
    labels = unit_labels
    label_matches = torch.tensor(
        [
            [left == right for right in labels]
            for left in labels
        ],
        dtype=torch.bool,
        device=target_audio.device,
    )
    score_mask = torch.tensor(
        [
            include_words is None or parent in include_words
            for parent in parent_words
        ],
        dtype=torch.bool,
        device=target_audio.device,
    )
    if not bool(score_mask.any()):
        zero = audio_features.sum() * 0.0
        return {
            "loss": zero,
            "accuracy": zero.detach(),
            "positive_cosine": zero.detach(),
            "margin": zero.detach(),
            "occurrences": 0,
            "distinct_units": 0,
        }

    logits = (
        audio_features @ unit_features.transpose(0, 1)
    ) / max(1e-4, float(temperature))
    audio_nll = (
        torch.logsumexp(logits, dim=1)
        - torch.logsumexp(
            logits.masked_fill(~label_matches, -torch.inf),
            dim=1,
        )
    ).mean()
    unit_nll = (
        torch.logsumexp(logits, dim=0)
        - torch.logsumexp(
            logits.masked_fill(~label_matches, -torch.inf),
            dim=0,
        )
    ).mean()
    positive_cosine = (
        audio_features * unit_features
    ).sum(dim=-1)
    loss = (
        0.5 * (audio_nll + unit_nll)
        + 0.25 * (1.0 - positive_cosine).mean()
    )
    predicted = logits.argmax(dim=1)
    accuracy = label_matches[
        torch.arange(len(labels), device=target_audio.device),
        predicted,
    ].float()[score_mask].mean()
    negative_logits = logits.masked_fill(label_matches, -torch.inf)
    positive_logits = logits.masked_fill(
        ~label_matches,
        -torch.inf,
    )
    has_negative = (~label_matches).any(dim=1)
    margin = torch.where(
        has_negative,
        positive_logits.max(dim=1).values
        - negative_logits.max(dim=1).values,
        torch.zeros_like(positive_cosine),
    )[score_mask].mean()
    scored_units = [
        unit
        for unit, parent in zip(unit_labels, parent_words, strict=True)
        if include_words is None or parent in include_words
    ]
    return {
        "loss": loss,
        "accuracy": accuracy.detach(),
        "positive_cosine": positive_cosine[score_mask].mean().detach(),
        "margin": margin.detach(),
        "occurrences": len(scored_units),
        "distinct_units": len(set(scored_units)),
    }


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
    lyric_frame_ids: torch.Tensor | None = None,
    lyric_semantic_weight: float = 0.0,
    lyric_denoised_semantic_weight: float = 0.0,
    lyric_phrase_semantic_weight: float = 0.0,
    lyric_phrase_denoised_semantic_weight: float = 0.0,
    lyric_semantic_temperature: float = 0.08,
    lyric_unit_semantic_weight: float = 0.0,
    lyric_unit_denoised_semantic_weight: float = 0.0,
    self_rollout_consistency_weight: float = 0.0,
    self_rollout_consistency_probability: float = 0.0,
    self_rollout_step_size: float = 0.125,
    self_rollout_solver_steps: int = 0,
    early_timestep_fraction: float = 0.0,
    early_timestep_max: float = 0.35,
    source_mel: torch.Tensor | None = None,
    source_noise_std: float = 0.0,
    refinement_mask: torch.Tensor | None = None,
    seed_full_frame_rewrite_probability: float = 0.0,
    seed_span_corruption_probability: float = 0.0,
    seed_span_corruption_fraction: float = 0.25,
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

    # In latent_mode, vocal_mel already IS the precomputed full-mix target (see
    # MusicDiffusionConfig.latent_mode's docstring) -- reconstruct_full_mix's
    # linear-magnitude-mel-sum formula only makes sense for actual log-mel
    # channels, not arbitrary learned latent channels.
    x1 = vocal_mel if config.latent_mode else reconstruct_full_mix(vocal_mel, backing_mel, config)

    # 1. Sample t uniformly in [0, 1]
    t = sample_cfm_timesteps(
        batch_size,
        device,
        early_fraction=early_timestep_fraction,
        early_max=early_timestep_max,
    )
    t_unsqueezed = t.view(-1, 1, 1) # Alignment for mel channels/frames

    # 2. Choose the source distribution x0. The established text-to-audio
    # path starts from Gaussian noise. A bounded retrieval refiner can instead
    # start from a target-free V80/unit-retrieval latent and learn only the
    # missing pronunciation plus boundary smoothing. This avoids asking a
    # small held-out pilot to relearn global music transport from scratch.
    if source_mel is None:
        x0 = torch.randn_like(x1)
    else:
        x0 = torch.as_tensor(
            source_mel,
            dtype=x1.dtype,
            device=device,
        )
        if x0.shape != x1.shape:
            raise ValueError(
                "source_mel must match the target shape "
                f"{tuple(x1.shape)}, got {tuple(x0.shape)}"
            )
        noise_std = max(0.0, float(source_noise_std))
        if noise_std > 0.0:
            x0 = x0 + noise_std * torch.randn_like(x0)

    resolved_refinement_mask = None
    if refinement_mask is not None:
        if source_mel is None:
            raise ValueError(
                "refinement_mask requires a target-free source_mel."
            )
        resolved_refinement_mask = torch.as_tensor(
            refinement_mask,
            dtype=x1.dtype,
            device=device,
        )
        if resolved_refinement_mask.ndim == 2:
            resolved_refinement_mask = resolved_refinement_mask.unsqueeze(-1)
        elif (
            resolved_refinement_mask.ndim == 3
            and resolved_refinement_mask.shape[1] == 1
            and resolved_refinement_mask.shape[2] == x1.shape[1]
        ):
            resolved_refinement_mask = resolved_refinement_mask.transpose(1, 2)
        if (
            resolved_refinement_mask.ndim != 3
            or resolved_refinement_mask.shape[:2] != x1.shape[:2]
            or resolved_refinement_mask.shape[2] not in (1, x1.shape[2])
        ):
            raise ValueError(
                "refinement_mask must have shape (batch, frames), "
                "(batch, frames, 1/channels), or (batch, 1, frames); "
                f"got {tuple(resolved_refinement_mask.shape)} for "
                f"target {tuple(x1.shape)}."
            )
        resolved_refinement_mask = resolved_refinement_mask.clamp(0.0, 1.0)

    corruption_mask = None
    if max(0.0, float(seed_span_corruption_probability)) > 0.0:
        if source_mel is None:
            raise ValueError(
                "seed span corruption requires a target-free source_mel."
            )
        if lyric_frame_ids is None:
            raise ValueError(
                "seed span corruption requires exact lyric_frame_ids."
            )
        x0, corruption_mask = corrupt_seed_source_spans(
            x0,
            lyric_frame_ids,
            probability=seed_span_corruption_probability,
            word_fraction=seed_span_corruption_fraction,
        )

    if resolved_refinement_mask is not None:
        rewrite_probability = max(
            0.0,
            min(1.0, float(seed_full_frame_rewrite_probability)),
        )
        if rewrite_probability > 0.0:
            rewrite_samples = (
                torch.rand(batch_size, device=device) < rewrite_probability
            )
            resolved_refinement_mask = torch.where(
                rewrite_samples[:, None, None],
                torch.ones_like(resolved_refinement_mask),
                resolved_refinement_mask,
            )
        if corruption_mask is not None:
            resolved_refinement_mask = torch.maximum(
                resolved_refinement_mask,
                corruption_mask,
            )

    path_target = x1
    if resolved_refinement_mask is not None:
        # Only the uncertain/fuzzy word spans travel toward the paired target.
        # Exact donor interiors remain the source latent, making preservation
        # part of the flow path rather than a soft downstream preference.
        path_target = (
            x0
            + resolved_refinement_mask * (x1 - x0)
        )

    # 3. Compute linear interpolation xt
    xt = (1.0 - t_unsqueezed) * x0 + t_unsqueezed * path_target

    # 4. Target velocity field vt = target - source
    target_velocity = path_target - x0

    # 5. Classifier-free condition dropout teaches the model all inference modes:
    # real reference conditions, missing style, and an empty lyric prompt.
    normalized_style = style_anchor
    model_texts = list(texts)
    model_frame_ids = (
        torch.as_tensor(lyric_frame_ids, dtype=torch.long, device=device).clone()
        if lyric_frame_ids is not None
        else None
    )
    default_dropout = max(0.0, min(1.0, float(condition_dropout_prob)))
    style_dropout = default_dropout if style_dropout_prob is None else max(0.0, min(1.0, float(style_dropout_prob)))
    text_dropout = default_dropout if text_dropout_prob is None else max(0.0, min(1.0, float(text_dropout_prob)))
    text_drop = torch.zeros(batch_size, dtype=torch.bool, device=device)
    if style_dropout > 0.0 or text_dropout > 0.0:
        # Most user generation has no reference MuQ anchor, while text is
        # always supplied. Train the zero-style path substantially more often
        # without also erasing the Vietnamese lyric at the same high rate.
        style_drop = torch.rand(batch_size, device=device) < style_dropout
        text_drop = torch.rand(batch_size, device=device) < text_dropout
        normalized_style = normalized_style.masked_fill(style_drop[:, None], 0.0)
        text_drop_flags = text_drop.detach().cpu().tolist()
        model_texts = ["" if text_drop_flags[index] else text for index, text in enumerate(model_texts)]
        if model_frame_ids is not None:
            model_frame_ids[text_drop] = 0

    # 6. Predict velocity field using MicroDiT (no cond passed)
    want_vocal_aux = lambda_vocal > 0.0
    if want_vocal_aux:
        predicted_velocity, vocal_aux = _call_model(
            model,
            x=xt,
            texts=model_texts,
            timestep=t,
            style_prompt=normalized_style,
            lyric_frame_ids=model_frame_ids,
            return_vocal_aux=True,
        )
    else:
        predicted_velocity = _call_model(
            model,
            x=xt,
            texts=model_texts,
            timestep=t,
            style_prompt=normalized_style,
            lyric_frame_ids=model_frame_ids,
        )

    # Vocal-active frames carry the consonants/formants needed for intelligible
    # words, while long silent spans otherwise dominate an unweighted mean.
    frame_energy = x1.float().mean(dim=-1)
    activity_threshold = torch.quantile(
        frame_energy.detach(),
        0.55,
        dim=1,
        keepdim=True,
    )
    activity = torch.sigmoid((frame_energy - activity_threshold) * 2.0)
    frame_weights = (1.0 + 2.0 * activity).unsqueeze(-1)
    if resolved_refinement_mask is not None:
        # Give fuzzy interiors most of the capacity while retaining a smaller
        # explicit zero-velocity loss outside the editable region.
        frame_weights = frame_weights * (
            0.25 + 3.75 * resolved_refinement_mask.float()
        )
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
    x1_fp32 = path_target.float()
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
        if resolved_refinement_mask is not None:
            vocal_target_velocity = (
                vocal_target_velocity
                * resolved_refinement_mask.float()
            )
        vocal_frame_energy = vocal_mel_fp32.mean(dim=-1)
        vocal_activity_threshold = torch.quantile(
            vocal_frame_energy.detach(),
            0.55,
            dim=1,
            keepdim=True,
        )
        vocal_activity = torch.sigmoid((vocal_frame_energy - vocal_activity_threshold) * 2.0)
        vocal_frame_weights = (1.0 + 2.0 * vocal_activity).unsqueeze(-1)
        vocal_frame_weights = vocal_frame_weights / vocal_frame_weights.mean(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        loss_vocal_aux = ((vocal_aux.float() - vocal_target_velocity).square() * vocal_frame_weights).mean()
        total_loss = total_loss + lambda_vocal * loss_vocal_aux

    semantic_weight = max(0.0, float(lyric_semantic_weight))
    if semantic_weight > 0.0:
        if lyric_frame_ids is None:
            raise ValueError(
                "lyric_semantic_weight requires exact lyric_frame_ids."
            )
        semantic = lyric_semantic_alignment_objective(
            model,
            x1,
            texts,
            lyric_frame_ids,
            temperature=lyric_semantic_temperature,
        )
        total_loss = total_loss + semantic_weight * semantic["loss"]

    # The clean-target semantic objective above teaches the compositional
    # lyric encoder and its acoustic projection, but it does not directly
    # constrain the velocity prediction. Couple that learned semantic space
    # to the denoiser by asking its reconstructed-clean estimate to retrieve
    # the exact word on the same timestamp span. Exclude CFG text-dropout
    # samples: forcing an unconditional prediction to recover a hidden lyric
    # would corrupt the classifier-free branch.
    denoised_semantic_weight = max(
        0.0,
        float(lyric_denoised_semantic_weight),
    )
    if denoised_semantic_weight > 0.0:
        if lyric_frame_ids is None:
            raise ValueError(
                "lyric_denoised_semantic_weight requires exact "
                "lyric_frame_ids."
            )
        conditioned_indices = torch.nonzero(
            ~text_drop,
            as_tuple=False,
        ).flatten()
        if conditioned_indices.numel() > 0:
            conditioned_texts = [
                texts[index]
                for index in conditioned_indices.detach().cpu().tolist()
            ]
            denoised_semantic = lyric_semantic_alignment_objective(
                model,
                predicted_clean.index_select(0, conditioned_indices),
                conditioned_texts,
                torch.as_tensor(
                    lyric_frame_ids,
                    dtype=torch.long,
                    device=device,
                ).index_select(0, conditioned_indices),
                temperature=lyric_semantic_temperature,
            )
            total_loss = (
                total_loss
                + denoised_semantic_weight
                * denoised_semantic["loss"]
            )

    # Isolated sung words are an unreliable acoustic target: the held-out
    # context oracle only became consistently recognizable on short ordered
    # phrases.  Preserve the gated phrase encoder on clean targets and, more
    # importantly, couple that same phrase space to the denoiser's clean
    # estimate.  This is the bridge from semantic-only pretraining into the
    # velocity field used by actual generation.
    phrase_semantic_weight = max(
        0.0,
        float(lyric_phrase_semantic_weight),
    )
    if phrase_semantic_weight > 0.0:
        if lyric_frame_ids is None:
            raise ValueError(
                "lyric_phrase_semantic_weight requires exact "
                "lyric_frame_ids."
            )
        clean_phrase_semantic = lyric_phrase_semantic_alignment_objective(
            model,
            x1,
            texts,
            lyric_frame_ids,
            temperature=lyric_semantic_temperature,
        )
        total_loss = (
            total_loss
            + phrase_semantic_weight * clean_phrase_semantic["loss"]
        )

    phrase_denoised_weight = max(
        0.0,
        float(lyric_phrase_denoised_semantic_weight),
    )
    if phrase_denoised_weight > 0.0:
        if lyric_frame_ids is None:
            raise ValueError(
                "lyric_phrase_denoised_semantic_weight requires exact "
                "lyric_frame_ids."
            )
        conditioned_indices = torch.nonzero(
            ~text_drop,
            as_tuple=False,
        ).flatten()
        if conditioned_indices.numel() > 0:
            conditioned_texts = [
                texts[index]
                for index in conditioned_indices.detach().cpu().tolist()
            ]
            denoised_phrase_semantic = (
                lyric_phrase_semantic_alignment_objective(
                    model,
                    predicted_clean.index_select(0, conditioned_indices),
                    conditioned_texts,
                    torch.as_tensor(
                        lyric_frame_ids,
                        dtype=torch.long,
                        device=device,
                    ).index_select(0, conditioned_indices),
                    temperature=lyric_semantic_temperature,
                )
            )
            total_loss = (
                total_loss
                + phrase_denoised_weight
                * denoised_phrase_semantic["loss"]
            )

    unit_semantic_weight = max(
        0.0,
        float(lyric_unit_semantic_weight),
    )
    if unit_semantic_weight > 0.0:
        if lyric_frame_ids is None:
            raise ValueError(
                "lyric_unit_semantic_weight requires exact "
                "lyric_frame_ids."
            )
        unit_semantic = lyric_unit_alignment_objective(
            model,
            x1,
            texts,
            lyric_frame_ids,
            temperature=lyric_semantic_temperature,
        )
        total_loss = (
            total_loss
            + unit_semantic_weight * unit_semantic["loss"]
        )

    # Clean-target unit alignment only teaches the acoustic/unit projection;
    # it does not require the diffusion velocity field to preserve reusable
    # phonetic units. Apply the same exact-frame retrieval objective to the
    # denoiser's predicted-clean estimate, excluding CFG text-dropout samples
    # whose lyrics are intentionally hidden.
    denoised_unit_weight = max(
        0.0,
        float(lyric_unit_denoised_semantic_weight),
    )
    if denoised_unit_weight > 0.0:
        if lyric_frame_ids is None:
            raise ValueError(
                "lyric_unit_denoised_semantic_weight requires exact "
                "lyric_frame_ids."
            )
        conditioned_indices = torch.nonzero(
            ~text_drop,
            as_tuple=False,
        ).flatten()
        if conditioned_indices.numel() > 0:
            conditioned_texts = [
                texts[index]
                for index in conditioned_indices.detach().cpu().tolist()
            ]
            denoised_units = lyric_unit_alignment_objective(
                model,
                predicted_clean.index_select(0, conditioned_indices),
                conditioned_texts,
                torch.as_tensor(
                    lyric_frame_ids,
                    dtype=torch.long,
                    device=device,
                ).index_select(0, conditioned_indices),
                temperature=lyric_semantic_temperature,
            )
            total_loss = (
                total_loss
                + denoised_unit_weight * denoised_units["loss"]
            )

    # Standard CFM only sees points on the analytic straight path between
    # noise and clean data. During inference the first imperfect Euler update
    # moves the state off that path, so later calls receive a distribution the
    # model was never trained to repair. The local mode runs one detached
    # model-produced step from the analytic state. When solver_steps >= 2, the
    # stronger on-policy mode instead starts at the actual source noise and
    # advances through a randomly selected prefix of a fixed-step Euler
    # solver under no_grad. Only the correction call retains activations, so
    # the objective sees accumulated inference errors without second-order
    # rollout gradients or an unbounded memory cost.
    rollout_weight = max(0.0, float(self_rollout_consistency_weight))
    rollout_probability = max(
        0.0,
        min(1.0, float(self_rollout_consistency_probability)),
    )
    if (
        rollout_weight > 0.0
        and rollout_probability > 0.0
        and torch.rand((), device=device) < rollout_probability
    ):
        solver_steps = max(0, int(self_rollout_solver_steps))
        if solver_steps >= 2:
            solver_dt = 1.0 / float(solver_steps)
            prefix_steps = int(
                torch.randint(
                    1,
                    solver_steps,
                    (),
                    device=device,
                ).item()
            )
            rollout_state = x0.detach().to(dtype=xt.dtype)
            with torch.no_grad():
                for prefix_index in range(prefix_steps):
                    prefix_t = torch.full_like(
                        t,
                        float(prefix_index) * solver_dt,
                    )
                    prefix_velocity = _call_model(
                        model,
                        x=rollout_state,
                        texts=model_texts,
                        timestep=prefix_t,
                        style_prompt=normalized_style,
                        lyric_frame_ids=model_frame_ids,
                    )
                    rollout_state = (
                        rollout_state.float()
                        + solver_dt * prefix_velocity.float()
                    ).to(dtype=xt.dtype)
            rollout_state = rollout_state.detach()
            rollout_t = torch.full_like(
                t,
                float(prefix_steps) * solver_dt,
            )
        else:
            rollout_step = max(
                1e-4,
                min(0.5, float(self_rollout_step_size)),
            )
            remaining = (1.0 - t).clamp_min(1e-4)
            rollout_dt = torch.minimum(
                torch.full_like(remaining, rollout_step),
                0.5 * remaining,
            )
            rollout_t = (t + rollout_dt).clamp(max=1.0 - 1e-4)
            rollout_state = (
                xt.float()
                + rollout_dt[:, None, None].float()
                * predicted_velocity_fp32
            ).detach().to(dtype=xt.dtype)
        rollout_velocity = _call_model(
            model,
            x=rollout_state,
            texts=model_texts,
            timestep=rollout_t,
            style_prompt=normalized_style,
            lyric_frame_ids=model_frame_ids,
        ).float()
        rollout_clean = (
            rollout_state.float()
            + (1.0 - rollout_t)[:, None, None].float()
            * rollout_velocity
        )
        rollout_reconstruction = (
            (rollout_clean - x1_fp32).abs() * frame_weights_fp32
        ).mean()
        rollout_time_delta = F.l1_loss(
            torch.diff(rollout_clean, dim=1),
            torch.diff(x1_fp32, dim=1),
        )
        rollout_frequency_delta = F.l1_loss(
            torch.diff(rollout_clean, dim=2),
            torch.diff(x1_fp32, dim=2),
        )
        rollout_loss = (
            rollout_reconstruction
            + 0.10 * (rollout_time_delta + rollout_frequency_delta)
        )
        total_loss = total_loss + rollout_weight * rollout_loss

    # Conditional flow matching can minimize its marginal audio loss while
    # reacting only to "text present" rather than to the requested phonemes.
    # Compare each correct lyric against a different lyric from the same batch.
    # Text dropout already trains the empty classifier-free branch; this extra
    # forward is reserved for the missing content-specific supervision.
    contrastive_weight = max(0.0, float(text_contrastive_weight))
    sensitivity_weight = max(0.0, float(text_sensitivity_weight))
    contrastive_probability = max(0.0, min(1.0, float(text_contrastive_prob)))
    (
        mismatched_texts,
        content_mask_flags,
        mismatched_source_indices,
    ) = build_mismatched_text_conditioning(model_texts)
    content_mask = torch.tensor(content_mask_flags, dtype=torch.bool, device=device)
    if (
        (contrastive_weight > 0.0 or sensitivity_weight > 0.0)
        and bool(content_mask.any())
        and torch.rand((), device=device) < contrastive_probability
    ):
        mismatched_frame_ids = None
        if model_frame_ids is not None:
            source_indices = torch.tensor(
                mismatched_source_indices,
                dtype=torch.long,
                device=device,
            )
            mismatched_frame_ids = model_frame_ids.index_select(0, source_indices)
            mismatched_frame_ids = mismatched_frame_ids.masked_fill(
                ~content_mask[:, None],
                0,
            )
        mismatched_velocity = _call_model(
            model,
            x=xt,
            texts=mismatched_texts,
            timestep=t,
            style_prompt=normalized_style,
            lyric_frame_ids=mismatched_frame_ids,
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
    lyric_frame_ids: torch.Tensor | None = None,
    initial_mel: torch.Tensor | None = None,
    refinement_mask: torch.Tensor | None = None,
    solver: str = "euler",
) -> torch.Tensor:
    """Sample a vocal mel, optionally using the style inputs from training."""
    model.eval()
    resolved_solver = str(solver).strip().casefold()
    if resolved_solver not in {"euler", "heun", "midpoint"}:
        raise ValueError(
            "solver must be one of: euler, heun, midpoint."
        )
    
    # Set seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        
        # Ensure numpy seed is aligned if needed
        import numpy as np
        np.random.seed(seed)
        
    batch_size = len(texts)
    
    # 1. Start with Gaussian noise for ordinary generation, or from a
    # target-free retrieval latent for seed-to-target refinement.
    if initial_mel is None:
        if refinement_mask is not None:
            raise ValueError(
                "refinement_mask requires an initial_mel seed."
            )
        xt = torch.randn(
            (batch_size, frames, config.latent_dim),
            device=device,
        )
    else:
        xt = torch.as_tensor(initial_mel, device=device)
        if xt.ndim != 3:
            raise ValueError(
                "initial_mel must be rank 3 in either (batch, frames, "
                "channels) or (batch, channels, frames) layout."
            )
        expected = (batch_size, int(frames), int(config.latent_dim))
        transposed = (
            batch_size,
            int(config.latent_dim),
            int(frames),
        )
        if tuple(xt.shape) == transposed:
            xt = xt.transpose(1, 2)
        elif tuple(xt.shape) != expected:
            raise ValueError(
                f"initial_mel must have shape {expected} or {transposed}, "
                f"got {tuple(xt.shape)}"
            )
        model_dtype = next(model.parameters()).dtype
        xt = xt.to(dtype=model_dtype)

    resolved_refinement_mask = None
    if refinement_mask is not None:
        resolved_refinement_mask = torch.as_tensor(
            refinement_mask,
            dtype=xt.dtype,
            device=device,
        )
        if resolved_refinement_mask.ndim == 2:
            resolved_refinement_mask = (
                resolved_refinement_mask.unsqueeze(-1)
            )
        elif (
            resolved_refinement_mask.ndim == 3
            and resolved_refinement_mask.shape[1] == 1
            and resolved_refinement_mask.shape[2] == int(frames)
        ):
            resolved_refinement_mask = (
                resolved_refinement_mask.transpose(1, 2)
            )
        if (
            resolved_refinement_mask.ndim != 3
            or resolved_refinement_mask.shape[:2] != xt.shape[:2]
            or resolved_refinement_mask.shape[2] not in (1, xt.shape[2])
        ):
            raise ValueError(
                "refinement_mask must align with initial_mel; got "
                f"{tuple(resolved_refinement_mask.shape)} for "
                f"{tuple(xt.shape)}."
            )
        resolved_refinement_mask = (
            resolved_refinement_mask.clamp(0.0, 1.0)
        )
    
    normalized_style = _prepare_style_condition(
        style_prompt,
        batch_size=batch_size,
        style_dim=int(getattr(model, "style_dim", 512)),
        device=device,
    )
    
    dt = 1.0 / steps
    
    def guided_velocity(state, timestep):
        velocity = _call_model(
            model,
            x=state,
            texts=texts,
            timestep=timestep,
            style_prompt=normalized_style,
            lyric_frame_ids=lyric_frame_ids,
        )
        if guidance_scale != 1.0:
            unconditional_frame_ids = (
                torch.zeros_like(lyric_frame_ids)
                if lyric_frame_ids is not None
                else None
            )
            unconditional = _call_model(
                model,
                x=state,
                texts=[""] * batch_size,
                timestep=timestep,
                style_prompt=normalized_style,
                lyric_frame_ids=unconditional_frame_ids,
            )
            velocity = unconditional + float(guidance_scale) * (
                velocity - unconditional
            )
        if resolved_refinement_mask is not None:
            velocity = velocity * resolved_refinement_mask
        return velocity

    # 2. Integrate the learned flow from t = 0 to t = 1. Euler remains the
    # checkpoint-compatible default; midpoint and Heun are inference-only
    # diagnostics for accumulated vector-field error.
    for step in range(steps):
        t_val = step / steps
        t = torch.full((batch_size,), t_val, device=device, dtype=torch.float32)
        first_velocity = guided_velocity(xt, t)
        if resolved_solver == "euler":
            xt = xt + first_velocity * dt
        elif resolved_solver == "midpoint":
            midpoint_t = torch.full_like(t, t_val + 0.5 * dt)
            midpoint_state = xt + first_velocity * (0.5 * dt)
            midpoint_velocity = guided_velocity(
                midpoint_state,
                midpoint_t,
            )
            xt = xt + midpoint_velocity * dt
        else:
            endpoint_t = torch.full_like(t, t_val + dt)
            endpoint_state = xt + first_velocity * dt
            endpoint_velocity = guided_velocity(
                endpoint_state,
                endpoint_t,
            )
            xt = xt + 0.5 * (
                first_velocity + endpoint_velocity
            ) * dt
        
    # Return the generated mel spectrogram (batch, n_mels, frames)
    # Match the output shape expected by the vocoders: (batch_size, n_mels, seq_len)
    return xt.transpose(1, 2)
