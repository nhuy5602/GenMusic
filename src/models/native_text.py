"""Trainable Vietnamese lyric and audio-content modules.

This module deliberately has no Hugging Face or TTS dependency.  Lyrics are
represented as UTF-8 bytes, which preserves every Vietnamese tone mark while
keeping the vocabulary fixed and deterministic.  The audio recognizer is a
training-time auxiliary head: CTC makes the predicted vocal component retain
the requested byte sequence instead of merely reacting to "some text".
"""

from __future__ import annotations

import math
import unicodedata

import torch
from torch import nn
from torch.nn import functional as F


CTC_BLANK_ID = 0
BYTE_OFFSET = 1
BYTE_TOKEN_COUNT = 256
ENCODER_BOS_ID = BYTE_OFFSET + BYTE_TOKEN_COUNT
ENCODER_EOS_ID = ENCODER_BOS_ID + 1
NATIVE_TEXT_VOCAB_SIZE = ENCODER_EOS_ID + 1

# UTF-8 remains an exact, compact input representation for the lyric encoder.
# CTC has a different constraint: after the audio frontend, a 4.096-second
# sample has only 96 time steps. Vietnamese tone marks take two or three UTF-8
# bytes, so byte targets can be longer than the available CTC alignment. A
# fixed NFC grapheme alphabet keeps this frontend fully trainable from scratch
# while representing each Vietnamese character with one target token.
_VIETNAMESE_LETTERS = (
    "abcdefghijklmnopqrstuvwxyz"
    "àáảãạăằắẳẵặâầấẩẫậ"
    "èéẻẽẹêềếểễệ"
    "ìíỉĩị"
    "òóỏõọôồốổỗộơờớởỡợ"
    "ùúủũụưừứửữự"
    "ỳýỷỹỵđ"
)
_CTC_EXTRA_SYMBOLS = " 0123456789'’-.,!?;:()"
NATIVE_CTC_SYMBOLS = tuple(dict.fromkeys(_VIETNAMESE_LETTERS + _CTC_EXTRA_SYMBOLS))
NATIVE_CTC_TO_ID = {
    symbol: index + 1 for index, symbol in enumerate(NATIVE_CTC_SYMBOLS)
}
NATIVE_CTC_FROM_ID = {
    index: symbol for symbol, index in NATIVE_CTC_TO_ID.items()
}
NATIVE_CTC_VOCAB_SIZE = 1 + len(NATIVE_CTC_SYMBOLS)


def _attention_heads(dim: int, maximum: int = 4) -> int:
    for heads in range(min(maximum, dim), 0, -1):
        if dim % heads == 0:
            return heads
    return 1


def utf8_token_ids(text: str, *, max_bytes: int | None = None) -> list[int]:
    """Encode exact Unicode text without a pretrained tokenizer."""
    values = list(str(text).strip().encode("utf-8"))
    if max_bytes is not None:
        values = values[: max(0, int(max_bytes))]
    return [value + BYTE_OFFSET for value in values]


def native_text_token_ids(text: str, *, max_tokens: int | None = None) -> list[int]:
    """Encode pronounceable Vietnamese graphemes for the trainable frontend.

    Earlier native checkpoints fed UTF-8 bytes to the lyric encoder.  A single
    Vietnamese letter can occupy three bytes, so the frame-aligned conditioning
    path gave accented letters two or three times as much duration as ASCII
    letters.  Reusing the fixed CTC grapheme ids keeps the frontend fully local
    and trainable while giving every written sound exactly one ordered token.
    The embedding table intentionally keeps its old size so acoustic checkpoint
    upgrades do not need an unsafe tensor-shape migration.
    """
    values = grapheme_token_ids(text)
    if max_tokens is not None:
        values = values[: max(0, int(max_tokens))]
    return values


def grapheme_token_ids(text: str) -> list[int]:
    """Encode Vietnamese NFC graphemes for feasible audio/text alignment."""
    normalized = unicodedata.normalize("NFC", str(text)).casefold()
    tokens: list[int] = []
    previous_was_space = True
    for character in normalized:
        if character.isspace() or character in _CTC_EXTRA_SYMBOLS:
            # Punctuation and written digits are not consistently pronounced
            # in singing. Treat them as word boundaries instead of asking the
            # audio recognizer to predict symbols absent from the waveform.
            character = " "
        token = NATIVE_CTC_TO_ID.get(character)
        if token is None:
            # Symbols outside the fixed Vietnamese alphabet (for example an
            # emoji) carry no pronounceable target and are safely ignored.
            continue
        if character == " " and previous_was_space:
            continue
        tokens.append(token)
        previous_was_space = character == " "
    if tokens and tokens[-1] == NATIVE_CTC_TO_ID[" "]:
        tokens.pop()
    return tokens


def ctc_target_text(text: str) -> str:
    """Return the exact pronounceable grapheme string used by native CTC."""
    return "".join(NATIVE_CTC_FROM_ID[token] for token in grapheme_token_ids(text))


def greedy_decode_ctc(logits: torch.Tensor) -> list[str]:
    """Collapse greedy CTC paths into Vietnamese grapheme strings."""
    if logits.dim() != 3:
        raise ValueError(f"CTC logits must be [batch,time,vocab], got {tuple(logits.shape)}")
    decoded: list[str] = []
    for path in logits.detach().argmax(dim=-1).cpu().tolist():
        symbols: list[str] = []
        previous = CTC_BLANK_ID
        for token in path:
            if token != CTC_BLANK_ID and token != previous:
                symbol = NATIVE_CTC_FROM_ID.get(int(token))
                if symbol is not None:
                    symbols.append(symbol)
            previous = int(token)
        decoded.append("".join(symbols).strip())
    return decoded


def greedy_decode_frame_text(logits: torch.Tensor) -> list[str]:
    """Collapse consecutive frame labels into an ordered grapheme string.

    Frame supervision deliberately repeats each target grapheme over an
    approximate acoustic span.  Unlike CTC, blanks are not required between
    adjacent labels, so decoding only removes consecutive duplicates and any
    accidental blank predictions.
    """
    if logits.dim() != 3:
        raise ValueError(
            f"Frame-text logits must be [batch,time,vocab], got {tuple(logits.shape)}"
        )
    decoded: list[str] = []
    for path in logits.detach().argmax(dim=-1).cpu().tolist():
        symbols: list[str] = []
        previous: int | None = None
        for token in path:
            token = int(token)
            if token != CTC_BLANK_ID and token != previous:
                symbol = NATIVE_CTC_FROM_ID.get(token)
                if symbol is not None:
                    symbols.append(symbol)
            previous = token
        decoded.append("".join(symbols).strip())
    return decoded


def build_ctc_targets(
    texts: list[str],
    *,
    device,
    max_target_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return concatenated CTC targets and a mask for non-empty samples."""
    sequences: list[list[int]] = []
    for text in texts:
        sequence: list[int] = []
        required_frames = 0
        for token in grapheme_token_ids(text):
            # CTC needs a blank frame only between identical adjacent labels.
            next_required = required_frames + 1 + int(bool(sequence) and sequence[-1] == token)
            if next_required > max(0, int(max_target_length)):
                break
            sequence.append(token)
            required_frames = next_required
        sequences.append(sequence)
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long, device=device)
    nonempty = lengths > 0
    flattened = [token for sequence in sequences for token in sequence]
    targets = torch.tensor(flattened, dtype=torch.long, device=device)
    return targets, lengths, nonempty


def native_ctc_loss(logits: torch.Tensor, texts: list[str]) -> torch.Tensor:
    """CTC loss from audio-only logits to the requested Vietnamese text."""
    if logits.dim() != 3:
        raise ValueError(f"CTC logits must be [batch,time,vocab], got {tuple(logits.shape)}")
    batch_size, time_steps, _ = logits.shape
    _, target_lengths, nonempty = build_ctc_targets(
        texts,
        device=logits.device,
        max_target_length=max(1, time_steps),
    )
    if not bool(nonempty.any()):
        return logits.sum() * 0.0

    selected_logits = logits[nonempty]
    selected_lengths = target_lengths[nonempty]
    # Rebuild targets because torch CTC expects only targets for selected rows.
    selected_texts = [texts[index] for index, keep in enumerate(nonempty.tolist()) if keep]
    selected_targets, selected_lengths, _ = build_ctc_targets(
        selected_texts,
        device=logits.device,
        max_target_length=max(1, time_steps),
    )
    input_lengths = torch.full(
        (selected_logits.shape[0],),
        time_steps,
        dtype=torch.long,
        device=logits.device,
    )
    return F.ctc_loss(
        selected_logits.float().log_softmax(dim=-1).transpose(0, 1),
        selected_targets,
        input_lengths,
        selected_lengths,
        blank=CTC_BLANK_ID,
        reduction="mean",
        zero_infinity=True,
    )


def build_frame_text_targets(
    texts: list[str],
    *,
    time_steps: int,
    device,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Spread ordered Vietnamese graphemes over an acoustic frame sequence.

    CTC was too under-constrained for this small singing corpus: its loss could
    fall by predicting common letters while greedy CER became worse.  The
    timestamped training crop already tells us which lyric span is present, so
    a monotonic uniform allocation is a useful, deterministic first alignment.
    It is intentionally approximate--the prosody module remains free to learn
    non-uniform sung durations--but every requested grapheme now contributes a
    direct frame-level content gradient.
    """
    steps = max(0, int(time_steps))
    targets = torch.full(
        (len(texts), steps),
        int(ignore_index),
        dtype=torch.long,
        device=device,
    )
    if steps == 0:
        return targets
    frame_positions = torch.arange(steps, device=device, dtype=torch.float32)
    for index, text in enumerate(texts):
        sequence = grapheme_token_ids(text)
        if not sequence:
            continue
        token_ids = torch.tensor(sequence, dtype=torch.long, device=device)
        token_positions = torch.floor(
            frame_positions * len(sequence) / steps
        ).long().clamp_max(len(sequence) - 1)
        targets[index] = token_ids[token_positions]
    return targets


def build_timestamped_frame_text_targets(
    segments_batch: list[list[dict]],
    *,
    crop_starts_seconds: list[float],
    crop_ends_seconds: list[float],
    time_steps: int,
    device,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Map exact word timestamps to recognizer frames, leaving silence ignored."""
    if not (
        len(segments_batch)
        == len(crop_starts_seconds)
        == len(crop_ends_seconds)
    ):
        raise ValueError("Timestamped frame-target batches must have matching lengths")
    steps = max(0, int(time_steps))
    targets = torch.full(
        (len(segments_batch), steps),
        int(ignore_index),
        dtype=torch.long,
        device=device,
    )
    if steps == 0:
        return targets

    for batch_index, segments in enumerate(segments_batch):
        crop_start = float(crop_starts_seconds[batch_index])
        crop_end = max(crop_start, float(crop_ends_seconds[batch_index]))
        crop_duration = max(1e-6, crop_end - crop_start)
        for segment in segments or []:
            for word in segment.get("words") or []:
                word_start = max(crop_start, float(word.get("start", crop_start)))
                word_end = min(crop_end, float(word.get("end", word_start)))
                sequence = grapheme_token_ids(str(word.get("word") or ""))
                if not sequence or word_end <= word_start:
                    continue
                first_step = max(
                    0,
                    min(
                        steps - 1,
                        math.floor((word_start - crop_start) * steps / crop_duration),
                    ),
                )
                last_step = max(
                    first_step + 1,
                    min(
                        steps,
                        math.ceil((word_end - crop_start) * steps / crop_duration),
                    ),
                )
                positions = torch.arange(
                    last_step - first_step,
                    device=device,
                    dtype=torch.float32,
                )
                token_ids = torch.tensor(sequence, dtype=torch.long, device=device)
                token_positions = torch.floor(
                    positions * len(sequence) / max(1, last_step - first_step)
                ).long().clamp_max(len(sequence) - 1)
                targets[batch_index, first_step:last_step] = token_ids[token_positions]
    return targets


def native_frame_text_loss(
    logits: torch.Tensor,
    texts: list[str],
    *,
    targets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-entropy for a monotonic frame-to-grapheme singing alignment."""
    if logits.dim() != 3:
        raise ValueError(
            f"Frame-text logits must be [batch,time,vocab], got {tuple(logits.shape)}"
        )
    if logits.shape[0] != len(texts):
        raise ValueError(
            "Frame-text batch size must match texts: "
            f"{logits.shape[0]} != {len(texts)}"
        )
    if targets is None:
        targets = build_frame_text_targets(
            texts,
            time_steps=logits.shape[1],
            device=logits.device,
        )
    else:
        targets = targets.to(device=logits.device, dtype=torch.long)
        if tuple(targets.shape) != tuple(logits.shape[:2]):
            raise ValueError(
                "Frame targets must match [batch,time]: "
                f"{tuple(targets.shape)} != {tuple(logits.shape[:2])}"
            )
    valid = targets != -100
    if not bool(valid.any()):
        return logits.sum() * 0.0
    return F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=-100,
    )


@torch.no_grad()
def ctc_guided_duration_targets(
    logits: torch.Tensor,
    texts: list[str],
    *,
    token_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build soft monotonic grapheme durations from an audio-only CTC head.

    The CTC recognizer is trained from real vocal stems.  Its posterior is only
    used as a detached alignment hint here; gradients still flow exclusively
    through ``NativeProsodyConditioner``.  A broad monotonic prior keeps this
    stable while the recognizer is imperfect early in training.
    """
    if logits.dim() != 3:
        raise ValueError(f"CTC logits must be [batch,time,vocab], got {tuple(logits.shape)}")
    batch_size, time_steps, _ = logits.shape
    result = torch.zeros((batch_size, token_width), device=logits.device, dtype=torch.float32)
    mask = torch.zeros_like(result, dtype=torch.bool)
    log_probs = logits.float().log_softmax(dim=-1)
    frame_positions = (torch.arange(time_steps, device=logits.device, dtype=torch.float32) + 0.5) / max(1, time_steps)
    for index, text in enumerate(texts):
        token_ids = grapheme_token_ids(text)[: max(0, time_steps)]
        if not token_ids or token_width <= 2:
            continue
        ids = torch.tensor(token_ids, device=logits.device, dtype=torch.long)
        posterior = log_probs[index][:, ids]
        centers = (torch.arange(len(token_ids), device=logits.device, dtype=torch.float32) + 0.5) / len(token_ids)
        monotonic_prior = -0.5 * ((frame_positions[:, None] - centers[None, :]) / 0.35).square()
        assignments = (posterior + monotonic_prior).softmax(dim=-1)
        durations = assignments.sum(dim=0)
        durations = durations / durations.sum().clamp_min(1e-6)
        end = min(token_width - 1, len(token_ids) + 1)
        result[index, 1:end] = durations[: end - 1]
        mask[index, 1:end] = True
    return result, mask


class NativeVietnameseTextEncoder(nn.Module):
    """Small Vietnamese grapheme encoder trained jointly with the music model."""

    is_native = True
    encoder_type = "native_utf8"

    def __init__(self, out_dim: int, *, max_length: int = 192, depth: int = 2):
        super().__init__()
        self.max_length = int(max_length)
        self.embedding = nn.Embedding(NATIVE_TEXT_VOCAB_SIZE, out_dim, padding_idx=CTC_BLANK_ID)
        self.position = nn.Embedding(self.max_length, out_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=out_dim,
            nhead=_attention_heads(out_dim),
            dim_feedforward=out_dim * 3,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=max(1, int(depth)), enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, texts: list[str], device) -> tuple[torch.Tensor, torch.Tensor]:
        sequences: list[list[int]] = []
        for text in texts:
            token_ids = native_text_token_ids(text, max_tokens=self.max_length - 2)
            # Keep an explicit non-padding token even for the unconditional
            # branch; text_present still tells the DiT that this prompt is empty.
            sequences.append([ENCODER_BOS_ID, *token_ids, ENCODER_EOS_ID])
        width = max(len(sequence) for sequence in sequences)
        tokens = torch.zeros((len(sequences), width), dtype=torch.long, device=device)
        mask = torch.zeros((len(sequences), width), dtype=torch.bool, device=device)
        for index, sequence in enumerate(sequences):
            length = len(sequence)
            tokens[index, :length] = torch.tensor(sequence, dtype=torch.long, device=device)
            mask[index, :length] = True
        positions = torch.arange(width, device=device).unsqueeze(0)
        hidden = self.embedding(tokens) + self.position(positions)
        hidden = self.encoder(hidden, src_key_padding_mask=~mask)
        return self.norm(hidden), mask


class NativeVocalMelPrior(nn.Module):
    """Predict a coarse vocal mel directly from frame-aligned lyric features.

    This is an internal, randomly initialized acoustic branch--not a TTS model
    or a pretrained pronunciation prior.  Its supervised real-vocal loss gives
    the joint flow a short path from graphemes to formant/energy structure, so
    the much easier backing target cannot make the generated vocal collapse to
    an unvoiced average.
    """

    def __init__(self, dim: int, n_mels: int):
        super().__init__()
        self.context = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(dim, dim, kernel_size=5, padding=2),
            nn.SiLU(),
        )
        self.norm = nn.LayerNorm(dim)
        self.projection = nn.Linear(dim, n_mels)
        # Start as a small, neutral residual when upgrading an acoustic
        # checkpoint. The direct prior loss quickly grows useful structure.
        nn.init.normal_(self.projection.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.projection.bias)

    def forward(
        self,
        aligned_text: torch.Tensor,
        text_present: torch.Tensor,
    ) -> torch.Tensor:
        if aligned_text.dim() != 3:
            raise ValueError(
                "aligned_text must be [batch,time,dim], "
                f"got {tuple(aligned_text.shape)}"
            )
        residual = self.context(aligned_text.transpose(1, 2)).transpose(1, 2)
        prior = self.projection(self.norm(aligned_text + residual))
        return prior * text_present.to(prior.dtype).view(-1, 1, 1)


class NativeProsodyConditioner(nn.Module):
    """Learn duration, pitch and voicing controls from graphemes and vocal mel.

    This is deliberately a small trainable acoustic module, not a TTS system.
    Durations are a monotonic soft alignment over the requested graphemes;
    pitch/voicing are frame-rate controls learned from the real vocal stem.
    The denoiser receives the controls during both training and generation.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.duration_head = nn.Linear(dim, 1)
        self.frame_encoder = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.prosody_head = nn.Linear(dim, 3)
        self.condition_projection = nn.Sequential(
            nn.Linear(3, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        # Warm-start old native checkpoints with the previous uniform
        # grapheme-to-frame alignment; the learned duration alignment grows
        # gradually instead of destroying a useful vocal prior on resume.
        self.alignment_gate = nn.Parameter(torch.tensor(-3.0))
        nn.init.zeros_(self.condition_projection[-1].weight)
        nn.init.zeros_(self.condition_projection[-1].bias)

    @staticmethod
    def _content_tokens(
        text_embed: torch.Tensor, text_mask: torch.Tensor, text_present: torch.Tensor
    ) -> list[torch.Tensor]:
        tokens: list[torch.Tensor] = []
        for index in range(text_embed.shape[0]):
            valid = text_embed[index][text_mask[index]]
            if not bool(text_present[index]) or valid.shape[0] <= 2:
                tokens.append(valid[:0])
            else:
                # NativeVietnameseTextEncoder wraps graphemes in BOS/EOS.
                tokens.append(valid[1:-1])
        return tokens

    def forward(
        self,
        text_embed: torch.Tensor,
        text_mask: torch.Tensor,
        text_present: torch.Tensor,
        frame_count: int,
    ) -> dict[str, torch.Tensor]:
        batch_size, dim = text_embed.shape[0], text_embed.shape[-1]
        device = text_embed.device
        frame_positions = (
            (torch.arange(frame_count, device=device, dtype=text_embed.dtype) + 0.5)
            / max(1, frame_count)
        )
        content = self._content_tokens(text_embed, text_mask, text_present)
        aligned: list[torch.Tensor] = []
        uniform_aligned: list[torch.Tensor] = []
        duration_maps: list[torch.Tensor] = []
        for index, tokens in enumerate(content):
            if tokens.shape[0] == 0:
                aligned.append(torch.zeros((frame_count, dim), device=device, dtype=text_embed.dtype))
                uniform_aligned.append(torch.zeros((frame_count, dim), device=device, dtype=text_embed.dtype))
                duration_maps.append(torch.zeros((text_embed.shape[1],), device=device, dtype=text_embed.dtype))
                continue
            logits = self.duration_head(tokens).squeeze(-1)
            durations = F.softmax(logits.float(), dim=0).to(text_embed.dtype)
            centers = torch.cumsum(durations.float(), dim=0) - 0.5 * durations.float()
            # A duration-proportional Gaussian gives a smooth monotonic
            # alignment and remains differentiable when a lyric is stretched.
            widths = (durations.float() * 0.75).clamp_min(1.0 / max(2, frame_count))
            scores = -0.5 * ((frame_positions[:, None].float() - centers[None, :]) / widths[None, :]).square()
            weights = F.softmax(scores, dim=-1).to(text_embed.dtype)
            aligned.append(weights @ tokens)
            uniform_aligned.append(
                F.interpolate(
                    tokens.transpose(0, 1).unsqueeze(0),
                    size=frame_count,
                    mode="linear",
                    align_corners=False,
                ).squeeze(0).transpose(0, 1)
            )
            duration_map = torch.zeros((text_embed.shape[1],), device=device, dtype=text_embed.dtype)
            duration_map[1 : 1 + tokens.shape[0]] = durations
            duration_maps.append(duration_map)
        aligned_text = torch.stack(aligned)
        uniform_text = torch.stack(uniform_aligned)
        duration_proportions = torch.stack(duration_maps)
        encoded = self.frame_encoder(aligned_text.transpose(1, 2)).transpose(1, 2)
        controls = self.prosody_head(encoded)
        pitch = torch.sigmoid(controls[..., 0])
        voicing = torch.sigmoid(controls[..., 1])
        energy = torch.tanh(controls[..., 2])
        prosody = torch.stack((pitch, voicing, energy), dim=-1)
        present = text_present.to(text_embed.dtype).view(batch_size, 1, 1)
        duration_strength = 0.05 + 0.95 * torch.sigmoid(self.alignment_gate)
        conditioned = uniform_text + duration_strength * (aligned_text - uniform_text)
        conditioned = conditioned + self.condition_projection(prosody) * present
        return {
            "aligned_text": conditioned * present,
            "duration_proportions": duration_proportions * text_present.to(text_embed.dtype).view(-1, 1),
            "pitch": pitch * present.squeeze(-1),
            "voicing": voicing * present.squeeze(-1),
            "voicing_logits": controls[..., 1] * present.squeeze(-1),
            "energy": energy * present.squeeze(-1),
            "controls": prosody * present,
        }


class NativeAudioTextRecognizer(nn.Module):
    """Audio-only CTC head trained from scratch on vocal and full-mix mels."""

    def __init__(self, n_mels: int, dim: int):
        super().__init__()
        hidden = max(64, min(256, int(dim)))
        self.frontend = nn.Sequential(
            nn.Conv1d(n_mels, hidden, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(1, hidden),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(1, hidden),
            nn.SiLU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=_attention_heads(hidden),
            dim_feedforward=hidden * 3,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context = nn.TransformerEncoder(
            layer, num_layers=2, enable_nested_tensor=False
        )
        self.classifier = nn.Linear(hidden, NATIVE_CTC_VOCAB_SIZE)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        if mel.dim() != 3:
            raise ValueError(f"vocal mel must be [batch,time,n_mels], got {tuple(mel.shape)}")
        hidden = self.frontend(mel.transpose(1, 2)).transpose(1, 2)
        return self.classifier(self.context(hidden))
