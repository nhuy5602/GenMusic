"""Compositional Vietnamese lyric conditioning for unseen words.

The existing XPhoneBERT path remains useful for global phonetic context, but a
frozen subword tokenizer can still hide whether a rare Vietnamese word is
represented compositionally.  This module provides a small trainable path
whose vocabulary is the Vietnamese alphabet decomposed into base letters and
tone marks.  It therefore does not need a word or phrase to have appeared in
the training corpus.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

import torch
from torch import nn

from .vietnamese_syllables import (
    CODA_NAMES,
    MAX_NUCLEUS_GRAPHEMES,
    NUCLEUS_GRAPHEMES,
    ONSET_NAMES,
    TONE_NAMES,
    decompose_vietnamese_syllable,
)

_WORD_RE = re.compile(r"[0-9A-Za-zÀ-ỹĐđ]+", flags=re.UNICODE)
_COMBINING_MARKS = ("\u0300", "\u0301", "\u0303", "\u0309", "\u0323", "\u0302", "\u0306", "\u031b")
_GRAPHEMES = (
    "<pad>",
    "<unk>",
    *tuple("abcdefghijklmnopqrstuvwxyz0123456789đ"),
    *_COMBINING_MARKS,
)
GRAPHEME_TO_ID = {token: index for index, token in enumerate(_GRAPHEMES)}
PAD_ID = GRAPHEME_TO_ID["<pad>"]
UNK_ID = GRAPHEME_TO_ID["<unk>"]


def split_lyric_words(text: str) -> list[str]:
    """Return normalized Vietnamese word units while preserving tone marks."""
    normalized = unicodedata.normalize("NFC", str(text)).casefold()
    return _WORD_RE.findall(normalized)


def grapheme_ids(word: str) -> list[int]:
    """Decompose one word into a fixed open Vietnamese grapheme inventory."""
    decomposed = unicodedata.normalize("NFD", str(word).casefold())
    ids = [
        GRAPHEME_TO_ID.get(character, UNK_ID)
        for character in decomposed
        if not character.isspace()
    ]
    return ids or [UNK_ID]


_VIETNAMESE_MULTI_GRAPHEMES = (
    "ngh",
    "ch",
    "gh",
    "gi",
    "kh",
    "ng",
    "nh",
    "ph",
    "qu",
    "th",
    "tr",
)


def split_vietnamese_grapheme_units(word: str) -> list[str]:
    """Split a word into reusable Vietnamese spelling/phonetic units.

    Tone and vowel-shape combining marks stay attached to their base letter,
    while common Vietnamese onset/coda digraphs remain single units. The
    resulting inventory is open: every unit is still encoded from the fixed
    character inventory rather than looked up in a closed word vocabulary.
    """
    decomposed = unicodedata.normalize("NFD", str(word).casefold())
    clusters: list[str] = []
    for character in decomposed:
        if character.isspace():
            continue
        if unicodedata.combining(character) and clusters:
            clusters[-1] += character
        else:
            clusters.append(character)
    if not clusters:
        return ["<unk>"]

    base_letters = [
        "".join(
            character
            for character in cluster
            if not unicodedata.combining(character)
        )
        for cluster in clusters
    ]
    units: list[str] = []
    index = 0
    while index < len(clusters):
        matched = 1
        for candidate in _VIETNAMESE_MULTI_GRAPHEMES:
            width = len(candidate)
            if "".join(base_letters[index : index + width]) == candidate:
                matched = width
                break
        units.append(
            unicodedata.normalize(
                "NFC",
                "".join(clusters[index : index + matched]),
            )
        )
        index += matched
    return units


def uniform_frame_word_ids(
    texts: Sequence[str],
    frames: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Allocate arbitrary input words over frames without corpus timing.

    Allocation is proportional to grapheme count, which gives longer
    Vietnamese syllables more space without consulting donor audio or target
    timestamps.  Zero remains reserved for silence; word indices start at one.
    """
    frame_count = max(1, int(frames))
    result = torch.zeros((len(texts), frame_count), dtype=torch.long, device=device)
    for batch_index, text in enumerate(texts):
        words = split_lyric_words(text)
        if not words:
            continue
        weights = torch.tensor(
            [max(1, len(grapheme_ids(word))) for word in words],
            dtype=torch.float64,
        )
        boundaries = torch.round(
            torch.cat(
                [
                    torch.zeros(1, dtype=torch.float64),
                    weights.cumsum(0) / weights.sum() * frame_count,
                ]
            )
        ).long()
        boundaries[0] = 0
        boundaries[-1] = frame_count
        for word_index in range(len(words)):
            start = min(frame_count - 1, int(boundaries[word_index]))
            end = max(start + 1, min(frame_count, int(boundaries[word_index + 1])))
            result[batch_index, start:end] = word_index + 1
    return result


class OpenVocabularyLyricEncoder(nn.Module):
    """Encode unseen words compositionally and align them to audio frames.

    The character BiGRU remains an unrestricted fallback for foreign or
    malformed tokens. Vietnamese syllables also receive an explicit
    onset/nucleus/coda/tone residual. This prevents a novel word from needing
    to be rediscovered as one opaque character sequence: every factor belongs
    to a small, fully covered linguistic inventory.

    ``factorized_projection`` ends in a zero-initialized layer. Adding this
    path therefore preserves old open-vocabulary checkpoints exactly when
    they are loaded with ``strict=False``, while still exposing a direct
    gradient that can learn the new residual during continuation.
    """

    def __init__(self, dim: int, *, char_dim: int = 64):
        super().__init__()
        hidden_dim = max(1, int(dim) // 2)
        self.dim = int(dim)
        self.embedding = nn.Embedding(
            len(_GRAPHEMES),
            int(char_dim),
            padding_idx=PAD_ID,
        )
        self.word_encoder = nn.GRU(
            int(char_dim),
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.word_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, self.dim),
            nn.SiLU(),
            nn.LayerNorm(self.dim),
        )
        factor_dim = max(16, int(char_dim))
        self.syllable_onset_embedding = nn.Embedding(
            len(ONSET_NAMES) + 1,
            factor_dim,
            padding_idx=0,
        )
        self.syllable_nucleus_embedding = nn.Embedding(
            len(NUCLEUS_GRAPHEMES) + 1,
            factor_dim,
            padding_idx=0,
        )
        self.syllable_nucleus_position_embedding = nn.Embedding(
            MAX_NUCLEUS_GRAPHEMES,
            factor_dim,
        )
        self.syllable_coda_embedding = nn.Embedding(
            len(CODA_NAMES) + 1,
            factor_dim,
            padding_idx=0,
        )
        self.syllable_tone_embedding = nn.Embedding(
            len(TONE_NAMES) + 1,
            factor_dim,
            padding_idx=0,
        )
        self.factorized_projection = nn.Sequential(
            nn.Linear(factor_dim * 4, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim),
        )
        nn.init.zeros_(self.factorized_projection[-1].weight)
        nn.init.zeros_(self.factorized_projection[-1].bias)
        self.silence_embedding = nn.Parameter(torch.zeros(self.dim))

    def _factorized_syllable_features(
        self,
        words: list[str],
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return explicit Vietnamese factors and a parse-validity mask."""
        count = len(words)
        onset_ids = torch.zeros(count, dtype=torch.long, device=device)
        nucleus_ids = torch.zeros(
            (count, MAX_NUCLEUS_GRAPHEMES),
            dtype=torch.long,
            device=device,
        )
        nucleus_mask = torch.zeros_like(nucleus_ids, dtype=torch.bool)
        coda_ids = torch.zeros(count, dtype=torch.long, device=device)
        tone_ids = torch.zeros(count, dtype=torch.long, device=device)
        valid = torch.zeros(count, dtype=torch.bool, device=device)

        for index, word in enumerate(words):
            syllable = decompose_vietnamese_syllable(word)
            if syllable is None:
                continue
            valid[index] = True
            # Zero is reserved for the invalid/padding representation.
            onset_ids[index] = syllable.onset_id + 1
            coda_ids[index] = syllable.coda_id + 1
            tone_ids[index] = syllable.tone_id + 1
            for position, nucleus_id in enumerate(syllable.nucleus_ids):
                nucleus_ids[index, position] = nucleus_id + 1
                nucleus_mask[index, position] = True

        onset = self.syllable_onset_embedding(onset_ids)
        coda = self.syllable_coda_embedding(coda_ids)
        tone = self.syllable_tone_embedding(tone_ids)
        nucleus = self.syllable_nucleus_embedding(nucleus_ids)
        positions = self.syllable_nucleus_position_embedding(
            torch.arange(MAX_NUCLEUS_GRAPHEMES, device=device)
        )
        nucleus = nucleus + positions.unsqueeze(0) * nucleus_mask.unsqueeze(-1)
        nucleus = (
            nucleus * nucleus_mask.unsqueeze(-1)
        ).sum(dim=1) / nucleus_mask.sum(dim=1, keepdim=True).clamp_min(1)

        factors = torch.cat([onset, nucleus, coda, tone], dim=-1)
        residual = self.factorized_projection(factors)
        residual = residual * valid.unsqueeze(-1)
        return residual, valid

    def _encode_words(
        self,
        words_by_text: list[list[str]],
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flattened = [
            word
            for words in words_by_text
            for word in words
        ]
        max_words = max(1, *(len(words) for words in words_by_text))
        memory = self.silence_embedding.new_zeros(
            (len(words_by_text), max_words, self.dim)
        )
        mask = torch.zeros(
            (len(words_by_text), max_words),
            dtype=torch.bool,
            device=device,
        )
        if not flattened:
            return memory, mask

        encoded = [grapheme_ids(word) for word in flattened]
        maximum_length = max(len(ids) for ids in encoded)
        character_ids = torch.full(
            (len(encoded), maximum_length),
            PAD_ID,
            dtype=torch.long,
            device=device,
        )
        lengths = torch.tensor(
            [len(ids) for ids in encoded],
            dtype=torch.long,
            device=device,
        )
        for row, ids in enumerate(encoded):
            character_ids[row, : len(ids)] = torch.tensor(
                ids,
                dtype=torch.long,
                device=device,
            )
        embedded = self.embedding(character_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.word_encoder(packed)
        word_features = self.word_projection(
            torch.cat([hidden[-2], hidden[-1]], dim=-1)
        )
        factorized_features, _ = self._factorized_syllable_features(
            flattened,
            device=device,
        )
        word_features = word_features + factorized_features

        offset = 0
        for batch_index, words in enumerate(words_by_text):
            count = len(words)
            if count:
                memory[batch_index, :count] = word_features[offset : offset + count]
                mask[batch_index, :count] = True
                offset += count
        return memory, mask

    def encode_units(
        self,
        units_by_text: list[list[str]],
        *,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode arbitrary subword units through the shared char BiGRU."""
        resolved_device = torch.device(
            device if device is not None else self.silence_embedding.device
        )
        return self._encode_words(
            units_by_text,
            device=resolved_device,
        )

    def forward(
        self,
        texts: Sequence[str],
        *,
        frames: int,
        frame_word_ids: torch.Tensor | None = None,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        resolved_device = torch.device(
            device if device is not None else self.silence_embedding.device
        )
        words_by_text = [split_lyric_words(text) for text in texts]
        word_memory, word_mask = self._encode_words(
            words_by_text,
            device=resolved_device,
        )
        if frame_word_ids is None:
            resolved_ids = uniform_frame_word_ids(
                texts,
                frames,
                device=resolved_device,
            )
        else:
            resolved_ids = torch.as_tensor(
                frame_word_ids,
                dtype=torch.long,
                device=resolved_device,
            )
            if resolved_ids.shape != (len(texts), int(frames)):
                raise ValueError(
                    "frame_word_ids must have shape "
                    f"{(len(texts), int(frames))}, got {tuple(resolved_ids.shape)}"
                )

        frame_features = self.silence_embedding.new_zeros(
            (len(texts), int(frames), self.dim)
        )
        for batch_index, words in enumerate(words_by_text):
            bank = torch.cat(
                [
                    self.silence_embedding.unsqueeze(0),
                    word_memory[batch_index, : len(words)],
                ],
                dim=0,
            )
            indices = resolved_ids[batch_index].clamp(
                min=0,
                max=max(0, len(words)),
            )
            frame_features[batch_index] = bank[indices]
        return word_memory, word_mask, frame_features, resolved_ids
