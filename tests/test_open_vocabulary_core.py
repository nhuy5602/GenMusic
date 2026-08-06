"""Focused tests for the report-aligned open-vocabulary diffusion core."""

from __future__ import annotations

import pytest
import torch

from src.models.open_vocabulary_lyrics import (
    OpenVocabularyLyricEncoder,
    grapheme_ids,
    split_lyric_words,
    uniform_frame_word_ids,
)
from src.models.v1_oobleck_codec import (
    V1_DOWNSAMPLE_RATIO,
    V1_LATENT_CHANNELS,
    decode_v1_oobleck_latent,
    encode_v1_oobleck_audio,
)
from src.models.vietnamese_syllables import decompose_vietnamese_syllable


def test_rare_vietnamese_words_use_compositional_graphemes() -> None:
    words = split_lyric_words(
        "Quỳnh, khuỷu và nghiêng!"
    )

    assert words == ["quỳnh", "khuỷu", "và", "nghiêng"]
    assert all(grapheme_ids(word) for word in words)
    assert grapheme_ids("ma") != grapheme_ids("má")


def test_vietnamese_syllable_factors_preserve_tone_and_coda() -> None:
    plain = decompose_vietnamese_syllable("thanh")
    accented = decompose_vietnamese_syllable("thành")

    assert plain is not None and accented is not None
    assert plain.onset == accented.onset == "th"
    assert plain.nucleus == accented.nucleus == ("a",)
    assert plain.coda == accented.coda == "nh"
    assert plain.tone == "ngang"
    assert accented.tone == "huyen"
    assert decompose_vietnamese_syllable("diffusion") is None


def test_uniform_alignment_and_encoder_cover_every_frame() -> None:
    texts = ["bình minh rực sáng", "mây trắng ngang trời"]
    frame_ids = uniform_frame_word_ids(texts, 24)
    encoder = OpenVocabularyLyricEncoder(dim=32, char_dim=16)

    memory, mask, frames, resolved_ids = encoder(
        texts,
        frames=24,
        frame_word_ids=frame_ids,
    )

    assert memory.shape == (2, 4, 32)
    assert mask.shape == (2, 4)
    assert frames.shape == (2, 24, 32)
    assert torch.equal(resolved_ids, frame_ids)
    assert bool((resolved_ids > 0).all())
    assert bool(torch.isfinite(frames).all())


def test_v1_codec_rejects_incompatible_shapes_before_download() -> None:
    with pytest.raises(ValueError, match="batch, 64, frames"):
        decode_v1_oobleck_latent(torch.zeros(1, V1_LATENT_CHANNELS - 1, 8), device="cpu")

    with pytest.raises(ValueError, match="batch, 2, samples"):
        encode_v1_oobleck_audio(torch.zeros(1, 1, V1_DOWNSAMPLE_RATIO), device="cpu")

    with pytest.raises(ValueError, match="divisible"):
        encode_v1_oobleck_audio(
            torch.zeros(1, 2, V1_DOWNSAMPLE_RATIO + 1),
            device="cpu",
        )
