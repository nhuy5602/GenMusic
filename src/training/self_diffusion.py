"""Dataset, DataLoader and training loop for the self-authored music diffusion model."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..models.open_vocabulary_lyrics import (
    split_lyric_words,
    uniform_frame_word_ids,
)
from ..models.text_to_music_diffusion import (
    MusicDiffusionConfig,
    _config_from_dict,
    normalize_mel,
    reconstruct_full_mix,
    structured_random_mel,
)

STYLE_EMBED_DIM = 512  # matches MuQ-MuLan / DiffRhythm2 teacher's cond_dim

# Ensure PyTorch helper works
def _torch():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise RuntimeError("Cần cài torch để chạy model sinh nhạc tự code.") from exc
    return torch, nn, Dataset, DataLoader

DEFAULT_TEXTS = [
    ("Mưa rơi trên mái hiên, lòng nghe bình yên.", "Vietnamese soft ballad, piano, warm strings, gentle beat"),
    ("Bước qua con phố, ta nhìn thấy bình minh.", "uplifting Vietnamese pop, acoustic guitar, bright drums"),
    ("Đêm nay thành phố ngủ quên trong tiếng gió.", "lonely ambient piano, slow pulse, spacious reverb"),
    ("Cùng nhau đi tới nơi ngày mai đang gọi.", "hopeful indie pop, steady rhythm, warm synths"),
]


def _is_usable_training_record(record: dict[str, Any]) -> bool:
    """Reject known silent Demucs failures and placeholder transcripts."""
    if record.get("has_vocal") is False or record.get("vocal_source") == "silence_fallback":
        return False
    text = str(record.get("text", "")).strip()
    if not text or text.casefold().startswith("vietnamese music track "):
        return False
    return True


def _filter_training_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if _is_usable_training_record(record)]


def split_training_records(
    records: list[dict[str, Any]],
    *,
    validation_fraction: float = 0.05,
    validation_max_records: int | None = 128,
    seed: int = 5602,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create a stable song-level split so validation never sees train crops."""
    usable = _filter_training_records(records)
    fraction = max(0.0, min(0.5, float(validation_fraction)))
    if fraction <= 0.0 or len(usable) < 2:
        return usable, []

    validation_count = max(1, round(len(usable) * fraction))
    if validation_max_records is not None:
        validation_count = min(validation_count, max(1, int(validation_max_records)))
    validation_count = min(validation_count, len(usable) - 1)
    ranked = sorted(
        enumerate(usable),
        key=lambda item: hashlib.sha256(f"{seed}:{item[1].get('id', item[0])}".encode("utf-8")).digest(),
    )
    validation_indices = {index for index, _ in ranked[:validation_count]}
    training_records = [record for index, record in enumerate(usable) if index not in validation_indices]
    validation_records = [record for index, record in enumerate(usable) if index in validation_indices]
    return training_records, validation_records


def _is_checkpoint_improvement(
    validation_loss: float | None,
    best_validation_loss: float,
    text_sensitivity: float | None,
    minimum_text_sensitivity: float,
    min_delta: float = 0.0,
    lexical_sensitivity: float | None = None,
    minimum_lexical_sensitivity: float = 0.0,
    lyric_semantic_accuracy: float | None = None,
    minimum_lyric_semantic_accuracy: float = 0.0,
    lyric_denoised_semantic_accuracy: float | None = None,
    minimum_lyric_denoised_semantic_accuracy: float = 0.0,
    lyric_unit_accuracy: float | None = None,
    minimum_lyric_unit_accuracy: float = 0.0,
    lyric_denoised_unit_accuracy: float | None = None,
    minimum_lyric_denoised_unit_accuracy: float = 0.0,
) -> bool:
    """Require acoustic improvement without accepting lyric-conditioning collapse."""
    if validation_loss is None:
        return False
    if text_sensitivity is not None and text_sensitivity < minimum_text_sensitivity:
        return False
    if (
        lexical_sensitivity is not None
        and lexical_sensitivity < minimum_lexical_sensitivity
    ):
        return False
    if (
        lyric_semantic_accuracy is not None
        and lyric_semantic_accuracy < minimum_lyric_semantic_accuracy
    ):
        return False
    if (
        lyric_denoised_semantic_accuracy is not None
        and lyric_denoised_semantic_accuracy
        < minimum_lyric_denoised_semantic_accuracy
    ):
        return False
    if (
        lyric_unit_accuracy is not None
        and lyric_unit_accuracy < minimum_lyric_unit_accuracy
    ):
        return False
    if (
        lyric_denoised_unit_accuracy is not None
        and lyric_denoised_unit_accuracy
        < minimum_lyric_denoised_unit_accuracy
    ):
        return False
    return validation_loss < best_validation_loss - max(0.0, float(min_delta))


def lyric_text_for_window(
    full_text: str,
    segments: list[dict[str, Any]],
    start_seconds: float,
    end_seconds: float,
) -> str:
    """Select timestamp-aligned words, approximating word times for old records."""
    if not segments:
        return str(full_text).strip()

    selected: list[str] = []
    for segment in segments:
        segment_start = float(segment.get("start", 0.0))
        segment_end = max(segment_start, float(segment.get("end", segment_start)))
        timestamped_words = segment.get("words") or []
        if timestamped_words:
            word_spans = [
                (
                    float(word.get("start", segment_start)),
                    float(word.get("end", segment_end)),
                    str(word.get("word") or word.get("text") or "").strip(),
                )
                for word in timestamped_words
            ]
        else:
            words = str(segment.get("text", "")).strip().split()
            duration = max(1e-3, segment_end - segment_start)
            word_spans = [
                (
                    segment_start + duration * index / max(1, len(words)),
                    segment_start + duration * (index + 1) / max(1, len(words)),
                    word,
                )
                for index, word in enumerate(words)
            ]
        selected.extend(
            word
            for word_start, word_end, word in word_spans
            if word and word_end > start_seconds and word_start < end_seconds
        )
    # An empty result is intentional: this crop lies in a non-vocal interval,
    # so conditioning it on the full-song transcript would teach false alignment.
    return " ".join(selected).strip()


def lyric_conditioning_for_window(
    full_text: str,
    segments: list[dict[str, Any]],
    start_seconds: float,
    end_seconds: float,
    frames: int,
) -> tuple[str, Any, float]:
    """Build exact word-to-frame IDs for one cropped training target.

    Word index zero is silence and positive IDs refer to the corresponding
    word in the returned lyric string. Records without word timestamps retain
    a deterministic text-only fallback, so old datasets remain trainable.
    """
    torch, _, _, _ = _torch()
    frame_count = max(1, int(frames))
    crop_duration = max(1e-6, float(end_seconds) - float(start_seconds))
    selected: list[tuple[str, float, float, bool]] = []
    for segment in segments:
        segment_start = float(segment.get("start", 0.0))
        segment_end = max(
            segment_start,
            float(segment.get("end", segment_start)),
        )
        timestamped_words = segment.get("words") or []
        if timestamped_words:
            spans = [
                (
                    str(word.get("word") or word.get("text") or "").strip(),
                    float(word.get("start", segment_start)),
                    float(word.get("end", segment_end)),
                    True,
                )
                for word in timestamped_words
            ]
        else:
            words = split_lyric_words(str(segment.get("text", "")))
            duration = max(1e-3, segment_end - segment_start)
            spans = [
                (
                    word,
                    segment_start + duration * index / max(1, len(words)),
                    segment_start
                    + duration * (index + 1) / max(1, len(words)),
                    False,
                )
                for index, word in enumerate(words)
            ]
        for raw_word, word_start, word_end, exact in spans:
            if word_end <= start_seconds or word_start >= end_seconds:
                continue
            normalized_words = split_lyric_words(raw_word)
            for normalized_word in normalized_words:
                selected.append(
                    (normalized_word, word_start, max(word_start, word_end), exact)
                )

    if not selected:
        fallback_text = lyric_text_for_window(
            full_text,
            segments,
            start_seconds,
            end_seconds,
        )
        fallback_ids = uniform_frame_word_ids(
            [fallback_text],
            frame_count,
        )[0]
        return fallback_text, fallback_ids, 0.0

    frame_word_ids = torch.zeros(frame_count, dtype=torch.long)
    for word_index, (_, word_start, word_end, _) in enumerate(selected, start=1):
        relative_start = (max(start_seconds, word_start) - start_seconds) / crop_duration
        relative_end = (min(end_seconds, word_end) - start_seconds) / crop_duration
        frame_start = min(
            frame_count - 1,
            max(0, int(math.floor(relative_start * frame_count))),
        )
        frame_end = min(
            frame_count,
            max(frame_start + 1, int(math.ceil(relative_end * frame_count))),
        )
        frame_word_ids[frame_start:frame_end] = word_index
    exact_fraction = sum(1 for *_, exact in selected if exact) / len(selected)
    return (
        " ".join(word for word, *_ in selected),
        frame_word_ids,
        float(exact_fraction),
    )


def split_lexical_holdout_records(
    records: list[dict[str, Any]],
    *,
    validation_fraction: float = 0.05,
    validation_max_records: int | None = 128,
    seed: int = 5602,
    minimum_word_frequency: int = 2,
    maximum_word_frequency: int = 8,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Hold out complete word types, not merely different song crops.

    Every record containing a selected held-out word goes to validation. Thus
    the returned training records contain zero occurrences of those words and
    validation directly measures compositional generalization.
    """
    usable = _filter_training_records(records)
    if len(usable) < 2:
        return usable, [], []
    word_sets = [
        set(split_lyric_words(str(record.get("text", ""))))
        for record in usable
    ]
    document_frequency = Counter(
        word
        for words in word_sets
        for word in words
    )
    target_count = max(1, round(len(usable) * max(0.0, float(validation_fraction))))
    if validation_max_records is not None:
        target_count = min(target_count, max(1, int(validation_max_records)))
    candidate_words = [
        word
        for word, frequency in document_frequency.items()
        if (
            len(word) >= 3
            and int(minimum_word_frequency)
            <= frequency
            <= int(maximum_word_frequency)
        )
    ]
    candidate_words.sort(
        key=lambda word: hashlib.sha256(
            f"{seed}:{word}".encode("utf-8")
        ).digest()
    )

    validation_indices: set[int] = set()
    heldout_words: list[str] = []
    maximum_validation = min(
        len(usable) - 1,
        max(target_count, target_count * 2),
    )
    for word in candidate_words:
        matching = {
            index
            for index, words in enumerate(word_sets)
            if word in words
        }
        expanded = validation_indices | matching
        if not matching or len(expanded) > maximum_validation:
            continue
        validation_indices = expanded
        heldout_words.append(word)
        if len(validation_indices) >= target_count:
            break

    if not validation_indices or not heldout_words:
        training, validation = split_training_records(
            usable,
            validation_fraction=validation_fraction,
            validation_max_records=validation_max_records,
            seed=seed,
        )
        return training, validation, []
    training = [
        record
        for index, record in enumerate(usable)
        if index not in validation_indices
    ]
    validation = [
        record
        for index, record in enumerate(usable)
        if index in validation_indices
    ]
    return training, validation, heldout_words


def existing_lexical_holdout_records(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]] | None:
    """Reuse a materialized lexical split when every record declares one.

    Retrieval/refinement datasets persist more than song membership: their
    validation records also carry target-free fuzzy-span masks for the exact
    held-out words. Re-splitting the same JSONL inside diffusion training can
    put an exact-only record into validation and silently measure a different
    lexical task. A complete ``refiner_split`` contract is therefore
    authoritative; partial metadata still falls back to the ordinary stable
    splitter.
    """
    usable = _filter_training_records(records)
    if not usable:
        return None
    declared = [str(record.get("refiner_split") or "") for record in usable]
    if not all(value in {"train", "validation"} for value in declared):
        return None
    training = [
        record
        for record, split in zip(usable, declared)
        if split == "train"
    ]
    validation = [
        record
        for record, split in zip(usable, declared)
        if split == "validation"
    ]
    if not training or not validation:
        return None
    training_words = {
        word
        for record in training
        for word in split_lyric_words(str(record.get("text", "")))
    }
    validation_words = {
        word
        for record in validation
        for word in split_lyric_words(str(record.get("text", "")))
    }
    heldout_words = sorted(validation_words - training_words)
    if not heldout_words:
        return None
    return training, validation, heldout_words


def audit_lexical_holdout_split(
    training_records: list[dict[str, Any]],
    validation_records: list[dict[str, Any]],
    heldout_words: list[str],
) -> dict[str, Any]:
    """Prove lexical and song isolation for an open-vocabulary split."""
    heldout = set(heldout_words)
    training_words = {
        word
        for record in training_records
        for word in split_lyric_words(str(record.get("text", "")))
    }
    validation_words = {
        word
        for record in validation_records
        for word in split_lyric_words(str(record.get("text", "")))
    }

    def song_ids(records: list[dict[str, Any]]) -> set[str]:
        return {
            str(record.get("song_id") or record.get("id") or "")
            for record in records
        } - {""}

    training_songs = song_ids(training_records)
    validation_songs = song_ids(validation_records)
    lexical_overlap = sorted(heldout & training_words)
    missing_validation_words = sorted(heldout - validation_words)
    song_overlap = sorted(training_songs & validation_songs)
    applicable = bool(heldout)
    return {
        "applicable": applicable,
        "passed": bool(
            applicable
            and not lexical_overlap
            and not missing_validation_words
            and not song_overlap
            and bool(training_records)
            and bool(validation_records)
        ),
        "training_records": len(training_records),
        "validation_records": len(validation_records),
        "heldout_word_count": len(heldout),
        "training_heldout_overlap_count": len(lexical_overlap),
        "training_heldout_overlap": lexical_overlap,
        "validation_missing_heldout_count": len(missing_validation_words),
        "validation_missing_heldout": missing_validation_words,
        "song_overlap_count": len(song_overlap),
        "song_overlap": song_overlap,
        "training_unique_songs": len(training_songs),
        "validation_unique_songs": len(validation_songs),
    }


def replace_heldout_words(
    text: str,
    heldout_words: set[str],
    *,
    replacement: str = "la",
) -> tuple[str, bool]:
    """Replace held-out words without changing word count or frame indices."""
    words = split_lyric_words(text)
    changed = any(word in heldout_words for word in words)
    return (
        " ".join(
            replacement if word in heldout_words else word
            for word in words
        ),
        changed,
    )


def seed_refinement_mask(
    record: dict[str, Any],
    frames: int,
    *,
    fuzzy_context_frames: int = 2,
    exact_boundary_frames: int = 2,
    exact_boundary_weight: float = 0.25,
):
    """Build the editable latent-frame mask for a target-free seed.

    Fuzzy/novel words are fully editable, including a short boundary context.
    Exact donor words expose only small boundary ramps so the refiner can
    remove clicks or interruptions without rewriting an already intelligible
    pronunciation. Records without the explicit masked-refiner flag preserve
    the historical all-frame CFM behaviour.
    """
    if not bool(record.get("masked_seed_refinement", False)):
        return None
    torch, _, _, _ = _torch()
    frame_count = max(1, int(frames))
    retrievals = record.get("seed_retrievals") or []
    if not retrievals:
        raise ValueError(
            "masked_seed_refinement requires seed_retrievals metadata."
        )
    mask = torch.zeros(frame_count, dtype=torch.float32)
    fuzzy_context = max(0, int(fuzzy_context_frames))
    boundary_frames = max(0, int(exact_boundary_frames))
    boundary_weight = max(
        0.0,
        min(1.0, float(exact_boundary_weight)),
    )
    fuzzy_spans = 0
    for retrieval in retrievals:
        start = max(
            0,
            min(frame_count - 1, int(retrieval["start_frame"])),
        )
        end = max(
            start + 1,
            min(frame_count, int(retrieval["end_frame"])),
        )
        if not bool(retrieval.get("exact", False)):
            fuzzy_spans += 1
            mask[
                max(0, start - fuzzy_context) :
                min(frame_count, end + fuzzy_context)
            ] = 1.0
            continue
        if boundary_frames > 0 and boundary_weight > 0.0:
            left_end = min(end, start + boundary_frames)
            right_start = max(start, end - boundary_frames)
            mask[start:left_end] = torch.maximum(
                mask[start:left_end],
                torch.full_like(mask[start:left_end], boundary_weight),
            )
            mask[right_start:end] = torch.maximum(
                mask[right_start:end],
                torch.full_like(mask[right_start:end], boundary_weight),
            )
    if fuzzy_spans < 1:
        # Exact-only training records still supervise continuity, but every
        # lexical-heldout validation record must include a fully editable span.
        split = str(record.get("refiner_split") or "")
        if split == "validation":
            raise ValueError(
                "Masked validation record has no fuzzy/held-out word span."
            )
    if not bool((mask > 0).any()):
        raise ValueError("Masked seed refiner produced an empty frame mask.")
    return mask


class MusicDiffusionDataset:
    """PyTorch Dataset mapping structured Mel-spectrograms and text/style prompts.

    dataset_dir accepts either one path or a list of paths -- multiple
    independently-preprocessed datasets (e.g. different raw-data batches/parts)
    are combined into a single training set, each record's mel/style paths
    resolved against its own source directory (see _with_absolute_paths).
    """
    def __init__(
        self,
        dataset_dir: str | Path | list[str | Path],
        config: MusicDiffusionConfig,
        max_records: int | None = None,
        additional_records: list[dict[str, Any]] | None = None,
        records: list[dict[str, Any]] | None = None,
        deterministic_crop: bool = False,
        crop_seed: int = 5602,
    ):
        _, _, Dataset, _ = _torch()
        dataset_dirs = [Path(dataset_dir)] if isinstance(dataset_dir, (str, Path)) else [Path(d) for d in dataset_dir]
        if not dataset_dirs:
            raise ValueError("dataset_dir must contain at least one path")
        self.root = dataset_dirs[0]
        self.config = config
        self.deterministic_crop = bool(deterministic_crop)
        self.crop_seed = int(crop_seed)
        if records is not None:
            # Caller already resolved/filtered these (e.g. split_training_records'
            # validation half) -- records are assumed already absolute-pathed via
            # _with_absolute_paths, so skip re-reading from disk entirely.
            self.excluded_record_count = 0
            resolved_records = list(records)
        else:
            self.excluded_record_count = 0
            resolved_records = []
            for root in dataset_dirs:
                all_records = _read_records(root)
                usable = _filter_training_records(all_records)
                self.excluded_record_count += len(all_records) - len(usable)
                resolved_records.extend(_with_absolute_paths(root, record) for record in usable)
        self.records = resolved_records[:max_records] if max_records is not None else resolved_records
        if additional_records:
            self.records.extend(additional_records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        torch, _, _, _ = _torch()
        record = self.records[idx]
        
        # Load vocal Mel (target x1) and backing Mel (condition cond)
        # Fallback to single mel if separated paths are not present in dataset
        has_backing_condition = "vocal_mel_path" in record and "backing_mel_path" in record
        if has_backing_condition:
            vocal_path = self.root / record["vocal_mel_path"]
            backing_path = self.root / record["backing_mel_path"]
            vocal_mel = _load_mel(vocal_path)
            backing_mel = _load_mel(backing_path)
        else:
            # Fallback for old/smoke dataset
            mel_path = _record_path(self.root, record)
            vocal_mel = _load_mel(mel_path)
            backing_mel = torch.zeros_like(vocal_mel)

        seed_mel = None
        refinement_mask = None
        seed_path = record.get("seed_mel_path")
        if seed_path:
            seed_mel = _load_mel(self.root / seed_path)
            refinement_mask = seed_refinement_mask(
                record,
                int(seed_mel.shape[1]),
            )
            
        # Crop both stems at the same offset so the text/audio condition stays aligned
        # (also a cheap augmentation: different epochs see different windows of longer songs).
        # Validation needs a *stable* measurement across epochs/resumes, so
        # deterministic_crop derives crop_start from a seeded hash of the
        # record id instead of the process-global random.randint used for
        # training augmentation.
        crop_start = 0
        shared_frames = min(vocal_mel.shape[1], backing_mel.shape[1])
        if seed_mel is not None:
            shared_frames = min(shared_frames, seed_mel.shape[1])
        if shared_frames > self.config.frames_per_chunk:
            max_start = shared_frames - self.config.frames_per_chunk
            if self.deterministic_crop:
                record_key = str(record.get("id") or idx)
                digest = hashlib.sha256(f"{self.crop_seed}:{record_key}".encode("utf-8")).digest()
                crop_start = int.from_bytes(digest[:8], "big") % (max_start + 1)
            else:
                crop_start = random.randint(0, max_start)
            vocal_mel = vocal_mel[:, crop_start:crop_start + self.config.frames_per_chunk]
            backing_mel = backing_mel[:, crop_start:crop_start + self.config.frames_per_chunk]
            if seed_mel is not None:
                seed_mel = seed_mel[
                    :,
                    crop_start:crop_start + self.config.frames_per_chunk,
                ]
            if refinement_mask is not None:
                refinement_mask = refinement_mask[
                    crop_start:crop_start + self.config.frames_per_chunk
                ]
        else:
            vocal_mel = _fit_mel_frames(vocal_mel, self.config.frames_per_chunk)
            backing_mel = _fit_mel_frames(backing_mel, self.config.frames_per_chunk)
            if seed_mel is not None:
                seed_mel = _fit_mel_frames(
                    seed_mel,
                    self.config.frames_per_chunk,
                )
            if refinement_mask is not None:
                refinement_mask = _fit_mel_frames(
                    refinement_mask.unsqueeze(0),
                    self.config.frames_per_chunk,
                )[0]

        # Style anchor: a precomputed MuQ-MuLan audio embedding of the whole song (see
        # preprocess_raw_vietnamese.py), the same contrastive audio-style space the
        # real DiffRhythm2 teacher conditions on. This is a fixed per-song summary
        # (unlike vocal_mel/backing_mel above, it does not need cropping) -- falls
        # back to a zero vector for older/synthetic datasets that never computed one.
        style_path = record.get("style_embed_path")
        if style_path and (self.root / style_path).exists():
            style_anchor = _load_mel(self.root / style_path).float().view(-1)
        else:
            style_anchor = torch.zeros(STYLE_EMBED_DIM)

        # Only keep the lyric words that actually fall within this crop's time window,
        # when word/segment-level timestamps are available -- otherwise every crop of a
        # long song would be conditioned on the full-song transcript, most of which the
        # cropped audio doesn't contain.
        vocal_mel = normalize_mel(vocal_mel, self.config)
        backing_mel = normalize_mel(backing_mel, self.config) if has_backing_condition else torch.zeros_like(vocal_mel)
        if seed_mel is not None:
            seed_mel = normalize_mel(seed_mel, self.config)

        lyric_text = str(record["text"])
        segments = record.get("segments") or []
        crop_start_seconds = (
            crop_start * self.config.hop_length / self.config.sample_rate
        )
        crop_end_seconds = (
            crop_start_seconds
            + self.config.frames_per_chunk
            * self.config.hop_length
            / self.config.sample_rate
        )
        if segments:
            (
                lyric_text,
                lyric_frame_ids,
                lyric_alignment_exact_fraction,
            ) = lyric_conditioning_for_window(
                lyric_text,
                segments,
                crop_start_seconds,
                crop_end_seconds,
                self.config.frames_per_chunk,
            )
        else:
            lyric_frame_ids = uniform_frame_word_ids(
                [lyric_text],
                self.config.frames_per_chunk,
            )[0]
            lyric_alignment_exact_fraction = 0.0
        item = {
            "vocal_mel": vocal_mel,
            "backing_mel": backing_mel,
            "style_anchor": style_anchor,
            "text": lyric_text,
            "lyric_frame_ids": lyric_frame_ids,
            "lyric_alignment_exact_fraction": (
                lyric_alignment_exact_fraction
            ),
        }
        if seed_mel is not None:
            item["seed_mel"] = seed_mel
        if refinement_mask is not None:
            item["refinement_mask"] = refinement_mask
        return item

class DiffusionTrainer:
    """Trainer orchestrating optimization steps and gradient descent for the diffusion denoiser."""
    def __init__(
        self,
        model,
        config: MusicDiffusionConfig,
        optimizer,
        device: str = "cpu",
        scheduler=None,
        ema_decay: float = 0.999,
        lambda_vocal: float = 1.0,
        style_dropout_prob: float = 0.5,
        text_dropout_prob: float = 0.1,
        text_contrastive_weight: float = 0.0,
        text_contrastive_margin: float = 0.03,
        text_contrastive_prob: float = 0.5,
        text_sensitivity_weight: float = 0.0,
        text_sensitivity_target: float = 0.20,
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
        seed_full_frame_rewrite_probability: float = 0.0,
        seed_span_corruption_probability: float = 0.0,
        seed_span_corruption_fraction: float = 0.25,
        semantic_pretrain_only: bool = False,
    ):
        torch, _, _, _ = _torch()
        self.model = model
        self.config = config
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.ema_decay = float(ema_decay)
        # Weight of the auxiliary vocal-only prediction loss ("Mixed Pro", see
        # MicroDiT.vocal_proj_out's docstring). 0.0 disables it.
        self.lambda_vocal = lambda_vocal
        # CFG dropout rates and lyric-content-sensitivity loss weights (see
        # cfm_loss's docstring); both loss weights default to 0.0 (disabled).
        self.style_dropout_prob = float(style_dropout_prob)
        self.text_dropout_prob = float(text_dropout_prob)
        self.text_contrastive_weight = max(0.0, float(text_contrastive_weight))
        self.text_contrastive_margin = max(0.0, float(text_contrastive_margin))
        self.text_contrastive_prob = max(0.0, min(1.0, float(text_contrastive_prob)))
        self.text_sensitivity_weight = max(0.0, float(text_sensitivity_weight))
        self.text_sensitivity_target = max(0.0, float(text_sensitivity_target))
        self.lyric_semantic_weight = max(
            0.0,
            float(lyric_semantic_weight),
        )
        self.lyric_denoised_semantic_weight = max(
            0.0,
            float(lyric_denoised_semantic_weight),
        )
        self.lyric_phrase_semantic_weight = max(
            0.0,
            float(lyric_phrase_semantic_weight),
        )
        self.lyric_phrase_denoised_semantic_weight = max(
            0.0,
            float(lyric_phrase_denoised_semantic_weight),
        )
        self.lyric_semantic_temperature = max(
            1e-4,
            float(lyric_semantic_temperature),
        )
        self.lyric_unit_semantic_weight = max(
            0.0,
            float(lyric_unit_semantic_weight),
        )
        self.lyric_unit_denoised_semantic_weight = max(
            0.0,
            float(lyric_unit_denoised_semantic_weight),
        )
        self.self_rollout_consistency_weight = max(
            0.0,
            float(self_rollout_consistency_weight),
        )
        self.self_rollout_consistency_probability = max(
            0.0,
            min(1.0, float(self_rollout_consistency_probability)),
        )
        self.self_rollout_step_size = max(
            1e-4,
            min(0.5, float(self_rollout_step_size)),
        )
        self.self_rollout_solver_steps = max(
            0,
            int(self_rollout_solver_steps),
        )
        self.early_timestep_fraction = max(
            0.0,
            min(1.0, float(early_timestep_fraction)),
        )
        self.early_timestep_max = max(
            1e-4,
            min(1.0, float(early_timestep_max)),
        )
        self.seed_full_frame_rewrite_probability = max(
            0.0,
            min(1.0, float(seed_full_frame_rewrite_probability)),
        )
        self.seed_span_corruption_probability = max(
            0.0,
            min(1.0, float(seed_span_corruption_probability)),
        )
        self.seed_span_corruption_fraction = max(
            0.0,
            min(1.0, float(seed_span_corruption_fraction)),
        )
        self.semantic_pretrain_only = bool(
            semantic_pretrain_only
        )
        # The exact-word InfoNCE objectives pool many word spans and backprop
        # through the denoiser's clean estimate. On T4, FP16 gradient scaling
        # can overflow every step even while the FP32 loss itself is finite;
        # GradScaler then silently skips all optimizer updates. Keep these
        # semantically supervised pilots in FP32. Plain acoustic CFM retains
        # the established AMP path.
        semantic_supervision = (
            self.lyric_semantic_weight > 0.0
            or self.lyric_denoised_semantic_weight > 0.0
            or self.lyric_phrase_semantic_weight > 0.0
            or self.lyric_phrase_denoised_semantic_weight > 0.0
            or self.lyric_unit_semantic_weight > 0.0
            or self.lyric_unit_denoised_semantic_weight > 0.0
        )
        self.use_amp = (
            str(device).startswith("cuda")
            and not semantic_supervision
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.optimizer_steps_completed = 0
        self.optimizer_steps_skipped = 0
        self.consecutive_nonfinite_gradients = 0
        self.ema_parameters = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    def _update_ema(self) -> None:
        torch, _, _, _ = _torch()
        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
                if (
                    self.semantic_pretrain_only
                    and not name.startswith(
                        (
                            "open_vocabulary_lyric.",
                            "lyric_semantic_",
                        )
                    )
                ):
                    continue
                if name in self.ema_parameters:
                    self.ema_parameters[name].lerp_(parameter.detach(), 1.0 - self.ema_decay)

    def apply_ema_weights(self) -> None:
        torch, _, _, _ = _torch()
        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
                if name in self.ema_parameters:
                    parameter.copy_(self.ema_parameters[name])

    def load_ema_state(self, state: dict[str, Any]) -> None:
        """Restore EMA tensors when a preempted training session resumes."""
        for name, value in state.items():
            if name in self.ema_parameters:
                self.ema_parameters[name] = value.detach().to(self.device).clone()

    def reset_semantic_ema(self) -> None:
        """Warm-start new semantic EMA tensors from their trained raw weights."""
        prefixes = (
            "open_vocabulary_lyric.",
            "lyric_semantic_",
        )
        for name, parameter in self.model.named_parameters():
            if (
                name in self.ema_parameters
                and name.startswith(prefixes)
            ):
                self.ema_parameters[name] = (
                    parameter.detach().to(self.device).clone()
                )

    def evaluate_ground_truth(self, dataloader, *, seed: int = 5602) -> float:
        """Measure a deterministic held-out loss using the EMA weights."""
        torch, _, _, _ = _torch()
        from ..models.cfm_flow import cfm_loss

        was_training = self.model.training
        raw_parameters = {
            name: parameter.detach().clone()
            for name, parameter in self.model.named_parameters()
            if name in self.ema_parameters
        }
        cuda_devices = [torch.cuda.current_device()] if str(self.device).startswith("cuda") else []
        losses: list[float] = []
        try:
            with torch.no_grad():
                for name, parameter in self.model.named_parameters():
                    if name in self.ema_parameters:
                        parameter.copy_(self.ema_parameters[name])
            self.model.eval()
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(int(seed))
                if cuda_devices:
                    torch.cuda.manual_seed_all(int(seed))
                with torch.no_grad():
                    for batch in dataloader:
                        vocal_mel = batch["vocal_mel"].to(self.device).transpose(1, 2)
                        backing_mel = batch["backing_mel"].to(self.device).transpose(1, 2)
                        style_anchor = batch["style_anchor"].to(self.device)
                        lyric_frame_ids = batch.get("lyric_frame_ids")
                        if lyric_frame_ids is not None:
                            lyric_frame_ids = lyric_frame_ids.to(self.device)
                        loss, _, _ = cfm_loss(
                            self.model,
                            vocal_mel,
                            backing_mel,
                            style_anchor,
                            batch["text"],
                            self.config,
                            style_dropout_prob=0.0,
                            text_dropout_prob=0.0,
                            lambda_vocal=self.lambda_vocal,
                            lyric_frame_ids=lyric_frame_ids,
                            source_mel=(
                                batch["seed_mel"]
                                .to(self.device)
                                .transpose(1, 2)
                                if "seed_mel" in batch
                                else None
                            ),
                            refinement_mask=(
                                batch["refinement_mask"].to(self.device)
                                if "refinement_mask" in batch
                                else None
                            ),
                            seed_full_frame_rewrite_probability=(
                                self.seed_full_frame_rewrite_probability
                            ),
                            seed_span_corruption_probability=(
                                self.seed_span_corruption_probability
                            ),
                            seed_span_corruption_fraction=(
                                self.seed_span_corruption_fraction
                            ),
                        )
                        losses.append(float(loss.detach().cpu()))
        finally:
            with torch.no_grad():
                for name, parameter in self.model.named_parameters():
                    if name in raw_parameters:
                        parameter.copy_(raw_parameters[name])
            self.model.train(was_training)
        if not losses:
            raise ValueError("Validation dataset produced no batches.")
        return sum(losses) / len(losses)

    def evaluate_text_sensitivity(self, dataloader, *, seed: int = 5602) -> float:
        """Measure whether EMA predictions change for a *different* lyric."""
        torch, _, _, _ = _torch()

        was_training = self.model.training
        raw_parameters = {
            name: parameter.detach().clone()
            for name, parameter in self.model.named_parameters()
            if name in self.ema_parameters
        }
        try:
            with torch.no_grad():
                for name, parameter in self.model.named_parameters():
                    if name in self.ema_parameters:
                        parameter.copy_(self.ema_parameters[name])
            self.model.eval()
            batch = next(iter(dataloader))
            model_dtype = next(self.model.parameters()).dtype
            clean_mel = (
                batch["vocal_mel"]
                .to(self.device, dtype=model_dtype)
                .transpose(1, 2)
            )
            generator = torch.Generator(device=self.device).manual_seed(int(seed))
            if "seed_mel" in batch:
                noise = (
                    batch["seed_mel"]
                    .to(self.device, dtype=model_dtype)
                    .transpose(1, 2)
                )
            else:
                noise = torch.randn(
                    clean_mel.shape,
                    generator=generator,
                    device=self.device,
                    dtype=model_dtype,
                )
            timestep = torch.full(
                (clean_mel.shape[0],),
                0.5,
                device=self.device,
                dtype=model_dtype,
            )
            noisy = 0.5 * noise + 0.5 * clean_mel
            zero_style = torch.zeros(
                (clean_mel.shape[0], int(getattr(self.model, "style_dim", STYLE_EMBED_DIM))),
                device=self.device,
                dtype=model_dtype,
            )
            lyric_frame_ids = batch.get("lyric_frame_ids")
            if lyric_frame_ids is not None:
                lyric_frame_ids = lyric_frame_ids.to(self.device)
            with torch.no_grad():
                from ..models.cfm_flow import build_mismatched_text_conditioning

                (
                    mismatched_texts,
                    content_mask_flags,
                    mismatched_source_indices,
                ) = build_mismatched_text_conditioning(batch["text"])
                content_mask = torch.tensor(content_mask_flags, dtype=torch.bool, device=self.device)
                if not bool(content_mask.any()):
                    return 0.0
                conditioned_kwargs = {
                    "x": noisy,
                    "texts": batch["text"],
                    "timestep": timestep,
                    "style_prompt": zero_style,
                }
                mismatched_kwargs = {
                    "x": noisy,
                    "texts": mismatched_texts,
                    "timestep": timestep,
                    "style_prompt": zero_style,
                }
                if lyric_frame_ids is not None:
                    conditioned_kwargs["lyric_frame_ids"] = lyric_frame_ids
                    source_indices = torch.tensor(
                        mismatched_source_indices,
                        dtype=torch.long,
                        device=self.device,
                    )
                    mismatched_kwargs["lyric_frame_ids"] = (
                        lyric_frame_ids.index_select(0, source_indices).masked_fill(
                            ~content_mask[:, None],
                            0,
                        )
                    )
                conditioned = self.model(**conditioned_kwargs)
                mismatched = self.model(**mismatched_kwargs)
            difference = ((conditioned - mismatched).square().mean(dim=(1, 2)).sqrt())[content_mask].mean()
            baseline = (conditioned.square().mean(dim=(1, 2)).sqrt())[content_mask].mean().clamp_min(1e-8)
            return float((difference / baseline).detach().cpu())
        finally:
            with torch.no_grad():
                for name, parameter in self.model.named_parameters():
                    if name in raw_parameters:
                        parameter.copy_(raw_parameters[name])
            self.model.train(was_training)

    def evaluate_lexical_sensitivity(
        self,
        dataloader,
        heldout_words: list[str],
        *,
        seed: int = 5602,
    ) -> float:
        """Measure response specifically to words absent from training."""
        torch, _, _, _ = _torch()
        heldout = set(heldout_words)
        if not heldout:
            return 0.0
        was_training = self.model.training
        raw_parameters = {
            name: parameter.detach().clone()
            for name, parameter in self.model.named_parameters()
            if name in self.ema_parameters
        }
        try:
            with torch.no_grad():
                for name, parameter in self.model.named_parameters():
                    if name in self.ema_parameters:
                        parameter.copy_(self.ema_parameters[name])
            self.model.eval()
            batch = next(iter(dataloader))
            mutated: list[str] = []
            changed_flags: list[bool] = []
            for text in batch["text"]:
                replaced, changed = replace_heldout_words(text, heldout)
                mutated.append(replaced)
                changed_flags.append(changed)
            changed_mask = torch.tensor(
                changed_flags,
                dtype=torch.bool,
                device=self.device,
            )
            if not bool(changed_mask.any()):
                return 0.0
            model_dtype = next(self.model.parameters()).dtype
            clean_mel = (
                batch["vocal_mel"]
                .to(self.device, dtype=model_dtype)
                .transpose(1, 2)
            )
            generator = torch.Generator(device=self.device).manual_seed(int(seed))
            if "seed_mel" in batch:
                noise = (
                    batch["seed_mel"]
                    .to(self.device, dtype=model_dtype)
                    .transpose(1, 2)
                )
            else:
                noise = torch.randn(
                    clean_mel.shape,
                    generator=generator,
                    device=self.device,
                    dtype=model_dtype,
                )
            timestep = torch.full(
                (clean_mel.shape[0],),
                0.5,
                device=self.device,
                dtype=model_dtype,
            )
            noisy = 0.5 * noise + 0.5 * clean_mel
            zero_style = torch.zeros(
                (
                    clean_mel.shape[0],
                    int(getattr(self.model, "style_dim", STYLE_EMBED_DIM)),
                ),
                device=self.device,
                dtype=model_dtype,
            )
            frame_ids = batch.get("lyric_frame_ids")
            kwargs = {}
            if frame_ids is not None:
                kwargs["lyric_frame_ids"] = frame_ids.to(self.device)
            with torch.no_grad():
                original = self.model(
                    x=noisy,
                    texts=batch["text"],
                    timestep=timestep,
                    style_prompt=zero_style,
                    **kwargs,
                )
                counterfactual = self.model(
                    x=noisy,
                    texts=mutated,
                    timestep=timestep,
                    style_prompt=zero_style,
                    **kwargs,
                )
            difference = (
                (original - counterfactual)
                .square()
                .mean(dim=(1, 2))
                .sqrt()[changed_mask]
                .mean()
            )
            baseline = (
                original.square()
                .mean(dim=(1, 2))
                .sqrt()[changed_mask]
                .mean()
                .clamp_min(1e-8)
            )
            return float((difference / baseline).detach().cpu())
        finally:
            with torch.no_grad():
                for name, parameter in self.model.named_parameters():
                    if name in raw_parameters:
                        parameter.copy_(raw_parameters[name])
            self.model.train(was_training)

    def evaluate_lyric_semantics(
        self,
        dataloader,
        *,
        heldout_words: list[str] | None = None,
    ) -> dict[str, float | int]:
        """Measure exact-span audio/word retrieval with EMA model weights."""
        torch, _, _, _ = _torch()
        from ..models.cfm_flow import lyric_semantic_alignment_objective
        from ..models.text_to_music_diffusion import reconstruct_full_mix

        was_training = self.model.training
        raw_parameters = {
            name: parameter.detach().clone()
            for name, parameter in self.model.named_parameters()
            if name in self.ema_parameters
        }
        weighted_accuracy = 0.0
        weighted_cosine = 0.0
        weighted_margin = 0.0
        weighted_loss = 0.0
        occurrences = 0
        distinct_words: set[str] = set()
        include_words = (
            set(heldout_words)
            if heldout_words
            else None
        )
        try:
            with torch.no_grad():
                for name, parameter in self.model.named_parameters():
                    if name in self.ema_parameters:
                        parameter.copy_(self.ema_parameters[name])
            self.model.eval()
            with torch.no_grad():
                for batch in dataloader:
                    vocal_mel = batch["vocal_mel"].to(self.device).transpose(1, 2)
                    backing_mel = batch["backing_mel"].to(self.device).transpose(1, 2)
                    target = (
                        vocal_mel
                        if self.config.latent_mode
                        else reconstruct_full_mix(
                            vocal_mel,
                            backing_mel,
                            self.config,
                        )
                    )
                    frame_ids = batch.get("lyric_frame_ids")
                    if frame_ids is None:
                        continue
                    result = lyric_semantic_alignment_objective(
                        self.model,
                        target,
                        batch["text"],
                        frame_ids.to(self.device),
                        temperature=self.lyric_semantic_temperature,
                        include_words=include_words,
                    )
                    count = int(result["occurrences"])
                    if count <= 0:
                        continue
                    occurrences += count
                    weighted_loss += float(result["loss"].detach().cpu()) * count
                    weighted_accuracy += float(result["accuracy"].detach().cpu()) * count
                    weighted_cosine += float(
                        result["positive_cosine"].detach().cpu()
                    ) * count
                    weighted_margin += float(result["margin"].detach().cpu()) * count
                    for text in batch["text"]:
                        distinct_words.update(
                            word
                            for word in split_lyric_words(text)
                            if include_words is None or word in include_words
                        )
        finally:
            with torch.no_grad():
                for name, parameter in self.model.named_parameters():
                    if name in raw_parameters:
                        parameter.copy_(raw_parameters[name])
            self.model.train(was_training)
        denominator = max(1, occurrences)
        return {
            "loss": weighted_loss / denominator,
            "accuracy": weighted_accuracy / denominator,
            "positive_cosine": weighted_cosine / denominator,
            "margin": weighted_margin / denominator,
            "occurrences": occurrences,
            "distinct_words": len(distinct_words),
        }

    def evaluate_denoised_lyric_semantics(
        self,
        dataloader,
        *,
        heldout_words: list[str] | None = None,
        timestep: float = 0.0,
        seed: int = 5602,
    ) -> dict[str, float | int]:
        """Score word retrieval from the EMA denoiser's clean estimate.

        Clean-target retrieval alone can improve while generation remains
        disconnected from that semantic space. This deterministic pure-noise
        probe sends the same inference starting distribution through the
        actual velocity head and scores the resulting one-step clean estimate
        on exact word spans, without leaking any clean target into the input.
        """
        torch, _, _, _ = _torch()
        from ..models.cfm_flow import lyric_semantic_alignment_objective
        from ..models.text_to_music_diffusion import reconstruct_full_mix

        resolved_timestep = max(0.0, min(1.0, float(timestep)))
        was_training = self.model.training
        raw_parameters = {
            name: parameter.detach().clone()
            for name, parameter in self.model.named_parameters()
            if name in self.ema_parameters
        }
        totals = {
            "loss": 0.0,
            "accuracy": 0.0,
            "positive_cosine": 0.0,
            "margin": 0.0,
        }
        occurrences = 0
        distinct_words: set[str] = set()
        include_words = set(heldout_words) if heldout_words else None
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        try:
            with torch.no_grad():
                for name, parameter in self.model.named_parameters():
                    if name in self.ema_parameters:
                        parameter.copy_(self.ema_parameters[name])
            self.model.eval()
            model_dtype = next(self.model.parameters()).dtype
            with torch.no_grad():
                for batch in dataloader:
                    vocal_mel = batch["vocal_mel"].to(
                        self.device,
                        dtype=model_dtype,
                    ).transpose(1, 2)
                    backing_mel = batch["backing_mel"].to(
                        self.device,
                        dtype=model_dtype,
                    ).transpose(1, 2)
                    target = (
                        vocal_mel
                        if self.config.latent_mode
                        else reconstruct_full_mix(
                            vocal_mel,
                            backing_mel,
                            self.config,
                        )
                    )
                    frame_ids = batch.get("lyric_frame_ids")
                    if frame_ids is None:
                        continue
                    frame_ids = frame_ids.to(self.device)
                    if "seed_mel" in batch:
                        noise = (
                            batch["seed_mel"]
                            .to(self.device, dtype=model_dtype)
                            .transpose(1, 2)
                        )
                    else:
                        noise = torch.randn(
                            target.shape,
                            generator=generator,
                            device=self.device,
                            dtype=model_dtype,
                        )
                    timestep_tensor = torch.full(
                        (target.shape[0],),
                        resolved_timestep,
                        device=self.device,
                        dtype=model_dtype,
                    )
                    noisy = (
                        (1.0 - resolved_timestep) * noise
                        + resolved_timestep * target
                    )
                    zero_style = torch.zeros(
                        (
                            target.shape[0],
                            int(
                                getattr(
                                    self.model,
                                    "style_dim",
                                    STYLE_EMBED_DIM,
                                )
                            ),
                        ),
                        device=self.device,
                        dtype=model_dtype,
                    )
                    predicted_velocity = self.model(
                        x=noisy,
                        texts=batch["text"],
                        timestep=timestep_tensor,
                        style_prompt=zero_style,
                        lyric_frame_ids=frame_ids,
                    )
                    predicted_clean = (
                        noisy
                        + (1.0 - resolved_timestep)
                        * predicted_velocity
                    )
                    result = lyric_semantic_alignment_objective(
                        self.model,
                        predicted_clean,
                        batch["text"],
                        frame_ids,
                        temperature=self.lyric_semantic_temperature,
                        include_words=include_words,
                    )
                    count = int(result["occurrences"])
                    if count <= 0:
                        continue
                    occurrences += count
                    for key in totals:
                        totals[key] += (
                            float(result[key].detach().cpu()) * count
                        )
                    for text in batch["text"]:
                        distinct_words.update(
                            word
                            for word in split_lyric_words(text)
                            if include_words is None or word in include_words
                        )
        finally:
            with torch.no_grad():
                for name, parameter in self.model.named_parameters():
                    if name in raw_parameters:
                        parameter.copy_(raw_parameters[name])
            self.model.train(was_training)
        denominator = max(1, occurrences)
        return {
            key: value / denominator
            for key, value in totals.items()
        } | {
            "occurrences": occurrences,
            "distinct_words": len(distinct_words),
            "timestep": resolved_timestep,
        }

    def evaluate_lyric_units(
        self,
        dataloader,
        *,
        heldout_words: list[str] | None = None,
    ) -> dict[str, float | int]:
        """Measure reusable unit retrieval inside lexical-heldout words."""
        torch, _, _, _ = _torch()
        from ..models.cfm_flow import lyric_unit_alignment_objective
        from ..models.text_to_music_diffusion import reconstruct_full_mix

        was_training = self.model.training
        raw_parameters = {
            name: parameter.detach().clone()
            for name, parameter in self.model.named_parameters()
            if name in self.ema_parameters
        }
        totals = {
            "loss": 0.0,
            "accuracy": 0.0,
            "positive_cosine": 0.0,
            "margin": 0.0,
        }
        occurrences = 0
        distinct_units = 0
        include_words = set(heldout_words) if heldout_words else None
        try:
            with torch.no_grad():
                for name, parameter in self.model.named_parameters():
                    if name in self.ema_parameters:
                        parameter.copy_(self.ema_parameters[name])
            self.model.eval()
            with torch.no_grad():
                for batch in dataloader:
                    vocal_mel = batch["vocal_mel"].to(
                        self.device
                    ).transpose(1, 2)
                    backing_mel = batch["backing_mel"].to(
                        self.device
                    ).transpose(1, 2)
                    target = (
                        vocal_mel
                        if self.config.latent_mode
                        else reconstruct_full_mix(
                            vocal_mel,
                            backing_mel,
                            self.config,
                        )
                    )
                    frame_ids = batch.get("lyric_frame_ids")
                    if frame_ids is None:
                        continue
                    result = lyric_unit_alignment_objective(
                        self.model,
                        target,
                        batch["text"],
                        frame_ids.to(self.device),
                        temperature=self.lyric_semantic_temperature,
                        include_words=include_words,
                    )
                    count = int(result["occurrences"])
                    if count <= 0:
                        continue
                    occurrences += count
                    distinct_units = max(
                        distinct_units,
                        int(result["distinct_units"]),
                    )
                    for key in totals:
                        totals[key] += (
                            float(result[key].detach().cpu()) * count
                        )
        finally:
            with torch.no_grad():
                for name, parameter in self.model.named_parameters():
                    if name in raw_parameters:
                        parameter.copy_(raw_parameters[name])
            self.model.train(was_training)
        denominator = max(1, occurrences)
        return {
            key: value / denominator
            for key, value in totals.items()
        } | {
            "occurrences": occurrences,
            "distinct_units": distinct_units,
        }

    def evaluate_denoised_lyric_units(
        self,
        dataloader,
        *,
        heldout_words: list[str] | None = None,
        timestep: float = 0.0,
        seed: int = 5602,
    ) -> dict[str, float | int]:
        """Score exact reusable units in an EMA predicted-clean latent."""
        torch, _, _, _ = _torch()
        from ..models.cfm_flow import lyric_unit_alignment_objective
        from ..models.text_to_music_diffusion import reconstruct_full_mix

        resolved_timestep = max(0.0, min(1.0, float(timestep)))
        was_training = self.model.training
        raw_parameters = {
            name: parameter.detach().clone()
            for name, parameter in self.model.named_parameters()
            if name in self.ema_parameters
        }
        totals = {
            "loss": 0.0,
            "accuracy": 0.0,
            "positive_cosine": 0.0,
            "margin": 0.0,
        }
        occurrences = 0
        distinct_units = 0
        include_words = set(heldout_words) if heldout_words else None
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        try:
            with torch.no_grad():
                for name, parameter in self.model.named_parameters():
                    if name in self.ema_parameters:
                        parameter.copy_(self.ema_parameters[name])
            self.model.eval()
            model_dtype = next(self.model.parameters()).dtype
            with torch.no_grad():
                for batch in dataloader:
                    vocal_mel = batch["vocal_mel"].to(
                        self.device,
                        dtype=model_dtype,
                    ).transpose(1, 2)
                    backing_mel = batch["backing_mel"].to(
                        self.device,
                        dtype=model_dtype,
                    ).transpose(1, 2)
                    target = (
                        vocal_mel
                        if self.config.latent_mode
                        else reconstruct_full_mix(
                            vocal_mel,
                            backing_mel,
                            self.config,
                        )
                    )
                    frame_ids = batch.get("lyric_frame_ids")
                    if frame_ids is None:
                        continue
                    frame_ids = frame_ids.to(self.device)
                    if "seed_mel" in batch:
                        noise = batch["seed_mel"].to(
                            self.device,
                            dtype=model_dtype,
                        ).transpose(1, 2)
                    else:
                        noise = torch.randn(
                            target.shape,
                            generator=generator,
                            device=self.device,
                            dtype=model_dtype,
                        )
                    timestep_tensor = torch.full(
                        (target.shape[0],),
                        resolved_timestep,
                        device=self.device,
                        dtype=model_dtype,
                    )
                    noisy = (
                        (1.0 - resolved_timestep) * noise
                        + resolved_timestep * target
                    )
                    zero_style = torch.zeros(
                        (
                            target.shape[0],
                            int(
                                getattr(
                                    self.model,
                                    "style_dim",
                                    STYLE_EMBED_DIM,
                                )
                            ),
                        ),
                        device=self.device,
                        dtype=model_dtype,
                    )
                    predicted_velocity = self.model(
                        x=noisy,
                        texts=batch["text"],
                        timestep=timestep_tensor,
                        style_prompt=zero_style,
                        lyric_frame_ids=frame_ids,
                    )
                    predicted_clean = (
                        noisy
                        + (1.0 - resolved_timestep) * predicted_velocity
                    )
                    result = lyric_unit_alignment_objective(
                        self.model,
                        predicted_clean,
                        batch["text"],
                        frame_ids,
                        temperature=self.lyric_semantic_temperature,
                        include_words=include_words,
                    )
                    count = int(result["occurrences"])
                    if count <= 0:
                        continue
                    occurrences += count
                    distinct_units = max(
                        distinct_units,
                        int(result["distinct_units"]),
                    )
                    for key in totals:
                        totals[key] += (
                            float(result[key].detach().cpu()) * count
                        )
        finally:
            with torch.no_grad():
                for name, parameter in self.model.named_parameters():
                    if name in raw_parameters:
                        parameter.copy_(raw_parameters[name])
            self.model.train(was_training)
        denominator = max(1, occurrences)
        return {
            key: value / denominator
            for key, value in totals.items()
        } | {
            "occurrences": occurrences,
            "distinct_units": distinct_units,
            "timestep": resolved_timestep,
        }

    def train_epoch(
        self,
        dataloader,
        *,
        epoch_index: int = 0,
        total_epochs: int = 1,
        start_batch: int = 0,
        log_every_steps: int = 10,
        on_step=None,
    ) -> list[float]:
        torch, _, _, _ = _torch()
        self.model.train()
        epoch_losses = []
        total_batches = len(dataloader)
        for batch_index, batch in enumerate(dataloader):
            if batch_index < start_batch:
                continue
            vocal_mel = batch["vocal_mel"].to(self.device)
            backing_mel = batch["backing_mel"].to(self.device)
            seed_mel = batch.get("seed_mel")
            if seed_mel is not None:
                seed_mel = seed_mel.to(self.device)
            refinement_mask = batch.get("refinement_mask")
            if refinement_mask is not None:
                refinement_mask = refinement_mask.to(self.device)
            style_anchor = batch["style_anchor"].to(self.device)
            texts = batch["text"]
            lyric_frame_ids = batch.get("lyric_frame_ids")
            if lyric_frame_ids is not None:
                lyric_frame_ids = lyric_frame_ids.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)

            # Transpose mels from (batch, n_mels, seq_len) to (batch, seq_len, n_mels) for DiT.
            # style_anchor is already a flat (batch, 512) MuQ-MuLan embedding, not
            # mel-shaped, so it needs no transpose.
            vocal_mel_t = vocal_mel.transpose(1, 2)
            backing_mel_t = backing_mel.transpose(1, 2)
            from ..models.cfm_flow import cfm_loss
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
                if self.semantic_pretrain_only:
                    from ..models.cfm_flow import (
                        lyric_semantic_alignment_objective,
                        lyric_unit_alignment_objective,
                    )
                    if lyric_frame_ids is None:
                        raise ValueError(
                            "Semantic pretraining requires exact "
                            "lyric_frame_ids."
                        )
                    target = (
                        vocal_mel_t
                        if self.config.latent_mode
                        else reconstruct_full_mix(
                            vocal_mel_t,
                            backing_mel_t,
                            self.config,
                        )
                    )
                    loss = target.sum() * 0.0
                    if self.lyric_semantic_weight > 0.0:
                        word_result = (
                            lyric_semantic_alignment_objective(
                                self.model,
                                target,
                                texts,
                                lyric_frame_ids,
                                temperature=(
                                    self.lyric_semantic_temperature
                                ),
                            )
                        )
                        loss = (
                            loss
                            + self.lyric_semantic_weight
                            * word_result["loss"]
                        )
                    if self.lyric_unit_semantic_weight > 0.0:
                        unit_result = lyric_unit_alignment_objective(
                            self.model,
                            target,
                            texts,
                            lyric_frame_ids,
                            temperature=(
                                self.lyric_semantic_temperature
                            ),
                        )
                        loss = (
                            loss
                            + self.lyric_unit_semantic_weight
                            * unit_result["loss"]
                        )
                    if (
                        self.lyric_semantic_weight <= 0.0
                        and self.lyric_unit_semantic_weight <= 0.0
                    ):
                        raise ValueError(
                            "Semantic pretraining requires a positive "
                            "semantic loss weight."
                        )
                    loss_gt = loss
                    loss_vocal_aux = None
                else:
                    loss, loss_gt, loss_vocal_aux = cfm_loss(
                        self.model, vocal_mel_t, backing_mel_t, style_anchor, texts, self.config,
                        lambda_vocal=self.lambda_vocal,
                        style_dropout_prob=self.style_dropout_prob,
                        text_dropout_prob=self.text_dropout_prob,
                        text_contrastive_weight=self.text_contrastive_weight,
                        text_contrastive_margin=self.text_contrastive_margin,
                        text_contrastive_prob=self.text_contrastive_prob,
                        text_sensitivity_weight=self.text_sensitivity_weight,
                        text_sensitivity_target=self.text_sensitivity_target,
                        lyric_frame_ids=lyric_frame_ids,
                        lyric_semantic_weight=self.lyric_semantic_weight,
                        lyric_denoised_semantic_weight=(
                            self.lyric_denoised_semantic_weight
                        ),
                        lyric_phrase_semantic_weight=(
                            self.lyric_phrase_semantic_weight
                        ),
                        lyric_phrase_denoised_semantic_weight=(
                            self.lyric_phrase_denoised_semantic_weight
                        ),
                        lyric_semantic_temperature=self.lyric_semantic_temperature,
                        lyric_unit_semantic_weight=(
                            self.lyric_unit_semantic_weight
                        ),
                        lyric_unit_denoised_semantic_weight=(
                            self.lyric_unit_denoised_semantic_weight
                        ),
                        self_rollout_consistency_weight=(
                            self.self_rollout_consistency_weight
                        ),
                        self_rollout_consistency_probability=(
                            self.self_rollout_consistency_probability
                        ),
                        self_rollout_step_size=(
                            self.self_rollout_step_size
                        ),
                        self_rollout_solver_steps=(
                            self.self_rollout_solver_steps
                        ),
                        early_timestep_fraction=(
                            self.early_timestep_fraction
                        ),
                        early_timestep_max=self.early_timestep_max,
                        source_mel=(
                            seed_mel.transpose(1, 2)
                            if seed_mel is not None
                            else None
                        ),
                        refinement_mask=refinement_mask,
                        seed_full_frame_rewrite_probability=(
                            self.seed_full_frame_rewrite_probability
                        ),
                        seed_span_corruption_probability=(
                            self.seed_span_corruption_probability
                        ),
                        seed_span_corruption_fraction=(
                            self.seed_span_corruption_fraction
                        ),
                    )

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()),
                1.0,
            )
            finite_gradient = bool(torch.isfinite(gradient_norm))
            optimizer_updated = False
            if finite_gradient:
                scale_before = float(self.scaler.get_scale())
                self.scaler.step(self.optimizer)
                self.scaler.update()
                scale_after = float(self.scaler.get_scale())
                optimizer_updated = (
                    not self.use_amp
                    or scale_after >= scale_before
                )
            else:
                # Enabled GradScaler must observe the inf flags so it can
                # back off. With scaling disabled, simply skip the corrupt
                # update; applying it would poison every later checkpoint.
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                self.scaler.update()
            if optimizer_updated:
                self.optimizer_steps_completed += 1
                self.consecutive_nonfinite_gradients = 0
                if self.scheduler is not None:
                    self.scheduler.step()
                self._update_ema()
            else:
                self.optimizer_steps_skipped += 1
                self.consecutive_nonfinite_gradients += 1
                print(
                    "optimizer_step_skipped "
                    f"gradient_norm={float(gradient_norm.detach().cpu())} "
                    f"consecutive={self.consecutive_nonfinite_gradients}",
                    flush=True,
                )
                if self.consecutive_nonfinite_gradients >= 8:
                    raise FloatingPointError(
                        "Eight consecutive non-finite gradient steps; "
                        "refusing to save an unchanged/corrupt model."
                    )
            loss_value = float(loss.detach().cpu())
            loss_record = {
                "loss": loss_value,
                "loss_gt": float(loss_gt.detach().cpu()),
                "loss_velocity": None,
                "loss_vocal_aux": float(loss_vocal_aux.detach().cpu()) if loss_vocal_aux is not None else None,
                "gradient_norm": float(
                    gradient_norm.detach().cpu()
                ),
                "optimizer_updated": optimizer_updated,
                "optimizer_steps_completed": (
                    self.optimizer_steps_completed
                ),
                "optimizer_steps_skipped": (
                    self.optimizer_steps_skipped
                ),
            }
            epoch_losses.append(loss_record)
            completed_batches = batch_index + 1
            should_log = (
                completed_batches == total_batches
                or completed_batches % max(1, int(log_every_steps)) == 0
            )
            if should_log:
                print(
                    f"epoch={epoch_index + 1}/{total_epochs} "
                    f"batch={completed_batches}/{total_batches} "
                    f"loss={loss_value:.6f}",
                    flush=True,
                )
            if on_step is not None:
                on_step(completed_batches, loss_record, should_log)
        return epoch_losses

def create_random_dataset(output_dir: str | Path, *, count: int = 16, frames: int = 128, seed: int = 5602, config: MusicDiffusionConfig | None = None, target_bytes: int | None = None, payload_frames: int = 2048) -> dict[str, Any]:
    config = config or MusicDiffusionConfig(frames_per_chunk=frames)
    root = Path(output_dir)
    mel_dir = root / "mels"
    mel_dir.mkdir(parents=True, exist_ok=True)
    if target_bytes:
        bytes_per_sample = config.latent_dim * max(frames, payload_frames) * 4
        count = max(int(count), math.ceil(int(target_bytes) / max(1, bytes_per_sample)))
    records = []
    random.seed(seed)
    for index in range(max(1, int(count))):
        text, style = DEFAULT_TEXTS[index % len(DEFAULT_TEXTS)]
        mel_path = mel_dir / f"sample_{index:05d}.pt"
        mel_path.parent.mkdir(parents=True, exist_ok=True)
        torch, _, _, _ = _torch()

        sample = structured_random_mel(config, frames, seed=seed + index)
        if target_bytes:
            sample = {"mel": sample, "augmentation_cache": structured_random_mel(config, max(frames, payload_frames), seed=seed + index + 100_000)}
        torch.save(sample, mel_path)
        records.append({"id": f"sample_{index:05d}", "text": text, "style": style, "mel_path": mel_path.relative_to(root).as_posix(), "frames": frames})
    (root / "records.jsonl").write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    (root / "config.json").write_text(json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8")
    dataset_bytes = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    report = {"status": "created", "backend": "genmusic-vn-self-diffusion", "count": len(records), "frames": frames, "seed": seed, "target_bytes": int(target_bytes or 0), "dataset_bytes": dataset_bytes, "dataset_gb": round(dataset_bytes / (1024 ** 3), 4), "records": str((root / "records.jsonl").resolve()), "config": str((root / "config.json").resolve())}
    (root / "dataset_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report

def _read_records(root: Path) -> list[dict[str, Any]]:
    path = root / "records.jsonl"
    if not path.exists():
        raise ValueError(f"Thiếu records.jsonl trong {root}")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError("Dataset không có record nào.")
    return records

def _load_mel(path: Path, *, device="cpu"):
    torch, _, _, _ = _torch()
    value = torch.load(path, map_location=device, weights_only=True)
    return value["mel"] if isinstance(value, dict) else value

def load_reference_conditioning(dataset_dir: str | Path, record_id: str | None = None) -> dict[str, Any]:
    """Pulls one record's real backing_mel + style_anchor out of an already-preprocessed
    dataset, for conditioned generation that matches what the model actually saw during
    training -- instead of generate_audio()'s zero-conditioned default (see
    prior measurements). Picks the first record if record_id is omitted.

    Returns raw (unbatched) tensors: backing_mel is (n_mels, frames) or None if the
    dataset has no separated backing stem; style_anchor is (512,) or None if the
    dataset never computed one (both are legitimate for older/synthetic datasets).
    """
    root = Path(dataset_dir)
    records = _read_records(root)
    record = next((r for r in records if r["id"] == record_id), records[0]) if record_id else records[0]

    backing_mel = _load_mel(root / record["backing_mel_path"]) if record.get("backing_mel_path") else None

    style_path = record.get("style_embed_path")
    style_anchor = _load_mel(root / style_path).float().view(-1) if style_path and (root / style_path).exists() else None

    return {
        "id": record["id"],
        "text": record["text"],
        "style": record["style"],
        "backing_mel": backing_mel,
        "style_anchor": style_anchor,
    }

def _record_path(root: Path, record: dict[str, Any]) -> Path:
    path_str = record.get("mel_path") or record.get("backing_mel_path") or record.get("vocal_mel_path")
    if not path_str:
        raise KeyError("Record missing 'mel_path', 'backing_mel_path', or 'vocal_mel_path'")
    path = Path(path_str)
    return path if path.is_absolute() else root / path

def _record_paths(root: Path, record: dict[str, Any]) -> list[tuple[str, Path]]:
    """Return every tensor path required by a record, including separated stems."""
    separated = (("vocal", "vocal_mel_path"), ("backing", "backing_mel_path"))
    if all(record.get(key) for _, key in separated):
        return [(name, _resolve_record_path(root, record[key])) for name, key in separated]
    return [("mel", _record_path(root, record))]

def _resolve_record_path(root: Path, path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else root / path

_MEL_PATH_FIELDS = (
    "mel_path", "backing_mel_path", "vocal_mel_path",
    "seed_mel_path", "backing_wav_path", "vocal_wav_path",
    "style_embed_path",
)

def _with_absolute_paths(root: Path, record: dict[str, Any]) -> dict[str, Any]:
    """Rewrite a record's mel/style path fields to absolute paths under its own
    source dataset root. Lets MusicDiffusionDataset combine records from several
    independently-preprocessed dataset directories (e.g. different raw-data
    batches) into one training set: once resolved, `self.root / record[field]`
    (used throughout __getitem__) returns the resolved absolute path unchanged
    regardless of which root `self.root` happens to be, since pathlib's `/`
    operator discards the left operand when the right one is already absolute.
    """
    resolved = dict(record)
    for field in _MEL_PATH_FIELDS:
        value = resolved.get(field)
        if value:
            resolved[field] = str(_resolve_record_path(root, value))
    return resolved

def _fit_mel_frames(mel, frames: int):
    torch, _, _, _ = _torch()
    if mel.shape[1] > frames:
        start = random.randint(0, mel.shape[1] - frames)
        return mel[:, start:start + frames]
    if mel.shape[1] < frames:
        return torch.nn.functional.pad(mel, (0, frames - mel.shape[1]))
    return mel

def validate_dataset(dataset_dir: str | Path, *, report_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(dataset_dir)
    report_destination = Path(report_path) if report_path else root / "validation_report.json"
    report_destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch, _, _, _ = _torch()
    except ImportError:
        report = {"status": "needs-torch", "dataset": str(root.resolve()), "missing": []}
        report_destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report
    missing = []
    invalid = []
    records = _read_records(root)
    config_data = json.loads((root / "config.json").read_text(encoding="utf-8")) if (root / "config.json").exists() else asdict(MusicDiffusionConfig())
    expected_mels = int(config_data.get("latent_dim", config_data.get("n_mels", 64)))
    for record in records:
        for stem_name, path in _record_paths(root, record):
            if not path.exists():
                missing.append({"stem": stem_name, "path": str(path)})
                continue
            tensor = _load_mel(path)
            if tuple(tensor.shape) != (expected_mels, int(record["frames"])):
                invalid.append({"stem": stem_name, "path": str(path), "shape": list(tensor.shape), "expected": [expected_mels, int(record["frames"])]})
    report = {"status": "valid" if not missing and not invalid else "invalid", "dataset": str(root.resolve()), "record_count": len(records), "missing": missing, "invalid": invalid, "format": "genmusic-self-diffusion-v1"}
    report_destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def estimate_vocal_mel_stats(
    dataset_dir: str | Path,
    records: list[dict[str, Any]],
    *,
    max_records: int = 256,
    max_frames_per_record: int = 2048,
) -> tuple[float, float]:
    """Estimate stable scalar target statistics from an even record sample."""
    torch, _, _, _ = _torch()
    root = Path(dataset_dir)
    if not records:
        raise ValueError("Dataset has no usable vocal records.")
    sample_count = min(len(records), max(1, int(max_records)))
    if sample_count == 1:
        sampled = [records[0]]
    else:
        sampled = [records[round(index * (len(records) - 1) / (sample_count - 1))] for index in range(sample_count)]
    value_sum = 0.0
    square_sum = 0.0
    value_count = 0
    for record in sampled:
        path = _resolve_record_path(root, record["vocal_mel_path"]) if record.get("vocal_mel_path") else _record_path(root, record)
        mel = _load_mel(path).float()
        if mel.shape[1] > max_frames_per_record:
            indices = torch.linspace(0, mel.shape[1] - 1, max_frames_per_record).long()
            mel = mel.index_select(1, indices)
        values = mel.double()
        value_sum += float(values.sum())
        square_sum += float(values.square().sum())
        value_count += values.numel()
    mean = value_sum / max(1, value_count)
    variance = max(1e-4, square_sum / max(1, value_count) - mean * mean)
    return float(mean), float(math.sqrt(variance))


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Keep the previous progress file valid if a worker is preempted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


SEMANTIC_PRETRAIN_SOURCE_CONTRACT = "semantic_heads_only_with_baseline_v1"


SEMANTIC_PRETRAIN_PREFIXES = (
    "open_vocabulary_lyric.",
    "lyric_semantic_audio_projection.",
    "lyric_semantic_word_projection.",
    "lyric_semantic_unit_projection.",
)


def configure_semantic_pretrain_parameters(model) -> dict[str, int]:
    """Freeze the acoustic denoiser while fitting exact-span semantic heads.

    Semantic-only loss has no acoustic velocity term. Letting it update the
    DiT backbone can therefore destroy a previously useful generator even if
    the semantic EMA intentionally skips those parameters. Only the open
    compositional lyric encoder and its training-time retrieval projections
    are required by the word/unit objectives.
    """
    trainable = 0
    frozen = 0
    trainable_tensors = 0
    for name, parameter in model.named_parameters():
        enabled = name.startswith(SEMANTIC_PRETRAIN_PREFIXES)
        parameter.requires_grad_(enabled)
        if enabled:
            trainable += int(parameter.numel())
            trainable_tensors += 1
        else:
            frozen += int(parameter.numel())
    if trainable_tensors == 0:
        raise ValueError(
            "Semantic pretraining requires open-vocabulary semantic heads."
        )
    return {
        "trainable_parameters": trainable,
        "trainable_tensors": trainable_tensors,
        "frozen_parameters": frozen,
    }


def train_model(
    dataset_dir: str | Path | list[str | Path],
    checkpoint_path: str | Path,
    *,
    epochs: int = 1,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    ema_decay: float = 0.999,
    device: str | None = None,
    max_records: int | None = None,
    additional_records: list[dict[str, Any]] | None = None,
    roberta_model: str = "vinai/xphonebert-base",
    dim: int = 256,
    depth: int = 4,
    heads: int = 4,
    ff_mult: int = 4,
    frames_per_chunk: int | None = None,
    resume: bool = False,
    save_every_epoch: bool = False,
    checkpoint_every_steps: int = 0,
    log_every_steps: int = 10,
    progress_path: str | Path | None = None,
    lambda_vocal: float = 1.0,
    style_dropout_prob: float = 0.5,
    text_dropout_prob: float = 0.1,
    text_contrastive_weight: float = 0.08,
    text_contrastive_margin: float = 0.03,
    text_contrastive_prob: float = 0.5,
    text_sensitivity_weight: float = 2.0,
    text_sensitivity_target: float = 0.20,
    validation_fraction: float = 0.05,
    validation_max_records: int | None = 128,
    validation_seed: int = 5602,
    early_stopping_patience: int = 4,
    minimum_epochs: int = 8,
    early_stopping_min_delta: float = 0.001,
    minimum_text_sensitivity: float | None = None,
    open_vocabulary_conditioning: bool = False,
    lexical_holdout_fraction: float = 0.0,
    minimum_lexical_sensitivity: float = 0.05,
    lyric_semantic_weight: float = 0.25,
    lyric_denoised_semantic_weight: float = 0.0,
    lyric_phrase_semantic_weight: float = 0.0,
    lyric_phrase_denoised_semantic_weight: float = 0.0,
    lyric_semantic_temperature: float = 0.08,
    minimum_lyric_semantic_accuracy: float = 0.05,
    minimum_lyric_denoised_semantic_accuracy: float = 0.0,
    lyric_unit_semantic_weight: float = 0.25,
    minimum_lyric_unit_accuracy: float = 0.10,
    lyric_unit_denoised_semantic_weight: float = 0.0,
    minimum_lyric_denoised_unit_accuracy: float = 0.0,
    self_rollout_consistency_weight: float = 0.0,
    self_rollout_consistency_probability: float = 0.0,
    self_rollout_step_size: float = 0.125,
    self_rollout_solver_steps: int = 0,
    early_timestep_fraction: float = 0.0,
    early_timestep_max: float = 0.35,
    seed_full_frame_rewrite_probability: float = 0.0,
    seed_span_corruption_probability: float = 0.0,
    seed_span_corruption_fraction: float = 0.25,
    semantic_pretrain_only: bool = False,
    reset_optimizer: bool = False,
    reset_ema: bool = False,
) -> dict[str, Any]:
    torch, _, _, DataLoaderClass = _torch()

    dataset_dirs = [Path(dataset_dir)] if isinstance(dataset_dir, (str, Path)) else [Path(d) for d in dataset_dir]
    root = dataset_dirs[0]
    checkpoint = Path(checkpoint_path)
    validations = [
        validate_dataset(d, report_path=checkpoint.parent / f"validation_report{'' if i == 0 else f'_{i}'}.json")
        for i, d in enumerate(dataset_dirs)
    ]
    invalid = [v for v in validations if v["status"] != "valid"]
    if invalid:
        raise ValueError(f"Dataset không hợp lệ; xem validation_report*.json. ({len(invalid)}/{len(dataset_dirs)} dataset(s) invalid)")

    # All combined datasets are assumed to share the same mel/config format (they
    # should, since they're preprocessed by the same pipeline) -- only the first
    # dataset's config.json is actually read.
    config = _config_from_dict(json.loads((root / "config.json").read_text(encoding="utf-8")))
    if frames_per_chunk is not None:
        frames = max(16, int(frames_per_chunk))
        config = replace(
            config,
            frames_per_chunk=frames,
            chunk_seconds=frames * config.hop_length / config.sample_rate,
        )
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    arch = {
        "dim": dim,
        "depth": depth,
        "heads": heads,
        "ff_mult": ff_mult,
        "open_vocabulary_conditioning": bool(
            open_vocabulary_conditioning
        ),
    }
    arch_to_save = {**arch, "roberta_model": roberta_model}
    resumed_payload: dict[str, Any] = {}
    start_epoch = 0
    resume_batch_in_epoch = 0

    # Song-level train/validation split, computed once regardless of resume,
    # so a resumed run keeps evaluating against the same held-out songs.
    # Records are pre-resolved to absolute paths (_with_absolute_paths), so the
    # `root` argument passed to estimate_vocal_mel_stats below is a no-op for
    # path resolution -- pathlib's `/` discards it whenever the right-hand
    # side is already absolute.
    all_records = [
        _with_absolute_paths(d, record)
        for d in dataset_dirs
        for record in _filter_training_records(_read_records(d))
    ]
    selected_records = all_records[:max_records] if max_records is not None else all_records
    if float(lexical_holdout_fraction) > 0.0:
        materialized_split = existing_lexical_holdout_records(
            selected_records
        )
        if materialized_split is not None:
            (
                training_records,
                validation_records,
                lexical_holdout_words,
            ) = materialized_split
        else:
            (
                training_records,
                validation_records,
                lexical_holdout_words,
            ) = split_lexical_holdout_records(
                selected_records,
                validation_fraction=lexical_holdout_fraction,
                validation_max_records=validation_max_records,
                seed=validation_seed,
            )
    else:
        training_records, validation_records = split_training_records(
            selected_records,
            validation_fraction=validation_fraction,
            validation_max_records=validation_max_records,
            seed=validation_seed,
        )
        lexical_holdout_words = []
    lexical_split_audit = audit_lexical_holdout_split(
        training_records,
        validation_records,
        lexical_holdout_words,
    )
    if lexical_holdout_words and not lexical_split_audit["passed"]:
        raise ValueError(
            "Lexical holdout split leaked training words or songs: "
            f"{lexical_split_audit}"
        )
    if not training_records:
        raise ValueError("Dataset has no usable training records after validation split.")

    if resume and checkpoint.is_file():
        from ..models.text_to_music_diffusion import load_checkpoint

        model, saved_config, resumed_payload = load_checkpoint(
            checkpoint,
            device=selected_device,
            roberta_model=roberta_model,
            use_ema=False,
            open_vocabulary_conditioning=open_vocabulary_conditioning,
        )
        saved_arch = resumed_payload.get("arch") or {}
        mismatched_arch = {
            key: (saved_arch.get(key), value)
            for key, value in arch.items()
            if int(saved_arch.get(key, value)) != int(value)
        }
        if mismatched_arch:
            raise ValueError(f"Resume checkpoint architecture mismatch: {mismatched_arch}")
        saved_open_vocabulary = bool(
            saved_arch.get("open_vocabulary_conditioning", False)
        )
        if saved_open_vocabulary and not open_vocabulary_conditioning:
            raise ValueError(
                "Cannot resume an open-vocabulary checkpoint with "
                "open_vocabulary_conditioning disabled."
            )
        if frames_per_chunk is not None and saved_config.frames_per_chunk != config.frames_per_chunk:
            raise ValueError(
                "Resume checkpoint frames_per_chunk does not match the requested value: "
                f"{saved_config.frames_per_chunk} != {config.frames_per_chunk}"
            )
        config = saved_config
        start_epoch = max(0, int(resumed_payload.get("epoch", 0)))
        saved_training_state = resumed_payload.get("training_state") or {}
        if int(saved_training_state.get("epoch", start_epoch)) == start_epoch:
            resume_batch_in_epoch = max(
                0,
                int(saved_training_state.get("batch_in_epoch", 0)),
            )
    else:
        mel_mean, mel_std = estimate_vocal_mel_stats(root, training_records)
        config = replace(config, mel_mean=mel_mean, mel_std=mel_std)
        from ..models.dit_transformer import MicroDiT

        model = MicroDiT(
            config,
            roberta_model=roberta_model,
            dim=dim,
            depth=depth,
            heads=heads,
            ff_mult=ff_mult,
            open_vocabulary_conditioning=open_vocabulary_conditioning,
        ).to(selected_device)

    semantic_parameter_report = None
    if semantic_pretrain_only:
        semantic_parameter_report = configure_semantic_pretrain_parameters(
            model
        )
        print(
            "semantic_pretrain_trainable_parameters="
            f"{semantic_parameter_report['trainable_parameters']} "
            "semantic_pretrain_frozen_parameters="
            f"{semantic_parameter_report['frozen_parameters']}",
            flush=True,
        )
    # Train only parameters that require gradients (the frozen RoBERTa weights
    # are re-downloaded on load and intentionally omitted from checkpoints).
    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
    optimizer_state_restored = False
    if resumed_payload.get("optimizer") and not reset_optimizer:
        try:
            optimizer.load_state_dict(resumed_payload["optimizer"])
            optimizer_state_restored = True
        except ValueError as exc:
            # New lyric-conditioning parameters change the optimizer parameter
            # count, but the old acoustic weights are still a valuable warm
            # start. Resume model/epoch/EMA and restart AdamW safely.
            print(f"optimizer_state_reset_for_model_upgrade={exc}", flush=True)
    elif resumed_payload.get("optimizer") and reset_optimizer:
        print(
            "optimizer_state_reset_for_objective_change=true",
            flush=True,
        )

    # Instantiate custom Dataset and DataLoader
    dataset = MusicDiffusionDataset(dataset_dirs, config, records=training_records, additional_records=additional_records)
    validation_dataset = (
        MusicDiffusionDataset(dataset_dirs, config, records=validation_records, deterministic_crop=True, crop_seed=validation_seed)
        if validation_records
        else None
    )
    if not dataset.records:
        raise ValueError("Dataset has no usable records after vocal/transcript quality filtering.")

    def collate_fn(batch):
        vocal_mels = torch.stack([item["vocal_mel"] for item in batch])
        backing_mels = torch.stack([item["backing_mel"] for item in batch])
        style_anchors = torch.stack([item["style_anchor"] for item in batch])
        lyric_frame_ids = torch.stack(
            [item["lyric_frame_ids"] for item in batch]
        )
        exact_fractions = torch.tensor(
            [item["lyric_alignment_exact_fraction"] for item in batch],
            dtype=torch.float32,
        )
        texts = [item["text"] for item in batch]
        result = {
            "vocal_mel": vocal_mels,
            "backing_mel": backing_mels,
            "style_anchor": style_anchors,
            "text": texts,
            "lyric_frame_ids": lyric_frame_ids,
            "lyric_alignment_exact_fraction": exact_fractions,
        }
        if all("seed_mel" in item for item in batch):
            result["seed_mel"] = torch.stack(
                [item["seed_mel"] for item in batch]
            )
        elif any("seed_mel" in item for item in batch):
            raise ValueError(
                "A batch cannot mix seeded and Gaussian CFM records."
            )
        if all("refinement_mask" in item for item in batch):
            result["refinement_mask"] = torch.stack(
                [item["refinement_mask"] for item in batch]
            )
        elif any("refinement_mask" in item for item in batch):
            raise ValueError(
                "A batch cannot mix masked and unmasked seed refiners."
            )
        return result

    batch_size_value = max(1, int(batch_size))

    def build_dataloader(epoch_index: int):
        # A deterministic per-epoch sampler lets a resumed worker skip batches
        # already covered by its latest mid-epoch checkpoint.
        generator = torch.Generator()
        generator.manual_seed(5602 + int(epoch_index))
        return DataLoaderClass(
            dataset,
            batch_size=batch_size_value,
            shuffle=True,
            collate_fn=collate_fn,
            generator=generator,
        )

    validation_dataloader = (
        DataLoaderClass(validation_dataset, batch_size=batch_size_value, shuffle=False, collate_fn=collate_fn)
        if validation_dataset is not None
        else None
    )

    epoch_count = max(1, int(epochs))
    steps_per_epoch = len(build_dataloader(0))
    total_steps = max(1, epoch_count * steps_per_epoch)
    # A model-objective upgrade intentionally resets AdamW. Do not then spend
    # 5% of all historical epochs warming up again; schedule over only the
    # remaining resume window. Restored schedulers keep the full target span.
    schedule_total_steps = (
        total_steps
        if optimizer_state_restored or start_epoch == 0
        else max(1, (epoch_count - start_epoch) * steps_per_epoch)
    )
    warmup_steps = min(max(1, int(schedule_total_steps * 0.05)), max(1, schedule_total_steps - 1))

    def learning_rate_multiplier(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, schedule_total_steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_multiplier)
    trainer = DiffusionTrainer(
        model,
        config,
        optimizer,
        device=selected_device,
        scheduler=scheduler,
        lambda_vocal=lambda_vocal,
        style_dropout_prob=style_dropout_prob,
        text_dropout_prob=text_dropout_prob,
        text_contrastive_weight=text_contrastive_weight,
        text_contrastive_margin=text_contrastive_margin,
        text_contrastive_prob=text_contrastive_prob,
        text_sensitivity_weight=text_sensitivity_weight,
        text_sensitivity_target=text_sensitivity_target,
        lyric_semantic_weight=(
            lyric_semantic_weight
            if open_vocabulary_conditioning
            else 0.0
        ),
        lyric_denoised_semantic_weight=(
            lyric_denoised_semantic_weight
            if open_vocabulary_conditioning
            else 0.0
        ),
        lyric_phrase_semantic_weight=(
            lyric_phrase_semantic_weight
            if open_vocabulary_conditioning
            else 0.0
        ),
        lyric_phrase_denoised_semantic_weight=(
            lyric_phrase_denoised_semantic_weight
            if open_vocabulary_conditioning
            else 0.0
        ),
        lyric_semantic_temperature=lyric_semantic_temperature,
        lyric_unit_semantic_weight=(
            lyric_unit_semantic_weight
            if open_vocabulary_conditioning
            else 0.0
        ),
        lyric_unit_denoised_semantic_weight=(
            lyric_unit_denoised_semantic_weight
            if open_vocabulary_conditioning
            else 0.0
        ),
        self_rollout_consistency_weight=(
            self_rollout_consistency_weight
        ),
        self_rollout_consistency_probability=(
            self_rollout_consistency_probability
        ),
        self_rollout_step_size=self_rollout_step_size,
        self_rollout_solver_steps=self_rollout_solver_steps,
        early_timestep_fraction=early_timestep_fraction,
        early_timestep_max=early_timestep_max,
        seed_full_frame_rewrite_probability=(
            seed_full_frame_rewrite_probability
        ),
        seed_span_corruption_probability=(
            seed_span_corruption_probability
        ),
        seed_span_corruption_fraction=seed_span_corruption_fraction,
        ema_decay=ema_decay,
        semantic_pretrain_only=semantic_pretrain_only,
    )
    if optimizer_state_restored and resumed_payload.get("scheduler"):
        scheduler.load_state_dict(resumed_payload["scheduler"])
    if resumed_payload.get("ema") and not reset_ema:
        trainer.load_ema_state(resumed_payload["ema"])
    elif resumed_payload.get("ema") and reset_ema:
        print(
            "ema_reset_for_source_target_distribution_change=true",
            flush=True,
        )
    if semantic_pretrain_only:
        # New/upgraded semantic heads have far fewer updates than the acoustic
        # backbone. Do not evaluate them through a stale .999 EMA initialized
        # mostly from random weights; keep the mature backbone EMA untouched.
        trainer.reset_semantic_ema()

    initial_lyric_semantics = None
    initial_lyric_units = None
    if semantic_pretrain_only and validation_dataloader is not None:
        initial_lyric_semantics = trainer.evaluate_lyric_semantics(
            validation_dataloader,
            heldout_words=lexical_holdout_words,
        )
        initial_lyric_units = trainer.evaluate_lyric_units(
            validation_dataloader,
            heldout_words=lexical_holdout_words,
        )
        print(
            "SEMANTIC_PRETRAIN_BASELINE "
            f"word_accuracy={initial_lyric_semantics['accuracy']:.6f} "
            f"unit_accuracy={initial_lyric_units['accuracy']:.6f}",
            flush=True,
        )

    saved_training_state = resumed_payload.get("training_state") or {}
    trainer.optimizer_steps_completed = max(
        0,
        int(saved_training_state.get("optimizer_steps_completed", 0)),
    )
    trainer.optimizer_steps_skipped = max(
        0,
        int(saved_training_state.get("optimizer_steps_skipped", 0)),
    )
    global_step = max(
        0,
        int(saved_training_state.get("global_step", start_epoch * steps_per_epoch)),
    )
    if resume_batch_in_epoch >= steps_per_epoch:
        start_epoch += resume_batch_in_epoch // steps_per_epoch
        resume_batch_in_epoch %= steps_per_epoch
    resumed_from_epoch = start_epoch
    resumed_from_batch = resume_batch_in_epoch
    progress_destination = (
        Path(progress_path)
        if progress_path is not None
        else checkpoint.parent / "training_progress.json"
    )

    if start_epoch >= epoch_count:
        return {
            "status": "complete",
            "backend": "genmusic-vn-self-diffusion",
            "dataset": str(root.resolve()),
            "dataset_dirs": [str(d.resolve()) for d in dataset_dirs],
            "checkpoint": str(checkpoint.resolve()),
            "device": selected_device,
            "epochs": epoch_count,
            "resumed_from_epoch": resumed_from_epoch,
            "resumed_from_batch": resumed_from_batch,
            "global_step": global_step,
            "message": "Checkpoint already reached the requested epoch count.",
        }

    started = time.perf_counter()
    losses = []
    loss_curve = []
    best_checkpoint = checkpoint.with_name(f"{checkpoint.stem}.best{checkpoint.suffix}")
    # A changed crop curriculum/model input path makes the old validation loss
    # incomparable; start a fresh best-checkpoint window after such an upgrade.
    saved_best_validation_loss = (
        saved_training_state.get("best_validation_loss")
        if not resumed_payload or optimizer_state_restored
        else None
    )
    best_validation_loss = float(saved_best_validation_loss) if saved_best_validation_loss is not None else float("inf")
    best_epoch = int(saved_training_state.get("best_epoch", 0)) if saved_best_validation_loss is not None else 0
    epochs_without_improvement = (
        int(saved_training_state.get("epochs_without_improvement", 0)) if saved_best_validation_loss is not None else 0
    )
    final_validation_loss: float | None = None
    final_text_sensitivity: float | None = None
    final_lexical_sensitivity: float | None = None
    final_lyric_semantics: dict[str, float | int] | None = None
    final_denoised_lyric_semantics: dict[str, float | int] | None = None
    final_lyric_units: dict[str, float | int] | None = None
    final_denoised_lyric_units: dict[str, float | int] | None = None
    # A validation-only selector could otherwise replace a text-responsive
    # model with a marginally lower-loss checkpoint that ignores lyrics. Keep
    # a small tolerance for measurement noise while making intelligibility a
    # hard gate.
    checkpoint_sensitivity_floor = max(
        0.0,
        float(minimum_text_sensitivity if minimum_text_sensitivity is not None else 0.90 * max(0.0, float(text_sensitivity_target))),
    )
    stopped_early = False
    completed_epochs = start_epoch

    from ..models.text_to_music_diffusion import save_checkpoint

    for epoch in range(start_epoch, epoch_count):
        dataloader = build_dataloader(epoch)
        start_batch = resume_batch_in_epoch if epoch == start_epoch else 0

        def on_step(
            completed_batches: int,
            loss_record: dict[str, Any],
            should_log: bool,
        ) -> None:
            nonlocal global_step
            global_step += 1
            training_state = {
                "status": "training",
                "epoch": epoch,
                "display_epoch": epoch + 1,
                "batch_in_epoch": completed_batches,
                "batches_per_epoch": steps_per_epoch,
                "global_step": global_step,
                "total_steps": total_steps,
                "loss": loss_record["loss"],
                "gradient_norm": loss_record["gradient_norm"],
                "optimizer_updated": loss_record["optimizer_updated"],
                "optimizer_steps_completed": (
                    loss_record["optimizer_steps_completed"]
                ),
                "optimizer_steps_skipped": (
                    loss_record["optimizer_steps_skipped"]
                ),
                "best_validation_loss": best_validation_loss if math.isfinite(best_validation_loss) else None,
                "best_epoch": best_epoch,
                "epochs_without_improvement": epochs_without_improvement,
                "checkpoint": str(checkpoint.resolve()),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            interval = max(0, int(checkpoint_every_steps))
            should_checkpoint = interval > 0 and global_step % interval == 0
            if should_log or should_checkpoint:
                _write_json_atomic(progress_destination, training_state)
            if should_checkpoint:
                save_checkpoint(
                    model,
                    checkpoint,
                    config,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    ema_state=trainer.ema_parameters,
                    epoch=epoch,
                    loss=loss_record["loss"],
                    arch=arch_to_save,
                    training_state=training_state,
                )
                print(
                    f"checkpoint_saved={checkpoint} global_step={global_step}",
                    flush=True,
                )

        epoch_losses = trainer.train_epoch(
            dataloader,
            epoch_index=epoch,
            total_epochs=epoch_count,
            start_batch=start_batch,
            log_every_steps=log_every_steps,
            on_step=on_step,
        )
        if not epoch_losses:
            resume_batch_in_epoch = 0
            continue
        losses.extend(epoch_losses)
        avg_loss = sum(d["loss"] for d in epoch_losses) / len(epoch_losses)
        avg_loss_gt = sum(d["loss_gt"] for d in epoch_losses) / len(epoch_losses)
        vocal_aux_values = [d["loss_vocal_aux"] for d in epoch_losses if d["loss_vocal_aux"] is not None]
        avg_loss_vocal_aux = sum(vocal_aux_values) / len(vocal_aux_values) if vocal_aux_values else None
        completed_epochs = epoch + 1
        final_validation_loss = (
            trainer.evaluate_ground_truth(validation_dataloader, seed=validation_seed)
            if validation_dataloader is not None
            else None
        )
        final_text_sensitivity = (
            trainer.evaluate_text_sensitivity(validation_dataloader, seed=validation_seed)
            if validation_dataloader is not None
            else None
        )
        final_lexical_sensitivity = (
            trainer.evaluate_lexical_sensitivity(
                validation_dataloader,
                lexical_holdout_words,
                seed=validation_seed,
            )
            if validation_dataloader is not None and lexical_holdout_words
            else None
        )
        final_lyric_semantics = (
            trainer.evaluate_lyric_semantics(
                validation_dataloader,
                heldout_words=lexical_holdout_words,
            )
            if (
                validation_dataloader is not None
                and open_vocabulary_conditioning
                and trainer.lyric_semantic_weight > 0.0
            )
            else None
        )
        final_denoised_lyric_semantics = (
            trainer.evaluate_denoised_lyric_semantics(
                validation_dataloader,
                heldout_words=lexical_holdout_words,
                seed=validation_seed,
            )
            if (
                validation_dataloader is not None
                and open_vocabulary_conditioning
                and trainer.lyric_denoised_semantic_weight > 0.0
            )
            else None
        )
        final_lyric_units = (
            trainer.evaluate_lyric_units(
                validation_dataloader,
                heldout_words=lexical_holdout_words,
            )
            if (
                validation_dataloader is not None
                and open_vocabulary_conditioning
                and trainer.lyric_unit_semantic_weight > 0.0
            )
            else None
        )
        final_denoised_lyric_units = (
            trainer.evaluate_denoised_lyric_units(
                validation_dataloader,
                heldout_words=lexical_holdout_words,
                seed=validation_seed,
            )
            if (
                validation_dataloader is not None
                and open_vocabulary_conditioning
                and trainer.lyric_unit_denoised_semantic_weight > 0.0
            )
            else None
        )
        improved = _is_checkpoint_improvement(
            final_validation_loss,
            best_validation_loss,
            final_text_sensitivity,
            checkpoint_sensitivity_floor,
            early_stopping_min_delta,
            final_lexical_sensitivity,
            max(0.0, float(minimum_lexical_sensitivity)),
            (
                float(final_lyric_semantics["accuracy"])
                if final_lyric_semantics is not None
                else None
            ),
            max(0.0, float(minimum_lyric_semantic_accuracy)),
            (
                float(final_denoised_lyric_semantics["accuracy"])
                if final_denoised_lyric_semantics is not None
                else None
            ),
            max(
                0.0,
                float(minimum_lyric_denoised_semantic_accuracy),
            ),
            (
                float(final_lyric_units["accuracy"])
                if final_lyric_units is not None
                else None
            ),
            max(0.0, float(minimum_lyric_unit_accuracy)),
            (
                float(final_denoised_lyric_units["accuracy"])
                if final_denoised_lyric_units is not None
                else None
            ),
            max(
                0.0,
                float(minimum_lyric_denoised_unit_accuracy),
            ),
        )
        if improved:
            best_validation_loss = final_validation_loss
            best_epoch = completed_epochs
            epochs_without_improvement = 0
            save_checkpoint(
                model,
                best_checkpoint,
                config,
                ema_state=trainer.ema_parameters,
                epoch=completed_epochs,
                loss=avg_loss,
                arch=arch_to_save,
                training_state={
                    "status": "best",
                    "epoch": completed_epochs,
                    "validation_loss": final_validation_loss,
                    "best_validation_loss": best_validation_loss,
                    "best_epoch": best_epoch,
                    "text_conditioning_sensitivity": final_text_sensitivity,
                    "minimum_text_sensitivity": checkpoint_sensitivity_floor,
                    "lexical_holdout_sensitivity": (
                        final_lexical_sensitivity
                    ),
                    "minimum_lexical_sensitivity": max(
                        0.0,
                        float(minimum_lexical_sensitivity),
                    ),
                    "lyric_semantic_alignment": final_lyric_semantics,
                    "minimum_lyric_semantic_accuracy": max(
                        0.0,
                        float(minimum_lyric_semantic_accuracy),
                    ),
                    "lyric_denoised_semantic_alignment": (
                        final_denoised_lyric_semantics
                    ),
                    "minimum_lyric_denoised_semantic_accuracy": max(
                        0.0,
                        float(
                            minimum_lyric_denoised_semantic_accuracy
                        ),
                    ),
                    "lyric_unit_alignment": final_lyric_units,
                    "minimum_lyric_unit_accuracy": max(
                        0.0,
                        float(minimum_lyric_unit_accuracy),
                    ),
                    "lyric_denoised_unit_alignment": (
                        final_denoised_lyric_units
                    ),
                    "minimum_lyric_denoised_unit_accuracy": max(
                        0.0,
                        float(minimum_lyric_denoised_unit_accuracy),
                    ),
                },
            )
        elif final_validation_loss is not None:
            epochs_without_improvement += 1
        loss_curve.append({
            "epoch": completed_epochs, "loss": avg_loss, "loss_gt": avg_loss_gt,
            "loss_velocity": None, "loss_vocal_aux": avg_loss_vocal_aux,
            "validation_loss": final_validation_loss,
            "best_validation_loss": best_validation_loss if math.isfinite(best_validation_loss) else None,
            "text_conditioning_sensitivity": final_text_sensitivity,
            "minimum_text_sensitivity": checkpoint_sensitivity_floor,
            "lexical_holdout_sensitivity": final_lexical_sensitivity,
            "minimum_lexical_sensitivity": max(
                0.0,
                float(minimum_lexical_sensitivity),
            ),
            "lyric_semantic_alignment": final_lyric_semantics,
            "minimum_lyric_semantic_accuracy": max(
                0.0,
                float(minimum_lyric_semantic_accuracy),
            ),
            "lyric_denoised_semantic_alignment": (
                final_denoised_lyric_semantics
            ),
            "minimum_lyric_denoised_semantic_accuracy": max(
                0.0,
                float(minimum_lyric_denoised_semantic_accuracy),
            ),
            "lyric_unit_alignment": final_lyric_units,
            "minimum_lyric_unit_accuracy": max(
                0.0,
                float(minimum_lyric_unit_accuracy),
            ),
            "lyric_denoised_unit_alignment": (
                final_denoised_lyric_units
            ),
            "minimum_lyric_denoised_unit_accuracy": max(
                0.0,
                float(minimum_lyric_denoised_unit_accuracy),
            ),
            "conditioning_gate_pass": (
                (
                    final_text_sensitivity is None
                    or final_text_sensitivity >= checkpoint_sensitivity_floor
                )
                and (
                    final_lexical_sensitivity is None
                    or final_lexical_sensitivity
                    >= max(0.0, float(minimum_lexical_sensitivity))
                )
                and (
                    final_lyric_semantics is None
                    or float(final_lyric_semantics["accuracy"])
                    >= max(
                        0.0,
                        float(minimum_lyric_semantic_accuracy),
                    )
                )
                and (
                    final_denoised_lyric_semantics is None
                    or float(
                        final_denoised_lyric_semantics["accuracy"]
                    )
                    >= max(
                        0.0,
                        float(
                            minimum_lyric_denoised_semantic_accuracy
                        ),
                    )
                )
                and (
                    final_lyric_units is None
                    or float(final_lyric_units["accuracy"])
                    >= max(
                        0.0,
                        float(minimum_lyric_unit_accuracy),
                    )
                )
                and (
                    final_denoised_lyric_units is None
                    or float(final_denoised_lyric_units["accuracy"])
                    >= max(
                        0.0,
                        float(minimum_lyric_denoised_unit_accuracy),
                    )
                )
            ),
        })
        print(
            f"epoch={completed_epochs}/{epoch_count} train_loss={avg_loss:.6f} "
            f"validation_loss={final_validation_loss if final_validation_loss is not None else 'disabled'} "
            f"text_sensitivity={final_text_sensitivity if final_text_sensitivity is not None else 'disabled'} "
            f"lexical_sensitivity={final_lexical_sensitivity if final_lexical_sensitivity is not None else 'disabled'} "
            f"semantic_accuracy={float(final_lyric_semantics['accuracy']) if final_lyric_semantics is not None else 'disabled'} "
            f"denoised_semantic_accuracy={float(final_denoised_lyric_semantics['accuracy']) if final_denoised_lyric_semantics is not None else 'disabled'} "
            f"unit_accuracy={float(final_lyric_units['accuracy']) if final_lyric_units is not None else 'disabled'} "
            f"denoised_unit_accuracy={float(final_denoised_lyric_units['accuracy']) if final_denoised_lyric_units is not None else 'disabled'} "
            f"best_epoch={best_epoch or 'n/a'}",
            flush=True,
        )
        if save_every_epoch:
            # Remote workers are preemptible. Persist raw weights, optimizer,
            # scheduler and EMA after each epoch so the next worker can resume.
            save_checkpoint(
                model,
                checkpoint,
                config,
                optimizer=optimizer,
                scheduler=scheduler,
                ema_state=trainer.ema_parameters,
                epoch=completed_epochs,
                loss=avg_loss,
                arch=arch_to_save,
                training_state={
                    "status": "training",
                    "epoch": completed_epochs,
                    "display_epoch": min(epoch + 2, epoch_count),
                    "batch_in_epoch": 0,
                    "batches_per_epoch": steps_per_epoch,
                    "global_step": global_step,
                    "optimizer_steps_completed": (
                        trainer.optimizer_steps_completed
                    ),
                    "optimizer_steps_skipped": (
                        trainer.optimizer_steps_skipped
                    ),
                    "total_steps": total_steps,
                    "loss": avg_loss,
                    "validation_loss": final_validation_loss,
                    "text_conditioning_sensitivity": final_text_sensitivity,
                    "minimum_text_sensitivity": checkpoint_sensitivity_floor,
                    "lexical_holdout_sensitivity": (
                        final_lexical_sensitivity
                    ),
                    "minimum_lexical_sensitivity": max(
                        0.0,
                        float(minimum_lexical_sensitivity),
                    ),
                    "lyric_semantic_alignment": final_lyric_semantics,
                    "minimum_lyric_semantic_accuracy": max(
                        0.0,
                        float(minimum_lyric_semantic_accuracy),
                    ),
                    "lyric_denoised_semantic_alignment": (
                        final_denoised_lyric_semantics
                    ),
                    "minimum_lyric_denoised_semantic_accuracy": max(
                        0.0,
                        float(
                            minimum_lyric_denoised_semantic_accuracy
                        ),
                    ),
                    "lyric_unit_alignment": final_lyric_units,
                    "minimum_lyric_unit_accuracy": max(
                        0.0,
                        float(minimum_lyric_unit_accuracy),
                    ),
                    "lyric_denoised_unit_alignment": (
                        final_denoised_lyric_units
                    ),
                    "minimum_lyric_denoised_unit_accuracy": max(
                        0.0,
                        float(minimum_lyric_denoised_unit_accuracy),
                    ),
                    "best_validation_loss": best_validation_loss if math.isfinite(best_validation_loss) else None,
                    "best_epoch": best_epoch,
                    "epochs_without_improvement": epochs_without_improvement,
                    "checkpoint": str(checkpoint.resolve()),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        resume_batch_in_epoch = 0
        if (
            validation_dataloader is not None
            and completed_epochs >= max(1, int(minimum_epochs))
            and epochs_without_improvement >= max(1, int(early_stopping_patience))
        ):
            stopped_early = True
            print(f"Early stopping at epoch {completed_epochs}; best validation epoch was {best_epoch}.", flush=True)
            break

    final_loss = (
        sum(d["loss"] for d in losses[-min(10, len(losses)):])
        / max(1, min(10, len(losses)))
        if losses
        else float(resumed_payload.get("loss") or 0.0)
    )
    completed_training_state = {
        "status": "complete",
        "epoch": completed_epochs,
        "display_epoch": completed_epochs,
        "batch_in_epoch": 0,
        "batches_per_epoch": steps_per_epoch,
        "global_step": global_step,
        "optimizer_steps_completed": trainer.optimizer_steps_completed,
        "optimizer_steps_skipped": trainer.optimizer_steps_skipped,
        "total_steps": total_steps,
        "loss": final_loss,
        "validation_loss": final_validation_loss,
        "best_validation_loss": best_validation_loss if math.isfinite(best_validation_loss) else None,
        "best_epoch": best_epoch,
        "epochs_without_improvement": epochs_without_improvement,
        "stopped_early": stopped_early,
        "semantic_pretrain_only": bool(semantic_pretrain_only),
        "semantic_parameter_report": semantic_parameter_report,
        "initial_lyric_semantic_alignment": initial_lyric_semantics,
        "initial_lyric_unit_alignment": initial_lyric_units,
        "optimizer_state_reset_requested": bool(reset_optimizer),
        "ema_state_reset_requested": bool(reset_ema),
        "text_conditioning_sensitivity": final_text_sensitivity,
        "lexical_holdout_sensitivity": final_lexical_sensitivity,
        "lyric_semantic_alignment": final_lyric_semantics,
        "minimum_lyric_semantic_accuracy": max(
            0.0,
            float(minimum_lyric_semantic_accuracy),
        ),
        "lyric_denoised_semantic_alignment": (
            final_denoised_lyric_semantics
        ),
        "minimum_lyric_denoised_semantic_accuracy": max(
            0.0,
            float(minimum_lyric_denoised_semantic_accuracy),
        ),
        "lyric_unit_alignment": final_lyric_units,
        "minimum_lyric_unit_accuracy": max(
            0.0,
            float(minimum_lyric_unit_accuracy),
        ),
        "lyric_denoised_unit_alignment": final_denoised_lyric_units,
        "minimum_lyric_denoised_unit_accuracy": max(
            0.0,
            float(minimum_lyric_denoised_unit_accuracy),
        ),
        "checkpoint": str(checkpoint.resolve()),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    save_checkpoint(
        model,
        checkpoint,
        config,
        optimizer=optimizer,
        scheduler=scheduler,
        ema_state=trainer.ema_parameters,
        epoch=completed_epochs,
        loss=final_loss,
        arch=arch_to_save,
        training_state=completed_training_state,
    )
    _write_json_atomic(progress_destination, completed_training_state)
    report = {
        "status": "complete",
        "backend": "genmusic-vn-self-diffusion",
        "dataset": str(root.resolve()),
        "dataset_dirs": [str(d.resolve()) for d in dataset_dirs],
        "checkpoint": str(checkpoint.resolve()),
        "best_checkpoint": (
            str(best_checkpoint.resolve())
            if best_checkpoint.is_file()
            else None
        ),
        "device": selected_device,
        "requested_epochs": epoch_count,
        "completed_epochs": completed_epochs,
        "stopped_early": stopped_early,
        "best_epoch": best_epoch,
        "best_validation_loss": (
            round(best_validation_loss, 6)
            if math.isfinite(best_validation_loss)
            else None
        ),
        "final_validation_loss": (
            round(final_validation_loss, 6)
            if final_validation_loss is not None
            else None
        ),
        "final_text_conditioning_sensitivity": (
            round(final_text_sensitivity, 6)
            if final_text_sensitivity is not None
            else None
        ),
        "final_lexical_holdout_sensitivity": (
            round(final_lexical_sensitivity, 6)
            if final_lexical_sensitivity is not None
            else None
        ),
        "final_lyric_semantic_alignment": final_lyric_semantics,
        "initial_lyric_semantic_alignment": initial_lyric_semantics,
        "minimum_lyric_semantic_accuracy": max(
            0.0,
            float(minimum_lyric_semantic_accuracy),
        ),
        "final_lyric_denoised_semantic_alignment": (
            final_denoised_lyric_semantics
        ),
        "minimum_lyric_denoised_semantic_accuracy": max(
            0.0,
            float(minimum_lyric_denoised_semantic_accuracy),
        ),
        "final_lyric_unit_alignment": final_lyric_units,
        "initial_lyric_unit_alignment": initial_lyric_units,
        "minimum_lyric_unit_accuracy": max(
            0.0,
            float(minimum_lyric_unit_accuracy),
        ),
        "final_lyric_denoised_unit_alignment": (
            final_denoised_lyric_units
        ),
        "minimum_lyric_denoised_unit_accuracy": max(
            0.0,
            float(minimum_lyric_denoised_unit_accuracy),
        ),
        "minimum_lexical_sensitivity": max(
            0.0,
            float(minimum_lexical_sensitivity),
        ),
        "lexical_holdout_fraction": max(
            0.0,
            float(lexical_holdout_fraction),
        ),
        "lexical_holdout_words": lexical_holdout_words,
        "lexical_holdout_word_count": len(lexical_holdout_words),
        "lexical_holdout_split_audit": lexical_split_audit,
        "open_vocabulary_conditioning": bool(
            open_vocabulary_conditioning
        ),
        "validation_record_count": len(validation_records),
        "resumed_from_epoch": resumed_from_epoch,
        "resumed_from_batch": resumed_from_batch,
        "optimizer_state_restored": optimizer_state_restored,
        "optimizer_state_reset_requested": bool(reset_optimizer),
        "ema_state_reset_requested": bool(reset_ema),
        "batch_size": batch_size_value,
        "record_count": len(dataset.records),
        "excluded_record_count": dataset.excluded_record_count,
        "additional_record_count": len(additional_records or []),
        "step_count": len(losses),
        "global_step": global_step,
        "optimizer_steps_completed": trainer.optimizer_steps_completed,
        "optimizer_steps_skipped": trainer.optimizer_steps_skipped,
        "checkpoint_every_steps": max(0, int(checkpoint_every_steps)),
        "final_loss": round(final_loss, 6),
        "loss_curve": loss_curve,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "dim": dim,
        "depth": depth,
        "heads": heads,
        "ff_mult": ff_mult,
        "frames_per_chunk": config.frames_per_chunk,
        "chunk_seconds": config.chunk_seconds,
        "mel_mean": round(config.mel_mean, 6),
        "mel_std": round(config.mel_std, 6),
        "warmup_steps": warmup_steps,
        "ema_decay": trainer.ema_decay,
        "amp_enabled": trainer.use_amp,
        "semantic_pretrain_only": trainer.semantic_pretrain_only,
        "semantic_parameter_report": semantic_parameter_report,
        "semantic_ema_reset_from_raw": bool(
            trainer.semantic_pretrain_only
        ),
        "style_dropout_prob": trainer.style_dropout_prob,
        "text_dropout_prob": trainer.text_dropout_prob,
        "text_contrastive_weight": trainer.text_contrastive_weight,
        "text_contrastive_margin": trainer.text_contrastive_margin,
        "text_contrastive_prob": trainer.text_contrastive_prob,
        "text_sensitivity_weight": trainer.text_sensitivity_weight,
        "text_sensitivity_target": trainer.text_sensitivity_target,
        "lyric_semantic_weight": trainer.lyric_semantic_weight,
        "lyric_denoised_semantic_weight": (
            trainer.lyric_denoised_semantic_weight
        ),
        "lyric_phrase_semantic_weight": (
            trainer.lyric_phrase_semantic_weight
        ),
        "lyric_phrase_denoised_semantic_weight": (
            trainer.lyric_phrase_denoised_semantic_weight
        ),
        "lyric_semantic_temperature": (
            trainer.lyric_semantic_temperature
        ),
        "lyric_unit_semantic_weight": (
            trainer.lyric_unit_semantic_weight
        ),
        "lyric_unit_denoised_semantic_weight": (
            trainer.lyric_unit_denoised_semantic_weight
        ),
        "self_rollout_consistency_weight": (
            trainer.self_rollout_consistency_weight
        ),
        "self_rollout_consistency_probability": (
            trainer.self_rollout_consistency_probability
        ),
        "self_rollout_step_size": trainer.self_rollout_step_size,
        "self_rollout_solver_steps": trainer.self_rollout_solver_steps,
        "early_timestep_fraction": trainer.early_timestep_fraction,
        "early_timestep_max": trainer.early_timestep_max,
        "seed_full_frame_rewrite_probability": (
            trainer.seed_full_frame_rewrite_probability
        ),
        "seed_span_corruption_probability": (
            trainer.seed_span_corruption_probability
        ),
        "seed_span_corruption_fraction": (
            trainer.seed_span_corruption_fraction
        ),
        "lyric_semantic_mode": (
            "compositional_word_to_exact_v1_latent_span"
            if trainer.lyric_semantic_weight > 0.0
            else "disabled"
        ),
        "minimum_text_sensitivity": checkpoint_sensitivity_floor,
        "text_sensitivity_mode": "matched_vs_mismatched_lyrics",
        "lyric_alignment_mode": (
            "exact_word_frames_with_uniform_inference_fallback"
            if open_vocabulary_conditioning
            else "xphonebert_sequence"
        ),
        "mixed_precision": trainer.use_amp,
        "lambda_vocal": lambda_vocal,
    }
    (checkpoint.parent / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
