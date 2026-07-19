"""Dataset, DataLoader and training loop for the GenMusic diffusion model."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import time
import unicodedata
from dataclasses import asdict, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..models.text_to_music_diffusion import MusicDiffusionConfig, normalize_mel, structured_random_mel

STYLE_EMBED_DIM = 512  # matches MuQ-MuLan / DiffRhythm2 teacher's cond_dim


def _music_config_from_json(path: str | Path) -> MusicDiffusionConfig:
    """Load model fields while allowing dataset provenance beside them."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    allowed = {field.name for field in fields(MusicDiffusionConfig)}
    return MusicDiffusionConfig(**{key: value for key, value in data.items() if key in allowed})

# Whisper-tiny labels from the original six Kaggle shards contain occasional
# English fragments, CJK characters and YouTube call-to-action speech.  Keeping
# those labels teaches the model that an arbitrary prompt may map to any vocal
# sound, which is exactly the text-conditioning collapse seen in evaluation.
_VIETNAMESE_MARKED_CHARS = frozenset(
    "ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệ"
    "íìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
)
_VIETNAMESE_COMMON_WORDS = frozenset(
    "ai anh ba bao bay ben biet bon buoc ca cac can chang chi cho chung co con cua "
    "da dang dau day dem den dieu doi dung duoc em gi giua hay hon khi khong "
    "hai la lai lam len long luc ma mai minh mot mua nam nay nghe ngay nguoi nhau "
    "nhieu nhung noi o qua ra rang roi sau se ta thay thi theo them tren trong "
    "troi tu van ve vi voi yeu".split()
)
_ENGLISH_COMMON_WORDS = frozenset(
    "a an and are be been but by can come comes do for from have i in is it "
    "let love me my of on open our say scale sign since subscribe taking the "
    "this to was we with you your".split()
)
_TRANSCRIPT_NOISE_PHRASES = (
    "dang ky kenh",
    "hay subscribe",
    "subscribe cho kenh",
    "cam on cac ban",
)


def _accentless_token(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", str(value).casefold()).replace("đ", "d")
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")


def clean_vietnamese_lyric(text: str) -> str:
    """Return a conservative Vietnamese transcript or an empty rejection.

    This is deliberately a training-label filter, not a general language
    detector.  It accepts unaccented Vietnamese when several common words are
    present, while rejecting mixed-script and clearly English-heavy ASR spans.
    """
    normalized = unicodedata.normalize("NFKC", str(text or "")).strip()
    if not normalized:
        return ""

    letters = [char for char in normalized if char.isalpha()]
    non_latin_letters = [
        char
        for char in letters
        if "LATIN" not in unicodedata.name(char, "") and char.casefold() != "đ"
    ]
    if non_latin_letters and len(non_latin_letters) / max(1, len(letters)) > 0.02:
        return ""

    # Retain word punctuation useful to the G2P frontend, replacing every
    # unsupported symbol with whitespace so hidden control characters cannot
    # become accidental tokens.
    cleaned_chars = [
        char if (char.isalpha() or char.isspace() or char in "'-.,!?") else " "
        for char in normalized
    ]
    cleaned = re.sub(r"\s+", " ", "".join(cleaned_chars)).strip(" -'.,!?")
    words = re.findall(r"[^\W\d_]+", cleaned.casefold(), flags=re.UNICODE)
    if len(words) < 2:
        return ""

    folded_words = [_accentless_token(word) for word in words]
    folded_text = " ".join(folded_words)
    if any(phrase in folded_text for phrase in _TRANSCRIPT_NOISE_PHRASES):
        return ""
    vietnamese_hits = sum(word in _VIETNAMESE_COMMON_WORDS for word in folded_words)
    english_hits = sum(word in _ENGLISH_COMMON_WORDS for word in folded_words)
    marked_count = sum(char.casefold() in _VIETNAMESE_MARKED_CHARS for char in cleaned)
    if vietnamese_hits < 2 and marked_count < 1:
        return ""
    if english_hits >= 2 and english_hits >= vietnamese_hits:
        return ""
    return cleaned


def usable_lyric_spans(record: dict[str, Any]) -> list[tuple[float, float, str]]:
    """Collect timestamped spans whose ASR label is useful for Vietnamese training."""
    spans: list[tuple[float, float, str]] = []
    for segment in record.get("segments") or []:
        cleaned = clean_vietnamese_lyric(str(segment.get("text", "")))
        start = max(0.0, float(segment.get("start", 0.0)))
        end = max(start, float(segment.get("end", start)))
        if cleaned and end > start:
            spans.append((start, end, cleaned))
    return spans

# Ensure PyTorch helper works
def _torch():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import Dataset, DataLoader
    except ImportError as exc:
        raise RuntimeError("Cần cài torch để chạy model sinh nhạc GenMusic.") from exc
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
    segments = record.get("segments") or []
    return bool(usable_lyric_spans(record) if segments else clean_vietnamese_lyric(text))


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
    groups: dict[str, list[int]] = {}
    for index, record in enumerate(usable):
        # Clean aligned datasets contain multiple chunks from the same song.
        # Splitting on chunk id would leak the singer, backing, and adjacent
        # lyrics into validation, so prefer a stable parent-song identifier.
        group_key = str(record.get("song_id") or record.get("id") or index)
        groups.setdefault(group_key, []).append(index)
    if len(groups) < 2:
        return usable, []
    ranked_groups = sorted(
        groups,
        key=lambda key: hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest(),
    )
    validation_indices: set[int] = set()
    for group_key in ranked_groups[:-1]:
        if len(validation_indices) >= validation_count:
            break
        validation_indices.update(groups[group_key])
    training_records = [
        record for index, record in enumerate(usable) if index not in validation_indices
    ]
    validation_records = [
        record for index, record in enumerate(usable) if index in validation_indices
    ]
    return training_records, validation_records


def _is_checkpoint_improvement(
    validation_loss: float | None,
    best_validation_loss: float,
    text_sensitivity: float | None,
    minimum_text_sensitivity: float,
    min_delta: float,
) -> bool:
    """Require acoustic improvement without accepting lyric-conditioning collapse."""
    if validation_loss is None:
        return False
    if text_sensitivity is not None and text_sensitivity < minimum_text_sensitivity:
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
        return clean_vietnamese_lyric(full_text)

    selected: list[str] = []
    for segment in segments:
        # Filter at segment level because isolated one-word timestamps often do
        # not contain enough evidence for a language decision.
        if not clean_vietnamese_lyric(str(segment.get("text", ""))):
            continue
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
    return clean_vietnamese_lyric(" ".join(selected))


def lyric_lines_for_window(
    full_text: str,
    segments: list[dict[str, Any]],
    start_seconds: float,
    end_seconds: float,
) -> str:
    """Keep timestamped segment boundaries when preparing a generation prompt.

    Training crops are at most ``frames_per_chunk`` (4.096 s by default), while
    quality checks and the web UI often render several crops back-to-back.  A
    flattened multi-segment prompt makes ``generate_audio`` put the entire
    lyric on one overlong denoising sequence, which is outside the training
    distribution and audibly rushes syllables.  Newlines preserve the natural
    segment/crop boundaries; the generator allocates each line independently.
    """
    if not segments:
        return clean_vietnamese_lyric(full_text)

    lines: list[str] = []
    for segment in segments:
        if not clean_vietnamese_lyric(str(segment.get("text", ""))):
            continue
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
        selected = [
            word
            for word_start, word_end, word in word_spans
            if word and word_end > start_seconds and word_start < end_seconds
        ]
        line = clean_vietnamese_lyric(" ".join(selected))
        if line:
            lines.append(line)
    return "\n".join(lines)

class MusicDiffusionDataset:
    """PyTorch Dataset mapping structured Mel-spectrograms and text/style prompts."""
    def __init__(
        self,
        dataset_dir: str | Path,
        config: MusicDiffusionConfig,
        max_records: int | None = None,
        additional_records: list[dict[str, Any]] | None = None,
        records: list[dict[str, Any]] | None = None,
        deterministic_crop: bool = False,
        crop_seed: int = 5602,
        lyric_aligned_crop_prob: float = 1.0,
    ):
        _, _, Dataset, _ = _torch()
        self.root = Path(dataset_dir)
        self.config = config
        all_records = list(records) if records is not None else _read_records(self.root)
        usable_records = _filter_training_records(all_records)
        self.excluded_record_count = len(all_records) - len(usable_records)
        self.records = usable_records[:max_records] if max_records is not None else usable_records
        self.deterministic_crop = bool(deterministic_crop)
        self.crop_seed = int(crop_seed)
        self.lyric_aligned_crop_prob = max(0.0, min(1.0, float(lyric_aligned_crop_prob)))
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
            
        # Crop both stems at the same offset.  Uniform whole-song crops mostly
        # land on instrumental/silent spans; selecting a valid lyric span makes
        # every optimization item teach an actual phoneme-to-vocal mapping.
        crop_start = 0
        shared_frames = min(vocal_mel.shape[1], backing_mel.shape[1])
        lyric_spans = usable_lyric_spans(record)
        if shared_frames > self.config.frames_per_chunk:
            max_start = shared_frames - self.config.frames_per_chunk
            use_lyric_span = bool(lyric_spans) and (
                self.deterministic_crop or random.random() < self.lyric_aligned_crop_prob
            )
            if use_lyric_span:
                record_key = str(record.get("id") or idx)
                if self.deterministic_crop:
                    digest = hashlib.sha256(
                        f"{self.crop_seed}:{record_key}:lyric".encode("utf-8")
                    ).digest()
                    span = lyric_spans[int.from_bytes(digest[:8], "big") % len(lyric_spans)]
                    fraction = int.from_bytes(digest[8:16], "big") / float(2**64 - 1)
                    focus_seconds = span[0] + fraction * (span[1] - span[0])
                else:
                    span = random.choice(lyric_spans)
                    focus_seconds = random.uniform(span[0], span[1])
                focus_frame = round(focus_seconds * self.config.sample_rate / self.config.hop_length)
                crop_start = max(
                    0,
                    min(max_start, focus_frame - self.config.frames_per_chunk // 2),
                )
            elif self.deterministic_crop:
                record_key = str(record.get("id") or idx)
                digest = hashlib.sha256(
                    f"{self.crop_seed}:{record_key}".encode("utf-8")
                ).digest()
                crop_start = int.from_bytes(digest[:8], "big") % (max_start + 1)
            else:
                crop_start = random.randint(0, max_start)
            vocal_mel = vocal_mel[:, crop_start:crop_start + self.config.frames_per_chunk]
            backing_mel = backing_mel[:, crop_start:crop_start + self.config.frames_per_chunk]
        else:
            vocal_mel = _fit_mel_frames(vocal_mel, self.config.frames_per_chunk)
            backing_mel = _fit_mel_frames(backing_mel, self.config.frames_per_chunk)

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

        lyric_text = clean_vietnamese_lyric(str(record["text"]))
        segments = record.get("segments") or []
        crop_start_seconds = crop_start * self.config.hop_length / self.config.sample_rate
        crop_end_seconds = crop_start_seconds + self.config.frames_per_chunk * self.config.hop_length / self.config.sample_rate
        if segments:
            lyric_text = lyric_text_for_window(lyric_text, segments, crop_start_seconds, crop_end_seconds)
        return {
            "vocal_mel": vocal_mel,
            "backing_mel": backing_mel,
            "style_anchor": style_anchor,
            "text": lyric_text,
            "frame_text_segments": (
                segments if record.get("exact_word_timestamps") else []
            ),
            "frame_text_crop_start_seconds": crop_start_seconds,
            "frame_text_crop_end_seconds": crop_end_seconds,
        }

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
        text_contrastive_weight: float = 0.08,
        text_contrastive_margin: float = 0.03,
        text_contrastive_prob: float = 0.5,
        text_sensitivity_weight: float = 2.0,
        text_sensitivity_target: float = 0.20,
        native_ctc_weight: float = 0.0,
        native_ctc_teacher_weight: float = 0.0,
        native_frame_text_weight: float = 0.0,
        native_frame_text_teacher_weight: float = 0.0,
        native_vocal_prior_weight: float = 0.0,
        vocal_structure_weight: float = 0.0,
        native_prosody_weight: float = 0.0,
        use_amp: bool = True,
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
        self.style_dropout_prob = float(style_dropout_prob)
        self.text_dropout_prob = float(text_dropout_prob)
        self.text_contrastive_weight = max(0.0, float(text_contrastive_weight))
        self.text_contrastive_margin = max(0.0, float(text_contrastive_margin))
        self.text_contrastive_prob = max(0.0, min(1.0, float(text_contrastive_prob)))
        self.text_sensitivity_weight = max(0.0, float(text_sensitivity_weight))
        self.text_sensitivity_target = max(0.0, float(text_sensitivity_target))
        self.native_ctc_weight = max(0.0, float(native_ctc_weight))
        self.native_ctc_teacher_weight = max(0.0, float(native_ctc_teacher_weight))
        self.native_frame_text_weight = max(
            0.0, float(native_frame_text_weight)
        )
        self.native_frame_text_teacher_weight = max(
            0.0, float(native_frame_text_teacher_weight)
        )
        self.native_vocal_prior_weight = max(0.0, float(native_vocal_prior_weight))
        self.vocal_structure_weight = max(0.0, float(vocal_structure_weight))
        self.native_prosody_weight = max(0.0, float(native_prosody_weight))
        # FP16 is optional even on CUDA. The content-conditioned objective can
        # legitimately need FP32 backward on Turing GPUs when GradScaler falls
        # below 1 yet data-dependent attention gradients still overflow.
        self.use_amp = bool(use_amp and str(device).startswith("cuda"))
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.ema_parameters = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    def _update_ema(self) -> None:
        torch, _, _, _ = _torch()
        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
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
            if (
                name in self.ema_parameters
                and tuple(value.shape) == tuple(self.ema_parameters[name].shape)
            ):
                self.ema_parameters[name] = value.detach().to(self.device).clone()

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
                        loss, _, _ = cfm_loss(
                            self.model,
                            vocal_mel,
                            backing_mel,
                            style_anchor,
                            batch["text"],
                            self.config,
                            condition_dropout_prob=0.0,
                            style_dropout_prob=0.0,
                            text_dropout_prob=0.0,
                            lambda_vocal=self.lambda_vocal,
                            native_frame_text_weight=self.native_frame_text_weight,
                            native_frame_text_teacher_weight=(
                                self.native_frame_text_teacher_weight
                            ),
                            frame_text_segments=batch.get("frame_text_segments"),
                            frame_text_crop_starts_seconds=batch.get(
                                "frame_text_crop_start_seconds"
                            ),
                            frame_text_crop_ends_seconds=batch.get(
                                "frame_text_crop_end_seconds"
                            ),
                            native_vocal_prior_weight=self.native_vocal_prior_weight,
                            vocal_structure_weight=self.vocal_structure_weight,
                            native_prosody_weight=self.native_prosody_weight,
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
        from ..models.cfm_flow import build_mismatched_texts
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
            vocal_mel = batch["vocal_mel"].to(self.device).transpose(1, 2)
            if bool(getattr(self.model, "joint_stem_generation", False)):
                backing_mel = batch["backing_mel"].to(self.device).transpose(1, 2)
                clean_mel = torch.cat((backing_mel, vocal_mel), dim=-1)
            else:
                clean_mel = vocal_mel
            generator = torch.Generator(device=self.device).manual_seed(int(seed))
            noise = torch.randn(
                clean_mel.shape,
                generator=generator,
                device=self.device,
                dtype=clean_mel.dtype,
            )
            timestep = torch.full(
                (clean_mel.shape[0],), 0.5, device=self.device, dtype=clean_mel.dtype
            )
            noisy = 0.5 * noise + 0.5 * clean_mel
            zero_style = torch.zeros(
                (clean_mel.shape[0], int(getattr(self.model, "style_dim", STYLE_EMBED_DIM))),
                device=self.device,
            )
            with torch.no_grad():
                mismatched_texts, content_mask_flags = build_mismatched_texts(batch["text"])
                content_mask = torch.tensor(
                    content_mask_flags,
                    dtype=torch.bool,
                    device=self.device,
                )
                if not bool(content_mask.any()):
                    return 0.0
                conditioned = self.model(
                    x=noisy,
                    texts=batch["text"],
                    timestep=timestep,
                    style_prompt=zero_style,
                )
                mismatched = self.model(
                    x=noisy,
                    texts=mismatched_texts,
                    timestep=timestep,
                    style_prompt=zero_style,
                )
            if bool(getattr(self.model, "joint_stem_generation", False)):
                mel_count = int(self.config.n_mels)
                conditioned = conditioned[..., mel_count:]
                mismatched = mismatched[..., mel_count:]
            difference = (
                (conditioned - mismatched).square().mean(dim=(1, 2)).sqrt()
            )[content_mask].mean()
            baseline = (
                conditioned.square().mean(dim=(1, 2)).sqrt()
            )[content_mask].mean().clamp_min(1e-8)
            return float((difference / baseline).detach().cpu())
        finally:
            with torch.no_grad():
                for name, parameter in self.model.named_parameters():
                    if name in raw_parameters:
                        parameter.copy_(raw_parameters[name])
            self.model.train(was_training)

    def evaluate_native_ctc(self, dataloader, *, use_ema: bool = True) -> float | None:
        """Held-out audio-only lyric loss; optionally bypass slow EMA warmup."""
        if not bool(getattr(self.model, "native_generation", False)):
            return None
        torch, _, _, _ = _torch()
        from ..models.native_text import native_ctc_loss

        was_training = self.model.training
        raw_parameters = {
            name: parameter.detach().clone()
            for name, parameter in self.model.named_parameters()
            if use_ema and name in self.ema_parameters
        }
        losses: list[float] = []
        try:
            with torch.no_grad():
                for name, parameter in self.model.named_parameters():
                    if use_ema and name in self.ema_parameters:
                        parameter.copy_(self.ema_parameters[name])
            self.model.eval()
            with torch.no_grad():
                for batch in dataloader:
                    vocal_mel = batch["vocal_mel"].to(self.device).transpose(1, 2)
                    logits = self.model.audio_text_logits(vocal_mel.float())
                    losses.append(float(native_ctc_loss(logits, batch["text"]).cpu()))
        finally:
            with torch.no_grad():
                for name, parameter in self.model.named_parameters():
                    if name in raw_parameters:
                        parameter.copy_(raw_parameters[name])
            self.model.train(was_training)
        return sum(losses) / len(losses) if losses else None

    def train_epoch(
        self,
        dataloader,
        *,
        epoch_index: int = 0,
        total_epochs: int = 1,
        start_batch: int = 0,
        batch_offset: int = 0,
        log_every_steps: int = 10,
        on_step=None,
    ) -> list[float]:
        torch, _, _, _ = _torch()
        self.model.train()
        epoch_losses = []
        total_batches = len(dataloader) + max(0, int(batch_offset))
        consecutive_overflow_steps = 0
        # GradScaler starts high and may need several halvings on the first
        # resumed batch. Sixteen attempts cover 65536 -> 1 without letting a
        # genuinely unstable run burn through an epoch.
        max_consecutive_overflow_steps = 16
        for batch_index, batch in enumerate(dataloader):
            if batch_index < start_batch:
                continue
            actual_batch = batch_index + 1 + max(0, int(batch_offset))
            vocal_mel = batch["vocal_mel"].to(self.device)
            backing_mel = batch["backing_mel"].to(self.device)
            style_anchor = batch["style_anchor"].to(self.device)
            texts = batch["text"]
            self.optimizer.zero_grad(set_to_none=True)

            # Transpose mels from (batch, n_mels, seq_len) to (batch, seq_len, n_mels) for DiT.
            # style_anchor is already a flat (batch, 512) MuQ-MuLan embedding, not
            # mel-shaped, so it needs no transpose.
            vocal_mel_t = vocal_mel.transpose(1, 2)
            backing_mel_t = backing_mel.transpose(1, 2)
            from ..models.cfm_flow import cfm_loss
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
                loss, loss_gt, loss_vocal_aux, loss_details = cfm_loss(
                    self.model, vocal_mel_t, backing_mel_t, style_anchor, texts, self.config,
                    lambda_vocal=self.lambda_vocal,
                    style_dropout_prob=self.style_dropout_prob,
                    text_dropout_prob=self.text_dropout_prob,
                    text_contrastive_weight=self.text_contrastive_weight,
                    text_contrastive_margin=self.text_contrastive_margin,
                    text_contrastive_prob=self.text_contrastive_prob,
                    text_sensitivity_weight=self.text_sensitivity_weight,
                    text_sensitivity_target=self.text_sensitivity_target,
                    native_ctc_weight=self.native_ctc_weight,
                    native_ctc_teacher_weight=self.native_ctc_teacher_weight,
                    native_frame_text_weight=self.native_frame_text_weight,
                    native_frame_text_teacher_weight=(
                        self.native_frame_text_teacher_weight
                    ),
                    frame_text_segments=batch.get("frame_text_segments"),
                    frame_text_crop_starts_seconds=batch.get(
                        "frame_text_crop_start_seconds"
                    ),
                    frame_text_crop_ends_seconds=batch.get(
                        "frame_text_crop_end_seconds"
                    ),
                    native_vocal_prior_weight=self.native_vocal_prior_weight,
                    vocal_structure_weight=self.vocal_structure_weight,
                    native_prosody_weight=self.native_prosody_weight,
                    return_details=True,
                )

            if not bool(torch.isfinite(loss)):
                current_lr = float(self.optimizer.param_groups[0]["lr"])
                raise FloatingPointError(
                    "Non-finite training loss detected; refusing to corrupt the next "
                    f"checkpoint (epoch={epoch_index + 1}, batch={actual_batch}, "
                    f"lr={current_lr:.8g}, amp_scale={float(self.scaler.get_scale()):.1f})."
                )

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()),
                1.0,
                error_if_nonfinite=False,
            )

            if not bool(torch.isfinite(gradient_norm)):
                current_lr = float(self.optimizer.param_groups[0]["lr"])
                previous_scale = float(self.scaler.get_scale())
                if not self.use_amp:
                    raise FloatingPointError(
                        "Non-finite gradients detected without AMP; refusing to "
                        "corrupt the next checkpoint "
                        f"(epoch={epoch_index + 1}, batch={actual_batch}, "
                        f"lr={current_lr:.8g})."
                    )

                # A transient FP16 overflow is expected while GradScaler finds a
                # safe scale. unscale_() has already recorded the non-finite
                # gradients, so step() skips the optimizer update and update()
                # lowers the scale. Do not advance LR/EMA for a skipped update.
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                consecutive_overflow_steps += 1
                next_scale = float(self.scaler.get_scale())
                print(
                    "AMP gradient overflow; skipped optimizer step "
                    f"(epoch={epoch_index + 1}, batch={actual_batch}, "
                    f"lr={current_lr:.8g}, amp_scale={previous_scale:.1f}"
                    f"->{next_scale:.1f}, consecutive="
                    f"{consecutive_overflow_steps}/{max_consecutive_overflow_steps}).",
                    flush=True,
                )
                if consecutive_overflow_steps >= max_consecutive_overflow_steps:
                    raise FloatingPointError(
                        "Gradients remained non-finite after repeated AMP scale "
                        "reductions; refusing to waste the session or corrupt a "
                        f"checkpoint (epoch={epoch_index + 1}, "
                        f"batch={actual_batch}, lr={current_lr:.8g}, "
                        f"amp_scale={next_scale:.1f})."
                    )
                continue

            consecutive_overflow_steps = 0
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler is not None:
                self.scheduler.step()
            self._update_ema()
            loss_value = float(loss.detach().cpu())
            loss_record = {
                "loss": loss_value,
                "loss_gt": float(loss_gt.detach().cpu()),
                "loss_velocity": None,
                "loss_vocal_aux": float(loss_vocal_aux.detach().cpu()) if loss_vocal_aux is not None else None,
                "loss_native_ctc_pred": (
                    float(loss_details["native_ctc_pred"].detach().cpu())
                    if loss_details["native_ctc_pred"] is not None else None
                ),
                "loss_native_ctc_teacher": (
                    float(loss_details["native_ctc_teacher"].detach().cpu())
                    if loss_details["native_ctc_teacher"] is not None else None
                ),
                "loss_native_frame_text_pred": (
                    float(
                        loss_details["native_frame_text_pred"].detach().cpu()
                    )
                    if loss_details["native_frame_text_pred"] is not None
                    else None
                ),
                "loss_native_frame_text_teacher": (
                    float(
                        loss_details["native_frame_text_teacher"].detach().cpu()
                    )
                    if loss_details["native_frame_text_teacher"] is not None
                    else None
                ),
                "loss_native_frame_text_prior": (
                    float(
                        loss_details["native_frame_text_prior"].detach().cpu()
                    )
                    if loss_details["native_frame_text_prior"] is not None
                    else None
                ),
                "loss_native_vocal_prior": (
                    float(loss_details["native_vocal_prior"].detach().cpu())
                    if loss_details["native_vocal_prior"] is not None else None
                ),
                "loss_native_vocal_prior_contrastive": (
                    float(
                        loss_details["native_vocal_prior_contrastive"]
                        .detach()
                        .cpu()
                    )
                    if loss_details["native_vocal_prior_contrastive"] is not None
                    else None
                ),
                "loss_vocal_structure": (
                    float(loss_details["vocal_structure"].detach().cpu())
                    if loss_details["vocal_structure"] is not None else None
                ),
                "loss_native_prosody": (
                    float(loss_details["native_prosody"].detach().cpu())
                    if loss_details["native_prosody"] is not None else None
                ),
            }
            epoch_losses.append(loss_record)
            completed_batches = actual_batch
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
        bytes_per_sample = config.n_mels * max(frames, payload_frames) * 4
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
    docs/PROJECT_REPORT.md). Picks the first record if record_id is omitted.

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

def _fit_mel_frames(mel, frames: int):
    torch, _, _, _ = _torch()
    if mel.shape[1] > frames:
        start = random.randint(0, mel.shape[1] - frames)
        return mel[:, start:start + frames]
    if mel.shape[1] < frames:
        return torch.nn.functional.pad(mel, (0, frames - mel.shape[1]))
    return mel

def validate_dataset(
    dataset_dir: str | Path,
    *,
    report_path: str | Path | None = None,
    max_tensor_records: int | None = None,
) -> dict[str, Any]:
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
    expected_mels = int(config_data.get("n_mels", 64))
    tensor_record_indices = set(range(len(records)))
    if max_tensor_records is not None and len(records) > max(0, int(max_tensor_records)):
        # Kaggle input tensors live behind a read-only FUSE mount. Loading every
        # vocal/backing tensor on each resumed run consumed several GPU-minutes
        # before optimization even started. Check every path, but inspect tensor
        # shape on an evenly distributed sample; preprocessing can retain the
        # default max_tensor_records=None for an exhaustive first validation.
        sample_count = max(0, int(max_tensor_records))
        if sample_count == 0:
            tensor_record_indices = set()
        elif sample_count == 1:
            tensor_record_indices = {0}
        else:
            tensor_record_indices = {
                round(index * (len(records) - 1) / (sample_count - 1))
                for index in range(sample_count)
            }
    for record_index, record in enumerate(records):
        for stem_name, path in _record_paths(root, record):
            if not path.exists():
                missing.append({"stem": stem_name, "path": str(path)})
                continue
            if record_index not in tensor_record_indices:
                continue
            tensor = _load_mel(path)
            if tuple(tensor.shape) != (expected_mels, int(record["frames"])):
                invalid.append({"stem": stem_name, "path": str(path), "shape": list(tensor.shape), "expected": [expected_mels, int(record["frames"])]})
    report = {"status": "valid" if not missing and not invalid else "invalid", "dataset": str(root.resolve()), "record_count": len(records), "path_checked_records": len(records), "tensor_shape_checked_records": len(tensor_record_indices), "tensor_shape_check_limit": max_tensor_records, "missing": missing, "invalid": invalid, "format": "genmusic-self-diffusion-v1"}
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


def train_model(
    dataset_dir: str | Path,
    checkpoint_path: str | Path,
    *,
    epochs: int = 1,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    device: str | None = None,
    max_records: int | None = None,
    additional_records: list[dict[str, Any]] | None = None,
    roberta_model: str = "vinai/xphonebert-base",
    text_encoder_type: str = "native_utf8",
    generation_target: str = "full_mix",
    dim: int = 256,
    depth: int = 4,
    heads: int = 4,
    ff_mult: int = 4,
    frames_per_chunk: int | None = None,
    resume: bool = False,
    reset_optimizer: bool = False,
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
    native_ctc_weight: float = 0.0,
    native_ctc_teacher_weight: float = 0.0,
    native_frame_text_weight: float = 0.0,
    native_frame_text_teacher_weight: float = 0.0,
    native_vocal_prior_weight: float = 0.0,
    vocal_structure_weight: float = 0.0,
    native_prosody_weight: float = 0.0,
    native_learning_rate_multiplier: float = 10.0,
    freeze_native_ctc: bool = False,
    validation_fraction: float = 0.05,
    validation_max_records: int | None = 128,
    validation_seed: int = 5602,
    early_stopping_patience: int = 4,
    minimum_epochs: int = 8,
    early_stopping_min_delta: float = 0.001,
    dataset_validation_max_records: int | None = 128,
    minimum_text_sensitivity: float | None = None,
    use_amp: bool = True,
) -> dict[str, Any]:
    torch, _, _, DataLoaderClass = _torch()

    root = Path(dataset_dir)
    checkpoint = Path(checkpoint_path)
    validation = validate_dataset(
        root,
        report_path=checkpoint.parent / "validation_report.json",
        max_tensor_records=dataset_validation_max_records,
    )
    if validation["status"] != "valid":
        raise ValueError("Dataset không hợp lệ; xem validation_report.json.")

    config = _music_config_from_json(root / "config.json")
    if frames_per_chunk is not None:
        frames = max(16, int(frames_per_chunk))
        config = replace(
            config,
            frames_per_chunk=frames,
            chunk_seconds=frames * config.hop_length / config.sample_rate,
        )
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    arch = {"dim": dim, "depth": depth, "heads": heads, "ff_mult": ff_mult}
    arch_to_save = {
        **arch,
        "roberta_model": roberta_model,
        "text_encoder_type": text_encoder_type,
        "generation_target": generation_target,
        "native_ctc_tokenizer": (
            "vi_grapheme_v1" if text_encoder_type == "native_utf8" else None
        ),
        "native_text_tokenizer": (
            "vi_grapheme_v2" if text_encoder_type == "native_utf8" else None
        ),
        "native_vocal_prior": text_encoder_type == "native_utf8",
        "native_prosody": text_encoder_type == "native_utf8",
    }
    resumed_payload: dict[str, Any] = {}
    start_epoch = 0
    resume_batch_in_epoch = 0
    all_records = _filter_training_records(_read_records(root))
    selected_records = all_records[:max_records] if max_records is not None else all_records
    training_records, validation_records = split_training_records(
        selected_records,
        validation_fraction=validation_fraction,
        validation_max_records=validation_max_records,
        seed=validation_seed,
    )
    if not training_records:
        raise ValueError("Dataset has no usable training records after validation split.")

    if resume and checkpoint.is_file():
        from ..models.text_to_music_diffusion import load_checkpoint

        model, saved_config, resumed_payload = load_checkpoint(
            checkpoint,
            device=selected_device,
            roberta_model=roberta_model,
            text_encoder_type=text_encoder_type,
            generation_target=generation_target,
            use_ema=False,
        )
        saved_arch = resumed_payload.get("arch") or {}
        mismatched_arch = {
            key: (saved_arch.get(key), value)
            for key, value in arch.items()
            if int(saved_arch.get(key, value)) != int(value)
        }
        if mismatched_arch:
            raise ValueError(f"Resume checkpoint architecture mismatch: {mismatched_arch}")
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
            text_encoder_type=text_encoder_type,
            generation_target=generation_target,
            dim=dim,
            depth=depth,
            heads=heads,
            ff_mult=ff_mult,
        ).to(selected_device)

    if freeze_native_ctc:
        recognizer = getattr(model, "audio_text_recognizer", None)
        if recognizer is None:
            raise ValueError("freeze_native_ctc requires a native_utf8 checkpoint")
        for parameter in recognizer.parameters():
            parameter.requires_grad_(False)
        print("native_ctc_frozen_for_diffusion=true", flush=True)

    # Native text/audio-content modules train faster than the warm-started
    # acoustic DiT. Legacy frozen RoBERTa parameters remain excluded because
    # requires_grad=False.
    native_modules = [
        module for module in (
            getattr(model, "text_encoder", None),
            getattr(model, "audio_text_recognizer", None),
            getattr(model, "native_prosody", None),
            getattr(model, "native_vocal_prior", None),
        )
        if module is not None and bool(getattr(model, "native_generation", False))
    ]
    native_parameter_ids = {
        id(parameter)
        for module in native_modules
        for parameter in module.parameters()
        if parameter.requires_grad
    }
    base_params = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in native_parameter_ids
    ]
    native_params = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) in native_parameter_ids
    ]
    parameter_groups = [{"params": base_params, "lr": learning_rate}]
    if native_params:
        parameter_groups.append({
            "params": native_params,
            "lr": learning_rate * max(1.0, float(native_learning_rate_multiplier)),
        })
    optimizer = torch.optim.AdamW(parameter_groups, lr=learning_rate)
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
        print("optimizer_state_reset_by_request=true", flush=True)

    # Instantiate custom Dataset and DataLoader
    dataset = MusicDiffusionDataset(
        root,
        config,
        records=training_records,
        additional_records=additional_records,
    )
    validation_dataset = (
        MusicDiffusionDataset(
            root,
            config,
            records=validation_records,
            deterministic_crop=True,
            crop_seed=validation_seed,
        )
        if validation_records
        else None
    )
    if not dataset.records:
        raise ValueError("Dataset has no usable records after vocal/transcript quality filtering.")
    
    def collate_fn(batch):
        vocal_mels = torch.stack([item["vocal_mel"] for item in batch])
        backing_mels = torch.stack([item["backing_mel"] for item in batch])
        style_anchors = torch.stack([item["style_anchor"] for item in batch])
        texts = [item["text"] for item in batch]
        return {
            "vocal_mel": vocal_mels,
            "backing_mel": backing_mels,
            "style_anchor": style_anchors,
            "text": texts,
            "frame_text_segments": [item["frame_text_segments"] for item in batch],
            "frame_text_crop_start_seconds": [
                item["frame_text_crop_start_seconds"] for item in batch
            ],
            "frame_text_crop_end_seconds": [
                item["frame_text_crop_end_seconds"] for item in batch
            ],
        }

    batch_size_value = max(1, int(batch_size))

    def build_dataloader(epoch_index: int, *, skip_batches: int = 0):
        # A deterministic per-epoch sampler lets a resumed worker skip batches
        # already covered by its latest mid-epoch checkpoint.
        generator = torch.Generator()
        generator.manual_seed(5602 + int(epoch_index))
        skipped = max(0, int(skip_batches)) * batch_size_value
        if skipped > 0:
            # Reproduce RandomSampler's permutation locally and remove completed
            # indices before DataLoader calls Dataset.__getitem__. The previous
            # `continue` loop re-read hundreds of full-song tensors from Kaggle
            # FUSE just to discard them, wasting several GPU-minutes on resume.
            # DataLoader consumes one int64 from its generator for `_base_seed`
            # before RandomSampler draws the epoch permutation. Mirror that
            # detail so a mid-epoch resume selects exactly the unseen records.
            torch.empty((), dtype=torch.int64).random_(generator=generator)
            indices = torch.randperm(len(dataset), generator=generator).tolist()
            remaining_indices = indices[min(skipped, len(indices)) :]
            return DataLoaderClass(
                dataset,
                batch_size=batch_size_value,
                sampler=remaining_indices,
                shuffle=False,
                collate_fn=collate_fn,
            )
        return DataLoaderClass(
            dataset,
            batch_size=batch_size_value,
            shuffle=True,
            collate_fn=collate_fn,
            generator=generator,
        )

    validation_dataloader = (
        DataLoaderClass(
            validation_dataset,
            batch_size=batch_size_value,
            shuffle=False,
            collate_fn=collate_fn,
        )
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
    warmup_steps = min(
        max(1, int(schedule_total_steps * 0.05)),
        max(1, schedule_total_steps - 1),
    )

    def learning_rate_multiplier(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(
            1.0,
            (step - warmup_steps) / max(1, schedule_total_steps - warmup_steps),
        )
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
        native_ctc_weight=native_ctc_weight,
        native_ctc_teacher_weight=native_ctc_teacher_weight,
        native_frame_text_weight=native_frame_text_weight,
        native_frame_text_teacher_weight=native_frame_text_teacher_weight,
        native_vocal_prior_weight=native_vocal_prior_weight,
        vocal_structure_weight=vocal_structure_weight,
        native_prosody_weight=native_prosody_weight,
        use_amp=use_amp,
    )
    if optimizer_state_restored and resumed_payload.get("scheduler"):
        scheduler.load_state_dict(resumed_payload["scheduler"])
    if resumed_payload.get("ema"):
        trainer.load_ema_state(resumed_payload["ema"])

    saved_training_state = resumed_payload.get("training_state") or {}
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
    best_validation_loss = (
        float(saved_best_validation_loss)
        if saved_best_validation_loss is not None
        else float("inf")
    )
    best_epoch = int(saved_training_state.get("best_epoch", 0)) if saved_best_validation_loss is not None else 0
    epochs_without_improvement = (
        int(saved_training_state.get("epochs_without_improvement", 0))
        if saved_best_validation_loss is not None
        else 0
    )
    final_validation_loss: float | None = None
    final_text_sensitivity: float | None = None
    final_native_ctc_validation: float | None = None
    final_native_ctc_validation_raw: float | None = None
    # A validation-only selector previously replaced a text-responsive model
    # with a marginally lower-loss checkpoint that ignored lyrics. Keep a small
    # tolerance for measurement noise while making intelligibility a hard gate.
    checkpoint_sensitivity_floor = max(
        0.0,
        float(
            minimum_text_sensitivity
            if minimum_text_sensitivity is not None
            else 0.90 * max(0.0, float(text_sensitivity_target))
        ),
    )
    stopped_early = False
    completed_epochs = start_epoch

    from ..models.text_to_music_diffusion import save_checkpoint

    for epoch in range(start_epoch, epoch_count):
        start_batch = resume_batch_in_epoch if epoch == start_epoch else 0
        dataloader = build_dataloader(epoch, skip_batches=start_batch)

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
            start_batch=0,
            batch_offset=start_batch,
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
        native_ctc_pred_values = [
            d["loss_native_ctc_pred"] for d in epoch_losses
            if d["loss_native_ctc_pred"] is not None
        ]
        native_ctc_teacher_values = [
            d["loss_native_ctc_teacher"] for d in epoch_losses
            if d["loss_native_ctc_teacher"] is not None
        ]
        native_frame_text_pred_values = [
            d["loss_native_frame_text_pred"] for d in epoch_losses
            if d["loss_native_frame_text_pred"] is not None
        ]
        native_frame_text_teacher_values = [
            d["loss_native_frame_text_teacher"] for d in epoch_losses
            if d["loss_native_frame_text_teacher"] is not None
        ]
        native_frame_text_prior_values = [
            d["loss_native_frame_text_prior"] for d in epoch_losses
            if d["loss_native_frame_text_prior"] is not None
        ]
        avg_native_ctc_pred = (
            sum(native_ctc_pred_values) / len(native_ctc_pred_values)
            if native_ctc_pred_values else None
        )
        avg_native_ctc_teacher = (
            sum(native_ctc_teacher_values) / len(native_ctc_teacher_values)
            if native_ctc_teacher_values else None
        )
        avg_native_frame_text_pred = (
            sum(native_frame_text_pred_values) / len(native_frame_text_pred_values)
            if native_frame_text_pred_values else None
        )
        avg_native_frame_text_teacher = (
            sum(native_frame_text_teacher_values)
            / len(native_frame_text_teacher_values)
            if native_frame_text_teacher_values else None
        )
        avg_native_frame_text_prior = (
            sum(native_frame_text_prior_values) / len(native_frame_text_prior_values)
            if native_frame_text_prior_values else None
        )
        native_vocal_prior_values = [
            d["loss_native_vocal_prior"] for d in epoch_losses
            if d["loss_native_vocal_prior"] is not None
        ]
        native_vocal_prior_contrastive_values = [
            d["loss_native_vocal_prior_contrastive"]
            for d in epoch_losses
            if d["loss_native_vocal_prior_contrastive"] is not None
        ]
        vocal_structure_values = [
            d["loss_vocal_structure"] for d in epoch_losses
            if d["loss_vocal_structure"] is not None
        ]
        native_prosody_values = [
            d["loss_native_prosody"] for d in epoch_losses
            if d.get("loss_native_prosody") is not None
        ]
        avg_native_vocal_prior = (
            sum(native_vocal_prior_values) / len(native_vocal_prior_values)
            if native_vocal_prior_values else None
        )
        avg_native_vocal_prior_contrastive = (
            sum(native_vocal_prior_contrastive_values)
            / len(native_vocal_prior_contrastive_values)
            if native_vocal_prior_contrastive_values else None
        )
        avg_vocal_structure = (
            sum(vocal_structure_values) / len(vocal_structure_values)
            if vocal_structure_values else None
        )
        avg_native_prosody = (
            sum(native_prosody_values) / len(native_prosody_values)
            if native_prosody_values else None
        )
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
        final_native_ctc_validation = (
            trainer.evaluate_native_ctc(validation_dataloader)
            if validation_dataloader is not None
            else None
        )
        final_native_ctc_validation_raw = (
            trainer.evaluate_native_ctc(validation_dataloader, use_ema=False)
            if validation_dataloader is not None
            else None
        )
        improved = _is_checkpoint_improvement(
            final_validation_loss,
            best_validation_loss,
            final_text_sensitivity,
            checkpoint_sensitivity_floor,
            early_stopping_min_delta,
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
                    "native_ctc_validation": final_native_ctc_validation,
                    "native_ctc_validation_raw": final_native_ctc_validation_raw,
                    "minimum_text_sensitivity": checkpoint_sensitivity_floor,
                },
            )
        elif final_validation_loss is not None:
            epochs_without_improvement += 1
        loss_curve.append({
            "epoch": completed_epochs,
            "loss": avg_loss,
            "loss_gt": avg_loss_gt,
            "loss_velocity": None,
            "loss_vocal_aux": avg_loss_vocal_aux,
            "loss_native_ctc_pred": avg_native_ctc_pred,
            "loss_native_ctc_teacher": avg_native_ctc_teacher,
            "loss_native_frame_text_pred": avg_native_frame_text_pred,
            "loss_native_frame_text_teacher": avg_native_frame_text_teacher,
            "loss_native_frame_text_prior": avg_native_frame_text_prior,
            "loss_native_vocal_prior": avg_native_vocal_prior,
            "loss_native_vocal_prior_contrastive": (
                avg_native_vocal_prior_contrastive
            ),
            "loss_vocal_structure": avg_vocal_structure,
            "loss_native_prosody": avg_native_prosody,
            "validation_loss": final_validation_loss,
            "best_validation_loss": best_validation_loss if math.isfinite(best_validation_loss) else None,
            "text_conditioning_sensitivity": final_text_sensitivity,
            "native_ctc_validation": final_native_ctc_validation,
            "native_ctc_validation_raw": final_native_ctc_validation_raw,
            "minimum_text_sensitivity": checkpoint_sensitivity_floor,
            "conditioning_gate_pass": (
                final_text_sensitivity is None
                or final_text_sensitivity >= checkpoint_sensitivity_floor
            ),
        })
        print(
            f"epoch={completed_epochs}/{epoch_count} train_loss={avg_loss:.6f} "
            f"validation_loss={final_validation_loss if final_validation_loss is not None else 'disabled'} "
            f"text_sensitivity={final_text_sensitivity if final_text_sensitivity is not None else 'disabled'} "
            f"native_ctc_validation={final_native_ctc_validation if final_native_ctc_validation is not None else 'disabled'} "
            f"native_ctc_validation_raw={final_native_ctc_validation_raw if final_native_ctc_validation_raw is not None else 'disabled'} "
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
                    "total_steps": total_steps,
                    "loss": avg_loss,
                    "validation_loss": final_validation_loss,
                    "text_conditioning_sensitivity": final_text_sensitivity,
                    "minimum_text_sensitivity": checkpoint_sensitivity_floor,
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
            print(
                f"Early stopping at epoch {completed_epochs}; best validation epoch was {best_epoch}.",
                flush=True,
            )
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
        "total_steps": total_steps,
        "loss": final_loss,
        "validation_loss": final_validation_loss,
        "best_validation_loss": best_validation_loss if math.isfinite(best_validation_loss) else None,
        "best_epoch": best_epoch,
        "epochs_without_improvement": epochs_without_improvement,
        "stopped_early": stopped_early,
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
    report = {"status": "complete", "backend": "genmusic-vn-self-diffusion", "dataset": str(root.resolve()), "checkpoint": str(checkpoint.resolve()), "best_checkpoint": str(best_checkpoint.resolve()) if best_checkpoint.is_file() else None, "device": selected_device, "requested_epochs": epoch_count, "completed_epochs": completed_epochs, "stopped_early": stopped_early, "best_epoch": best_epoch, "best_validation_loss": round(best_validation_loss, 6) if math.isfinite(best_validation_loss) else None, "final_validation_loss": round(final_validation_loss, 6) if final_validation_loss is not None else None, "final_text_conditioning_sensitivity": round(final_text_sensitivity, 6) if final_text_sensitivity is not None else None, "validation_record_count": len(validation_records), "dataset_validation": validation, "resumed_from_epoch": resumed_from_epoch, "resumed_from_batch": resumed_from_batch, "optimizer_state_restored": optimizer_state_restored, "batch_size": batch_size_value, "record_count": len(dataset.records), "excluded_record_count": dataset.excluded_record_count, "additional_record_count": len(additional_records or []), "step_count": len(losses), "global_step": global_step, "checkpoint_every_steps": max(0, int(checkpoint_every_steps)), "final_loss": round(final_loss, 6), "loss_curve": loss_curve, "elapsed_seconds": round(time.perf_counter() - started, 3), "dim": dim, "depth": depth, "heads": heads, "ff_mult": ff_mult, "frames_per_chunk": config.frames_per_chunk, "chunk_seconds": config.chunk_seconds, "mel_mean": round(config.mel_mean, 6), "mel_std": round(config.mel_std, 6), "warmup_steps": warmup_steps, "ema_decay": trainer.ema_decay, "style_dropout_prob": trainer.style_dropout_prob, "text_dropout_prob": trainer.text_dropout_prob, "text_contrastive_weight": trainer.text_contrastive_weight, "text_contrastive_margin": trainer.text_contrastive_margin, "text_contrastive_prob": trainer.text_contrastive_prob, "text_sensitivity_weight": trainer.text_sensitivity_weight, "text_sensitivity_target": trainer.text_sensitivity_target, "lyric_aligned_crop_prob": dataset.lyric_aligned_crop_prob, "mixed_precision": trainer.use_amp}
    report["lambda_vocal"] = lambda_vocal
    report["text_encoder_type"] = text_encoder_type
    report["native_generation"] = text_encoder_type == "native_utf8"
    report["generation_target"] = getattr(model, "generation_target", generation_target)
    report["joint_stem_checkpoint_migrated"] = bool(
        resumed_payload.get("joint_stem_checkpoint_migrated", False)
    )
    report["native_ctc_tokenizer"] = (
        "vi_grapheme_v1" if text_encoder_type == "native_utf8" else None
    )
    report["native_ctc_classifier_migrated"] = bool(
        resumed_payload.get("native_ctc_classifier_migrated", False)
    )
    report["native_ctc_weight"] = trainer.native_ctc_weight
    report["native_ctc_teacher_weight"] = trainer.native_ctc_teacher_weight
    report["native_frame_text_weight"] = trainer.native_frame_text_weight
    report["native_frame_text_teacher_weight"] = (
        trainer.native_frame_text_teacher_weight
    )
    report["native_vocal_prior_weight"] = trainer.native_vocal_prior_weight
    report["vocal_structure_weight"] = trainer.vocal_structure_weight
    report["native_prosody_weight"] = trainer.native_prosody_weight
    report["native_text_tokenizer"] = (
        "vi_grapheme_v2" if text_encoder_type == "native_utf8" else None
    )
    report["native_ctc_frozen"] = bool(freeze_native_ctc)
    report["optimizer_reset_requested"] = bool(reset_optimizer)
    report["native_learning_rate_multiplier"] = max(
        1.0, float(native_learning_rate_multiplier)
    )
    report["final_native_ctc_validation"] = (
        round(final_native_ctc_validation, 6)
        if final_native_ctc_validation is not None else None
    )
    report["final_native_ctc_validation_raw"] = (
        round(final_native_ctc_validation_raw, 6)
        if final_native_ctc_validation_raw is not None else None
    )
    report["minimum_text_sensitivity"] = checkpoint_sensitivity_floor
    report["text_sensitivity_mode"] = "matched_vs_mismatched_lyrics"
    (checkpoint.parent / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
