"""Dataset, DataLoader and training loop for the self-authored music diffusion model."""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..models.text_to_music_diffusion import MusicDiffusionConfig, normalize_mel, structured_random_mel

STYLE_EMBED_DIM = 512  # matches MuQ-MuLan / DiffRhythm2 teacher's cond_dim

# Ensure PyTorch helper works
def _torch():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import Dataset, DataLoader
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

class MusicDiffusionDataset:
    """PyTorch Dataset mapping structured Mel-spectrograms and text/style prompts."""
    def __init__(self, dataset_dir: str | Path, config: MusicDiffusionConfig, max_records: int | None = None, additional_records: list[dict[str, Any]] | None = None):
        _, _, Dataset, _ = _torch()
        self.root = Path(dataset_dir)
        self.config = config
        all_records = _read_records(self.root)
        records = _filter_training_records(all_records)
        self.excluded_record_count = len(all_records) - len(records)
        self.records = records[:max_records] if max_records is not None else records
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
            
        # Crop both stems at the same offset so the text/audio condition stays aligned
        # (also a cheap augmentation: different epochs see different windows of longer songs).
        crop_start = 0
        shared_frames = min(vocal_mel.shape[1], backing_mel.shape[1])
        if shared_frames > self.config.frames_per_chunk:
            crop_start = random.randint(0, shared_frames - self.config.frames_per_chunk)
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

        lyric_text = str(record["text"])
        segments = record.get("segments") or []
        if segments:
            crop_start_seconds = crop_start * self.config.hop_length / self.config.sample_rate
            crop_end_seconds = crop_start_seconds + self.config.frames_per_chunk * self.config.hop_length / self.config.sample_rate
            lyric_text = lyric_text_for_window(lyric_text, segments, crop_start_seconds, crop_end_seconds)
        return {"vocal_mel": vocal_mel, "backing_mel": backing_mel, "style_anchor": style_anchor, "text": lyric_text}

class DiffusionTrainer:
    """Trainer orchestrating optimization steps and gradient descent for the diffusion denoiser."""
    def __init__(self, model, config: MusicDiffusionConfig, optimizer, device: str = "cpu", scheduler=None, ema_decay: float = 0.999):
        torch, _, _, _ = _torch()
        self.model = model
        self.config = config
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.ema_decay = float(ema_decay)
        self.use_amp = str(device).startswith("cuda")
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
            if name in self.ema_parameters:
                self.ema_parameters[name] = value.detach().to(self.device).clone()

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
            style_anchor = batch["style_anchor"].to(self.device)
            texts = batch["text"]
            self.optimizer.zero_grad(set_to_none=True)

            # Transpose mels from (batch, n_mels, seq_len) to (batch, seq_len, n_mels) for DiT.
            # style_anchor is already a flat (batch, 512) MuQ-MuLan embedding, not
            # mel-shaped, so it needs no transpose.
            vocal_mel_t = vocal_mel.transpose(1, 2)
            from ..models.cfm_flow import cfm_loss
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
                loss = cfm_loss(self.model, vocal_mel_t, style_anchor, texts, self.config)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(list(self.model.parameters()), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler is not None:
                self.scheduler.step()
            self._update_ema()
            # "loss_gt" mirrors distill_training's field name (there is no teacher
            # here, so loss == loss_gt) so baseline and distilled runs can be
            # compared on the same axis -- see docs/experiments/*.md.
            loss_value = float(loss.detach().cpu())
            loss_record = {
                "loss": loss_value,
                "loss_gt": loss_value,
                "loss_velocity": None,
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
    expected_mels = int(config_data.get("n_mels", 64))
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
    roberta_model: str = "xlm-roberta-base",
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
) -> dict[str, Any]:
    torch, _, _, DataLoaderClass = _torch()

    root = Path(dataset_dir)
    checkpoint = Path(checkpoint_path)
    validation = validate_dataset(root, report_path=checkpoint.parent / "validation_report.json")
    if validation["status"] != "valid":
        raise ValueError("Dataset không hợp lệ; xem validation_report.json.")

    config = MusicDiffusionConfig(**json.loads((root / "config.json").read_text(encoding="utf-8")))
    if frames_per_chunk is not None:
        frames = max(16, int(frames_per_chunk))
        config = replace(
            config,
            frames_per_chunk=frames,
            chunk_seconds=frames * config.hop_length / config.sample_rate,
        )
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    arch = {"dim": dim, "depth": depth, "heads": heads, "ff_mult": ff_mult}
    resumed_payload: dict[str, Any] = {}
    start_epoch = 0
    resume_batch_in_epoch = 0

    if resume and checkpoint.is_file():
        from ..models.text_to_music_diffusion import load_checkpoint

        model, saved_config, resumed_payload = load_checkpoint(
            checkpoint,
            device=selected_device,
            roberta_model=roberta_model,
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
        usable_records = _filter_training_records(_read_records(root))
        records_for_stats = usable_records[:max_records] if max_records is not None else usable_records
        mel_mean, mel_std = estimate_vocal_mel_stats(root, records_for_stats)
        config = replace(config, mel_mean=mel_mean, mel_std=mel_std)
        from ..models.dit_transformer import MicroDiT

        model = MicroDiT(
            config,
            roberta_model=roberta_model,
            dim=dim,
            depth=depth,
            heads=heads,
            ff_mult=ff_mult,
        ).to(selected_device)

    # Train only parameters that require gradients (the frozen RoBERTa weights
    # are re-downloaded on load and intentionally omitted from checkpoints).
    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
    if resumed_payload.get("optimizer"):
        optimizer.load_state_dict(resumed_payload["optimizer"])

    # Instantiate custom Dataset and DataLoader
    dataset = MusicDiffusionDataset(root, config, max_records=max_records, additional_records=additional_records)
    if not dataset.records:
        raise ValueError("Dataset has no usable records after vocal/transcript quality filtering.")
    
    def collate_fn(batch):
        vocal_mels = torch.stack([item["vocal_mel"] for item in batch])
        backing_mels = torch.stack([item["backing_mel"] for item in batch])
        style_anchors = torch.stack([item["style_anchor"] for item in batch])
        texts = [item["text"] for item in batch]
        return {"vocal_mel": vocal_mels, "backing_mel": backing_mels, "style_anchor": style_anchors, "text": texts}

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

    epoch_count = max(1, int(epochs))
    steps_per_epoch = len(build_dataloader(0))
    total_steps = max(1, epoch_count * steps_per_epoch)
    warmup_steps = min(max(1, int(total_steps * 0.05)), max(1, total_steps - 1))

    def learning_rate_multiplier(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_multiplier)
    trainer = DiffusionTrainer(model, config, optimizer, device=selected_device, scheduler=scheduler)
    if resumed_payload.get("scheduler"):
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
                    arch=arch,
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
        loss_curve.append({"epoch": epoch + 1, "loss": avg_loss, "loss_gt": avg_loss, "loss_velocity": None})
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
                epoch=epoch + 1,
                loss=avg_loss,
                arch=arch,
                training_state={
                    "status": "training",
                    "epoch": epoch + 1,
                    "display_epoch": min(epoch + 2, epoch_count),
                    "batch_in_epoch": 0,
                    "batches_per_epoch": steps_per_epoch,
                    "global_step": global_step,
                    "total_steps": total_steps,
                    "loss": avg_loss,
                    "checkpoint": str(checkpoint.resolve()),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        resume_batch_in_epoch = 0

    final_loss = (
        sum(d["loss"] for d in losses[-min(10, len(losses)):])
        / max(1, min(10, len(losses)))
        if losses
        else float(resumed_payload.get("loss") or 0.0)
    )
    completed_training_state = {
        "status": "complete",
        "epoch": epoch_count,
        "display_epoch": epoch_count,
        "batch_in_epoch": 0,
        "batches_per_epoch": steps_per_epoch,
        "global_step": global_step,
        "total_steps": total_steps,
        "loss": final_loss,
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
        epoch=epoch_count,
        loss=final_loss,
        arch=arch,
        training_state=completed_training_state,
    )
    _write_json_atomic(progress_destination, completed_training_state)
    report = {"status": "complete", "backend": "genmusic-vn-self-diffusion", "dataset": str(root.resolve()), "checkpoint": str(checkpoint.resolve()), "device": selected_device, "epochs": epoch_count, "resumed_from_epoch": resumed_from_epoch, "resumed_from_batch": resumed_from_batch, "batch_size": batch_size_value, "record_count": len(dataset.records), "excluded_record_count": dataset.excluded_record_count, "additional_record_count": len(additional_records or []), "step_count": len(losses), "global_step": global_step, "checkpoint_every_steps": max(0, int(checkpoint_every_steps)), "final_loss": round(final_loss, 6), "loss_curve": loss_curve, "elapsed_seconds": round(time.perf_counter() - started, 3), "dim": dim, "depth": depth, "heads": heads, "ff_mult": ff_mult, "frames_per_chunk": config.frames_per_chunk, "chunk_seconds": config.chunk_seconds, "mel_mean": round(config.mel_mean, 6), "mel_std": round(config.mel_std, 6), "warmup_steps": warmup_steps, "ema_decay": trainer.ema_decay, "mixed_precision": trainer.use_amp}
    (checkpoint.parent / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
