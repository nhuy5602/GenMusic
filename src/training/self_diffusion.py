"""Dataset, DataLoader and training loop for the self-authored music diffusion model."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..models.text_to_music_diffusion import MusicDiffusionConfig, diffusion_loss, make_model, structured_random_mel

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

class MusicDiffusionDataset:
    """PyTorch Dataset mapping structured Mel-spectrograms and text/style prompts."""
    def __init__(self, dataset_dir: str | Path, config: MusicDiffusionConfig, max_records: int | None = None, additional_records: list[dict[str, Any]] | None = None):
        _, _, Dataset, _ = _torch()
        self.root = Path(dataset_dir)
        self.config = config
        self.records = _read_records(self.root)[:max_records or None]
        if additional_records:
            self.records.extend(additional_records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        torch, _, _, _ = _torch()
        record = self.records[idx]
        
        # Load vocal Mel (target x1) and backing Mel (condition cond)
        # Fallback to single mel if separated paths are not present in dataset
        if "vocal_mel_path" in record and "backing_mel_path" in record:
            vocal_path = self.root / record["vocal_mel_path"]
            backing_path = self.root / record["backing_mel_path"]
            vocal_mel = _load_mel(vocal_path)
            backing_mel = _load_mel(backing_path)
        else:
            # Fallback for old/smoke dataset
            mel_path = _record_path(self.root, record)
            vocal_mel = _load_mel(mel_path)
            backing_mel = torch.zeros_like(vocal_mel)
            
        vocal_mel = _fit_mel_frames(vocal_mel, self.config.frames_per_chunk)
        backing_mel = _fit_mel_frames(backing_mel, self.config.frames_per_chunk)

        # Style anchor: a precomputed MuQ-MuLan audio embedding of the song (see
        # preprocess_raw_vietnamese.py), the same contrastive audio-style space
        # the real DiffRhythm2 teacher conditions on. Falls back to a zero vector
        # for older/synthetic datasets that never computed one.
        style_path = record.get("style_embed_path")
        if style_path and (self.root / style_path).exists():
            style_anchor = _load_mel(self.root / style_path).float().view(-1)
        else:
            style_anchor = torch.zeros(STYLE_EMBED_DIM)

        text = f"{record['style']}. {record['text']}"
        return {"vocal_mel": vocal_mel, "backing_mel": backing_mel, "style_anchor": style_anchor, "text": text}

class DiffusionTrainer:
    """Trainer orchestrating optimization steps and gradient descent for the diffusion denoiser."""
    def __init__(self, model, config: MusicDiffusionConfig, optimizer, device: str = "cpu"):
        self.model = model
        self.config = config
        self.optimizer = optimizer
        self.device = device

    def train_epoch(self, dataloader) -> list[float]:
        torch, _, _, _ = _torch()
        self.model.train()
        epoch_losses = []
        for batch in dataloader:
            vocal_mel = batch["vocal_mel"].to(self.device)
            backing_mel = batch["backing_mel"].to(self.device)
            style_anchor = batch["style_anchor"].to(self.device)
            texts = batch["text"]
            self.optimizer.zero_grad(set_to_none=True)
            
            # Check if using the new MicroDiT model
            is_dit = self.model.__class__.__name__ == "MicroDiT"
            if is_dit:
                # Transpose mels from (batch, n_mels, seq_len) to (batch, seq_len, n_mels) for DiT
                vocal_mel_t = vocal_mel.transpose(1, 2)
                backing_mel_t = backing_mel.transpose(1, 2)
                from ..models.cfm_flow import cfm_loss
                loss = cfm_loss(self.model, vocal_mel_t, backing_mel_t, style_anchor, texts, self.config)
            else:
                # Fallback to single mel input for the old Conv1D model
                loss = diffusion_loss(self.model, vocal_mel, texts, self.config)
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.model.parameters()), 1.0)
            self.optimizer.step()
            # "loss_gt" mirrors distill_training's field name (there is no teacher
            # here, so loss == loss_gt) so baseline and distilled runs can be
            # compared on the same axis -- see docs/experiments/*.md.
            epoch_losses.append({"loss": float(loss.detach().cpu()), "loss_gt": float(loss.detach().cpu()), "loss_velocity": None})
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

def _record_path(root: Path, record: dict[str, Any]) -> Path:
    path_str = record.get("mel_path") or record.get("backing_mel_path") or record.get("vocal_mel_path")
    if not path_str:
        raise KeyError("Record missing 'mel_path', 'backing_mel_path', or 'vocal_mel_path'")
    path = Path(path_str)
    return path if path.is_absolute() else root / path

def _fit_mel_frames(mel, frames: int):
    torch, _, _, _ = _torch()
    if mel.shape[1] > frames:
        return mel[:, :frames]
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
        path = _record_path(root, record)
        if not path.exists():
            missing.append(str(path))
            continue
        tensor = _load_mel(path)
        if tuple(tensor.shape) != (expected_mels, int(record["frames"])):
            invalid.append({"path": str(path), "shape": list(tensor.shape), "expected": [expected_mels, int(record["frames"])]})
    report = {"status": "valid" if not missing and not invalid else "invalid", "dataset": str(root.resolve()), "record_count": len(records), "missing": missing, "invalid": invalid, "format": "genmusic-self-diffusion-v1"}
    report_destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report

def train_model(dataset_dir: str | Path, checkpoint_path: str | Path, *, epochs: int = 1, batch_size: int = 4, learning_rate: float = 2e-4, device: str | None = None, max_records: int | None = None, additional_records: list[dict[str, Any]] | None = None, model_type: str = "conv1d", roberta_model: str = "xlm-roberta-base", dim: int = 256, depth: int = 4, heads: int = 4, ff_mult: int = 4) -> dict[str, Any]:
    torch, _, DatasetClass, DataLoaderClass = _torch()

    root = Path(dataset_dir)
    checkpoint = Path(checkpoint_path)
    validation = validate_dataset(root, report_path=checkpoint.parent / "validation_report.json")
    if validation["status"] != "valid":
        raise ValueError("Dataset không hợp lệ; xem validation_report.json.")
    
    config = MusicDiffusionConfig(**json.loads((root / "config.json").read_text(encoding="utf-8")))
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_type == "dit":
        from ..models.dit_transformer import MicroDiT
        model = MicroDiT(config, roberta_model=roberta_model, dim=dim, depth=depth, heads=heads, ff_mult=ff_mult).to(selected_device)
        # Train only parameters that requires_grad (i.e. exclude frozen RoBERTa weights)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
    else:
        model = make_model(config).to(selected_device)
        optimizer = torch.optim.AdamW(list(model.parameters()), lr=learning_rate)
    
    # Instantiate custom Dataset and DataLoader
    dataset = MusicDiffusionDataset(root, config, max_records=max_records, additional_records=additional_records)
    
    def collate_fn(batch):
        vocal_mels = torch.stack([item["vocal_mel"] for item in batch])
        backing_mels = torch.stack([item["backing_mel"] for item in batch])
        style_anchors = torch.stack([item["style_anchor"] for item in batch])
        texts = [item["text"] for item in batch]
        return {"vocal_mel": vocal_mels, "backing_mel": backing_mels, "style_anchor": style_anchors, "text": texts}

    dataloader = DataLoaderClass(
        dataset, 
        batch_size=max(1, int(batch_size)), 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    trainer = DiffusionTrainer(model, config, optimizer, device=selected_device)
    
    started = time.perf_counter()
    losses = []
    loss_curve = []

    for epoch in range(max(1, int(epochs))):
        epoch_losses = trainer.train_epoch(dataloader)
        losses.extend(epoch_losses)
        avg_loss = sum(d["loss"] for d in epoch_losses) / len(epoch_losses)
        loss_curve.append({"epoch": epoch + 1, "loss": avg_loss, "loss_gt": avg_loss, "loss_velocity": None})

    final_loss = sum(d["loss"] for d in losses[-min(10, len(losses)):]) / max(1, min(10, len(losses)))
    from ..models.text_to_music_diffusion import save_checkpoint

    arch = {"dim": dim, "depth": depth, "heads": heads, "ff_mult": ff_mult} if model_type == "dit" else None
    save_checkpoint(model, checkpoint, config, optimizer=optimizer, epoch=max(1, int(epochs)), loss=final_loss, arch=arch)
    report = {"status": "complete", "backend": "genmusic-vn-self-diffusion", "dataset": str(root.resolve()), "checkpoint": str(checkpoint.resolve()), "device": selected_device, "epochs": max(1, int(epochs)), "batch_size": max(1, int(batch_size)), "additional_record_count": len(additional_records or []), "step_count": len(losses), "final_loss": round(final_loss, 6), "loss_curve": loss_curve, "elapsed_seconds": round(time.perf_counter() - started, 3), "model_type": model_type, "dim": dim, "depth": depth, "heads": heads, "ff_mult": ff_mult}
    (checkpoint.parent / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
