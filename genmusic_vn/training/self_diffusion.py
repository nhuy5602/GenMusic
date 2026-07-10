"""Dataset and training loop for the self-authored music diffusion model."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..models.text_to_music_diffusion import MusicDiffusionConfig, diffusion_loss, make_model, structured_random_mel


DEFAULT_TEXTS = [
    ("Mưa rơi trên mái hiên, lòng nghe bình yên.", "Vietnamese soft ballad, piano, warm strings, gentle beat"),
    ("Bước qua con phố, ta nhìn thấy bình minh.", "uplifting Vietnamese pop, acoustic guitar, bright drums"),
    ("Đêm nay thành phố ngủ quên trong tiếng gió.", "lonely ambient piano, slow pulse, spacious reverb"),
    ("Cùng nhau đi tới nơi ngày mai đang gọi.", "hopeful indie pop, steady rhythm, warm synths"),
]


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
        import torch

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
    import torch

    value = torch.load(path, map_location=device, weights_only=True)
    return value["mel"] if isinstance(value, dict) else value


def validate_dataset(dataset_dir: str | Path, *, report_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(dataset_dir)
    report_destination = Path(report_path) if report_path else root / "validation_report.json"
    report_destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        import torch
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
        path = root / record["mel_path"]
        if not path.exists():
            missing.append(str(path))
            continue
        tensor = _load_mel(path)
        if tuple(tensor.shape) != (expected_mels, int(record["frames"])):
            invalid.append({"path": str(path), "shape": list(tensor.shape), "expected": [expected_mels, int(record["frames"])]})
    report = {"status": "valid" if not missing and not invalid else "invalid", "dataset": str(root.resolve()), "record_count": len(records), "missing": missing, "invalid": invalid, "format": "genmusic-self-diffusion-v1"}
    report_destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def train_model(dataset_dir: str | Path, checkpoint_path: str | Path, *, epochs: int = 1, batch_size: int = 4, learning_rate: float = 2e-4, device: str | None = None, max_records: int | None = None) -> dict[str, Any]:
    import torch

    root = Path(dataset_dir)
    checkpoint = Path(checkpoint_path)
    validation = validate_dataset(root, report_path=checkpoint.parent / "validation_report.json")
    if validation["status"] != "valid":
        raise ValueError("Dataset không hợp lệ; xem validation_report.json.")
    config = MusicDiffusionConfig(**json.loads((root / "config.json").read_text(encoding="utf-8")))
    records = _read_records(root)[:max_records or None]
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(config).to(selected_device)
    optimizer = torch.optim.AdamW(list(model.parameters()), lr=learning_rate)
    started = time.perf_counter()
    losses = []
    model.train()
    for epoch in range(max(1, int(epochs))):
        random.shuffle(records)
        for start in range(0, len(records), max(1, int(batch_size))):
            batch = records[start : start + max(1, int(batch_size))]
            mel = torch.stack([_load_mel(root / record["mel_path"], device=selected_device) for record in batch])
            texts = [f"{record['style']}. {record['text']}" for record in batch]
            optimizer.zero_grad(set_to_none=True)
            loss = diffusion_loss(model, mel, texts, config)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    final_loss = sum(losses[-min(10, len(losses)) :]) / max(1, min(10, len(losses)))
    model_path = checkpoint
    from ..models.text_to_music_diffusion import save_checkpoint

    save_checkpoint(model, model_path, config, optimizer=optimizer, epoch=max(1, int(epochs)), loss=final_loss)
    report = {"status": "complete", "backend": "genmusic-vn-self-diffusion", "dataset": str(root.resolve()), "checkpoint": str(checkpoint.resolve()), "device": selected_device, "epochs": max(1, int(epochs)), "batch_size": max(1, int(batch_size)), "step_count": len(losses), "final_loss": round(final_loss, 6), "elapsed_seconds": round(time.perf_counter() - started, 3)}
    (checkpoint.parent / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
