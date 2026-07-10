"""Supervised flow-matching training over preprocessed latent records."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    DataLoader = Any  # type: ignore[misc,assignment]
    Dataset = object  # type: ignore[assignment]

from ..models.jam_diffrhythm import ConditionalDiT, ConditionalFlowMatching, DiTConfig, count_parameters, model_config, require_torch


@dataclass(frozen=True)
class SFTConfig:
    preset: str = "demo"
    steps: int = 1000
    batch_size: int = 2
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    max_frames: int = 6144
    num_workers: int = 0
    device: str = "auto"
    seed: int = 5602
    save_every: int = 250


def _load_records(manifest: str | Path) -> list[dict[str, Any]]:
    base_dir = Path(manifest).parent
    records: list[dict[str, Any]] = []
    for line in Path(manifest).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        for field in ("latent_path", "style_path", "lrc_path"):
            if record.get(field) and not Path(record[field]).is_absolute():
                record[field] = str((base_dir / record[field]).resolve())
        records.append(record)
    return records


def _load_pt(path: str | Path) -> Any:
    value = torch.load(path, map_location="cpu", weights_only=False)
    return value["tensor"] if isinstance(value, dict) and "tensor" in value else value


class LatentManifestDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], *, max_frames: int):
        require_torch()
        self.records = records
        self.max_frames = max_frames
        tokens = sorted({token for record in records for token in record.get("phoneme_tokens", [])})
        self.vocabulary = {"<pad>": 0, "<unk>": 1, **{token: index + 2 for index, token in enumerate(tokens)}}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        latent = _load_pt(record["latent_path"]).float()[: self.max_frames]
        style = _load_pt(record["style_path"]).float().reshape(-1)
        text = torch.tensor([self.vocabulary.get(token, 1) for token in record.get("phoneme_tokens", [])], dtype=torch.long)
        return {"latent": latent, "style": style, "text": text, "id": record["id"]}


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    target_frames = max(item["latent"].shape[0] for item in batch)
    latent_dim = batch[0]["latent"].shape[-1]
    latent = torch.zeros(len(batch), target_frames, latent_dim, dtype=torch.float32)
    for index, item in enumerate(batch):
        latent[index, : item["latent"].shape[0]] = item["latent"]
    text = torch.nn.utils.rnn.pad_sequence([item["text"] for item in batch], batch_first=True, padding_value=0)
    style = torch.stack([item["style"] for item in batch])
    return {"latent": latent, "cond": torch.zeros_like(latent), "style": style, "text": text, "ids": [item["id"] for item in batch]}


def _device(requested: str) -> Any:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _save_checkpoint(path: Path, model: ConditionalDiT, config: DiTConfig, training: SFTConfig, history: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "model_config": config.as_dict(), "training_config": asdict(training), "history": history, "parameter_count": count_parameters(model)}, path)


def train_sft_from_manifest(manifest: str | Path, output_dir: str | Path, *, config: SFTConfig | None = None) -> dict[str, Any]:
    """Train the project-owned CFM DiT and write a reproducible report."""

    require_torch()
    config = config or SFTConfig()
    torch.manual_seed(config.seed)
    records = _load_records(manifest)
    if not records:
        raise ValueError("Manifest không có record hợp lệ để train.")
    model_cfg = model_config(config.preset)
    model = ConditionalDiT(model_cfg).to(_device(config.device))
    flow = ConditionalFlowMatching(model)
    dataset = LatentManifestDataset(records, max_frames=config.max_frames)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, collate_fn=_collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    device = _device(config.device)
    history: list[float] = []
    iterator = iter(loader)
    model.train()
    for step in range(1, config.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        latent = batch["latent"].to(device)
        cond = batch["cond"].to(device)
        text = batch["text"].to(device)
        style = batch["style"].to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = flow.loss(latent, cond, text, style)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        history.append(float(loss.detach().cpu()))
        if step % max(1, config.save_every) == 0 or step == config.steps:
            _save_checkpoint(Path(output_dir) / "sft_checkpoint.pt", model, model_cfg, config, history)
    vocabulary_path = Path(output_dir) / "phoneme_vocabulary.json"
    vocabulary_path.parent.mkdir(parents=True, exist_ok=True)
    vocabulary_path.write_text(json.dumps(dataset.vocabulary, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        "stage": "sft-cfm-dit",
        "records": len(records),
        "steps": config.steps,
        "initial_loss": history[0],
        "final_loss": history[-1],
        "parameter_count": count_parameters(model),
        "model_config": model_cfg.as_dict(),
        "training_config": asdict(config),
        "checkpoint": str((Path(output_dir) / "sft_checkpoint.pt").resolve()),
        "vocabulary": str(vocabulary_path.resolve()),
        "status": "trained",
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "sft_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
