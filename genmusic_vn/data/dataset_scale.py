from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .training_dataset import generate_diverse_training_records


def write_large_diverse_dataset(
    output_root: str | Path,
    *,
    target_bytes: int,
    seed: int = 5602,
    shard_bytes: int = 128 * 1024 * 1024,
    batch_size: int = 2000,
    max_records: int | None = None,
) -> dict[str, Any]:
    """Stream original synthetic records into shards until a byte target is reached."""
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)
    target_bytes = max(1, int(target_bytes))
    shard_bytes = max(1_048_576, int(shard_bytes))
    batch_size = max(1, int(batch_size))
    records_written = 0
    bytes_written = 0
    shard_index = 0
    current_handle = None
    current_shard_bytes = 0
    shard_paths: list[str] = []

    try:
        while bytes_written < target_bytes and (max_records is None or records_written < max_records):
            remaining_records = None if max_records is None else max_records - records_written
            batch_count = batch_size if remaining_records is None else min(batch_size, remaining_records)
            batch = generate_diverse_training_records(
                batch_count,
                seed=seed + shard_index + records_written,
                start_index=records_written + 1,
            )
            for record in batch:
                line = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
                if current_handle is None or current_shard_bytes and current_shard_bytes + len(line) > shard_bytes:
                    if current_handle is not None:
                        current_handle.close()
                    shard_path = output_path / f"diverse_shard_{shard_index:04d}.jsonl"
                    current_handle = shard_path.open("wb")
                    shard_paths.append(str(shard_path))
                    current_shard_bytes = 0
                    shard_index += 1
                current_handle.write(line)
                current_shard_bytes += len(line)
                bytes_written += len(line)
                records_written += 1
                if bytes_written >= target_bytes or (max_records is not None and records_written >= max_records):
                    break
            if not batch:
                break
    finally:
        if current_handle is not None:
            current_handle.close()

    manifest = {
        "status": "complete",
        "output_root": str(output_path),
        "target_bytes": target_bytes,
        "target_gb_decimal": round(target_bytes / 1_000_000_000, 4),
        "bytes_written": bytes_written,
        "size_gb_decimal": round(bytes_written / 1_000_000_000, 4),
        "records_written": records_written,
        "shard_count": len(shard_paths),
        "shards": shard_paths,
        "seed": seed,
        "batch_size": batch_size,
        "shard_bytes": shard_bytes,
        "source": "generated_diverse_training_dataset",
        "training_note": "Use directory sampling when training; do not load all shards into RAM at once.",
    }
    manifest_path = output_path / "dataset_manifest.json"
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def gigabytes_to_bytes(gigabytes: float) -> int:
    return max(1, math.ceil(float(gigabytes) * 1_000_000_000))
