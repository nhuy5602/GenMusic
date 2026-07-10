"""Dependency-light WebDataset shard writer for preprocessed records."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path
from typing import Any, Iterator


def _read_manifest(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def pack_jam_webdataset(manifest_path: str | Path, output_dir: str | Path, *, shard_size: int = 1000) -> dict[str, Any]:
    """Pack ``.pt``, ``.lrc``, phoneme text and JSON metadata into tar shards."""

    manifest_source = Path(manifest_path)
    records = _read_manifest(manifest_source)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    shards: list[str] = []
    for shard_index in range(0, len(records), max(1, shard_size)):
        shard_records = records[shard_index : shard_index + max(1, shard_size)]
        shard_path = destination / f"shard-{shard_index // max(1, shard_size):06d}.tar"
        with tarfile.open(shard_path, "w") as archive:
            for record in shard_records:
                key = str(record["id"])
                metadata = {key_name: value for key_name, value in record.items() if key_name not in {"latent_path", "style_path", "lrc_path"}}
                _add_bytes(archive, f"{key}.json", json.dumps(metadata, ensure_ascii=False).encode("utf-8"))
                _add_file(archive, f"{key}.latent.pt", _resolve_path(record["latent_path"], manifest_source.parent))
                _add_file(archive, f"{key}.style.pt", _resolve_path(record["style_path"], manifest_source.parent))
                _add_file(archive, f"{key}.lrc", _resolve_path(record["lrc_path"], manifest_source.parent))
                _add_bytes(archive, f"{key}.phonemes.txt", " ".join(record.get("phoneme_tokens", [])).encode("utf-8"))
        shards.append(str(shard_path.resolve()))
    index = {"manifest": str(Path(manifest_path).resolve()), "shard_count": len(shards), "record_count": len(records), "shards": shards}
    (destination / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return index


def _resolve_path(path: str | Path, base_dir: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else base_dir / candidate


def _add_bytes(archive: tarfile.TarFile, name: str, value: bytes) -> None:
    import io

    info = tarfile.TarInfo(name)
    info.size = len(value)
    archive.addfile(info, io.BytesIO(value))


def _add_file(archive: tarfile.TarFile, name: str, path: str | Path) -> None:
    archive.add(str(path), arcname=name, recursive=False)


def iter_webdataset_records(shard_dir: str | Path) -> Iterator[dict[str, Any]]:
    """Read metadata-only records for inspection without importing WebDataset."""

    for shard in sorted(Path(shard_dir).glob("shard-*.tar")):
        with tarfile.open(shard, "r") as archive:
            for member in archive.getmembers():
                if not member.name.endswith(".json"):
                    continue
                handle = archive.extractfile(member)
                if handle is not None:
                    yield json.loads(handle.read().decode("utf-8"))
