from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable

from .training_dataset import normalize_training_record


def load_reference_records(
    paths: Iterable[str | Path],
    *,
    max_records: int | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Load labeled reference rows supplied by the caller or crawler.

    Reference content is intentionally not embedded in the package. Rows must
    contain text, an emotion label, and a genre label so the model cannot
    silently invent missing reference metadata.
    """
    candidates: list[dict[str, Any]] = []
    for path_value in paths:
        path = Path(path_value)
        if not path.exists():
            continue
        source_paths = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
        for source_path in source_paths:
            for raw in _read_json_rows(source_path):
                if _is_labeled_reference(raw):
                    candidates.append(raw)

    if max_records is None or max_records <= 0 or len(candidates) <= max_records:
        return candidates
    rng = random.Random(seed)
    return rng.sample(candidates, max_records)


def load_reference_training_records(
    paths: Iterable[str | Path],
    *,
    max_records: int | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Normalize externally supplied reference rows for text-model training."""
    records: list[dict[str, Any]] = []
    for raw in load_reference_records(paths, max_records=max_records, seed=seed):
        normalized = normalize_training_record(raw)
        if normalized:
            normalized["source"] = str(raw.get("source") or "external_reference_dataset")
            records.append(normalized)
    return records


def load_reference_eval_records(
    paths: Iterable[str | Path],
    *,
    max_records: int | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Adapt externally supplied rows to the evaluation schema without inventing lyric data."""
    records: list[dict[str, Any]] = []
    for raw in load_reference_records(paths, max_records=max_records, seed=seed):
        normalized = normalize_training_record(raw)
        if not normalized:
            continue
        record = dict(raw)
        record["id"] = str(record.get("id") or normalized["id"])
        record["input_text"] = normalized["input_text"]
        record["expected_emotions"] = list(
            record.get("expected_emotions") or [normalized["emotion"]]
        )
        record["expected_keywords"] = list(
            record.get("expected_keywords") or normalized["expected_keywords"]
        )
        record["expected_vocal_gender"] = str(
            record.get("expected_vocal_gender") or normalized.get("expected_vocal_gender") or ""
        )
        record["genre"] = str(record.get("genre") or normalized["style_prompt"])
        record["genre_label"] = normalized["genre_label"]
        record["length_bucket"] = str(record.get("length_bucket") or "reference_dataset")
        record["source"] = str(record.get("source") or "external_reference_dataset")
        records.append(record)
    return records


def _read_json_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return rows
    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return rows
        values = payload if isinstance(payload, list) else [payload]
        return [dict(item) for item in values if isinstance(item, dict)]
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _is_labeled_reference(record: dict[str, Any]) -> bool:
    text = str(record.get("input_text") or record.get("text") or record.get("chorus") or "").strip()
    emotion = str(record.get("emotion") or "").strip()
    if not emotion:
        expected_emotions = record.get("expected_emotions")
        emotion = str(expected_emotions[0]).strip() if isinstance(expected_emotions, list) and expected_emotions else ""
    genre_label = str(record.get("genre_label") or "").strip()
    return bool(text and emotion and genre_label)
