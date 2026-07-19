"""Build GenMusic tensors from clean, word-aligned Vietnamese song chunks."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from src.data.preprocess_raw_vietnamese import (
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    SAMPLE_RATE,
    process_file,
    run_demucs_batch,
)
from src.training.self_diffusion import clean_vietnamese_lyric


DEFAULT_REPO_ID = "sunbv56/song_dataset_training_20s_cleaned"
DEFAULT_SHARD = "data/train-00000-of-00063.parquet"


def aligned_segments_from_row(row: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Convert absolute millisecond word times to chunk-relative seconds."""
    chunk_start_ms = int(row.get("chunk_start_ms") or 0)
    chunk_end_ms = int(row.get("chunk_end_ms") or chunk_start_ms)
    duration_seconds = max(0.0, (chunk_end_ms - chunk_start_ms) / 1000.0)
    segments: list[dict[str, Any]] = []

    for sentence in row.get("chunk_word_timestamps") or []:
        words: list[dict[str, Any]] = []
        for item in sentence or []:
            word = str(item.get("word") or "").strip()
            if not word:
                continue
            start = max(0.0, (float(item.get("start", chunk_start_ms)) - chunk_start_ms) / 1000.0)
            end = max(start, (float(item.get("end", chunk_start_ms)) - chunk_start_ms) / 1000.0)
            start = min(start, duration_seconds)
            end = min(end, duration_seconds)
            if end > start:
                words.append(
                    {
                        "start": round(start, 3),
                        "end": round(end, 3),
                        "word": word,
                    }
                )
        if not words:
            continue
        text = clean_vietnamese_lyric(" ".join(word["word"] for word in words))
        if not text:
            continue
        segments.append(
            {
                "start": words[0]["start"],
                "end": words[-1]["end"],
                "text": text,
                "words": words,
            }
        )

    full_text = clean_vietnamese_lyric(" ".join(segment["text"] for segment in segments))
    return full_text, segments


def aligned_row_is_usable(
    row: dict[str, Any],
    *,
    minimum_duration_seconds: float = 8.0,
    maximum_duration_seconds: float = 22.0,
    minimum_words: int = 6,
    maximum_words_per_second: float = 3.5,
) -> bool:
    """Reject silent, tiny, or unnaturally dense chunks before GPU work."""
    start_ms = int(row.get("chunk_start_ms") or 0)
    end_ms = int(row.get("chunk_end_ms") or start_ms)
    duration = max(0.0, (end_ms - start_ms) / 1000.0)
    if duration < minimum_duration_seconds or duration > maximum_duration_seconds:
        return False
    text, segments = aligned_segments_from_row(row)
    word_count = sum(len(segment["words"]) for segment in segments)
    if not text or word_count < minimum_words:
        return False
    return word_count / max(duration, 1e-6) <= maximum_words_per_second


def _audio_bytes(row: dict[str, Any]) -> bytes:
    value = row.get("audio")
    if isinstance(value, dict):
        payload = value.get("bytes")
        if isinstance(payload, (bytes, bytearray, memoryview)):
            return bytes(payload)
        path = value.get("path")
        if path and Path(path).is_file():
            return Path(path).read_bytes()
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    raise ValueError("Aligned row does not contain embedded audio bytes")


def iter_parquet_rows(path: str | Path, *, batch_size: int = 32) -> Iterable[dict[str, Any]]:
    """Stream rows so a 500 MB audio shard is never expanded fully in RAM."""
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - Kaggle ships pyarrow.
        raise RuntimeError("pyarrow is required for aligned Parquet preprocessing") from exc

    columns = [
        "chunk_id",
        "song_id",
        "audio",
        "title",
        "artist",
        "album",
        "chunk_start_ms",
        "chunk_end_ms",
        "chunk_lyrics",
        "chunk_word_timestamps",
    ]
    parquet_file = parquet.ParquetFile(str(path))
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
        yield from batch.to_pylist()


def select_aligned_rows(
    rows: Iterable[dict[str, Any]],
    *,
    max_records: int,
    max_chunks_per_song: int = 2,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Take a diverse deterministic subset instead of adjacent chunks per song."""
    selected: list[dict[str, Any]] = []
    per_song: Counter[str] = Counter()
    scanned = rejected = duplicate_song_limit = 0
    for row in rows:
        scanned += 1
        if not aligned_row_is_usable(row):
            rejected += 1
            continue
        song_id = str(row.get("song_id") or row.get("chunk_id") or "").strip()
        if per_song[song_id] >= max_chunks_per_song:
            duplicate_song_limit += 1
            continue
        per_song[song_id] += 1
        selected.append(row)
        if len(selected) >= max_records:
            break
    return selected, {
        "scanned": scanned,
        "selected": len(selected),
        "songs": len(per_song),
        "rejected": rejected,
        "song_limit_rejected": duplicate_song_limit,
    }


def preprocess_aligned_parquet(
    parquet_path: str | Path,
    output_dir: str | Path,
    *,
    max_records: int = 160,
    max_chunks_per_song: int = 2,
    demucs_device: str = "cuda",
    batch_size: int = 8,
) -> dict[str, Any]:
    """Separate stems, compute mels, and preserve exact lyric timestamps."""
    output_root = Path(output_dir)
    incoming_root = output_root / "incoming"
    separated_root = output_root / "separated"
    output_root.mkdir(parents=True, exist_ok=True)
    incoming_root.mkdir(parents=True, exist_ok=True)

    selected, selection = select_aligned_rows(
        iter_parquet_rows(parquet_path),
        max_records=max_records,
        max_chunks_per_song=max_chunks_per_song,
    )
    if not selected:
        raise RuntimeError("No usable aligned Vietnamese rows were found")

    records_path = output_root / "records.jsonl"
    records_path.write_text("", encoding="utf-8")
    failures: list[dict[str, str]] = []
    processed = 0

    for batch_start in range(0, len(selected), batch_size):
        rows = selected[batch_start : batch_start + batch_size]
        staged: list[tuple[dict[str, Any], Path]] = []
        for row in rows:
            chunk_id = str(row.get("chunk_id") or f"aligned_{batch_start:05d}")
            audio_path = incoming_root / f"{chunk_id}.wav"
            audio_path.write_bytes(_audio_bytes(row))
            staged.append((row, audio_path))

        run_demucs_batch(
            [audio_path for _, audio_path in staged],
            separated_root,
            demucs_device,
        )
        for row, audio_path in staged:
            try:
                text, segments = aligned_segments_from_row(row)
                record = process_file(
                    audio_path,
                    output_root,
                    whisper_model=None,
                    keep_separated=False,
                    use_demucs=True,
                    transcribe=False,
                    demucs_device=demucs_device,
                    device=demucs_device,
                    compute_style=False,
                )
                if not record.get("demucs_separated"):
                    raise RuntimeError("Demucs did not produce both aligned stems")
                record.update(
                    {
                        "text": text,
                        "segments": segments,
                        "song_id": str(row.get("song_id") or ""),
                        "title": str(row.get("title") or ""),
                        "artist": str(row.get("artist") or ""),
                        "album": str(row.get("album") or ""),
                        "chunk_start_ms": int(row.get("chunk_start_ms") or 0),
                        "chunk_end_ms": int(row.get("chunk_end_ms") or 0),
                        "exact_word_timestamps": True,
                        "source_dataset": DEFAULT_REPO_ID,
                        "source_license": "apache-2.0",
                    }
                )
                with records_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                processed += 1
                print(
                    f"aligned_processed={processed}/{len(selected)} id={record['id']} "
                    f"words={sum(len(segment['words']) for segment in segments)}",
                    flush=True,
                )
            except Exception as exc:
                failures.append({"id": str(row.get("chunk_id") or ""), "error": str(exc)})
                print(f"[WARNING] aligned row failed: {row.get('chunk_id')}: {exc}", flush=True)
            finally:
                audio_path.unlink(missing_ok=True)

    shutil.rmtree(incoming_root, ignore_errors=True)
    config = {
        "sample_rate": SAMPLE_RATE,
        "n_mels": N_MELS,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "source_dataset": DEFAULT_REPO_ID,
        "source_shard": str(Path(parquet_path).name),
        "exact_word_timestamps": True,
    }
    (output_root / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    report = {
        "status": "complete" if processed == len(selected) else "completed_with_warnings",
        **selection,
        "processed": processed,
        "failures": failures,
        "output_dir": str(output_root.resolve()),
    }
    (output_root / "aligned_preprocess_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if processed < max(8, math.ceil(len(selected) * 0.8)):
        raise RuntimeError(f"Only {processed}/{len(selected)} aligned rows were processed")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--shard", default=DEFAULT_SHARD)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-records", type=int, default=160)
    parser.add_argument("--max-chunks-per-song", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--demucs-device", default="cuda")
    args = parser.parse_args()

    if args.max_records < 8 or args.max_chunks_per_song < 1 or args.batch_size < 1:
        raise ValueError("max-records >= 8 and positive song/batch limits are required")
    from huggingface_hub import hf_hub_download

    parquet_path = hf_hub_download(
        repo_id=args.repo_id,
        filename=args.shard,
        repo_type="dataset",
    )
    report = preprocess_aligned_parquet(
        parquet_path,
        args.output,
        max_records=args.max_records,
        max_chunks_per_song=args.max_chunks_per_song,
        demucs_device=args.demucs_device,
        batch_size=args.batch_size,
    )
    print("aligned_preprocess_result=" + json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
