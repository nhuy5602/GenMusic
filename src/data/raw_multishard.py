"""Materialize an exact-aligned raw Vietnamese stem corpus.

V68 is a data gate only.  It streams eight Hugging Face shards, chooses one
usable chunk per globally unique song, separates raw vocal/backing waveforms
with Demucs, and writes exactly 512 records.  It performs no generator or
recognizer training and uses no ASR.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
from pathlib import Path
from typing import Any, Iterable

import soundfile as sf
import torch

from src.data.aligned_raw import validate_raw_records
from src.data.preprocess_aligned_vietnamese import (
    DEFAULT_REPO_ID,
    _audio_bytes,
    aligned_segments_from_row,
    iter_parquet_rows,
    select_aligned_rows,
)
from src.data.preprocess_raw_vietnamese import (
    SAMPLE_RATE,
    process_file,
    run_demucs_batch,
)


STATE_NAME = "master_raw_multishard_v68_state.json"
DEFAULT_SHARDS = tuple(
    f"data/train-{index:05d}-of-00063.parquet"
    for index in range(8)
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _song_id(row: dict[str, Any]) -> str:
    value = str(row.get("song_id") or row.get("chunk_id") or "").strip()
    if not value:
        raise ValueError("Aligned row has no song/chunk identifier")
    return value


def select_balanced_unique_rows(
    candidate_groups: list[list[dict[str, Any]]],
    *,
    target_records: int,
    records_per_shard: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select balanced globally unique songs, then backfill deterministically."""
    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[str, str]] = set()
    selected_songs: set[str] = set()
    per_shard = [0 for _ in candidate_groups]

    def add(row: dict[str, Any], shard_index: int) -> bool:
        song = _song_id(row)
        chunk = str(row.get("chunk_id") or "").strip()
        key = (song, chunk)
        if song in selected_songs or key in selected_keys:
            return False
        copied = dict(row)
        copied["_v68_shard_index"] = shard_index
        selected.append(copied)
        selected_songs.add(song)
        selected_keys.add(key)
        per_shard[shard_index] += 1
        return True

    for shard_index, rows in enumerate(candidate_groups):
        for row in rows:
            if per_shard[shard_index] >= records_per_shard:
                break
            add(row, shard_index)

    if len(selected) < target_records:
        for shard_index, rows in enumerate(candidate_groups):
            for row in rows:
                if len(selected) >= target_records:
                    break
                add(row, shard_index)
            if len(selected) >= target_records:
                break

    selected = selected[:target_records]
    if len(selected) != target_records:
        raise RuntimeError(
            f"V68 found only {len(selected)}/{target_records} globally unique songs"
        )
    return selected, {
        "target_records": target_records,
        "selected_records": len(selected),
        "unique_songs": len({_song_id(row) for row in selected}),
        "selected_per_shard": per_shard,
        "candidate_counts": [len(rows) for rows in candidate_groups],
    }


def raw_materialization_gate(
    validation: dict[str, Any],
    *,
    target_records: int,
) -> dict[str, Any]:
    gate = {
        **validation,
        "target_records": int(target_records),
    }
    gate["pass"] = bool(
        int(validation["records"]) == target_records
        and int(validation["unique_songs"]) == target_records
        and float(validation["exact_timestamp_fraction"]) >= 0.99
        and float(validation["finite_fraction"]) == 1.0
        and float(validation["mono_waveform_fraction"]) == 1.0
        and float(validation["duration_valid_fraction"]) >= 0.99
        and float(validation["non_silent_vocal_fraction"]) >= 0.99
        and float(validation["non_silent_backing_fraction"]) >= 0.99
    )
    return gate


def _scan_candidates(
    parquet_paths: Iterable[Path],
    *,
    candidate_records_per_shard: int,
    progress_label: str = "V68",
    shard_index_offset: int = 0,
) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    reports: list[dict[str, Any]] = []
    for local_index, parquet_path in enumerate(parquet_paths):
        index = shard_index_offset + local_index
        rows, report = select_aligned_rows(
            iter_parquet_rows(parquet_path),
            max_records=candidate_records_per_shard,
            max_chunks_per_song=1,
        )
        for row in rows:
            row["_v68_source_shard"] = parquet_path.name
        groups.append(rows)
        reports.append({
            "shard_index": index,
            "shard": parquet_path.name,
            **report,
        })
        print(
            f"{progress_label}_SHARD_SCAN",
            index,
            parquet_path.name,
            f"selected={len(rows)}",
            f"scanned={report['scanned']}",
            flush=True,
        )
    return groups, reports


def _validate_staged_audio(audio_path: Path) -> None:
    """Reject empty/undecodable candidates before they poison a Demucs batch."""
    info = sf.info(str(audio_path))
    if int(info.frames) <= 0 or int(info.samplerate) <= 0:
        raise ValueError(
            "Aligned row contains empty audio "
            f"(frames={info.frames}, sample_rate={info.samplerate})"
        )


def _materialize_selected_rows(
    rows: list[dict[str, Any]],
    *,
    dataset: Path,
    state: dict[str, Any],
    state_path: Path,
    batch_size: int,
    demucs_device: str,
    progress_label: str = "V68",
    target_successes: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    if target_successes is not None:
        if target_successes <= 0:
            raise ValueError("target_successes must be positive")
        if target_successes > len(rows):
            raise ValueError(
                "target_successes cannot exceed candidate row count"
            )
    incoming_root = dataset / "incoming"
    separated_root = dataset / "separated"
    incoming_root.mkdir(parents=True, exist_ok=True)
    records_path = dataset / "records.jsonl"
    records_path.write_text("", encoding="utf-8")
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    row_cursor = 0
    while row_cursor < len(rows):
        if (
            target_successes is not None
            and len(records) >= target_successes
        ):
            break
        batch_start = row_cursor
        batch_capacity = batch_size
        if target_successes is not None:
            batch_capacity = min(
                batch_capacity,
                target_successes - len(records),
            )
        batch = rows[row_cursor : row_cursor + batch_capacity]
        row_cursor += len(batch)
        staged: list[tuple[dict[str, Any], Path]] = []
        for offset, row in enumerate(batch):
            chunk_id = str(
                row.get("chunk_id")
                or f"v68_{batch_start + offset:05d}"
            )
            audio_path = incoming_root / f"{chunk_id}.wav"
            try:
                audio_path.write_bytes(_audio_bytes(row))
                _validate_staged_audio(audio_path)
                staged.append((row, audio_path))
            except Exception as exc:
                audio_path.unlink(missing_ok=True)
                failures.append({
                    "id": str(row.get("chunk_id") or ""),
                    "error": f"audio preflight failed: {exc}",
                })
                print(
                    f"[WARNING] V68 audio preflight failed: "
                    f"{row.get('chunk_id')}: {exc}",
                    flush=True,
                )

        if staged:
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
                    dataset,
                    whisper_model=None,
                    keep_separated=False,
                    use_demucs=True,
                    transcribe=False,
                    demucs_device=demucs_device,
                    device=demucs_device,
                    compute_style=False,
                    raw_audio=True,
                )
                if not record.get("demucs_separated"):
                    raise RuntimeError("Demucs did not produce both V68 stems")
                record.update({
                    "text": text,
                    "segments": segments,
                    "song_id": _song_id(row),
                    "title": str(row.get("title") or ""),
                    "artist": str(row.get("artist") or ""),
                    "album": str(row.get("album") or ""),
                    "chunk_start_ms": int(row.get("chunk_start_ms") or 0),
                    "chunk_end_ms": int(row.get("chunk_end_ms") or 0),
                    "exact_word_timestamps": True,
                    "source_dataset": DEFAULT_REPO_ID,
                    "source_shard": str(
                        row.get("_v68_source_shard") or ""
                    ),
                    "source_license": "apache-2.0",
                })
                with records_path.open("a", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps(record, ensure_ascii=False) + "\n"
                    )
                records.append(record)
            except Exception as exc:
                failures.append({
                    "id": str(row.get("chunk_id") or ""),
                    "error": str(exc),
                })
                print(
                    f"[WARNING] V68 row failed: {row.get('chunk_id')}: {exc}",
                    flush=True,
                )
            finally:
                audio_path.unlink(missing_ok=True)

        state["processed_records"] = len(records)
        state["failed_records"] = len(failures)
        state["attempted_records"] = row_cursor
        state["last_batch_start"] = batch_start
        _write_json(state_path, state)
        progress_target = (
            target_successes
            if target_successes is not None
            else len(rows)
        )
        print(
            f"{progress_label}_MATERIALIZED "
            f"{len(records)}/{progress_target} "
            f"failures={len(failures)}",
            flush=True,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    shutil.rmtree(incoming_root, ignore_errors=True)
    shutil.rmtree(separated_root, ignore_errors=True)
    return records, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument(
        "--shard",
        action="append",
        dest="shards",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target-records", type=int, default=512)
    parser.add_argument("--records-per-shard", type=int, default=64)
    parser.add_argument("--candidate-records-per-shard", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--demucs-device", default="cuda")
    args = parser.parse_args()

    shards = tuple(args.shards or DEFAULT_SHARDS)
    if len(shards) < 8 or len(set(shards)) != len(shards):
        raise ValueError("V68 requires at least eight unique shards")
    if args.target_records < 256:
        raise ValueError("V68 target-records must be at least 256")
    if args.candidate_records_per_shard < args.records_per_shard:
        raise ValueError("V68 candidate quota must cover the balanced quota")

    from huggingface_hub import hf_hub_download

    output_root = args.output_root.resolve()
    dataset = output_root / "dataset"
    dataset.mkdir(parents=True, exist_ok=True)
    state_path = output_root / STATE_NAME
    state: dict[str, Any] = {
        "status": "downloading_and_scanning",
        "training": False,
        "goal_eligible_prediction": False,
        "data_gate_only": True,
        "design": (
            "eight-shard globally unique exact-timestamp raw 24kHz "
            "Demucs vocal/backing corpus for a later raw-phone gate"
        ),
        "repo_id": args.repo_id,
        "shards": list(shards),
        "target_records": args.target_records,
        "records_per_shard": args.records_per_shard,
        "candidate_records_per_shard": (
            args.candidate_records_per_shard
        ),
        "pretrained_tts_used": False,
        "pretrained_asr_used": False,
        "pretrained_source_separator_used": True,
        "asr_evaluation_only": False,
        "processed_records": 0,
        "failed_records": 0,
    }
    _write_json(state_path, state)

    parquet_paths = [
        Path(hf_hub_download(
            repo_id=args.repo_id,
            filename=shard,
            repo_type="dataset",
        ))
        for shard in shards
    ]
    groups, scan_reports = _scan_candidates(
        parquet_paths,
        candidate_records_per_shard=(
            args.candidate_records_per_shard
        ),
    )
    selected, selection_report = select_balanced_unique_rows(
        groups,
        target_records=args.target_records,
        records_per_shard=args.records_per_shard,
    )
    state.update({
        "status": "materializing",
        "scan_reports": scan_reports,
        "selection": selection_report,
    })
    _write_json(state_path, state)

    records, failures = _materialize_selected_rows(
        selected,
        dataset=dataset,
        state=state,
        state_path=state_path,
        batch_size=args.batch_size,
        demucs_device=args.demucs_device,
    )
    config = {
        "sample_rate": SAMPLE_RATE,
        "source_dataset": args.repo_id,
        "source_shards": list(shards),
        "exact_word_timestamps": True,
        "raw_audio_mode": True,
        "records": len(records),
        "unique_songs": len({
            str(record["song_id"]) for record in records
        }),
    }
    _write_json(dataset / "config.json", config)
    _write_json(dataset / "aligned_preprocess_report.json", {
        "status": (
            "complete"
            if len(records) == args.target_records and not failures
            else "completed_with_warnings"
        ),
        "processed": len(records),
        "failures": failures,
        "scan_reports": scan_reports,
        "selection": selection_report,
    })

    state["status"] = "validating"
    _write_json(state_path, state)
    validation = validate_raw_records(dataset, records)
    gate = raw_materialization_gate(
        validation,
        target_records=args.target_records,
    )
    state.update({
        "status": (
            "materialization_passed"
            if gate["pass"]
            else "materialization_failed"
        ),
        "records": len(records),
        "failures": failures,
        "validation": validation,
        "gate": gate,
        "next_if_pass": (
            "run a bounded raw-phrase phone recognizer gate before any "
            "generator training"
        ),
        "next_if_fail": (
            "diagnose shard selection or Demucs failures; do not train"
        ),
    })
    _write_json(state_path, state)
    print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)
    if not gate["pass"]:
        raise RuntimeError("V68 raw multi-shard materialization gate failed")


if __name__ == "__main__":
    main()
