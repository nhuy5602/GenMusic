"""Materialize the native waveform dataset with reserve backfill.

V70's raw exact-word recognizer learned a diverse phone inventory but
generalized poorly to words absent from the 512-song training corpus.  V71
changes only data coverage: four times as many globally unique songs and
shards, with the same raw 24 kHz Demucs representation and validation rules.
No recognizer or generator is trained in this phase.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
from pathlib import Path
from typing import Any

import torch

from src.data.raw_multishard import (
    _materialize_selected_rows,
    _scan_candidates,
    _write_json,
    raw_materialization_gate,
    select_balanced_unique_rows,
    validate_raw_records,
)
from src.data.preprocess_aligned_vietnamese import DEFAULT_REPO_ID


STATE_NAME = "master_raw_multishard_v71_state.json"
DEFAULT_SHARDS = tuple(
    f"data/train-{index:05d}-of-00063.parquet"
    for index in range(32)
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--shard", action="append", dest="shards")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target-records", type=int, default=2_048)
    parser.add_argument("--records-per-shard", type=int, default=64)
    parser.add_argument("--candidate-records-per-shard", type=int, default=128)
    parser.add_argument("--reserve-records", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--demucs-device", default="cuda")
    args = parser.parse_args()
    shards = tuple(args.shards or DEFAULT_SHARDS)
    if len(shards) != 32 or len(set(shards)) != 32:
        raise ValueError("V71 requires exactly 32 unique shards")
    if args.target_records != 2_048:
        raise ValueError("V71 requires exactly 2,048 target records")
    if args.records_per_shard != 64:
        raise ValueError("V71 requires 64 balanced records per shard")
    if args.candidate_records_per_shard < args.records_per_shard:
        raise ValueError("V71 candidate quota must cover balanced quota")
    if args.reserve_records < args.batch_size:
        raise ValueError("V71 reserve must cover at least one Demucs batch")

    from huggingface_hub import hf_hub_download

    output_root = args.output_root.resolve()
    dataset = output_root / "dataset"
    dataset.mkdir(parents=True, exist_ok=True)
    state_path = output_root / STATE_NAME
    state: dict[str, Any] = {
        "status": "downloading_and_scanning",
        "training": False,
        "generator_training": False,
        "data_gate_only": True,
        "design": (
            "32-shard 2048-song exact-timestamp raw 24kHz Demucs "
            "vocal/backing corpus for a higher-coverage phone gate"
        ),
        "repo_id": args.repo_id,
        "shards": list(shards),
        "target_records": args.target_records,
        "records_per_shard": args.records_per_shard,
        "candidate_records_per_shard": args.candidate_records_per_shard,
        "reserve_records": args.reserve_records,
        "pretrained_tts_used": False,
        "pretrained_asr_used": False,
        "pretrained_source_separator_used": True,
        "processed_records": 0,
        "failed_records": 0,
    }
    _write_json(state_path, state)

    # A 32-shard all-at-once HF cache can exceed Kaggle's writable disk once
    # the virtual environment and raw outputs coexist.  Scan one shard, keep
    # only its selected in-memory rows, then delete that shard's cache before
    # downloading the next one.
    groups: list[list[dict[str, Any]]] = []
    scan_reports: list[dict[str, Any]] = []
    shard_cache = output_root / "hf_shard_cache"
    for shard_index, shard in enumerate(shards):
        shutil.rmtree(shard_cache, ignore_errors=True)
        parquet_path = Path(hf_hub_download(
            repo_id=args.repo_id,
            filename=shard,
            repo_type="dataset",
            cache_dir=shard_cache,
        ))
        shard_groups, shard_reports = _scan_candidates(
            [parquet_path],
            candidate_records_per_shard=args.candidate_records_per_shard,
            progress_label="V71",
            shard_index_offset=shard_index,
        )
        groups.extend(shard_groups)
        scan_reports.extend(shard_reports)
        state["scanned_shards"] = shard_index + 1
        state["scan_reports"] = scan_reports
        _write_json(state_path, state)
        shutil.rmtree(shard_cache, ignore_errors=True)
        gc.collect()
    shutil.rmtree(shard_cache, ignore_errors=True)
    materialization_candidates = args.target_records + args.reserve_records
    selected, selection_report = select_balanced_unique_rows(
        groups,
        target_records=materialization_candidates,
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
        progress_label="V71",
        target_successes=args.target_records,
    )
    config = {
        "sample_rate": 24_000,
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
            if len(records) == args.target_records
            else "completed_with_warnings"
        ),
        "processed": len(records),
        "failures": failures,
        "scan_reports": scan_reports,
        "selection": selection_report,
        "materialization_candidates": materialization_candidates,
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
            "rerun one bounded exact-word phone gate with a frozen "
            "song-heldout split; do not train a generator yet"
        ),
        "next_if_fail": (
            "diagnose shard coverage or Demucs failures; do not train"
        ),
    })
    _write_json(state_path, state)
    print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)
    if not gate["pass"]:
        raise RuntimeError("V71 wide raw materialization gate failed")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
