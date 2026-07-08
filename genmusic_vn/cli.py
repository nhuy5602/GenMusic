from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .chorus_ablation import (
    DEFAULT_CHORUS_ABLATION_DATASET,
    evaluate_chorus_ablation_dataset,
    write_chorus_ablation_report,
)
from .evaluation import DEFAULT_EVAL_DATASET, evaluate_dataset
from .kaggle_auto import (
    DEFAULT_MUSICGEN_MODEL,
    KaggleAutoError,
    KaggleJobConfig,
    submit_text_to_music_job,
    sync_kaggle_artifact,
)
from .synthetic_dataset import generate_synthetic_records, write_jsonl
from .xlsx_dataset import records_from_xlsx, write_jsonl as write_xlsx_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="genmusic-vn", description="Vietnamese text-to-MP3 Kaggle MusicGen client.")
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="Submit Vietnamese text to Kaggle MusicGen and return an MP3 artifact.")
    generate.add_argument("--text", required=True, help="Vietnamese input text.")
    generate.add_argument("--duration", type=int, default=30, help="Target duration in seconds.")
    generate.add_argument("--out", default="outputs", help="Output directory.")
    generate.add_argument("--genre", default=None, help="Optional style/genre hint.")
    generate.add_argument("--model", default=DEFAULT_MUSICGEN_MODEL, help="MusicGen model on Kaggle.")
    generate.add_argument("--username", default=None, help="Kaggle username. Defaults to kaggle.json or KAGGLE_USERNAME.")
    generate.add_argument("--machine-shape", default="NvidiaTeslaT4")
    generate.add_argument("--no-submit", action="store_true", help="Only stage Kaggle files locally.")
    generate.add_argument("--wait", action="store_true", help="Poll Kaggle until the kernel finishes, then download MP3.")
    generate.add_argument("--poll-seconds", type=int, default=60)
    generate.add_argument("--timeout-seconds", type=int, default=10_800)

    sync = sub.add_parser("sync-kaggle", help="Download a model/artifact from Kaggle to local storage.")
    sync.add_argument("--source", required=True, choices=["dataset", "kernel"], help="Download from a Kaggle dataset or kernel output.")
    sync.add_argument("--ref", required=True, help="Kaggle ref, e.g. username/dataset-slug or username/kernel-slug.")
    sync.add_argument("--out", default="models/current", help="Local target directory.")

    evaluate = sub.add_parser("evaluate", help="Evaluate text-to-lyrics/vocal planning on the benchmark dataset.")
    evaluate.add_argument("--dataset", default=str(DEFAULT_EVAL_DATASET), help="JSONL evaluation dataset.")
    evaluate.add_argument("--out", default="outputs/evaluation", help="Output directory for evaluation artifacts.")
    evaluate.add_argument("--duration", type=int, default=12, help="Duration used for pipeline planning.")

    chorus_ablation = sub.add_parser("chorus-ablation", help="Compare pasted chorus planning with and without the original style hint.")
    chorus_ablation.add_argument("--dataset", default=str(DEFAULT_CHORUS_ABLATION_DATASET), help="JSONL dataset with chorus and style fields.")
    chorus_ablation.add_argument("--out", default="outputs/chorus_ablation", help="Output directory for the ablation report.")
    chorus_ablation.add_argument("--duration", type=int, default=45, help="Target duration used for planning.")

    import_xlsx = sub.add_parser("import-xlsx-dataset", help="Convert the Vietnamese XLSX benchmark into JSONL.")
    import_xlsx.add_argument("--xlsx", required=True, help="Workbook path.")
    import_xlsx.add_argument("--out", default="datasets/evaluation/vietnamese_musicgen_input_dataset.jsonl", help="Output JSONL path.")

    evaluate_xlsx = sub.add_parser("evaluate-xlsx", help="Evaluate planning directly from the Vietnamese XLSX benchmark.")
    evaluate_xlsx.add_argument("--xlsx", required=True, help="Workbook path.")
    evaluate_xlsx.add_argument("--out", default="outputs/evaluation_xlsx", help="Output directory for evaluation artifacts.")

    batch_xlsx = sub.add_parser("batch-generate-xlsx", help="Submit XLSX rows to Kaggle and generate MP3 files.")
    batch_xlsx.add_argument("--xlsx", required=True, help="Workbook path.")
    batch_xlsx.add_argument("--out", default="outputs/xlsx_batch", help="Output directory.")
    batch_xlsx.add_argument("--model", default=DEFAULT_MUSICGEN_MODEL, help="MusicGen model on Kaggle.")
    batch_xlsx.add_argument("--username", default=None, help="Kaggle username. Defaults to kaggle.json or KAGGLE_USERNAME.")
    batch_xlsx.add_argument("--machine-shape", default="NvidiaTeslaT4")
    batch_xlsx.add_argument("--limit", type=int, default=0, help="Maximum rows to submit. 0 means all selected rows.")
    batch_xlsx.add_argument("--ids", default="", help="Comma-separated row IDs, e.g. VN001,VN008.")
    batch_xlsx.add_argument("--sample-per-mood", type=int, default=0, help="Keep up to N rows per expected mood.")
    batch_xlsx.add_argument("--no-submit", action="store_true", help="Only stage Kaggle files locally.")
    batch_xlsx.add_argument("--wait", action="store_true", help="Wait for each Kaggle job and download output before submitting the next.")
    batch_xlsx.add_argument("--poll-seconds", type=int, default=60)
    batch_xlsx.add_argument("--timeout-seconds", type=int, default=10_800)

    synth = sub.add_parser("make-eval-dataset", help="Generate a labeled synthetic JSONL benchmark dataset.")
    synth.add_argument("--count", type=int, default=24, help="Number of synthetic records.")
    synth.add_argument("--seed", type=int, default=42, help="Deterministic random seed.")
    synth.add_argument("--out", default="datasets/evaluation/synthetic_eval.jsonl", help="Output JSONL path.")
    synth.add_argument("--emotions", default="", help="Comma-separated emotion labels. Defaults to all labels.")
    synth.add_argument("--lengths", default="", help="Comma-separated length buckets: short,medium,long.")
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = build_parser().parse_args(argv)
    if args.command == "sync-kaggle":
        try:
            manifest = sync_kaggle_artifact(source=args.source, ref=args.ref, output_dir=args.out)
        except KaggleAutoError as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2))
            return 1
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0

    if args.command == "evaluate":
        report = evaluate_dataset(args.dataset, output_root=args.out, duration_seconds=args.duration)
        report_path = Path(args.out) / "evaluation_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report["report_path"] = str(report_path)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.command == "chorus-ablation":
        report = evaluate_chorus_ablation_dataset(args.dataset, output_root=Path(args.out) / "runs", duration_seconds=args.duration)
        json_path, md_path = write_chorus_ablation_report(report, args.out)
        report["report_path"] = str(json_path)
        report["markdown_path"] = str(md_path)
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.command == "import-xlsx-dataset":
        records = records_from_xlsx(args.xlsx)
        output_path = write_xlsx_jsonl(records, args.out)
        print(json.dumps({"path": str(output_path), "count": len(records)}, ensure_ascii=False, indent=2))
        return 0

    if args.command == "evaluate-xlsx":
        records = records_from_xlsx(args.xlsx)
        jsonl_path = Path(args.out) / "xlsx_eval_dataset.jsonl"
        write_xlsx_jsonl(records, jsonl_path)
        report = evaluate_dataset(jsonl_path, output_root=args.out, duration_seconds=30)
        report_path = Path(args.out) / "evaluation_report.json"
        report["report_path"] = str(report_path)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.command == "make-eval-dataset":
        emotions = [item.strip() for item in args.emotions.split(",") if item.strip()] or None
        lengths = [item.strip() for item in args.lengths.split(",") if item.strip()] or None
        records = generate_synthetic_records(args.count, seed=args.seed, emotions=emotions, lengths=lengths)
        output_path = write_jsonl(records, args.out)
        print(json.dumps({"path": str(output_path), "count": len(records), "seed": args.seed}, ensure_ascii=False, indent=2))
        return 0

    if args.command == "batch-generate-xlsx":
        records = _select_xlsx_records(
            records_from_xlsx(args.xlsx),
            ids=args.ids,
            sample_per_mood=args.sample_per_mood,
            limit=args.limit,
        )
        batch_dir = Path(args.out)
        batch_dir.mkdir(parents=True, exist_ok=True)
        jobs = []
        for record in records:
            job = submit_text_to_music_job(
                text=record["input_text"],
                output_root=batch_dir,
                duration_seconds=int(record.get("duration_seconds") or 30),
                genre=record.get("genre") or None,
                config=KaggleJobConfig(
                    model=args.model,
                    username=args.username,
                    machine_shape=args.machine_shape,
                    submit=not args.no_submit,
                    wait=args.wait,
                    poll_seconds=args.poll_seconds,
                    timeout_seconds=args.timeout_seconds,
                ),
            )
            job["dataset_id"] = record["id"]
            job["expected_mood_text"] = record.get("expected_mood_text", "")
            jobs.append(job)
        manifest = {
            "xlsx": args.xlsx,
            "out": str(batch_dir),
            "count": len(jobs),
            "submitted": not args.no_submit,
            "wait": args.wait,
            "jobs": jobs,
        }
        manifest_path = batch_dir / "batch_manifest.json"
        manifest["manifest_path"] = str(manifest_path)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0

    job = submit_text_to_music_job(
        text=args.text,
        output_root=Path(args.out),
        duration_seconds=args.duration,
        genre=args.genre,
        config=KaggleJobConfig(
            model=args.model,
            username=args.username,
            machine_shape=args.machine_shape,
            submit=not args.no_submit,
            wait=args.wait,
            poll_seconds=args.poll_seconds,
            timeout_seconds=args.timeout_seconds,
        ),
    )
    print(json.dumps(job, ensure_ascii=False, indent=2))
    return 0


def _select_xlsx_records(
    records: list[dict],
    *,
    ids: str = "",
    sample_per_mood: int = 0,
    limit: int = 0,
) -> list[dict]:
    selected = list(records)
    requested_ids = [item.strip() for item in ids.split(",") if item.strip()]
    if requested_ids:
        allowed = set(requested_ids)
        selected = [record for record in selected if record.get("id") in allowed]
    if sample_per_mood > 0:
        counts: dict[str, int] = {}
        sampled: list[dict] = []
        for record in selected:
            mood = str(record.get("expected_mood_text") or "unknown")
            if counts.get(mood, 0) >= sample_per_mood:
                continue
            sampled.append(record)
            counts[mood] = counts.get(mood, 0) + 1
        selected = sampled
    if limit > 0:
        selected = selected[:limit]
    return selected


if __name__ == "__main__":
    raise SystemExit(main())
