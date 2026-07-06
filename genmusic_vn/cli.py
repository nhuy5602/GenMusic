from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .evaluation import DEFAULT_EVAL_DATASET, evaluate_dataset
from .kaggle_auto import KaggleAutoError, KaggleJobConfig, submit_text_to_music_job, sync_kaggle_artifact
from .synthetic_dataset import generate_synthetic_records, write_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="genmusic-vn", description="Vietnamese text-to-MP3 Kaggle MusicGen client.")
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="Submit Vietnamese text to Kaggle MusicGen and return an MP3 artifact.")
    generate.add_argument("--text", required=True, help="Vietnamese input text.")
    generate.add_argument("--duration", type=int, default=30, help="Target duration in seconds.")
    generate.add_argument("--out", default="outputs", help="Output directory.")
    generate.add_argument("--genre", default=None, help="Optional style/genre hint.")
    generate.add_argument("--model", default="facebook/musicgen-small", help="MusicGen model on Kaggle.")
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

    if args.command == "make-eval-dataset":
        emotions = [item.strip() for item in args.emotions.split(",") if item.strip()] or None
        lengths = [item.strip() for item in args.lengths.split(",") if item.strip()] or None
        records = generate_synthetic_records(args.count, seed=args.seed, emotions=emotions, lengths=lengths)
        output_path = write_jsonl(records, args.out)
        print(json.dumps({"path": str(output_path), "count": len(records), "seed": args.seed}, ensure_ascii=False, indent=2))
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


if __name__ == "__main__":
    raise SystemExit(main())
