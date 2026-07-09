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
from .dataset_scale import gigabytes_to_bytes, write_large_diverse_dataset
from .evaluation import DEFAULT_EVAL_DATASET, evaluate_dataset
from .kaggle_auto import (
    DEFAULT_CUSTOM_MUSIC_MODEL,
    KaggleAutoError,
    KaggleJobConfig,
    submit_text_to_music_job,
    sync_kaggle_artifact,
)
from .pipeline import create_music_project
from .reference_dataset import write_reference_datasets
from .self_improve import run_self_improvement
from .synthetic_dataset import generate_synthetic_records, write_jsonl
from .trained_text_model import DEFAULT_LOCAL_MODEL_PATH, trained_model_status, train_text_model, write_text_model
from .training_auto import submit_text_model_training_job
from .training_dataset import generate_diverse_training_records, generate_training_records, load_training_records, write_training_jsonl
from .xlsx_dataset import records_from_xlsx, write_jsonl as write_xlsx_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="genmusic-vn", description="Vietnamese text-to-MP3 custom composer client.")
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="Submit Vietnamese text to Kaggle custom composer and return an MP3 artifact.")
    generate.add_argument("--text", required=True, help="Vietnamese input text.")
    generate.add_argument("--duration", type=int, default=30, help="Target duration in seconds.")
    generate.add_argument("--out", default="outputs", help="Output directory.")
    generate.add_argument("--genre", default=None, help="Optional style/genre hint.")
    generate.add_argument("--model", default=DEFAULT_CUSTOM_MUSIC_MODEL, help="Custom music model identifier.")
    generate.add_argument("--username", default=None, help="Kaggle username. Defaults to kaggle.json or KAGGLE_USERNAME.")
    generate.add_argument("--machine-shape", default="NvidiaTeslaT4")
    generate.add_argument("--no-submit", action="store_true", help="Only stage Kaggle files locally.")
    generate.add_argument("--wait", action="store_true", help="Poll Kaggle until the kernel finishes, then download MP3.")
    generate.add_argument("--poll-seconds", type=int, default=60)
    generate.add_argument("--timeout-seconds", type=int, default=10_800)

    generate_local = sub.add_parser("generate-local", help="Generate a local custom composer track.")
    generate_local.add_argument("--text", required=True, help="Vietnamese input text.")
    generate_local.add_argument("--duration", type=int, default=30, help="Target duration in seconds.")
    generate_local.add_argument("--out", default="outputs/local", help="Output directory.")
    generate_local.add_argument("--genre", default=None, help="Optional style/genre hint.")

    sync = sub.add_parser("sync-kaggle", help="Download a model/artifact from Kaggle to local storage.")
    sync.add_argument("--source", required=True, choices=["dataset", "kernel"], help="Download from a Kaggle dataset or kernel output.")
    sync.add_argument("--ref", required=True, help="Kaggle ref, e.g. username/dataset-slug or username/kernel-slug.")
    sync.add_argument("--out", default="models/current", help="Local target directory.")

    model_status = sub.add_parser("model-status", help="Show the currently loaded trained text model artifact.")
    model_status.add_argument("--model", default=None, help="Optional model JSON path.")

    train_data = sub.add_parser("make-train-dataset", help="Generate a supervised Vietnamese text model training dataset.")
    train_data.add_argument("--count", type=int, default=480, help="Number of generated training records.")
    train_data.add_argument("--seed", type=int, default=42, help="Deterministic random seed.")
    train_data.add_argument("--profile", choices=["standard", "diverse"], default="diverse", help="Generation profile; diverse combines independent context/style pools.")
    train_data.add_argument("--out", default="datasets/training/generated_text_model_train.jsonl", help="Output JSONL path.")

    reference_data = sub.add_parser("make-reference-dataset", help="Generate safe reference train/eval datasets for self-improvement.")
    reference_data.add_argument("--count", type=int, default=24, help="Number of reference-style records.")
    reference_data.add_argument("--seed", type=int, default=5602, help="Deterministic random seed.")
    reference_data.add_argument("--train-out", default="datasets/training/reference_song_train.jsonl", help="Training JSONL path.")
    reference_data.add_argument("--eval-out", default="datasets/evaluation/reference_song_eval.jsonl", help="Evaluation JSONL path.")

    large_data = sub.add_parser("make-large-dataset", help="Stream a large diverse synthetic dataset into JSONL shards.")
    large_data.add_argument("--target-gb", type=float, default=1.0, help="Target decimal gigabytes on disk.")
    large_data.add_argument("--out", default="datasets/training/diverse_1gb", help="Output shard directory.")
    large_data.add_argument("--seed", type=int, default=5602, help="Deterministic random seed.")
    large_data.add_argument("--shard-mb", type=int, default=128, help="Approximate shard size in MB.")
    large_data.add_argument("--batch-size", type=int, default=2000, help="Records generated per streaming batch.")
    large_data.add_argument("--max-records", type=int, default=0, help="Optional hard record cap; 0 means no cap.")

    train_model = sub.add_parser("train-text-model", help="Train the local text emotion/style model. Defaults to Kaggle.")
    train_model.add_argument("--samples", type=int, default=480, help="Synthetic records to generate before training.")
    train_model.add_argument("--seed", type=int, default=42, help="Deterministic random seed.")
    train_model.add_argument("--dataset", default="", help="Optional extra JSONL training dataset.")
    train_model.add_argument("--dataset-limit", type=int, default=60000, help="Maximum records sampled from extra files/directories.")
    train_model.add_argument("--out", default="outputs/model_training", help="Training job output directory.")
    train_model.add_argument("--model-out", default=str(DEFAULT_LOCAL_MODEL_PATH), help="Local model artifact path.")
    train_model.add_argument("--local", action="store_true", help="Train locally for smoke tests instead of submitting to Kaggle.")
    train_model.add_argument("--username", default=None, help="Kaggle username. Defaults to kaggle.json or KAGGLE_USERNAME.")
    train_model.add_argument("--no-submit", action="store_true", help="Only stage Kaggle files locally.")
    train_model.add_argument("--wait", action="store_true", help="Wait for Kaggle training and download the model artifact.")
    train_model.add_argument("--poll-seconds", type=int, default=60)
    train_model.add_argument("--timeout-seconds", type=int, default=7200)

    evaluate = sub.add_parser("evaluate", help="Evaluate text-to-lyrics/vocal planning on the benchmark dataset.")
    evaluate.add_argument("--dataset", default=str(DEFAULT_EVAL_DATASET), help="JSONL evaluation dataset.")
    evaluate.add_argument("--out", default="outputs/evaluation", help="Output directory for evaluation artifacts.")
    evaluate.add_argument("--duration", type=int, default=12, help="Duration used for pipeline planning.")

    self_improve = sub.add_parser("self-improve", help="Iteratively train, simulate user prompts, evaluate quality, and add targeted records.")
    self_improve.add_argument("--iterations", type=int, default=3, help="Maximum local self-improvement rounds.")
    self_improve.add_argument("--samples", type=int, default=640, help="Base generated training samples per round.")
    self_improve.add_argument("--eval-count", type=int, default=24, help="Synthetic evaluation samples per round.")
    self_improve.add_argument("--seed", type=int, default=5602, help="Deterministic random seed.")
    self_improve.add_argument("--out", default="outputs/self_improve", help="Output directory for loop reports.")
    self_improve.add_argument("--model-out", default=str(DEFAULT_LOCAL_MODEL_PATH), help="Model artifact path to update.")
    self_improve.add_argument("--extra-dataset", default="", help="Optional comma-separated local JSONL datasets you have permission to use.")
    self_improve.add_argument("--extra-dataset-limit", type=int, default=60000, help="Maximum records sampled from extra files/directories.")
    self_improve.add_argument("--duration", type=int, default=30, help="Duration used for simulated user prompts.")
    self_improve.add_argument("--render-audio", action="store_true", help="Render local backing WAV/MP3 for clarity checks.")
    self_improve.add_argument("--stop-score", type=float, default=0.88, help="Stop early when combined score reaches this value.")

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
    batch_xlsx.add_argument("--model", default=DEFAULT_CUSTOM_MUSIC_MODEL, help="Custom music model identifier.")
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
    if args.command == "model-status":
        print(json.dumps(trained_model_status(args.model), ensure_ascii=False, indent=2))
        return 0

    if args.command == "sync-kaggle":
        try:
            manifest = sync_kaggle_artifact(source=args.source, ref=args.ref, output_dir=args.out)
        except KaggleAutoError as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2))
            return 1
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0

    if args.command == "make-train-dataset":
        generator = generate_diverse_training_records if args.profile == "diverse" else generate_training_records
        records = generator(args.count, seed=args.seed)
        output_path = write_training_jsonl(records, args.out)
        unique_inputs = len({str(record.get("input_text", "")) for record in records})
        print(json.dumps({"path": str(output_path), "count": len(records), "unique_inputs": unique_inputs, "seed": args.seed, "profile": args.profile}, ensure_ascii=False, indent=2))
        return 0

    if args.command == "make-reference-dataset":
        train_path, eval_path = write_reference_datasets(
            train_out=args.train_out,
            eval_out=args.eval_out,
            count=args.count,
            seed=args.seed,
        )
        print(json.dumps({"train_path": str(train_path), "eval_path": str(eval_path), "count": args.count, "seed": args.seed}, ensure_ascii=False, indent=2))
        return 0

    if args.command == "make-large-dataset":
        manifest = write_large_diverse_dataset(
            args.out,
            target_bytes=gigabytes_to_bytes(args.target_gb),
            seed=args.seed,
            shard_bytes=max(1, args.shard_mb) * 1_000_000,
            batch_size=args.batch_size,
            max_records=args.max_records or None,
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0

    if args.command == "train-text-model":
        extra_paths = [item.strip() for item in args.dataset.split(",") if item.strip()]
        if args.local:
            records = generate_training_records(args.samples, seed=args.seed) + load_training_records(
                extra_paths,
                max_records=args.dataset_limit,
                seed=args.seed,
            )
            model, report = train_text_model(records, seed=args.seed)
            model_path = write_text_model(model, args.model_out)
            report_path = Path(args.out) / "local_training_report.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_payload = {"model_path": str(model_path), **report}
            report_path.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({"model_path": str(model_path), "report_path": str(report_path), "report": report}, ensure_ascii=False, indent=2))
            return 0
        job = submit_text_model_training_job(
            output_root=args.out,
            sample_count=args.samples,
            seed=args.seed,
            extra_datasets=extra_paths,
            extra_dataset_max_records=args.extra_dataset_limit,
            config=KaggleJobConfig(
                username=args.username,
                machine_shape="CPU",
                submit=not args.no_submit,
                wait=args.wait,
                poll_seconds=args.poll_seconds,
                timeout_seconds=args.timeout_seconds,
            ),
            local_model_path=args.model_out,
        )
        print(json.dumps(job, ensure_ascii=False, indent=2))
        return 0

    if args.command == "generate-local":
        result = create_music_project(
            args.text,
            output_root=args.out,
            backend="custom",
            duration_seconds=args.duration,
            genre=args.genre or None,
            render_audio=True,
        )
        print(json.dumps(
            {
                "run_id": result.run_id,
                "backend": result.backend,
                "emotion": result.emotion.label,
                "harmony": {
                    "key": result.harmony.key,
                    "scale": result.harmony.scale,
                    "bpm": result.harmony.bpm,
                    "progression": result.harmony.chord_progression,
                },
                "files": [file.__dict__ for file in result.files],
                "prompt": result.prompt,
            },
            ensure_ascii=False,
            indent=2,
        ))
        return 0

    if args.command == "evaluate":
        report = evaluate_dataset(args.dataset, output_root=args.out, duration_seconds=args.duration)
        report_path = Path(args.out) / "evaluation_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report["report_path"] = str(report_path)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.command == "self-improve":
        extra_paths = [item.strip() for item in args.extra_dataset.split(",") if item.strip()]
        report = run_self_improvement(
            iterations=args.iterations,
            samples=args.samples,
            eval_count=args.eval_count,
            seed=args.seed,
            output_root=args.out,
            model_out=args.model_out,
            extra_datasets=extra_paths,
            duration_seconds=args.duration,
            render_audio=args.render_audio,
            stop_score=args.stop_score,
        )
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
