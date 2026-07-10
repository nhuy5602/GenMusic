from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .evaluation.chorus_ablation import (
    DEFAULT_CHORUS_ABLATION_DATASET,
    evaluate_chorus_ablation_dataset,
    write_chorus_ablation_report,
)
from .data.dataset_scale import gigabytes_to_bytes, write_large_diverse_dataset
from .evaluation.evaluation import DEFAULT_EVAL_DATASET, evaluate_dataset
from .integrations.kaggle_auto import (
    DEFAULT_CUSTOM_MUSIC_MODEL,
    KaggleAutoError,
    KaggleJobConfig,
    submit_text_to_music_job,
    sync_kaggle_artifact,
)
from .data.licensed_lyric_crawler import (
    build_rhyme_profile,
    crawl_licensed_sources,
    load_source_specs,
)
from .core.pipeline import create_music_project
from .evaluation.project_metrics import build_project_report
from .data.reference_dataset import write_reference_datasets
from .evaluation.self_improve import run_self_improvement
from .data.synthetic_dataset import generate_synthetic_records, write_jsonl
from .integrations.trained_text_model import DEFAULT_LOCAL_MODEL_PATH, trained_model_status, train_text_model, write_text_model
from .integrations.training_auto import submit_text_model_training_job
from .data.training_dataset import generate_diverse_training_records, generate_training_records, load_training_records, write_training_jsonl
from .data.xlsx_dataset import records_from_xlsx, write_jsonl as write_xlsx_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="genmusic-vn", description="Công cụ custom composer tạo MP3 từ văn bản tiếng Việt.")
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="Gửi văn bản tiếng Việt lên custom composer Kaggle và nhận artifact MP3.")
    generate.add_argument("--text", required=True, help="Văn bản input tiếng Việt.")
    generate.add_argument("--duration", type=int, default=30, help="Thời lượng mục tiêu, tính bằng giây.")
    generate.add_argument("--out", default="outputs", help="Thư mục output.")
    generate.add_argument("--genre", default=None, help="Gợi ý phong cách/thể loại, không bắt buộc.")
    generate.add_argument("--model", default=DEFAULT_CUSTOM_MUSIC_MODEL, help="Mã định danh custom music model.")
    generate.add_argument("--username", default=None, help="Username Kaggle; mặc định đọc từ kaggle.json hoặc KAGGLE_USERNAME.")
    generate.add_argument("--machine-shape", default="NvidiaTeslaT4")
    generate.add_argument("--no-submit", action="store_true", help="Chỉ chuẩn bị file Kaggle ở local.")
    generate.add_argument("--wait", action="store_true", help="Chờ kernel Kaggle hoàn tất rồi tải MP3.")
    generate.add_argument("--poll-seconds", type=int, default=60)
    generate.add_argument("--timeout-seconds", type=int, default=10_800)

    generate_local = sub.add_parser("generate-local", help="Tạo backing track custom composer ở local.")
    generate_local.add_argument("--text", required=True, help="Văn bản input tiếng Việt.")
    generate_local.add_argument("--duration", type=int, default=30, help="Thời lượng mục tiêu, tính bằng giây.")
    generate_local.add_argument("--out", default="outputs/local", help="Thư mục output.")
    generate_local.add_argument("--genre", default=None, help="Gợi ý phong cách/thể loại, không bắt buộc.")

    sync = sub.add_parser("sync-kaggle", help="Tải model/artifact từ Kaggle về local.")
    sync.add_argument("--source", required=True, choices=["dataset", "kernel"], help="Tải từ dataset Kaggle hoặc output kernel.")
    sync.add_argument("--ref", required=True, help="Kaggle ref, ví dụ username/dataset-slug hoặc username/kernel-slug.")
    sync.add_argument("--out", default="models/current", help="Thư mục đích local.")

    model_status = sub.add_parser("model-status", help="Hiển thị artifact text model đang được load.")
    model_status.add_argument("--model", default=None, help="Đường dẫn model JSON, không bắt buộc.")

    train_data = sub.add_parser("make-train-dataset", help="Tạo dataset train text model tiếng Việt có giám sát.")
    train_data.add_argument("--count", type=int, default=480, help="Số record train cần tạo.")
    train_data.add_argument("--seed", type=int, default=42, help="Seed ngẫu nhiên xác định.")
    train_data.add_argument("--profile", choices=["standard", "diverse"], default="diverse", help="Profile sinh dữ liệu; diverse kết hợp nhiều pool context/style độc lập.")
    train_data.add_argument("--out", default="datasets/training/generated_text_model_train.jsonl", help="Đường dẫn JSONL output.")

    reference_data = sub.add_parser("make-reference-dataset", help="Tạo dataset train/eval tham chiếu an toàn cho self-improve.")
    reference_data.add_argument("--count", type=int, default=24, help="Số record kiểu tham chiếu.")
    reference_data.add_argument("--seed", type=int, default=5602, help="Seed ngẫu nhiên xác định.")
    reference_data.add_argument("--train-out", default="datasets/training/reference_song_train.jsonl", help="Đường dẫn JSONL train.")
    reference_data.add_argument("--eval-out", default="datasets/evaluation/reference_song_eval.jsonl", help="Đường dẫn JSONL đánh giá.")

    crawl_lyrics = sub.add_parser("crawl-licensed-lyrics", help="Thu thập nguyên verse/chorus có license kèm provenance.")
    crawl_lyrics.add_argument("--sources", required=True, help="Manifest JSON/JSONL có URL được phê duyệt và trường license.")
    crawl_lyrics.add_argument("--out", default="datasets/training/licensed_lyric_sections.jsonl", help="Đường dẫn JSONL output.")
    crawl_lyrics.add_argument("--max-sources", type=int, default=0, help="Số nguồn tối đa được dùng; 0 nghĩa là tất cả.")
    crawl_lyrics.add_argument("--max-sections", type=int, default=12, help="Số section đầy đủ tối đa trên mỗi nguồn.")

    rhyme_profile = sub.add_parser("train-rhyme-profile", help="Học profile vần/điệp âm gọn từ các section có license.")
    rhyme_profile.add_argument("--dataset", required=True, help="JSONL section có license do crawl-licensed-lyrics tạo.")
    rhyme_profile.add_argument("--out", default="models/rhyme_profile.json", help="Đường dẫn profile output.")

    large_data = sub.add_parser("make-large-dataset", help="Sinh dataset tổng hợp đa dạng dạng stream thành các shard JSONL.")
    large_data.add_argument("--target-gb", type=float, default=5.0, help="Dung lượng mục tiêu theo GB thập phân.")
    large_data.add_argument("--out", default="datasets/training/diverse_5gb", help="Thư mục shard output.")
    large_data.add_argument("--seed", type=int, default=5602, help="Seed ngẫu nhiên xác định.")
    large_data.add_argument("--shard-mb", type=int, default=128, help="Kích thước shard gần đúng theo MB.")
    large_data.add_argument("--batch-size", type=int, default=2000, help="Số record tạo trong mỗi batch stream.")
    large_data.add_argument("--max-records", type=int, default=0, help="Giới hạn record cứng, không bắt buộc; 0 nghĩa là không giới hạn.")

    train_model = sub.add_parser("train-text-model", help="Train text model cảm xúc/phong cách local; mặc định submit lên Kaggle.")
    train_model.add_argument("--samples", type=int, default=480, help="Số record tổng hợp tạo trước khi train.")
    train_model.add_argument("--seed", type=int, default=42, help="Seed ngẫu nhiên xác định.")
    train_model.add_argument("--dataset", default="", help="Dataset train JSONL bổ sung, không bắt buộc.")
    train_model.add_argument("--dataset-limit", type=int, default=60000, help="Số record tối đa lấy từ file/thư mục bổ sung.")
    train_model.add_argument("--out", default="outputs/model_training", help="Thư mục output của job train.")
    train_model.add_argument("--model-out", default=str(DEFAULT_LOCAL_MODEL_PATH), help="Đường dẫn artifact model local.")
    train_model.add_argument("--local", action="store_true", help="Train local để smoke test thay vì submit lên Kaggle.")
    train_model.add_argument("--username", default=None, help="Username Kaggle; mặc định đọc từ kaggle.json hoặc KAGGLE_USERNAME.")
    train_model.add_argument("--no-submit", action="store_true", help="Chỉ chuẩn bị file Kaggle ở local.")
    train_model.add_argument("--wait", action="store_true", help="Chờ Kaggle train xong rồi tải artifact model.")
    train_model.add_argument("--poll-seconds", type=int, default=60)
    train_model.add_argument("--timeout-seconds", type=int, default=7200)

    evaluate = sub.add_parser("evaluate", help="Đánh giá lập kế hoạch text-to-lyrics/vocal trên dataset benchmark.")
    evaluate.add_argument("--dataset", default=str(DEFAULT_EVAL_DATASET), help="Dataset đánh giá JSONL.")
    evaluate.add_argument("--out", default="outputs/evaluation", help="Thư mục output artifact đánh giá.")
    evaluate.add_argument("--duration", type=int, default=12, help="Thời lượng dùng khi lập kế hoạch pipeline.")

    project_report = sub.add_parser("project-report", help="Tạo plot latency/retry/error thật từ trạng thái job Kaggle.")
    project_report.add_argument("--source", default="outputs", help="Thư mục gốc chứa các file job_state.json.")
    project_report.add_argument("--out", default="outputs/project_report", help="Thư mục output báo cáo telemetry project.")

    self_improve = sub.add_parser("self-improve", help="Lặp lại train, giả lập prompt user, đánh giá chất lượng và thêm record mục tiêu.")
    self_improve.add_argument("--iterations", type=int, default=3, help="Số vòng tự cải thiện local tối đa.")
    self_improve.add_argument("--samples", type=int, default=640, help="Số mẫu train nền tạo trong mỗi vòng.")
    self_improve.add_argument("--eval-count", type=int, default=24, help="Số mẫu đánh giá tổng hợp trong mỗi vòng.")
    self_improve.add_argument("--seed", type=int, default=5602, help="Seed ngẫu nhiên xác định.")
    self_improve.add_argument("--out", default="outputs/self_improve", help="Thư mục output báo cáo vòng lặp.")
    self_improve.add_argument("--model-out", default=str(DEFAULT_LOCAL_MODEL_PATH), help="Đường dẫn artifact model cần cập nhật.")
    self_improve.add_argument("--extra-dataset", default="", help="Dataset JSONL local bổ sung mà bạn có quyền sử dụng, phân cách bằng dấu phẩy.")
    self_improve.add_argument("--extra-dataset-limit", type=int, default=60000, help="Số record tối đa lấy từ file/thư mục bổ sung.")
    self_improve.add_argument("--duration", type=int, default=30, help="Thời lượng dùng cho prompt user giả lập.")
    self_improve.add_argument("--render-audio", action="store_true", help="Render WAV/MP3 backing local để kiểm tra độ rõ.")
    self_improve.add_argument("--stop-score", type=float, default=0.88, help="Dừng sớm khi điểm tổng hợp đạt ngưỡng này.")

    chorus_ablation = sub.add_parser("chorus-ablation", help="So sánh lập kế hoạch chorus có và không có gợi ý style gốc.")
    chorus_ablation.add_argument("--dataset", default=str(DEFAULT_CHORUS_ABLATION_DATASET), help="Dataset JSONL có trường chorus và style.")
    chorus_ablation.add_argument("--out", default="outputs/chorus_ablation", help="Thư mục output báo cáo ablation.")
    chorus_ablation.add_argument("--duration", type=int, default=45, help="Thời lượng mục tiêu dùng khi lập kế hoạch.")

    import_xlsx = sub.add_parser("import-xlsx-dataset", help="Chuyển benchmark XLSX tiếng Việt thành JSONL.")
    import_xlsx.add_argument("--xlsx", required=True, help="Đường dẫn workbook.")
    import_xlsx.add_argument("--out", default="datasets/evaluation/vietnamese_musicgen_input_dataset.jsonl", help="Đường dẫn JSONL output.")

    evaluate_xlsx = sub.add_parser("evaluate-xlsx", help="Đánh giá lập kế hoạch trực tiếp từ benchmark XLSX tiếng Việt.")
    evaluate_xlsx.add_argument("--xlsx", required=True, help="Đường dẫn workbook.")
    evaluate_xlsx.add_argument("--out", default="outputs/evaluation_xlsx", help="Thư mục output artifact đánh giá.")

    batch_xlsx = sub.add_parser("batch-generate-xlsx", help="Gửi các dòng XLSX lên Kaggle và tạo file MP3.")
    batch_xlsx.add_argument("--xlsx", required=True, help="Đường dẫn workbook.")
    batch_xlsx.add_argument("--out", default="outputs/xlsx_batch", help="Thư mục output.")
    batch_xlsx.add_argument("--model", default=DEFAULT_CUSTOM_MUSIC_MODEL, help="Mã định danh custom music model.")
    batch_xlsx.add_argument("--username", default=None, help="Username Kaggle; mặc định đọc từ kaggle.json hoặc KAGGLE_USERNAME.")
    batch_xlsx.add_argument("--machine-shape", default="NvidiaTeslaT4")
    batch_xlsx.add_argument("--limit", type=int, default=0, help="Số dòng submit tối đa; 0 nghĩa là toàn bộ dòng đã chọn.")
    batch_xlsx.add_argument("--ids", default="", help="ID dòng phân cách bằng dấu phẩy, ví dụ VN001,VN008.")
    batch_xlsx.add_argument("--sample-per-mood", type=int, default=0, help="Giữ tối đa N dòng cho mỗi mood dự kiến.")
    batch_xlsx.add_argument("--no-submit", action="store_true", help="Chỉ chuẩn bị file Kaggle ở local.")
    batch_xlsx.add_argument("--wait", action="store_true", help="Chờ từng job Kaggle và tải output trước khi submit job tiếp theo.")
    batch_xlsx.add_argument("--poll-seconds", type=int, default=60)
    batch_xlsx.add_argument("--timeout-seconds", type=int, default=10_800)

    synth = sub.add_parser("make-eval-dataset", help="Tạo dataset benchmark JSONL tổng hợp có nhãn.")
    synth.add_argument("--count", type=int, default=24, help="Số record tổng hợp.")
    synth.add_argument("--seed", type=int, default=42, help="Seed ngẫu nhiên xác định.")
    synth.add_argument("--out", default="datasets/evaluation/synthetic_eval.jsonl", help="Đường dẫn JSONL output.")
    synth.add_argument("--emotions", default="", help="Nhãn cảm xúc phân cách bằng dấu phẩy; mặc định dùng tất cả.")
    synth.add_argument("--lengths", default="", help="Nhóm độ dài phân cách bằng dấu phẩy: short,medium,long.")
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

    if args.command == "crawl-licensed-lyrics":
        specs = load_source_specs(args.sources)
        report = crawl_licensed_sources(
            specs,
            args.out,
            max_sources=args.max_sources or None,
            max_sections_per_source=args.max_sections,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.command == "train-rhyme-profile":
        profile = build_rhyme_profile(args.dataset, args.out)
        print(json.dumps(profile, ensure_ascii=False, indent=2))
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

    if args.command == "project-report":
        report = build_project_report(args.source, output_root=args.out)
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
