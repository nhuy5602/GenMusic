from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .data.lyric_alignment import align_wav_to_lyrics, load_segments, write_lrc
from .data.vietnamese_g2p import vietnamese_g2p
from .data.vietnamese_text import normalize_vietnamese_lyrics
from .evaluation.jam_metrics import objective_metrics, write_metric_report
from .evaluation.jam_plots import write_jam_plots
from .evaluation.project_metrics import build_project_report
from .integrations.kaggle_auto import DEFAULT_MODEL, KaggleJobConfig, refresh_kaggle_job, run_local_generation, submit_text_to_music_job, upload_dataset_to_kaggle
from .training.self_diffusion import create_random_dataset, train_model, validate_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="genmusic-vn", description="Model sinh nhạc từ text do GenMusic VN tự code.")
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="Đóng gói hoặc submit request lên Kaggle.")
    generate.add_argument("--text", required=True)
    generate.add_argument("--duration", type=int, default=12)
    generate.add_argument("--out", default="outputs")
    generate.add_argument("--genre", default=None)
    generate.add_argument("--model", default=DEFAULT_MODEL)
    generate.add_argument("--username", default=None)
    generate.add_argument("--machine-shape", default="NvidiaTeslaT4")
    generate.add_argument("--no-submit", action="store_true")
    generate.add_argument("--wait", action="store_true")
    generate.add_argument("--poll-seconds", type=int, default=60)
    generate.add_argument("--timeout-seconds", type=int, default=21_600)
    generate.add_argument("--dataset-ref", default=None, help="Dataset training cố định dạng owner/slug; mặc định lấy từ Kaggle username và slug project.")
    generate.add_argument("--dataset-gb", type=float, default=None, help="Tương thích lệnh cũ; dataset phải được tạo trước bằng make-and-upload-dataset.")

    refresh = sub.add_parser("refresh-kaggle", help="Cập nhật trạng thái và tải output Kaggle.")
    refresh.add_argument("--state", required=True)

    local = sub.add_parser("generate-local", help="Sinh WAV/MP3 bằng model tự code tại local.")
    local.add_argument("--text", required=True)
    local.add_argument("--style", default="Vietnamese pop, warm piano, clear melody")
    local.add_argument("--duration", type=float, default=4.0)
    local.add_argument("--checkpoint", default=None)
    local.add_argument("--steps", type=int, default=6)
    local.add_argument("--seed", type=int, default=5602)
    local.add_argument("--device", default=None)
    local.add_argument("--out", required=True)

    random_data = sub.add_parser("make-random-dataset", help="Tạo dataset mel random cho smoke training.")
    random_data.add_argument("--out", required=True)
    random_data.add_argument("--count", type=int, default=16)
    random_data.add_argument("--frames", type=int, default=128)
    random_data.add_argument("--seed", type=int, default=5602)
    random_data.add_argument("--target-gb", type=float, default=0.0, help="Mục tiêu dung lượng dataset thực trên đĩa.")

    validate = sub.add_parser("validate-dataset", help="Kiểm tra records.jsonl và các tensor mel.")
    validate.add_argument("--dataset", required=True)

    upload = sub.add_parser("upload-dataset", help="Upload dataset self-diffusion lên Kaggle.")
    upload.add_argument("--dataset", required=True)
    upload.add_argument("--username", default=None)
    upload.add_argument("--slug", default=None)
    upload.add_argument("--dataset-ref", default=None, help="Dataset ref cố định dạng owner/slug.")
    upload.add_argument("--timeout-seconds", type=int, default=3_600)

    prepare = sub.add_parser("make-and-upload-dataset", help="Tạo dataset random theo dung lượng rồi upload vào dataset ref cố định.")
    prepare.add_argument("--out", default="datasets/random_self_diffusion_training")
    prepare.add_argument("--count", type=int, default=16)
    prepare.add_argument("--frames", type=int, default=128)
    prepare.add_argument("--seed", type=int, default=5602)
    prepare.add_argument("--target-gb", type=float, default=1.0, help="Dung lượng dataset mục tiêu trên đĩa, ví dụ 1 hoặc 5.")
    prepare.add_argument("--username", default=None)
    prepare.add_argument("--dataset-ref", default=None, help="Dataset ref cố định dạng owner/slug.")
    prepare.add_argument("--timeout-seconds", type=int, default=3_600)

    train = sub.add_parser("train-self", help="Train conditional diffusion model tự code.")
    train.add_argument("--dataset", required=True)
    train.add_argument("--checkpoint", required=True)
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--learning-rate", type=float, default=2e-4)
    train.add_argument("--device", default=None)
    train.add_argument("--max-records", type=int, default=None)

    normalize = sub.add_parser("normalize-lyrics", help="Chuẩn hóa lyric tiếng Việt.")
    normalize.add_argument("--input", required=True)
    normalize.add_argument("--out", required=True)

    g2p = sub.add_parser("lyrics-g2p", help="Xuất token G2P tiếng Việt kèm số thanh điệu.")
    g2p.add_argument("--input", required=True)
    g2p.add_argument("--out", required=True)
    g2p.add_argument("--no-phonemizer", action="store_true")

    align = sub.add_parser("align-lyrics", help="Tạo LRC từ ASR segment hoặc heuristic.")
    align.add_argument("--audio", required=True)
    align.add_argument("--lyrics", required=True)
    align.add_argument("--out", required=True)
    align.add_argument("--segments", default="")
    align.add_argument("--asr-model", default=None)
    align.add_argument("--allow-heuristic", action="store_true")

    evaluation = sub.add_parser("evaluate-self", help="Đánh giá khách quan, không yêu cầu MOS.")
    evaluation.add_argument("--generated", required=True)
    evaluation.add_argument("--reference", default=None)
    evaluation.add_argument("--generated-text", default=None)
    evaluation.add_argument("--reference-text", default=None)
    evaluation.add_argument("--out", required=True)

    project = sub.add_parser("project-report", help="Sinh telemetry từ các job state.")
    project.add_argument("--source", default="outputs")
    project.add_argument("--out", default="outputs/project_report")
    return parser


def _read_optional_text(value: str | None) -> str | None:
    if not value:
        return None
    path = Path(value)
    return path.read_text(encoding="utf-8") if path.exists() else value


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)

    if args.command == "generate":
        report = submit_text_to_music_job(
            text=args.text,
            output_root=args.out,
            duration_seconds=args.duration,
            genre=args.genre,
            config=KaggleJobConfig(model=args.model, username=args.username, machine_shape=args.machine_shape, submit=not args.no_submit, wait=args.wait, poll_seconds=args.poll_seconds, timeout_seconds=args.timeout_seconds, training_dataset_ref=args.dataset_ref),
        )
    elif args.command == "refresh-kaggle":
        report = refresh_kaggle_job(args.state)
    elif args.command == "generate-local":
        report = run_local_generation(text=args.text, style=args.style, output_dir=args.out, duration_seconds=args.duration, checkpoint=args.checkpoint, steps=args.steps, seed=args.seed, device=args.device)
    elif args.command == "make-random-dataset":
        target_bytes = int(args.target_gb * (1024 ** 3)) if args.target_gb > 0 else None
        report = create_random_dataset(args.out, count=args.count, frames=args.frames, seed=args.seed, target_bytes=target_bytes)
    elif args.command == "validate-dataset":
        report = validate_dataset(args.dataset)
    elif args.command == "upload-dataset":
        report = upload_dataset_to_kaggle(args.dataset, username=args.username, slug=args.slug, dataset_ref=args.dataset_ref, timeout_seconds=args.timeout_seconds)
    elif args.command == "make-and-upload-dataset":
        target_bytes = int(args.target_gb * (1024 ** 3)) if args.target_gb > 0 else None
        dataset_report = create_random_dataset(args.out, count=args.count, frames=args.frames, seed=args.seed, target_bytes=target_bytes)
        upload_report = upload_dataset_to_kaggle(args.out, username=args.username, dataset_ref=args.dataset_ref, timeout_seconds=args.timeout_seconds)
        report = {"status": upload_report["status"], "dataset_report": dataset_report, "upload": upload_report}
    elif args.command == "train-self":
        report = train_model(args.dataset, args.checkpoint, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, device=args.device, max_records=args.max_records)
    elif args.command == "normalize-lyrics":
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(normalize_vietnamese_lyrics(Path(args.input).read_text(encoding="utf-8")) + "\n", encoding="utf-8")
        report = {"status": "normalized", "path": str(output.resolve())}
    elif args.command == "lyrics-g2p":
        result = vietnamese_g2p(Path(args.input).read_text(encoding="utf-8"), use_phonemizer=not args.no_phonemizer)
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        report = result.as_dict()
    elif args.command == "align-lyrics":
        segments = load_segments(args.segments) if args.segments else None
        lines = align_wav_to_lyrics(args.audio, Path(args.lyrics).read_text(encoding="utf-8"), segments=segments, asr_model=args.asr_model, allow_heuristic=args.allow_heuristic)
        output = write_lrc(lines, args.out)
        report = {"status": "aligned", "path": str(output.resolve()), "line_count": len(lines)}
    elif args.command == "evaluate-self":
        report = {"objective": objective_metrics(generated_audio=args.generated, reference_audio=args.reference, generated_transcript=_read_optional_text(args.generated_text), reference_transcript=_read_optional_text(args.reference_text)), "subjective": {"status": "skipped-by-request", "MOS": None, "CMOS": None}, "protocol": {"backend": "genmusic-vn-self-diffusion", "mos_required": False}}
        write_metric_report(report, args.out)
        report["plots"] = write_jam_plots(report, Path(args.out) / "plots")
    elif args.command == "project-report":
        report = build_project_report(args.source, output_root=args.out)
    else:  # pragma: no cover - argparse enforces command choices
        raise ValueError(args.command)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report.get("status") in {"failed", "needs_setup", "pending", "invalid", "needs-torch"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
