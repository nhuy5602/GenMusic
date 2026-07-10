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
from .integrations.diffrhythm_official import (
    DiffRhythmConfig,
    create_random_official_dataset,
    ensure_official_checkout,
    run_official_inference,
    run_official_training,
    validate_official_dataset,
    write_official_distillation_plan,
)
from .integrations.kaggle_auto import (
    DEFAULT_DIFFRHYTHM_MODEL,
    KaggleJobConfig,
    refresh_kaggle_job,
    submit_text_to_music_job,
)
from .evaluation.project_metrics import build_project_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="genmusic-vn", description="Pipeline GenMusic VN dùng model chính thức ASLP-lab/DiffRhythm.")
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="Stage/submit input lyric lên Kaggle để DiffRhythm sinh bài hát.")
    generate.add_argument("--text", required=True)
    generate.add_argument("--duration", type=int, default=95, help="DiffRhythm hỗ trợ 95 hoặc 96-285 giây.")
    generate.add_argument("--out", default="outputs")
    generate.add_argument("--genre", default=None)
    generate.add_argument("--model", default=DEFAULT_DIFFRHYTHM_MODEL)
    generate.add_argument("--username", default=None)
    generate.add_argument("--machine-shape", default="NvidiaTeslaT4")
    generate.add_argument("--no-submit", action="store_true")
    generate.add_argument("--wait", action="store_true")
    generate.add_argument("--poll-seconds", type=int, default=60)
    generate.add_argument("--timeout-seconds", type=int, default=21_600)

    refresh = sub.add_parser("refresh-kaggle", help="Cập nhật trạng thái và tải output job DiffRhythm.")
    refresh.add_argument("--state", required=True)

    normalize = sub.add_parser("normalize-lyrics", help="Chuẩn hóa lyric tiếng Việt.")
    normalize.add_argument("--input", required=True)
    normalize.add_argument("--out", required=True)

    g2p = sub.add_parser("lyrics-g2p", help="Xuất token G2P tiếng Việt kèm số thanh điệu.")
    g2p.add_argument("--input", required=True)
    g2p.add_argument("--out", required=True)
    g2p.add_argument("--no-phonemizer", action="store_true")

    align = sub.add_parser("align-lyrics", help="Tạo LRC từ ASR segment hoặc timestamp heuristic.")
    align.add_argument("--audio", required=True)
    align.add_argument("--lyrics", required=True)
    align.add_argument("--out", required=True)
    align.add_argument("--segments", default="")
    align.add_argument("--asr-model", default=None)
    align.add_argument("--allow-heuristic", action="store_true")

    random_data = sub.add_parser("make-random-diffrhythm-dataset", help="Tạo dataset random đúng format train.scp của DiffRhythm.")
    random_data.add_argument("--out", required=True)
    random_data.add_argument("--count", type=int, default=4)
    random_data.add_argument("--max-frames", type=int, default=64)
    random_data.add_argument("--seed", type=int, default=5602)

    validate = sub.add_parser("validate-diffrhythm-dataset", help="Kiểm tra train.scp và artifact .pt của DiffRhythm.")
    validate.add_argument("--dataset", required=True)

    train = sub.add_parser("train-diffrhythm", help="Chạy train/train.py chính thức của DiffRhythm.")
    train.add_argument("--dataset", required=True)
    train.add_argument("--repo", default=None)
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument("--batch-size", type=int, default=1)
    train.add_argument("--timeout-seconds", type=int, default=21_600)

    distill = sub.add_parser("distill-diffrhythm", help="Tạo kế hoạch distillation teacher 32 bước -> student 4 bước.")
    distill.add_argument("--out", required=True)
    distill.add_argument("--teacher-ref", default=DEFAULT_DIFFRHYTHM_MODEL)
    distill.add_argument("--teacher-steps", type=int, default=32)
    distill.add_argument("--student-steps", type=int, default=4)

    local = sub.add_parser("generate-local-diffrhythm", help="Chạy infer/infer.py chính thức tại local.")
    local.add_argument("--lyrics", required=True)
    local.add_argument("--style", required=True)
    local.add_argument("--out", required=True)
    local.add_argument("--repo", default=None)
    local.add_argument("--duration", type=int, default=95)
    local.add_argument("--no-chunked", action="store_true")

    evaluation = sub.add_parser("evaluate-diffrhythm", help="Đánh giá khách quan, không yêu cầu MOS.")
    evaluation.add_argument("--generated", required=True)
    evaluation.add_argument("--reference", default=None)
    evaluation.add_argument("--generated-text", default=None)
    evaluation.add_argument("--reference-text", default=None)
    evaluation.add_argument("--out", required=True)

    project = sub.add_parser("project-report", help="Sinh telemetry từ job_state.json.")
    project.add_argument("--source", default="outputs")
    project.add_argument("--out", default="outputs/project_report")
    return parser


def _read_optional_text(value: str | None) -> str | None:
    if not value:
        return None
    try:
        path = Path(value)
        return path.read_text(encoding="utf-8") if path.exists() else value
    except OSError:
        return value


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
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.command == "refresh-kaggle":
        print(json.dumps(refresh_kaggle_job(args.state), ensure_ascii=False, indent=2))
        return 0

    if args.command == "normalize-lyrics":
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(normalize_vietnamese_lyrics(Path(args.input).read_text(encoding="utf-8")) + "\n", encoding="utf-8")
        print(json.dumps({"status": "normalized", "path": str(output.resolve())}, ensure_ascii=False, indent=2))
        return 0

    if args.command == "lyrics-g2p":
        result = vietnamese_g2p(Path(args.input).read_text(encoding="utf-8"), use_phonemizer=not args.no_phonemizer)
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
        return 0

    if args.command == "align-lyrics":
        segments = load_segments(args.segments) if args.segments else None
        lines = align_wav_to_lyrics(
            args.audio,
            Path(args.lyrics).read_text(encoding="utf-8"),
            segments=segments,
            asr_model=args.asr_model,
            allow_heuristic=args.allow_heuristic,
        )
        output = write_lrc(lines, args.out)
        print(json.dumps({"status": "aligned", "path": str(output.resolve()), "line_count": len(lines)}, ensure_ascii=False, indent=2))
        return 0

    if args.command == "make-random-diffrhythm-dataset":
        report = create_random_official_dataset(args.out, count=args.count, max_frames=args.max_frames, seed=args.seed)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.command == "validate-diffrhythm-dataset":
        print(json.dumps(validate_official_dataset(args.dataset), ensure_ascii=False, indent=2))
        return 0

    if args.command == "train-diffrhythm":
        report = run_official_training(args.dataset, repo_path=args.repo, epochs=args.epochs, batch_size=args.batch_size, timeout_seconds=args.timeout_seconds)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.command == "distill-diffrhythm":
        report = write_official_distillation_plan(args.out, teacher_ref=args.teacher_ref, teacher_steps=args.teacher_steps, student_steps=args.student_steps)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.command == "generate-local-diffrhythm":
        report = run_official_inference(
            lyrics=Path(args.lyrics).read_text(encoding="utf-8"),
            style_prompt=args.style,
            output_dir=args.out,
            config=DiffRhythmConfig(repo_path=args.repo, audio_length=args.duration, chunked=not args.no_chunked),
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.command == "evaluate-diffrhythm":
        report = {
            "objective": objective_metrics(
                generated_audio=args.generated,
                reference_audio=args.reference,
                generated_transcript=_read_optional_text(args.generated_text),
                reference_transcript=_read_optional_text(args.reference_text),
            ),
            "subjective": {"status": "skipped-by-request", "MOS": None, "CMOS": None},
            "protocol": {"backend": "ASLP-lab/DiffRhythm", "mos_required": False},
        }
        write_metric_report(report, args.out)
        report["plots"] = write_jam_plots(report, Path(args.out) / "plots")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.command == "project-report":
        print(json.dumps(build_project_report(args.source, output_root=args.out), ensure_ascii=False, indent=2))
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
