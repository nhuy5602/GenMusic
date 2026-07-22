from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.data.lyric_alignment import align_wav_to_lyrics, load_segments, write_lrc
from src.data.vietnamese_text import normalize_vietnamese_lyrics
from src.evaluation.jam_metrics import objective_metrics, write_metric_report
from src.evaluation.jam_plots import write_jam_plots
from src.evaluation.project_metrics import build_project_report
from src.integrations.kaggle_auto import DEFAULT_MODEL, KaggleJobConfig, refresh_kaggle_job, run_local_generation, submit_text_to_music_job, upload_dataset_to_kaggle
from src.training.self_diffusion import create_random_dataset, train_model, validate_dataset


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
    local.add_argument("--backing-mel", default=None, help="Backing mel condition saved by preprocessing.")
    local.add_argument("--style-anchor", default=None, help="MuQ-MuLan style embedding saved by preprocessing.")
    local.add_argument("--style", default="Vietnamese pop, warm piano, clear melody")
    local.add_argument("--duration", type=float, default=4.0)
    local.add_argument("--checkpoint", default=None)
    local.add_argument("--steps", type=int, default=6)
    local.add_argument("--guidance-scale", type=float, default=1.0, help="Classifier-free lyric guidance; 1.5 is recommended for new checkpoints.")
    local.add_argument("--seed", type=int, default=5602)
    local.add_argument("--device", default=None)
    local.add_argument("--out", required=True)
    local.add_argument("--vocoder", default="vocos", choices=["vocos", "griffinlim"], help="Chọn bộ giải mã spectrogram thành âm thanh.")
    local.add_argument("--roberta-model", default="vinai/xphonebert-base", help="Tên model RoBERTa dùng làm Text Encoder.")
    local.add_argument("--reference-dataset", default=None, help="Thư mục dataset đã preprocess -- lấy backing_mel + style_anchor thật từ một record để sinh nhạc có điều kiện đúng như lúc train, thay vì mặc định zero-conditioned.")
    local.add_argument("--reference-id", default=None, help="ID record cụ thể trong --reference-dataset (mặc định lấy record đầu tiên).")

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
    train.add_argument("--dataset", required=True, nargs="+", help="Một hoặc nhiều thư mục dataset đã preprocess (kết hợp lại thành một tập huấn luyện).")
    train.add_argument("--checkpoint", required=True)
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--learning-rate", type=float, default=2e-4)
    train.add_argument("--device", default=None)
    train.add_argument("--max-records", type=int, default=None)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--save-every-epoch", action="store_true")
    train.add_argument("--checkpoint-every-steps", type=int, default=0)
    train.add_argument("--log-every-steps", type=int, default=10)
    train.add_argument("--progress-file", default=None)
    train.add_argument("--roberta-model", default="vinai/xphonebert-base", help="Tên model RoBERTa dùng làm Text Encoder.")
    train.add_argument("--dim", type=int, default=256, help="Hidden dim của MicroDiT.")
    train.add_argument("--depth", type=int, default=4, help="Số lớp transformer block.")
    train.add_argument("--heads", type=int, default=4, help="Số attention head.")
    train.add_argument("--ff-mult", type=int, default=4, help="Hệ số feed-forward.")
    train.add_argument("--frames-per-chunk", type=int, default=None, help="Override độ dài crop train; 384 tương đương khoảng bốn giây ở 24 kHz.")
    train.add_argument("--lambda-vocal", type=float, default=1.0, help="Weight of auxiliary vocal-only prediction loss (Mixed Pro style, 0.0 disables it).")
    train.add_argument("--architecture", default="microdit", choices=("microdit", "native_dit"), help="microdit = this project's cross-attention lyric conditioning; native_dit = DiffRhythm2's real concatenated self-attention DiT, vendored (src/models/diffrhythm2_native.py), trained from scratch at student scale.")

    distill = sub.add_parser("train-distill", help="Huấn luyện chưng cất tri thức từ DiffRhythm gốc sang MicroDiT.")
    distill.add_argument("--dataset", required=True, nargs="+", help="Một hoặc nhiều thư mục dataset đã preprocess (kết hợp lại thành một tập huấn luyện).")
    distill.add_argument("--student-checkpoint", required=True)
    distill.add_argument("--teacher-checkpoint", default=None)
    distill.add_argument("--epochs", type=int, default=5)
    distill.add_argument("--batch-size", type=int, default=4)
    distill.add_argument("--learning-rate", type=float, default=1e-4)
    distill.add_argument("--device", default=None)
    distill.add_argument("--alpha-feature", type=float, default=0.5)
    distill.add_argument("--beta-repa", type=float, default=0.0, help="Weight of REPA-style representation-alignment loss against a frozen MuQ encoder (0.0 disables it).")
    distill.add_argument("--repo-id", default="ASLP-lab/DiffRhythm2")
    distill.add_argument("--dim", type=int, default=256, help="Hidden dim của MicroDiT student.")
    distill.add_argument("--depth", type=int, default=4, help="Số lớp transformer block.")
    distill.add_argument("--heads", type=int, default=4, help="Số attention head.")
    distill.add_argument("--ff-mult", type=int, default=4, help="Hệ số feed-forward.")
    distill.add_argument("--roberta-model", default="vinai/xphonebert-base", help="Tên model RoBERTa dùng làm Text Encoder.")
    distill.add_argument("--max-records", type=int, default=None, help="Limit training to the first N usable records (for cheap smoke tests).")
    distill.add_argument("--lambda-vocal", type=float, default=1.0, help="Weight of auxiliary vocal-only prediction loss (Mixed Pro style, 0.0 disables it).")

    latent_encoder = sub.add_parser("train-latent-encoder", help="Pretrain một encoder mới (mel/audio -> latent 64 chiều, 5Hz) khớp với decoder BigVGAN thật của DiffRhythm2 (đóng băng, tải từ HuggingFace).")
    latent_encoder.add_argument("--dataset", required=True)
    latent_encoder.add_argument("--checkpoint", required=True)
    latent_encoder.add_argument("--epochs", type=int, default=1)
    latent_encoder.add_argument("--batch-size", type=int, default=4)
    latent_encoder.add_argument("--learning-rate", type=float, default=1e-4)
    latent_encoder.add_argument("--device", default=None)
    latent_encoder.add_argument("--max-records", type=int, default=None, help="Limit training to the first N usable records (for cheap smoke tests).")
    latent_encoder.add_argument("--log-every-steps", type=int, default=10)
    latent_encoder.add_argument("--repo-id", default="ASLP-lab/DiffRhythm2")
    latent_encoder.add_argument("--crop-seconds", type=float, default=1.0, help="Độ dài đoạn audio train mỗi step (ngắn -- backprop qua cả BigVGAN decoder tốn VRAM nhiều nếu crop dài).")
    latent_encoder.add_argument("--warmup-steps", type=int, default=200, help="Linear LR warmup steps before cosine decay -- stabilizes early training against the frozen decoder.")
    latent_encoder.add_argument("--grad-clip-norm", type=float, default=1.0)

    precompute_latent = sub.add_parser("precompute-latent-dataset", help="Chuyển dataset mel đã có sang không gian latent thật của DiffRhythm2 (64 chiều, 5Hz), dùng LatentAudioEncoder đã pretrain.")
    precompute_latent.add_argument("--source-dataset", required=True)
    precompute_latent.add_argument("--encoder-checkpoint", required=True)
    precompute_latent.add_argument("--out", required=True)
    precompute_latent.add_argument("--device", default=None)
    precompute_latent.add_argument("--max-records", type=int, default=None)
    precompute_latent.add_argument("--crop-seconds", type=float, default=4.096)

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

    preprocess = sub.add_parser("preprocess-raw", help="Tiền xử lý và gán nhãn dataset âm thanh thô.")
    preprocess.add_argument("--input", default="dataset/vietnamese_songs")
    preprocess.add_argument("--output", default="dataset/diff_rhythm_dataset")
    preprocess.add_argument("--whisper-model", default="small")
    preprocess.add_argument("--keep-separated-count", type=int, default=10, help="Số lượng bài hát giữ lại thư mục âm thanh đã tách để hậu kiểm.")
    preprocess.add_argument("--max-files", type=int, default=None, help="Giới hạn số lượng bài hát tối đa sẽ xử lý.")
    preprocess.add_argument("--skip-demucs", action="store_true", help="Bỏ tách vocal, dùng bản phối thật làm mục tiêu nhanh.")
    preprocess.add_argument("--skip-asr", action="store_true", help="Bỏ Whisper ASR và dùng nhãn text mặc định.")
    preprocess.add_argument("--demucs-device", default="auto", choices=("auto", "cuda", "cpu"))
    preprocess.add_argument("--whisper-device", default="auto", choices=("auto", "cpu", "cuda"))

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
        report = run_local_generation(
            text=args.text,
            style=args.style,
            output_dir=args.out,
            duration_seconds=args.duration,
            checkpoint=args.checkpoint,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            device=args.device,
            vocoder=args.vocoder,
            roberta_model=args.roberta_model,
            reference_dataset=args.reference_dataset,
            reference_id=args.reference_id,
            backing_mel=args.backing_mel,
            style_anchor=args.style_anchor,
        )
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
        report = train_model(args.dataset, args.checkpoint, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, device=args.device, max_records=args.max_records, roberta_model=args.roberta_model, dim=args.dim, depth=args.depth, heads=args.heads, ff_mult=args.ff_mult, frames_per_chunk=args.frames_per_chunk, resume=args.resume, save_every_epoch=args.save_every_epoch, checkpoint_every_steps=args.checkpoint_every_steps, log_every_steps=args.log_every_steps, progress_path=args.progress_file, lambda_vocal=args.lambda_vocal, architecture=args.architecture)
    elif args.command == "train-distill":
        from src.training.distill_training import run_distillation_training
        report = run_distillation_training(args.dataset, args.student_checkpoint, args.teacher_checkpoint, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, device=args.device, alpha_feature=args.alpha_feature, beta_repa=args.beta_repa, repo_id=args.repo_id, dim=args.dim, depth=args.depth, heads=args.heads, ff_mult=args.ff_mult, roberta_model=args.roberta_model, max_records=args.max_records, lambda_vocal=args.lambda_vocal)
    elif args.command == "train-latent-encoder":
        from src.training.latent_encoder_training import train_latent_encoder
        report = train_latent_encoder(args.dataset, args.checkpoint, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, device=args.device, max_records=args.max_records, log_every_steps=args.log_every_steps, repo_id=args.repo_id, crop_seconds=args.crop_seconds, warmup_steps=args.warmup_steps, grad_clip_norm=args.grad_clip_norm)
    elif args.command == "precompute-latent-dataset":
        from src.data.precompute_latent_dataset import precompute_latent_dataset
        report = precompute_latent_dataset(args.source_dataset, args.encoder_checkpoint, args.out, device=args.device, max_records=args.max_records, crop_seconds=args.crop_seconds)
    elif args.command == "normalize-lyrics":
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(normalize_vietnamese_lyrics(Path(args.input).read_text(encoding="utf-8")) + "\n", encoding="utf-8")
        report = {"status": "normalized", "path": str(output.resolve())}
    elif args.command == "lyrics-g2p":
        from text2phonemesequence import Text2PhonemeSequence
        text2phone = Text2PhonemeSequence(language='vie', is_cuda=False)
        text_content = Path(args.input).read_text(encoding="utf-8")
        phonemes = text2phone.infer_sentence(text_content)
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        result = {"text": text_content, "phonemes": phonemes, "backend": "text2phonemesequence"}
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        report = result
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
    elif args.command == "preprocess-raw":
        from src.data.preprocess_raw_vietnamese import preprocess_raw_audio
        report = preprocess_raw_audio(
            args.input,
            args.output,
            args.whisper_model,
            keep_separated_count=args.keep_separated_count,
            max_files=args.max_files,
            use_demucs=not args.skip_demucs,
            transcribe=not args.skip_asr,
            demucs_device=args.demucs_device,
            whisper_device=args.whisper_device,
        )
    else:  # pragma: no cover - argparse enforces command choices
        raise ValueError(args.command)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report.get("status") in {"failed", "completed_with_warnings", "needs_setup", "pending", "invalid", "needs-torch"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
