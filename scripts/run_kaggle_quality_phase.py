"""Submit only checkpoint generation/ASR evaluation to a Kaggle GPU.

This phase never trains and never creates report plots. It can therefore test
new generation durations, CFG values, or conditioning modes against an existing
checkpoint without spending another multi-hour training reservation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.kaggle_phase_submit import (
    ensure_source_dataset,
    new_run_dir,
    require_complete_kernels,
    submit_context,
    submit_phase_kernel,
)
from scripts.run_kaggle_iterative_self import _wait_for_dataset_visible
from scripts.run_kaggle_multi_part_training import _parse_kernel_refs


def _kernel_code(
    *,
    source_count: int,
    expected_records: int,
    evaluation_records: int,
    duration: float,
    steps: int,
    guidance_scales: str,
    pronunciation_prior_strengths: str,
    native_prior_start_strength: float,
    conditioning_modes: list[str],
    native_only: bool,
) -> str:
    template = r'''import json
import shutil
import sys
import traceback
from pathlib import Path

INPUT_ROOT = Path("/kaggle/input")
WORKING_ROOT = Path("/kaggle/working")

try:
    source_cli = next(
        path
        for path in INPUT_ROOT.rglob("cli.py")
        if (path.parent / "scripts/kaggle_phase_runtime.py").is_file()
    )
    source_root = WORKING_ROOT / "GenMusic"
    shutil.copytree(source_cli.parent, source_root, dirs_exist_ok=True)
    sys.path.insert(0, str(source_root))

    from scripts.kaggle_phase_runtime import (
        build_combined_dataset,
        find_checkpoint,
        gpu_preflight,
        install_online_audio_dependencies,
        run_logged,
    )

    environment = install_online_audio_dependencies(
        source_root, native_only=__NATIVE_ONLY__
    )
    gpu_preflight()
    combined_root = build_combined_dataset(
        source_count=__SOURCE_COUNT__,
        expected_records=__EXPECTED_RECORDS__,
    )
    checkpoint = find_checkpoint()
    print(f"evaluation_checkpoint={checkpoint}", flush=True)

    reports = {}
    modes = __CONDITIONING_MODES__
    for mode in modes:
        output_dir = WORKING_ROOT / "quality_evaluation" / mode
        command = [
            sys.executable,
            str(source_root / "scripts/evaluate_generation_quality.py"),
            str(checkpoint),
            str(combined_root),
            str(output_dir),
            str(__EVALUATION_RECORDS__),
            "--whisper-model",
            "small",
            "--duration",
            str(__DURATION__),
            "--steps",
            str(__STEPS__),
            "--guidance-scales",
            __GUIDANCE_SCALES__,
            "--pronunciation-prior-strengths",
            __PRONUNCIATION_PRIOR_STRENGTHS__,
            "--native-prior-start-strength",
            str(__NATIVE_PRIOR_START_STRENGTH__),
        ]
        if mode == "reference_style":
            command.append("--use-style-anchor")
        if __NATIVE_ONLY__:
            command.append("--native-only")
        run_logged(
            command,
            f"quality_evaluation_{mode}",
            cwd=source_root,
            env=environment,
        )
        reports[mode] = json.loads(
            (output_dir / "quality_report.json").read_text(encoding="utf-8")
        )

    def report_rank(item):
        _, report = item
        summary = report.get("summary") or {}
        pitch = float(summary.get("mean_pitch_std_semitones_generated") or 0.0)
        real_pitch = float(summary.get("mean_pitch_std_semitones_real") or 2.5)
        flatness = float(summary.get("mean_flatness_generated") or 0.0)
        real_flatness = float(summary.get("mean_flatness_real") or 0.02)
        acoustic_distance = (
            abs(pitch - real_pitch) / max(0.25, real_pitch)
            + abs(flatness - real_flatness) / max(0.002, real_flatness)
        )
        return (
            bool(summary.get("intelligibility_pass")),
            float(summary.get("mean_word_accuracy_generated") or 0.0),
            -float(summary.get("mean_cer_generated") or 1.0),
            -acoustic_distance,
        )

    selected_mode, selected_report = max(reports.items(), key=report_rank)
    selected_dir = WORKING_ROOT / "quality_evaluation" / selected_mode
    shutil.copy2(selected_dir / "quality_report.json", WORKING_ROOT / "quality_report.json")
    samples = selected_report.get("samples") or []

    def sample_rank(sample):
        asr = sample.get("generated_asr") or {}
        vocal_asr = sample.get("generated_vocal_asr") or {}
        metrics = sample.get("generated") or {}
        real = sample.get("real_vocal_same_vocoder") or {}
        pitch = float(metrics.get("pitch_std_semitones") or 0.0)
        real_pitch = float(real.get("pitch_std_semitones") or 2.5)
        return (
            float(asr.get("word_accuracy") or 0.0),
            -float(asr.get("cer") or 1.0),
            float(vocal_asr.get("word_accuracy") or 0.0),
            -float(vocal_asr.get("cer") or 1.0),
            -abs(pitch - real_pitch),
        )

    if samples:
        best_sample = max(samples, key=sample_rank)
        best_wav = selected_dir / f"{best_sample['id']}_generated.wav"
        if best_wav.is_file():
            shutil.copy2(best_wav, WORKING_ROOT / "best_generated.wav")
            ffmpeg = shutil.which("ffmpeg")
            if ffmpeg:
                run_logged(
                    [
                        ffmpeg,
                        "-y",
                        "-i",
                        str(best_wav),
                        "-codec:a",
                        "libmp3lame",
                        "-q:a",
                        "2",
                        str(WORKING_ROOT / "best_generated.mp3"),
                    ],
                    "encode_best_mp3",
                )
        for stem_name in ("vocal", "backing"):
            stem_wav = selected_dir / f"{best_sample['id']}_generated_{stem_name}.wav"
            if stem_wav.is_file():
                shutil.copy2(stem_wav, WORKING_ROOT / f"best_generated_{stem_name}.wav")

    comparison = {
        mode: report.get("summary") or {}
        for mode, report in reports.items()
    }
    (WORKING_ROOT / "quality_comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (WORKING_ROOT / "evaluation_result.json").write_text(
        json.dumps(
            {
                "phase": "quality_evaluation",
                "checkpoint": str(checkpoint),
                "selected_conditioning_mode": selected_mode,
                "duration_seconds": __DURATION__,
                "diffusion_steps": __STEPS__,
                "guidance_scales": __GUIDANCE_SCALES__,
                "pronunciation_prior_strengths": __PRONUNCIATION_PRIOR_STRENGTHS__,
                "intelligibility_pass": bool(
                    (selected_report.get("summary") or {}).get("intelligibility_pass")
                ),
                "quality_summary": selected_report.get("summary") or {},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    shutil.rmtree(combined_root, ignore_errors=True)
    shutil.rmtree(source_root, ignore_errors=True)
    (WORKING_ROOT / "success.txt").write_text("success", encoding="utf-8")
except Exception:
    traceback_text = traceback.format_exc()
    print(traceback_text, flush=True)
    (WORKING_ROOT / "error.txt").write_text(traceback_text, encoding="utf-8")
    raise
'''
    return (
        template.replace("__SOURCE_COUNT__", str(source_count))
        .replace("__EXPECTED_RECORDS__", str(expected_records))
        .replace("__EVALUATION_RECORDS__", str(evaluation_records))
        .replace("__DURATION__", repr(float(duration)))
        .replace("__STEPS__", str(steps))
        .replace("__GUIDANCE_SCALES__", repr(guidance_scales))
        .replace(
            "__PRONUNCIATION_PRIOR_STRENGTHS__",
            repr(pronunciation_prior_strengths),
        )
        .replace("__NATIVE_PRIOR_START_STRENGTH__", repr(float(native_prior_start_strength)))
        .replace("__CONDITIONING_MODES__", repr(conditioning_modes))
        .replace("__NATIVE_ONLY__", repr(bool(native_only)))
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", action="append", default=[], metavar="PART=KERNEL_REF")
    parser.add_argument("--checkpoint-kernel-ref", default="")
    parser.add_argument("--checkpoint-dataset-ref", default="")
    parser.add_argument("--source-dataset-ref", default="")
    parser.add_argument("--expected-records", type=int, default=1843)
    parser.add_argument("--evaluation-records", type=int, default=6)
    parser.add_argument("--duration", type=float, default=4.096)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--guidance-scales", default="0.75,1.0,1.25")
    parser.add_argument(
        "--pronunciation-prior-strengths",
        default="0.0",
        help="Comma-separated Vietnamese pronunciation-prior strengths in [0,1]",
    )
    parser.add_argument(
        "--native-prior-start-strength",
        type=float,
        default=0.0,
        help="Internal native vocal-prior interpolation at CFM start; no pretrained TTS.",
    )
    parser.add_argument(
        "--conditioning-modes",
        default="no_style,reference_style",
        help="Comma-separated: no_style,reference_style",
    )
    parser.add_argument(
        "--native-only",
        action="store_true",
        help="Fail unless the checkpoint generates with native_utf8 and prior strength 0.",
    )
    parser.add_argument("--accelerator", default="NvidiaTeslaT4")
    parser.add_argument("--session-timeout-seconds", type=int, default=7200)
    parser.add_argument("--kernel-slug", default="")
    args = parser.parse_args()

    if args.native_only:
        prior_values = [
            float(value.strip())
            for value in args.pronunciation_prior_strengths.split(",")
            if value.strip()
        ]
        if prior_values != [0.0]:
            raise ValueError("--native-only requires --pronunciation-prior-strengths 0.0")

    if bool(args.checkpoint_kernel_ref) == bool(args.checkpoint_dataset_ref):
        raise ValueError(
            "Pass exactly one of --checkpoint-kernel-ref or --checkpoint-dataset-ref"
        )
    modes = [value.strip() for value in args.conditioning_modes.split(",") if value.strip()]
    invalid_modes = set(modes) - {"no_style", "reference_style"}
    if not modes or invalid_modes:
        raise ValueError(f"Invalid conditioning modes: {sorted(invalid_modes)}")

    refs_by_part = _parse_kernel_refs(args.kernel)
    context = submit_context()
    complete_refs = list(refs_by_part.values())
    if args.checkpoint_kernel_ref:
        complete_refs.append(args.checkpoint_kernel_ref)
    require_complete_kernels(context, complete_refs)
    if args.checkpoint_dataset_ref:
        _wait_for_dataset_visible(
            context.cli,
            args.checkpoint_dataset_ref,
            context.environment,
            expected_marker="self_all_parts.pt",
        )

    timestamp, run_dir = new_run_dir(context, "quality")
    source_ref = ensure_source_dataset(
        context,
        source_ref=args.source_dataset_ref,
        run_dir=run_dir,
        timestamp=timestamp,
        phase="quality",
    )
    kernel_slug = args.kernel_slug or f"genmusic-quality-{timestamp}"
    dataset_sources = [source_ref]
    if args.checkpoint_dataset_ref:
        dataset_sources.append(args.checkpoint_dataset_ref)
    kernel_sources = [refs_by_part[part] for part in sorted(refs_by_part)]
    if args.checkpoint_kernel_ref:
        kernel_sources.append(args.checkpoint_kernel_ref)

    submit_phase_kernel(
        context,
        phase="quality",
        run_dir=run_dir,
        kernel_slug=kernel_slug,
        code=_kernel_code(
            source_count=len(refs_by_part),
            expected_records=args.expected_records,
            evaluation_records=args.evaluation_records,
            duration=args.duration,
            steps=args.steps,
            guidance_scales=args.guidance_scales,
            pronunciation_prior_strengths=args.pronunciation_prior_strengths,
            native_prior_start_strength=args.native_prior_start_strength,
            conditioning_modes=modes,
            native_only=args.native_only,
        ),
        dataset_sources=dataset_sources,
        kernel_sources=kernel_sources,
        enable_gpu=True,
        enable_internet=True,
        accelerator=args.accelerator,
        timeout_seconds=args.session_timeout_seconds,
        state={
            "source_dataset_ref": source_ref,
            "processed_kernel_refs": refs_by_part,
            "checkpoint_kernel_ref": args.checkpoint_kernel_ref or None,
            "checkpoint_dataset_ref": args.checkpoint_dataset_ref or None,
            "expected_records": args.expected_records,
            "evaluation_records": args.evaluation_records,
            "duration": args.duration,
            "steps": args.steps,
            "guidance_scales": args.guidance_scales,
            "pronunciation_prior_strengths": args.pronunciation_prior_strengths,
            "native_prior_start_strength": args.native_prior_start_strength,
            "conditioning_modes": modes,
            "accelerator": args.accelerator,
            "session_timeout_seconds": args.session_timeout_seconds,
        },
    )


if __name__ == "__main__":
    main()
