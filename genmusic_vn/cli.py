from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .kaggle_auto import KaggleAutoError, KaggleJobConfig, run_or_stage_kaggle_job, sync_kaggle_artifact
from .pipeline import create_music_project
from .schemas import to_plain_data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="genmusic-vn", description="Vietnamese text-to-music pipeline.")
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="Analyze text and render a local guide track.")
    _add_common_args(generate)
    generate.add_argument("--backend", default="guide", choices=["guide", "musicgen", "stable-audio"])

    export = sub.add_parser("export-kaggle", help="Create report and prompt_pack.json for Kaggle GPU generation.")
    _add_common_args(export)

    analyze = sub.add_parser("analyze", help="Print pipeline analysis without rendering audio.")
    _add_common_args(analyze)

    kaggle_auto = sub.add_parser("kaggle-auto", help="Create a local guide track, submit a Kaggle GPU job, and optionally download output.")
    _add_common_args(kaggle_auto)
    kaggle_auto.add_argument("--kaggle-backend", default="musicgen", choices=["musicgen", "stable-audio-open"])
    kaggle_auto.add_argument("--model", default="facebook/musicgen-small")
    kaggle_auto.add_argument("--username", default=None, help="Kaggle username. Defaults to kaggle.json or KAGGLE_USERNAME.")
    kaggle_auto.add_argument("--machine-shape", default="NvidiaTeslaP100")
    kaggle_auto.add_argument("--no-submit", action="store_true", help="Only stage Kaggle files locally.")
    kaggle_auto.add_argument("--wait", action="store_true", help="Poll Kaggle until the kernel finishes, then download output.")
    kaggle_auto.add_argument("--poll-seconds", type=int, default=60)
    kaggle_auto.add_argument("--timeout-seconds", type=int, default=10_800)

    sync = sub.add_parser("sync-kaggle", help="Download a trained model/artifact from Kaggle to local storage.")
    sync.add_argument("--source", required=True, choices=["dataset", "kernel"], help="Download from a Kaggle dataset or kernel output.")
    sync.add_argument("--ref", required=True, help="Kaggle ref, e.g. username/dataset-slug or username/kernel-slug.")
    sync.add_argument("--out", default="models/current", help="Local target directory.")
    return parser


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--text", required=True, help="Vietnamese input text.")
    parser.add_argument("--duration", type=int, default=30, help="Target duration in seconds.")
    parser.add_argument("--out", default="outputs", help="Output directory.")
    parser.add_argument("--genre", default=None, help="Optional style/genre hint for the model prompt.")


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

    if args.command == "kaggle-auto":
        result = create_music_project(
            text=args.text,
            output_root=Path(args.out),
            backend="guide",
            duration_seconds=args.duration,
            genre=args.genre,
            render_audio=True,
        )
        job = run_or_stage_kaggle_job(
            result,
            output_root=Path(args.out),
            config=KaggleJobConfig(
                kaggle_backend=args.kaggle_backend,
                model=args.model,
                username=args.username,
                machine_shape=args.machine_shape,
                submit=not args.no_submit,
                wait=args.wait,
                poll_seconds=args.poll_seconds,
                timeout_seconds=args.timeout_seconds,
            ),
        )
        data = to_plain_data(result)
        data["kaggle_job"] = job
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0

    render_audio = args.command == "generate"
    backend = getattr(args, "backend", "guide")
    result = create_music_project(
        text=args.text,
        output_root=Path(args.out),
        backend=backend,
        duration_seconds=args.duration,
        genre=args.genre,
        render_audio=render_audio,
    )
    data = to_plain_data(result)
    print(json.dumps(data, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
