from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .kaggle_auto import KaggleAutoError, KaggleJobConfig, submit_text_to_music_job, sync_kaggle_artifact


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
