"""Generate the standalone Google Colab notebook without changing Kaggle guides."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.integrations.colab_auto import (
    DEFAULT_COLAB_NOTEBOOK_URL,
    write_colab_notebook,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="colab/GenMusic_Full_Training.ipynb",
    )
    parser.add_argument("--colab-url", default=DEFAULT_COLAB_NOTEBOOK_URL)
    parser.add_argument("--repo-ref", default="master")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--cache-data-on-drive",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--checkpoint-every-steps", type=int, default=25)
    args = parser.parse_args()
    report = write_colab_notebook(
        args.out,
        colab_url=args.colab_url,
        repo_ref=args.repo_ref,
        epochs=args.epochs,
        batch_size=args.batch_size,
        cache_data_on_drive=args.cache_data_on_drive,
        checkpoint_every_steps=args.checkpoint_every_steps,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
