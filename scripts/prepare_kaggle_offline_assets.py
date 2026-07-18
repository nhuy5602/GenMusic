"""Build the reusable offline model/dependency bundle for Kaggle training.

Some Kaggle accounts receive a GPU worker without working DNS even when the
kernel metadata enables internet.  Keep large pretrained files in one private
dataset and attach it to every training iteration instead of downloading them
inside each GPU session.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path


HF_MODELS = {
    "xphonebert": ("vinai/xphonebert-base", None),
    "charsiu_g2p": ("charsiu/g2p_multilingual_byT5_small_100", None),
    # Text2PhonemeSequence uses ByT5 only as a tokenizer. Downloading its
    # PyTorch/TF/Flax language-model weights would add over 1 GB that is never
    # opened by the pipeline.
    "byt5": (
        "google/byt5-small",
        ["config.json", "tokenizer_config.json", "special_tokens_map.json"],
    ),
    "vocos": ("charactr/vocos-mel-24khz", None),
}
PURE_PYTHON_PACKAGES = (
    "vocos==0.1.0",
    "encodec==0.1.1",
    "text2phonemesequence==0.1.4",
    "segments==2.4.0",
    "csvw",
    "isodate",
    "python-dateutil",
    "rfc3986<2",
    "uritemplate",
    "babel",
    "language-tags",
    "rdflib",
    "termcolor",
    "jsonschema",
    "openai-whisper==20250625",
    "more-itertools",
)


def _download(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    if not destination.is_file():
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(destination)
    return destination


def _download_whisper_small(destination_dir: Path) -> Path:
    import whisper

    url = whisper._MODELS["small"]
    destination = _download(url, destination_dir / "small.pt")
    expected_sha256 = url.rsplit("/", 2)[-2]
    hasher = hashlib.sha256()
    with destination.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    if digest != expected_sha256:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"Whisper small checksum mismatch: {digest} != {expected_sha256}")
    return destination


def _download_wheelhouse(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    uvx = shutil.which("uvx")
    if not uvx:
        raise RuntimeError("uvx is required to build the offline wheelhouse")
    subprocess.run(
        [
            uvx,
            "--from",
            "pip",
            "pip",
            "download",
            "--disable-pip-version-check",
            "--no-deps",
            "--dest",
            str(destination),
            *PURE_PYTHON_PACKAGES,
        ],
        check=True,
    )
    # tiktoken is compiled. Include wheels for both Python versions seen in
    # Kaggle images so pip can select the compatible one without network.
    for python_version, abi in (("311", "cp311"), ("312", "cp312")):
        subprocess.run(
            [
                uvx,
                "--from",
                "pip",
                "pip",
                "download",
                "--disable-pip-version-check",
                "--no-deps",
                "--only-binary=:all:",
                "--platform",
                "manylinux2014_x86_64",
                "--implementation",
                "cp",
                "--python-version",
                python_version,
                "--abi",
                abi,
                "--dest",
                str(destination),
                "tiktoken",
            ],
            check=True,
        )


def prepare_offline_assets(output_dir: str | Path, *, force: bool = False) -> Path:
    """Materialize Hub snapshots and return one uncompressed portable tar."""
    from huggingface_hub import snapshot_download

    root = Path(output_dir)
    bundle = root / "genmusic_offline_assets.tar"
    if bundle.is_file() and not force:
        return bundle

    staging = root / "staging"
    if force and staging.exists():
        shutil.rmtree(staging)
    models_dir = staging / "models"
    for local_name, (repo_id, allow_patterns) in HF_MODELS.items():
        destination = models_dir / local_name
        if not destination.is_dir() or not any(destination.iterdir()):
            snapshot_download(
                repo_id=repo_id,
                local_dir=destination,
                allow_patterns=allow_patterns,
            )

    _download_whisper_small(staging / "whisper")
    _download(
        "https://raw.githubusercontent.com/lingjzhu/CharsiuG2P/main/dicts/vie-c.tsv",
        staging / "vie-c.tsv",
    )
    _download_wheelhouse(staging / "wheelhouse")

    temporary = bundle.with_name(bundle.name + ".part")
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(temporary, "w") as archive:
        for child in sorted(staging.iterdir()):
            archive.add(child, arcname=child.name, recursive=True)
    temporary.replace(bundle)
    return bundle


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/kaggle_offline_assets/current")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(prepare_offline_assets(args.output, force=args.force))
