from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


PUBLIC_MUSIC_DATASET_REF = "sonlest/vietnamese-music-dataset-version3-part3"
PUBLIC_MUSIC_DATASET_LICENSE = "CC0: Public Domain"
SUPPORTED_AUDIO_SUFFIXES = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}


def build_custom_music_audio_manifest(
    input_root: str | Path,
    output_path: str | Path,
    *,
    max_files: int = 0,
    source_dataset: str = PUBLIC_MUSIC_DATASET_REF,
) -> dict[str, Any]:
    """Create captioned audio records without inventing lyric transcripts.

    The public dataset contains audio but no lyric labels. Captions therefore
    come from filenames and conservative Vietnamese music defaults. An ASR
    transcript can be attached later as an optional vocal add-on field.
    """
    root = Path(input_root)
    paths = [path for path in sorted(root.rglob("*")) if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES]
    if max_files > 0:
        paths = paths[:max_files]
    records: list[dict[str, Any]] = []
    for index, path in enumerate(paths, start=1):
        duration = probe_duration_seconds(path)
        records.append(
            {
                "id": f"music_{index:06d}",
                "audio": str(path.resolve()),
                "caption": caption_from_filename(path),
                "duration_seconds": duration,
                "source_dataset": source_dataset,
                "license": PUBLIC_MUSIC_DATASET_LICENSE,
                "lyrics_status": "not_provided",
                "record_type": "text_to_music_audio_conditioning",
            }
        )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {
        "path": str(destination),
        "count": len(records),
        "source_dataset": source_dataset,
        "license": PUBLIC_MUSIC_DATASET_LICENSE,
        "audio_root": str(root),
        "lyric_policy": "ASR is optional and is not required for text-to-music training.",
    }


def caption_from_filename(path: str | Path) -> str:
    stem = Path(path).stem
    cleaned = re.sub(r"[_\-.]+", " ", stem)
    cleaned = re.sub(r"\b\d{1,5}\b", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        cleaned = "bản nhạc Việt Nam"
    return f"Vietnamese music, {cleaned}, expressive melody, clean studio arrangement"


def probe_duration_seconds(path: str | Path) -> float | None:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return round(float(result.stdout.strip()), 4)
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None


def load_custom_music_audio_manifest(path: str | Path, *, max_records: int = 0) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if max_records > 0 and len(records) >= max_records:
                break
    return records
