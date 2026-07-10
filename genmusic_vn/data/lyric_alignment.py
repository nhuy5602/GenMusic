"""Sentence-level lyric alignment and LRC serialization."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Sequence

from .vietnamese_text import lyric_content_lines, normalize_vietnamese_lyrics


class AlignmentUnavailable(RuntimeError):
    """Raised when no real ASR/alignment backend is configured."""


@dataclass(frozen=True)
class AlignedLine:
    text: str
    start: float
    end: float
    source: str = "asr-segment"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _tokens(value: str) -> list[str]:
    return re.findall(r"[0-9A-Za-zÀ-ỹĐđ]+", normalize_vietnamese_lyrics(value).casefold())


def _lrc_timestamp(seconds: float) -> str:
    minutes = max(0, int(seconds // 60))
    remainder = max(0.0, seconds - minutes * 60)
    return f"[{minutes:02d}:{remainder:05.2f}]"


def write_lrc(lines: Sequence[AlignedLine], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(f"{_lrc_timestamp(line.start)} {line.text}" for line in lines) + "\n"
    destination.write_text(content, encoding="utf-8")
    return destination


def read_lrc(path: str | Path) -> list[AlignedLine]:
    lines: list[AlignedLine] = []
    pattern = re.compile(r"^\[(\d+):(\d{2}(?:\.\d+)?)\]\s*(.*)$")
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        match = pattern.match(raw.strip())
        if not match:
            continue
        start = int(match.group(1)) * 60 + float(match.group(2))
        lines.append(AlignedLine(text=match.group(3).strip(), start=start, end=start))
    for index, line in enumerate(lines[:-1]):
        lines[index] = AlignedLine(line.text, line.start, lines[index + 1].start, "lrc")
    if lines:
        last = lines[-1]
        lines[-1] = AlignedLine(last.text, last.start, last.start, "lrc")
    return lines


def align_lyrics_to_segments(
    lyrics: str,
    segments: Iterable[dict[str, Any]],
) -> list[AlignedLine]:
    """Map lyric lines to timestamped ASR segments without inventing time.

    The matcher is intentionally conservative: it chooses the best remaining
    segment span based on token similarity and keeps unmatched lines marked by
    a zero-length interval for manual review.
    """

    lyric_lines = lyric_content_lines(lyrics)
    asr_segments = [
        {
            "start": float(item["start"]),
            "end": float(item["end"]),
            "text": str(item.get("text", "")),
        }
        for item in segments
        if float(item.get("end", 0.0)) >= float(item.get("start", 0.0))
    ]
    result: list[AlignedLine] = []
    cursor = 0
    for lyric in lyric_lines:
        target = " ".join(_tokens(lyric))
        best_index: int | None = None
        best_score = 0.0
        for index in range(cursor, len(asr_segments)):
            candidate = " ".join(_tokens(asr_segments[index]["text"]))
            score = SequenceMatcher(None, target, candidate).ratio()
            if score > best_score:
                best_score = score
                best_index = index
            if score >= 0.96:
                break
        if best_index is None or best_score < 0.18:
            result.append(AlignedLine(lyric, 0.0, 0.0, "unmatched"))
            continue
        segment = asr_segments[best_index]
        result.append(AlignedLine(lyric, segment["start"], segment["end"], "asr-segment"))
        cursor = best_index + 1
    return result


def heuristic_alignment(lyrics: str, duration_seconds: float) -> list[AlignedLine]:
    """Create an explicitly labelled rough alignment for pipeline smoke tests."""

    content = lyric_content_lines(lyrics)
    if not content:
        return []
    span = max(0.1, float(duration_seconds)) / len(content)
    return [
        AlignedLine(text=line, start=index * span, end=(index + 1) * span, source="heuristic")
        for index, line in enumerate(content)
    ]


def load_segments(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if source.suffix.casefold() == ".jsonl":
        return [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    value = json.loads(source.read_text(encoding="utf-8"))
    return value if isinstance(value, list) else value["segments"]


def align_wav_to_lyrics(
    audio_path: str | Path,
    lyrics: str,
    *,
    segments: Iterable[dict[str, Any]] | None = None,
    asr_model: str | None = None,
    allow_heuristic: bool = False,
) -> list[AlignedLine]:
    """Align a WAV with lyrics using supplied ASR segments or faster-whisper.

    A model name is required for automatic ASR so a download cannot happen by
    accident. The heuristic path is available only when explicitly enabled and
    is always labelled in the resulting LRC metadata.
    """

    if segments is not None:
        return align_lyrics_to_segments(lyrics, segments)
    if asr_model:
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except ImportError as exc:
            raise AlignmentUnavailable("Cần cài faster-whisper để căn chỉnh bằng ASR.") from exc
        model = WhisperModel(asr_model)
        detected, _ = model.transcribe(str(audio_path), language="vi", word_timestamps=False)
        return align_lyrics_to_segments(lyrics, ({"start": item.start, "end": item.end, "text": item.text} for item in detected))
    if allow_heuristic:
        try:
            import soundfile as sf  # type: ignore

            duration = float(sf.info(str(audio_path)).duration)
        except Exception:
            try:
                import librosa  # type: ignore

                duration = float(librosa.get_duration(path=str(audio_path)))
            except Exception as exc:
                raise AlignmentUnavailable("Cần soundfile hoặc librosa để đọc thời lượng audio.") from exc
        return heuristic_alignment(lyrics, duration)
    raise AlignmentUnavailable("Thiếu ASR segments hoặc --asr-model; không tạo timestamp giả.")
