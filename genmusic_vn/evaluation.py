from __future__ import annotations

import json
import tempfile
import unicodedata
from pathlib import Path
from typing import Any

from .pipeline import create_music_project
from .schemas import MusicResult


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_DATASET = PROJECT_ROOT / "datasets" / "evaluation" / "vi_text_to_music_eval.jsonl"
ROMANIZED_TEMPLATE_PHRASES = {
    "ngay xua",
    "o lai",
    "binh yen",
    "mot lan",
    "cau hat",
    "duong ve",
    "trai tim",
    "anh den",
}


def load_eval_dataset(path: str | Path = DEFAULT_EVAL_DATASET) -> list[dict[str, Any]]:
    dataset_path = Path(path)
    records: list[dict[str, Any]] = []
    for line in dataset_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def evaluate_dataset(
    dataset_path: str | Path = DEFAULT_EVAL_DATASET,
    *,
    output_root: str | Path | None = None,
    duration_seconds: int = 12,
) -> dict[str, Any]:
    records = load_eval_dataset(dataset_path)
    if output_root is None:
        temp_dir = tempfile.TemporaryDirectory()
        output_path = Path(temp_dir.name)
    else:
        temp_dir = None
        output_path = Path(output_root)
        output_path.mkdir(parents=True, exist_ok=True)

    try:
        items = [
            evaluate_record(record, output_path, duration_seconds=duration_seconds)
            for record in records
        ]
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()

    summary = _aggregate(items)
    return {
        "dataset": str(Path(dataset_path)),
        "sample_count": len(items),
        "summary": summary,
        "by_length": _aggregate_by(items, "length_bucket"),
        "by_expected_emotion": _aggregate_by(items, "expected_emotion"),
        "items": items,
    }


def evaluate_record(record: dict[str, Any], output_root: Path, *, duration_seconds: int) -> dict[str, Any]:
    record_duration = int(record.get("duration_seconds") or duration_seconds)
    result = create_music_project(
        record["input_text"],
        output_root=output_root,
        duration_seconds=record_duration,
        genre=record.get("genre") or None,
        render_audio=False,
    )
    lyric_text = "\n".join(result.lyrics.full_song)
    lyric_lines = _content_lyric_lines(result)

    expected_emotions = set(record.get("expected_emotions") or [])
    expected_keywords = list(record.get("expected_keywords") or [])
    expected_phrases = list(record.get("expected_lyric_phrases") or [])
    expected_vocal_gender = record.get("expected_vocal_gender")

    keyword_hits = _match_count(expected_keywords, lyric_text)
    phrase_hits = _match_count(expected_phrases, lyric_text)
    prompt_keyword_hits = _match_count(expected_keywords, result.prompt)
    romanized_violations = sorted(
        phrase for phrase in ROMANIZED_TEMPLATE_PHRASES if phrase in lyric_text.lower()
    )

    metrics = {
        "emotion_match": int(not expected_emotions or result.emotion.label in expected_emotions),
        "keyword_recall": _ratio(keyword_hits, len(expected_keywords)),
        "prompt_keyword_recall": _ratio(prompt_keyword_hits, len(expected_keywords)),
        "phrase_recall": _ratio(phrase_hits, len(expected_phrases)),
        "scene_cue_density": _ratio(min(len(result.scene.prompt_cues), 4), 4),
        "no_title": int(result.lyrics.title == "" and not any(line.startswith("[Title]") for line in result.lyrics.full_song)),
        "diacritic_line_rate": _ratio(sum(1 for line in lyric_lines if _has_vietnamese_diacritic(line)), len(lyric_lines)),
        "romanized_violation_count": len(romanized_violations),
        "vocal_recommendation_match": int(not expected_vocal_gender or result.vocal.gender == expected_vocal_gender),
    }
    metrics["overall_score"] = _mean(
        [
            metrics["emotion_match"],
            metrics["keyword_recall"],
            metrics["prompt_keyword_recall"],
            metrics["phrase_recall"],
            metrics["scene_cue_density"],
            metrics["no_title"],
            metrics["diacritic_line_rate"],
            int(metrics["romanized_violation_count"] == 0),
            metrics["vocal_recommendation_match"],
        ]
    )

    return {
        "id": record.get("id", result.run_id),
        "length_bucket": record.get("length_bucket", "unknown"),
        "expected_emotion": sorted(expected_emotions)[0] if expected_emotions else "",
        "expected_mood_text": record.get("expected_mood_text", ""),
        "expected": {
            "emotions": sorted(expected_emotions),
            "keywords": expected_keywords,
            "phrases": expected_phrases,
            "vocal_gender": expected_vocal_gender,
            "duration_seconds": record_duration,
            "genre": record.get("genre", ""),
        },
        "predicted": {
            "run_id": result.run_id,
            "emotion": result.emotion.label,
            "vocal_gender": result.vocal.gender,
            "lyrics": result.lyrics.full_song,
            "prompt": result.prompt,
            "scene": result.scene.labels,
        },
        "metrics": metrics,
        "romanized_violations": romanized_violations,
    }


def _content_lyric_lines(result: MusicResult) -> list[str]:
    return [
        line.strip()
        for line in result.lyrics.full_song
        if line.strip() and not line.startswith("[")
    ]


def _match_count(expected: list[str], text: str) -> int:
    normalized_text = _strip_accents(text).lower()
    return sum(1 for item in expected if _strip_accents(item).lower() in normalized_text)


def _strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    return stripped.replace("đ", "d").replace("Đ", "D")


def _has_vietnamese_diacritic(text: str) -> bool:
    return _strip_accents(text) != text


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 1.0
    return round(numerator / denominator, 4)


def _mean(values: list[float | int]) -> float:
    if not values:
        return 0.0
    return round(sum(float(value) for value in values) / len(values), 4)


def _aggregate(items: list[dict[str, Any]]) -> dict[str, float]:
    metric_names = sorted({name for item in items for name in item["metrics"]})
    return {
        name: _mean([item["metrics"][name] for item in items])
        for name in metric_names
    }


def _aggregate_by(items: list[dict[str, Any]], key: str) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        grouped.setdefault(str(item.get(key) or "unknown"), []).append(item)
    return {group: _aggregate(group_items) for group, group_items in sorted(grouped.items())}
