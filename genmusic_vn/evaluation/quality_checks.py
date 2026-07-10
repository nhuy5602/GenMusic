from __future__ import annotations

import json
import math
import wave
from pathlib import Path
from typing import Any

from ..core.pipeline import create_music_project
from ..core.rhyme import (
    natural_rhyme_score,
    section_end_pair_rhyme_rate,
    section_assonance_rate,
    section_head_tail_rhyme_rate,
    section_luc_bat_rhyme_rate,
    section_vietnamese_rhyme_rate,
)
from ..core.schemas import MusicResult, to_plain_data
from ..core.text_utils import tokenize_words


EMOTION_BPM_RANGES = {
    "joy": (104, 134),
    "sadness": (60, 84),
    "anger": (112, 152),
    "fear": (50, 96),
    "calm": (62, 92),
    "romantic": (72, 104),
    "hope": (86, 122),
    "nostalgic": (66, 92),
}


def evaluate_music_result_quality(
    result: MusicResult,
    *,
    expected: dict[str, Any] | None = None,
    audio_required: bool = False,
) -> dict[str, Any]:
    expected = expected or {}
    sections = _content_lyric_sections(result.lyrics.full_song)
    lines = [line for section in sections for line in section]
    words = [word for line in lines for word in tokenize_words(line)]
    target_lines = _target_lyric_lines(result.duration_seconds, result.text_plan.input_kind)

    line_count_score = _clamp(len(lines) / max(1, target_lines))
    word_count_score = _clamp(len(words) / max(16, target_lines * 6))
    singable_line_rate = _singable_line_rate(lines)
    lyric_completeness = _mean([line_count_score, word_count_score, singable_line_rate])

    rhyme_pair_rate = section_end_pair_rhyme_rate(sections)
    rhyme_profile = {
        "pair": rhyme_pair_rate,
        "head_tail": section_head_tail_rhyme_rate(sections),
        "luc_bat": section_luc_bat_rhyme_rate(sections),
        "vietnamese": section_vietnamese_rhyme_rate(sections),
    }
    rhyme_score = natural_rhyme_score(lines)

    beat_mood_score, beat_note = _beat_mood_score(result.emotion.label, result.harmony.bpm)
    vocal_score, vocal_note = _vocal_score(result, audio_required=audio_required)
    clarity = _audio_clarity_summary(result)
    flow_style_score = _flow_style_score(result, singable_line_rate)
    keyword_score = _keyword_score(expected, result)

    metrics = {
        "lyric_completeness": round(lyric_completeness, 4),
        "lyric_line_count": len(lines),
        "target_lyric_lines": target_lines,
        "singable_line_rate": round(singable_line_rate, 4),
        "rhyme_score": round(rhyme_score, 4),
        "rhyme_pair_rate": round(rhyme_pair_rate, 4),
        "rhyme_assonance_rate": round(section_assonance_rate(sections), 4),
        "beat_mood_score": round(beat_mood_score, 4),
        "vocal_presence_score": round(vocal_score, 4),
        "audio_clarity_score": round(clarity["score"], 4),
        "flow_style_score": round(flow_style_score, 4),
        "keyword_recall": round(keyword_score, 4),
    }
    metrics["overall_quality_score"] = round(
        _mean(
            [
                metrics["lyric_completeness"],
                metrics["rhyme_score"],
                metrics["beat_mood_score"],
                metrics["vocal_presence_score"],
                metrics["audio_clarity_score"],
                metrics["flow_style_score"],
                metrics["keyword_recall"],
            ]
        ),
        4,
    )
    supplied_rating = expected.get("user_rating")
    try:
        supplied_rating = float(supplied_rating)
    except (TypeError, ValueError):
        supplied_rating = None
    if supplied_rating is not None and 1.0 <= supplied_rating <= 5.0:
        user_rating = round(supplied_rating, 2)
        user_rating_source = "user_supplied"
    else:
        user_rating = _quality_score_to_rating(metrics["overall_quality_score"])
        user_rating_source = "objective_quality_proxy"

    issues = _quality_issues(metrics, audio_required=audio_required)
    return {
        "id": expected.get("id") or result.run_id,
        "run_id": result.run_id,
        "input_kind": result.text_plan.input_kind,
        "emotion": result.emotion.label,
        "bpm": result.harmony.bpm,
        "vocal": to_plain_data(result.vocal),
        "files": [to_plain_data(file) for file in result.files],
        "metrics": metrics,
        "user_rating": user_rating,
        "user_rating_source": user_rating_source,
        "rhyme_profile": {key: round(value, 4) for key, value in rhyme_profile.items()},
        "audio_clarity": clarity,
        "notes": {
            "beat": beat_note,
            "vocal": vocal_note,
        },
        "issues": issues,
        "rating_note": (
            "Đây là rating proxy khách quan đã hiệu chỉnh; nên thu thập rating thật trước khi quyết định cho sản phẩm."
            if user_rating_source == "objective_quality_proxy"
            else "Rating do người dùng cung cấp."
        ),
    }


def evaluate_simulated_cases(
    cases: list[dict[str, Any]],
    *,
    output_root: str | Path,
    duration_seconds: int = 30,
    render_audio: bool = False,
    audio_required: bool = False,
) -> dict[str, Any]:
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        text = str(case.get("input_text") or case.get("text") or case.get("chorus") or "").strip()
        if not text:
            continue
        result = create_music_project(
            text,
            output_root=output_path / "runs",
            duration_seconds=int(case.get("duration_seconds") or duration_seconds),
            genre=case.get("genre") or case.get("style_prompt") or None,
            render_audio=render_audio,
        )
        expected = dict(case)
        expected.setdefault("id", f"sim_{index:03d}")
        items.append(
            evaluate_music_result_quality(
                result,
                expected=expected,
                audio_required=audio_required or render_audio,
            )
        )
    report = {
        "case_count": len(items),
        "summary": _aggregate_metrics(items),
        "user_rating": {
            "mean": round(_mean([float(item.get("user_rating", 0.0)) for item in items]), 2),
            "median": _median_rating(items),
            "min": _min_rating(items),
            "max": _max_rating(items),
            "distribution": _rating_distribution(items),
            "source": "user_supplied_or_objective_quality_proxy",
        },
        "items": items,
    }
    report_path = output_path / "quality_report.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _content_lyric_sections(full_song: list[str]) -> list[list[str]]:
    sections: list[list[str]] = []
    current: list[str] = []
    for raw_line in full_song:
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("["):
            if current:
                sections.append(current)
                current = []
            continue
        current.append(line)
    if current:
        sections.append(current)
    return sections


def _target_lyric_lines(duration_seconds: int, input_kind: str) -> int:
    if input_kind == "lyrics":
        return max(4, min(12, math.ceil(duration_seconds / 6)))
    if duration_seconds <= 16:
        return 4
    if duration_seconds <= 30:
        return 6
    if duration_seconds <= 60:
        return 10
    return 12


def _singable_line_rate(lines: list[str]) -> float:
    if not lines:
        return 0.0
    hits = sum(1 for line in lines if 4 <= len(tokenize_words(line)) <= 12)
    return hits / len(lines)


def _beat_mood_score(emotion: str, bpm: int) -> tuple[float, str]:
    low, high = EMOTION_BPM_RANGES.get(emotion, (64, 128))
    if low <= bpm <= high:
        return 1.0, f"BPM {bpm} nằm trong khoảng dự kiến của cảm xúc {emotion}: {low}-{high}."
    distance = low - bpm if bpm < low else bpm - high
    score = _clamp(1.0 - distance / 36.0)
    return score, f"BPM {bpm} lệch {distance} BPM so với khoảng dự kiến của cảm xúc {emotion}: {low}-{high}."


def _vocal_score(result: MusicResult, *, audio_required: bool) -> tuple[float, str]:
    has_vocal_plan = bool(result.vocal.gender and result.vocal.delivery and "vocal plan:" in result.prompt)
    has_vocal_file = any(
        file.kind == "vocal" or "vocal" in file.description.lower()
        for file in result.files
    )
    prompt_demands_clear_vocal = "garbled" in result.negative_prompt.lower() and "wrong-language vocals" in result.negative_prompt.lower()
    if has_vocal_file:
        return 1.0, "Output đã render có artifact vocal."
    if audio_required:
        planned = 0.35 if has_vocal_plan else 0.0
        return planned, "Audio đã render local nhưng không có artifact vocal; vẫn cần TTS Kaggle để có giọng hát thật."
    if has_vocal_plan and prompt_demands_clear_vocal:
        return 0.85, "Vocal đã được lập kế hoạch và ràng buộc trong prompt; lượt này chưa render TTS thật."
    if has_vocal_plan:
        return 0.65, "Vocal đã được lập kế hoạch nhưng ràng buộc trong prompt còn yếu."
    return 0.0, "Không tìm thấy kế hoạch vocal."


def _audio_clarity_summary(result: MusicResult) -> dict[str, Any]:
    wav_files = [
        Path(file.path)
        for file in result.files
        if Path(file.path).suffix.lower() == ".wav" and Path(file.path).exists()
    ]
    if not wav_files:
        return {
            "score": 0.85,
            "status": "not_rendered",
            "note": "Không có artifact WAV; độ rõ chỉ được xem là dự kiến, chưa được xác minh.",
        }
    analyses = [_analyze_wav(path) for path in wav_files]
    score = _mean([item["score"] for item in analyses])
    return {
        "score": round(score, 4),
        "status": "verified_wav",
        "files": analyses,
    }


def _analyze_wav(path: Path) -> dict[str, Any]:
    try:
        with wave.open(str(path), "rb") as wav:
            width = wav.getsampwidth()
            channels = wav.getnchannels()
            frame_count = wav.getnframes()
            frames = wav.readframes(frame_count)
    except (OSError, wave.Error) as exc:
        return {"path": str(path), "score": 0.0, "error": str(exc)}

    if width != 2 or not frames:
        return {"path": str(path), "score": 0.4, "note": f"Unsupported WAV width {width}."}
    sample_count = len(frames) // 2
    values = [
        int.from_bytes(frames[index : index + 2], "little", signed=True)
        for index in range(0, len(frames), 2)
    ]
    peak = max(abs(value) for value in values) if values else 0
    rms = math.sqrt(sum(value * value for value in values) / max(1, len(values))) / 32768.0
    clipped = sum(1 for value in values if abs(value) >= 32600) / max(1, len(values))
    too_quiet = rms < 0.015
    score = 1.0
    score -= min(0.7, clipped * 80.0)
    if too_quiet:
        score -= 0.25
    if peak == 0:
        score = 0.0
    return {
        "path": str(path),
        "score": round(_clamp(score), 4),
        "channels": channels,
        "samples": sample_count,
        "peak": peak,
        "rms": round(rms, 5),
        "clipped_sample_rate": round(clipped, 6),
    }


def _flow_style_score(result: MusicResult, singable_line_rate: float) -> float:
    prompt = result.prompt.lower()
    has_song_form = "song form:" in prompt
    has_melody = "singer-ready melody" in prompt
    has_instruments = bool(result.harmony.instruments)
    has_traits = bool(result.harmony.music_traits)
    arrangement_score = _mean([has_song_form, has_melody, has_instruments, has_traits])
    return _mean([singable_line_rate, arrangement_score])


def _keyword_score(expected: dict[str, Any], result: MusicResult) -> float:
    keywords = list(expected.get("expected_keywords") or [])
    if not keywords:
        return 1.0
    text = "\n".join(result.lyrics.full_song).lower()
    hits = sum(1 for keyword in keywords if str(keyword).lower() in text or str(keyword).lower() in result.prompt.lower())
    return hits / len(keywords)


def _quality_issues(metrics: dict[str, float | int], *, audio_required: bool) -> list[str]:
    issues: list[str] = []
    if float(metrics["lyric_completeness"]) < 0.72:
        issues.append("lyrics_not_enough")
    if float(metrics["rhyme_score"]) < 0.5:
        issues.append("weak_rhyme")
    if float(metrics["beat_mood_score"]) < 0.72:
        issues.append("beat_mood_mismatch")
    if float(metrics["vocal_presence_score"]) < (0.7 if audio_required else 0.6):
        issues.append("vocal_missing_or_unverified")
    if float(metrics["audio_clarity_score"]) < 0.7:
        issues.append("audio_unclear_or_clipping")
    if float(metrics["flow_style_score"]) < 0.72:
        issues.append("flow_style_mismatch")
    return issues


def _aggregate_metrics(items: list[dict[str, Any]]) -> dict[str, float]:
    metric_names = sorted({name for item in items for name in item["metrics"]})
    return {
        name: round(_mean([float(item["metrics"][name]) for item in items]), 4)
        for name in metric_names
    }


def _mean(values: list[float | int | bool]) -> float:
    if not values:
        return 0.0
    return sum(float(value) for value in values) / len(values)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _quality_score_to_rating(score: float) -> float:
    """Calibrate objective quality so uncertainty and weak outputs are visible in the plot."""
    calibrated = _clamp((float(score) - 0.35) / 0.65)
    return round(1.0 + 4.0 * calibrated, 2)


def _rating_values(items: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for item in items:
        try:
            values.append(float(item.get("user_rating", 0.0)))
        except (TypeError, ValueError):
            continue
    return values


def _median_rating(items: list[dict[str, Any]]) -> float:
    values = sorted(_rating_values(items))
    if not values:
        return 0.0
    middle = len(values) // 2
    if len(values) % 2:
        return round(values[middle], 2)
    return round((values[middle - 1] + values[middle]) / 2.0, 2)


def _min_rating(items: list[dict[str, Any]]) -> float:
    values = _rating_values(items)
    return round(min(values), 2) if values else 0.0


def _max_rating(items: list[dict[str, Any]]) -> float:
    values = _rating_values(items)
    return round(max(values), 2) if values else 0.0


def _rating_distribution(items: list[dict[str, Any]]) -> dict[str, int]:
    distribution = {str(rating): 0 for rating in range(1, 6)}
    for value in _rating_values(items):
        bucket = min(5, max(1, int(value + 0.5)))
        distribution[str(bucket)] += 1
    return distribution
