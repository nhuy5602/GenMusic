from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Any

from .pipeline import create_music_project
from .rhyme import section_vietnamese_rhyme_rate, strip_accents
from .text_utils import extract_lyric_lines, tokenize_words


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHORUS_ABLATION_DATASET = PROJECT_ROOT / "datasets" / "evaluation" / "chorus_ablation_safe.jsonl"
GENERATED_PHRASE_GUARDS = {
    "ở lại thêm một lần",
    "bình yên nằm lại",
    "cho câu hát tìm thấy",
}


def load_chorus_ablation_dataset(path: str | Path = DEFAULT_CHORUS_ABLATION_DATASET) -> list[dict[str, Any]]:
    dataset_path = Path(path)
    records: list[dict[str, Any]] = []
    for line in dataset_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def evaluate_chorus_ablation_dataset(
    dataset_path: str | Path = DEFAULT_CHORUS_ABLATION_DATASET,
    *,
    output_root: str | Path | None = None,
    duration_seconds: int = 45,
) -> dict[str, Any]:
    records = load_chorus_ablation_dataset(dataset_path)
    if output_root is None:
        temp_dir = tempfile.TemporaryDirectory()
        output_path = Path(temp_dir.name)
    else:
        temp_dir = None
        output_path = Path(output_root)
        output_path.mkdir(parents=True, exist_ok=True)

    try:
        pairs = [
            evaluate_chorus_ablation_record(record, output_path, duration_seconds=duration_seconds)
            for record in records
        ]
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()

    cases = [case for pair in pairs for case in pair["cases"]]
    summary = _aggregate(cases)
    variant_summary = {
        variant: _aggregate([case for case in cases if case["variant"] == variant])
        for variant in ("no_style", "style_on")
    }
    pair_summary = {
        "avg_style_prompt_gain": _mean([pair["deltas"]["style_prompt_gain"] for pair in pairs]),
        "avg_harmony_style_gain": _mean([pair["deltas"]["harmony_style_gain"] for pair in pairs]),
        "avg_preservation_delta": _mean([pair["deltas"]["preservation_delta"] for pair in pairs]),
        "pair_count": len(pairs),
    }
    return {
        "dataset": str(Path(dataset_path)),
        "case_count": len(cases),
        "pair_count": len(pairs),
        "summary": summary,
        "variant_summary": variant_summary,
        "pair_summary": pair_summary,
        "pairs": pairs,
    }


def evaluate_chorus_ablation_record(record: dict[str, Any], output_root: Path, *, duration_seconds: int) -> dict[str, Any]:
    chorus = str(record["chorus"]).strip()
    duration = int(record.get("duration_seconds") or duration_seconds)
    expected_style = str(record.get("style") or "").strip()
    expected_terms = list(record.get("expected_style_terms") or _style_terms(expected_style))
    cases = [
        _evaluate_case(record, chorus, "", "no_style", expected_terms, output_root, duration),
        _evaluate_case(record, chorus, expected_style, "style_on", expected_terms, output_root, duration),
    ]
    by_variant = {case["variant"]: case for case in cases}
    no_style = by_variant["no_style"]["metrics"]
    style_on = by_variant["style_on"]["metrics"]
    return {
        "id": record.get("id", ""),
        "title_hint": record.get("title_hint", ""),
        "style": expected_style,
        "cases": cases,
        "deltas": {
            "style_prompt_gain": round(style_on["style_prompt_recall"] - no_style["style_prompt_recall"], 4),
            "harmony_style_gain": round(style_on["harmony_style_recall"] - no_style["harmony_style_recall"], 4),
            "preservation_delta": round(style_on["lyric_preservation"] - no_style["lyric_preservation"], 4),
        },
    }


def _evaluate_case(
    record: dict[str, Any],
    chorus: str,
    style: str,
    variant: str,
    expected_terms: list[str],
    output_root: Path,
    duration: int,
) -> dict[str, Any]:
    case_output_root = output_root / _safe_path_part(str(record.get("id") or "case")) / variant
    case_output_root.mkdir(parents=True, exist_ok=True)
    result = create_music_project(
        chorus,
        output_root=case_output_root,
        duration_seconds=duration,
        genre=style or None,
        render_audio=False,
    )
    lyric_lines = _content_lines(result.lyrics.full_song)
    lyric_sections = _content_sections(result.lyrics.full_song)
    prompt_blob = "\n".join(
        [
            result.prompt,
            " ".join(result.harmony.instruments),
            " ".join(result.harmony.music_traits),
            " ".join(result.scene.prompt_cues),
            " ".join(result.scene.arrangement_cues),
        ]
    )
    source_phrases = _source_phrases(chorus)
    phrase_hits = [phrase for phrase in source_phrases if _contains(prompt_blob + "\n" + "\n".join(lyric_lines), phrase)]
    style_prompt_hits = [term for term in expected_terms if _contains(result.prompt, term)]
    harmony_hits = [term for term in expected_terms if _contains(prompt_blob, term)]
    generated_violations = sorted(phrase for phrase in GENERATED_PHRASE_GUARDS if phrase in "\n".join(lyric_lines).lower())
    max_line_words = max((len(tokenize_words(line)) for line in lyric_lines), default=0)
    content_word_count = sum(len(tokenize_words(line)) for line in lyric_lines)
    input_kind_is_lyrics = result.text_plan.input_kind == "lyrics"
    metrics = {
        "input_kind_is_lyrics": int(input_kind_is_lyrics),
        "lyric_preservation": _ratio(len(phrase_hits), len(source_phrases)),
        "style_prompt_recall": _ratio(len(style_prompt_hits), len(expected_terms)),
        "harmony_style_recall": _ratio(len(harmony_hits), len(expected_terms)),
        "generated_phrase_violation_count": len(generated_violations),
        "singable_line_rate": _ratio(sum(1 for line in lyric_lines if 3 <= len(tokenize_words(line)) <= 15), len(lyric_lines)),
        "vietnamese_rhyme_rate": section_vietnamese_rhyme_rate(lyric_sections),
        "duration_pressure": round(content_word_count / max(1, duration), 3),
    }
    metrics["overall_score"] = _mean(
        [
            metrics["input_kind_is_lyrics"],
            metrics["lyric_preservation"],
            metrics["style_prompt_recall"] if style else 1.0,
            metrics["harmony_style_recall"] if style else 1.0,
            int(metrics["generated_phrase_violation_count"] == 0),
            metrics["singable_line_rate"],
            max(0.0, 1.0 - max(0.0, metrics["duration_pressure"] - 1.8) / 1.8),
        ]
    )
    return {
        "variant": variant,
        "style": style,
        "run_id": result.run_id,
        "output_root": str(case_output_root),
        "emotion": result.emotion.label,
        "key": result.harmony.key,
        "scale": result.harmony.scale,
        "bpm": result.harmony.bpm,
        "text_plan": {
            "input_kind": result.text_plan.input_kind,
            "mode": result.text_plan.mode,
            "line_count": result.text_plan.sentence_count,
        },
        "song_form": result.lyrics.song_form,
        "line_count": len(lyric_lines),
        "max_line_words": max_line_words,
        "content_word_count": content_word_count,
        "metrics": metrics,
        "hits": {
            "source_phrases": phrase_hits,
            "style_prompt": style_prompt_hits,
            "harmony_style": harmony_hits,
            "generated_violations": generated_violations,
        },
        "prompt_preview": result.prompt[:900],
        "lyrics_preview": result.lyrics.full_song[:12],
    }


def write_chorus_ablation_report(report: dict[str, Any], output_root: str | Path) -> tuple[Path, Path]:
    out = Path(output_root)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "chorus_ablation_report.json"
    md_path = out / "chorus_ablation_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Chorus Style Ablation Report", "", "## Summary", ""]
    for key, value in report["summary"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Variant Summary", ""])
    for variant, metrics in report.get("variant_summary", {}).items():
        lines.append(f"### {variant}")
        for key, value in metrics.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
    lines.extend(["", "## Pair Summary", ""])
    for key, value in report["pair_summary"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Cases",
            "",
            "| ID | Variant | Style | Emotion | BPM | Preserve | Style prompt | Harmony style | Violations | Overall |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for pair in report["pairs"]:
        for case in pair["cases"]:
            metrics = case["metrics"]
            lines.append(
                f"| {pair['id']} | {case['variant']} | {case['style'] or '(none)'} | {case['emotion']} | "
                f"{case['bpm']} | {metrics['lyric_preservation']:.2f} | "
                f"{metrics['style_prompt_recall']:.2f} | {metrics['harmony_style_recall']:.2f} | "
                f"{metrics['generated_phrase_violation_count']} | {metrics['overall_score']:.2f} |"
            )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def _source_phrases(text: str) -> list[str]:
    phrases: list[str] = []
    for line in extract_lyric_lines(text):
        chunks = re.split(r"[,;]|\s+(?:đến khi|dẫu|nếu|rồi|từ ngày|từ người)\s+", line, flags=re.IGNORECASE)
        for chunk in chunks:
            words = tokenize_words(chunk)
            if len(words) >= 3:
                phrases.append(" ".join(words[:8]))
    return _dedupe(phrases)


def _content_lines(full_song: list[str]) -> list[str]:
    return [line.strip() for line in full_song if line.strip() and not line.strip().startswith("[")]


def _content_sections(full_song: list[str]) -> list[list[str]]:
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


def _style_terms(style: str) -> list[str]:
    terms = []
    for chunk in re.split(r"[,;/|]+", style.lower().replace("_", " ").replace("-", " ")):
        cleaned = " ".join(chunk.split())
        if cleaned:
            terms.append(cleaned)
    return _dedupe(terms)


def _contains(text: str, phrase: str) -> bool:
    normalized_text = strip_accents(text).lower()
    normalized_phrase = strip_accents(phrase).lower()
    return normalized_phrase in normalized_text


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 1.0
    return round(numerator / denominator, 4)


def _mean(values: list[float | int]) -> float:
    if not values:
        return 0.0
    return round(sum(float(value) for value in values) / len(values), 4)


def _aggregate(cases: list[dict[str, Any]]) -> dict[str, float]:
    metric_names = sorted({name for case in cases for name in case["metrics"]})
    return {
        name: _mean([case["metrics"][name] for case in cases])
        for name in metric_names
    }


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = value.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


def _safe_path_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return cleaned.strip("-") or "case"
