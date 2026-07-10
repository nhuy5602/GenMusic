from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .evaluation import evaluate_dataset
from .quality_checks import evaluate_simulated_cases
from ..data.reference_dataset import generate_reference_eval_records, generate_reference_training_records
from .report_plots import generate_evaluation_plots, generate_self_improvement_plots
from ..data.synthetic_dataset import generate_synthetic_records, write_jsonl
from ..integrations.trained_text_model import DEFAULT_LOCAL_MODEL_PATH, train_text_model, write_text_model
from ..data.training_dataset import (
    GENRE_SCENES,
    generate_diverse_training_records,
    load_training_records,
    style_prompt_for_genre,
    write_training_jsonl,
)


EMOTION_TO_GENRE = {
    "joy": "edm",
    "sadness": "pop_ballad",
    "anger": "rock",
    "fear": "horror",
    "calm": "ambient",
    "romantic": "rnb",
    "hope": "trap",
    "nostalgic": "folk",
}


def run_self_improvement(
    *,
    iterations: int = 3,
    samples: int = 640,
    eval_count: int = 24,
    seed: int = 5602,
    output_root: str | Path = "outputs/self_improve",
    model_out: str | Path = DEFAULT_LOCAL_MODEL_PATH,
    extra_datasets: list[str | Path] | None = None,
    extra_dataset_max_records: int | None = 60000,
    duration_seconds: int = 30,
    render_audio: bool = False,
    stop_score: float = 0.88,
) -> dict[str, Any]:
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = Path(model_out)
    extra_records = load_training_records(
        extra_datasets or [],
        max_records=extra_dataset_max_records,
        seed=seed,
    )
    reference_train = generate_reference_training_records(max(16, eval_count // 2), seed=seed)
    reference_eval = generate_reference_eval_records(max(8, eval_count // 3), seed=seed)

    targeted_records: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_model: dict[str, Any] | None = None

    for iteration in range(1, max(1, iterations) + 1):
        iter_seed = seed + iteration - 1
        iter_dir = output_path / f"iteration_{iteration:02d}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        training_records = (
            generate_diverse_training_records(samples, seed=iter_seed)
            + reference_train
            + extra_records
            + targeted_records
        )
        training_data_path = write_training_jsonl(training_records, iter_dir / "training_data.jsonl")
        model, train_report = train_text_model(training_records, seed=iter_seed)
        iteration_model_path = write_text_model(model, iter_dir / "genmusic_text_model.json")
        saved_model_path = write_text_model(model, model_path)
        if saved_model_path.resolve() != DEFAULT_LOCAL_MODEL_PATH.resolve():
            write_text_model(model, DEFAULT_LOCAL_MODEL_PATH)

        eval_records = generate_synthetic_records(eval_count, seed=iter_seed) + reference_eval
        eval_dataset_path = write_jsonl(eval_records, iter_dir / "eval_dataset.jsonl")
        eval_report = evaluate_dataset(
            eval_dataset_path,
            output_root=iter_dir / "eval_runs",
            duration_seconds=duration_seconds,
        )
        eval_report_path = iter_dir / "evaluation_report.json"
        eval_report["report_path"] = str(eval_report_path)
        eval_report_path.write_text(json.dumps(eval_report, ensure_ascii=False, indent=2), encoding="utf-8")

        simulated_cases = _select_simulated_cases(eval_records, limit=min(12, len(eval_records)))
        quality_report = evaluate_simulated_cases(
            simulated_cases,
            output_root=iter_dir / "quality",
            duration_seconds=duration_seconds,
            render_audio=render_audio,
            audio_required=render_audio,
        )
        eval_report["plots"] = generate_evaluation_plots(
            eval_report,
            iter_dir / "plots",
            quality_report=quality_report,
            title_prefix=f"Vòng lặp {iteration}",
        )
        eval_report_path.write_text(json.dumps(eval_report, ensure_ascii=False, indent=2), encoding="utf-8")

        combined = _combined_summary(
            iteration=iteration,
            train_record_count=len(training_records),
            training_data_path=training_data_path,
            model_path=iteration_model_path,
            train_report=train_report,
            eval_report=eval_report,
            quality_report=quality_report,
        )
        new_targeted = _targeted_records_from_reports(
            eval_records,
            eval_report,
            quality_report,
            iteration=iteration,
            seed=iter_seed,
        )
        targeted_records = _dedupe_records(targeted_records + new_targeted)[-64:]
        write_training_jsonl(targeted_records, iter_dir / "targeted_next_records.jsonl")

        history.append(
            {
                "iteration": iteration,
                "summary": combined,
                "train_report": train_report,
                "evaluation_report_path": str(eval_report_path),
                "quality_report_path": quality_report["report_path"],
                "targeted_next_count": len(targeted_records),
            }
        )
        if best is None or combined["combined_score"] > best["summary"]["combined_score"]:
            best = history[-1]
            best_model = model
        if _can_stop(combined, stop_score=stop_score):
            break

    if best_model is not None:
        final_model_path = write_text_model(best_model, model_path)
        if final_model_path.resolve() != DEFAULT_LOCAL_MODEL_PATH.resolve():
            write_text_model(best_model, DEFAULT_LOCAL_MODEL_PATH)
        if best is not None:
            best["summary"]["selected_model_path"] = str(final_model_path)

    report = {
        "status": "complete",
        "iterations_requested": iterations,
        "iterations_run": len(history),
        "seed": seed,
        "model_path": str(model_path),
        "default_model_path": str(DEFAULT_LOCAL_MODEL_PATH),
        "output_root": str(output_path),
        "extra_dataset_record_count": len(extra_records),
        "extra_dataset_max_records": extra_dataset_max_records,
        "copyright_note": (
            "No copyrighted web lyrics are bundled. Use --extra-dataset only for local lyrics/data "
            "you have permission to use."
        ),
        "best_iteration": best,
        "history": history,
    }
    report["plots"] = generate_self_improvement_plots(report, output_path / "plots")
    report_path = output_path / "self_improve_report.json"
    markdown_path = output_path / "self_improve_report.md"
    report["report_path"] = str(report_path)
    report["markdown_path"] = str(markdown_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(_markdown_report(report), encoding="utf-8")
    return report


def _combined_summary(
    *,
    iteration: int,
    train_record_count: int,
    training_data_path: Path,
    model_path: Path,
    train_report: dict[str, Any],
    eval_report: dict[str, Any],
    quality_report: dict[str, Any],
) -> dict[str, Any]:
    text_model_score = _mean(
        [
            float(train_report.get("emotion_accuracy", 0.0)),
            float(train_report.get("genre_accuracy", 0.0)),
        ]
    )
    planning_score = float(eval_report.get("summary", {}).get("overall_score", 0.0))
    quality_score = float(quality_report.get("summary", {}).get("overall_quality_score", 0.0))
    combined_score = _mean([text_model_score, planning_score, quality_score])
    issue_counts: dict[str, int] = {}
    for item in quality_report.get("items", []):
        for issue in item.get("issues", []):
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
    return {
        "iteration": iteration,
        "combined_score": round(combined_score, 4),
        "text_model_score": round(text_model_score, 4),
        "planning_score": round(planning_score, 4),
        "quality_score": round(quality_score, 4),
        "emotion_accuracy": train_report.get("emotion_accuracy", 0.0),
        "genre_accuracy": train_report.get("genre_accuracy", 0.0),
        "overall_eval_score": planning_score,
        "overall_quality_score": quality_score,
        "train_record_count": train_record_count,
        "training_data_path": str(training_data_path),
        "model_path": str(model_path),
        "quality_issue_counts": dict(sorted(issue_counts.items())),
    }


def _targeted_records_from_reports(
    eval_records: list[dict[str, Any]],
    eval_report: dict[str, Any],
    quality_report: dict[str, Any],
    *,
    iteration: int,
    seed: int,
) -> list[dict[str, Any]]:
    by_id = {str(record.get("id")): record for record in eval_records}
    weak_ids: set[str] = set()
    for item in eval_report.get("items", []):
        metrics = item.get("metrics", {})
        if float(metrics.get("emotion_match", 1.0)) < 1.0 or float(metrics.get("keyword_recall", 1.0)) < 0.55:
            weak_ids.add(str(item.get("id")))
    for item in quality_report.get("items", []):
        if item.get("issues"):
            weak_ids.add(str(item.get("id")))

    records: list[dict[str, Any]] = []
    for offset, record_id in enumerate(sorted(weak_ids)):
        source = by_id.get(record_id)
        if not source:
            continue
        expected_emotions = source.get("expected_emotions") or []
        emotion = str(expected_emotions[0] if expected_emotions else source.get("emotion") or "calm")
        if emotion not in EMOTION_TO_GENRE:
            emotion = "calm"
        genre_label = str(source.get("genre_label") or EMOTION_TO_GENRE[emotion])
        if genre_label not in GENRE_SCENES:
            genre_label = EMOTION_TO_GENRE[emotion]
        keywords = list(source.get("expected_keywords") or GENRE_SCENES[genre_label]["keywords"])
        text = _targeted_text(source, genre_label=genre_label)
        records.append(
            {
                "id": f"self_improve_{iteration:02d}_{seed}_{offset:03d}_{record_id}",
                "input_text": text,
                "emotion": emotion,
                "genre_label": genre_label,
                "style_prompt": style_prompt_for_genre(genre_label),
                "expected_keywords": _dedupe_values(keywords + ["vocal rõ", "đủ lời", "có vần"]),
                "expected_vocal_gender": source.get("expected_vocal_gender", ""),
                "source": "self_improve_failure_case",
            }
        )
    return records[:24]


def _targeted_text(source: dict[str, Any], *, genre_label: str) -> str:
    style = GENRE_SCENES[genre_label]["style_prompt"]
    text = str(source.get("input_text") or source.get("text") or source.get("chorus") or "").strip()
    keywords = ", ".join(str(item) for item in list(source.get("expected_keywords") or [])[:6])
    return (
        f"{text} Yêu cầu cải thiện: lời phải đủ câu, có vần tiếng Việt, vocal rõ chữ không rè, "
        f"beat hợp mood, flow đúng style {genre_label}. Style tham chiếu: {style}. Từ khóa cần giữ: {keywords}."
    )


def _select_simulated_cases(records: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    if len(records) <= limit:
        return records
    selected: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    for record in records:
        source = str(record.get("source") or record.get("length_bucket") or "unknown")
        if source in seen_sources:
            continue
        selected.append(record)
        seen_sources.add(source)
        if len(selected) >= limit:
            return selected
    for record in records:
        if len(selected) >= limit:
            break
        if record not in selected:
            selected.append(record)
    return selected


def _can_stop(summary: dict[str, Any], *, stop_score: float) -> bool:
    if float(summary["combined_score"]) < stop_score:
        return False
    severe = {"lyrics_not_enough", "weak_rhyme", "beat_mood_mismatch"}
    issue_counts = summary.get("quality_issue_counts", {})
    return not any(issue_counts.get(issue, 0) for issue in severe)


def _dedupe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for record in records:
        key = str(record.get("input_text") or record.get("id"))
        if key in seen:
            continue
        seen.add(key)
        result.append(record)
    return result


def _dedupe_values(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = str(value).strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Báo cáo tự cải thiện GenMusic VN",
        "",
        f"- Trạng thái: {report['status']}",
        f"- Số vòng đã chạy: {report['iterations_run']} / {report['iterations_requested']}",
        f"- Đường dẫn model: `{report['model_path']}`",
        f"- Thư mục output: `{report['output_root']}`",
        f"- Dashboard plot: `{report.get('plots', {}).get('plot_data_path', '')}`",
        "",
        "## Điểm số",
        "",
        "| Vòng | Tổng hợp | Text model | Lập kế hoạch | Chất lượng | Đúng cảm xúc | Đúng thể loại | Vấn đề |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in report["history"]:
        summary = item["summary"]
        issues = ", ".join(f"{key}:{value}" for key, value in summary["quality_issue_counts"].items()) or "none"
        lines.append(
            "| {iteration} | {combined_score:.4f} | {text_model_score:.4f} | "
            "{planning_score:.4f} | {quality_score:.4f} | {emotion_accuracy:.4f} | "
            "{genre_accuracy:.4f} | {issues} |".format(**summary, issues=issues)
        )
    lines.extend(
        [
            "",
            "## Biểu đồ",
            "",
            *[
                f"- {name}: `{path}`"
                for name, path in report.get("plots", {}).get("files", {}).items()
            ],
            "",
            "## Ghi chú",
            "",
            "- Vòng lặp dùng train/evaluate local nên có thể chạy khi không còn quota Kaggle.",
            "- Vocal hát thật vẫn cần output TTS Kaggle; guide track local chỉ xác minh độ rõ nhạc nền.",
            f"- {report['copyright_note']}",
            "",
        ]
    )
    return "\n".join(lines)


def _mean(values: list[float | int]) -> float:
    if not values:
        return 0.0
    return sum(float(value) for value in values) / len(values)
