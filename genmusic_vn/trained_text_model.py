from __future__ import annotations

import json
import math
import random
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .text_utils import extract_keywords, normalize_text, tokenize_words
from .training_dataset import GENRE_SCENES, style_prompt_for_genre


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_MODEL_PATH = PROJECT_ROOT / "models" / "current" / "genmusic_text_model.json"
DEFAULT_BOOTSTRAP_MODEL_PATH = PROJECT_ROOT / "datasets" / "trained_models" / "genmusic_text_model.json"
MODEL_CANDIDATES = [DEFAULT_LOCAL_MODEL_PATH, DEFAULT_BOOTSTRAP_MODEL_PATH]
MODEL_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TextModelPrediction:
    emotion: str
    emotion_confidence: float
    emotion_scores: dict[str, float]
    genre_label: str
    genre_confidence: float
    genre_scores: dict[str, float]
    style_prompt: str
    model_path: str
    model_version: str


_MODEL_CACHE: dict[str, dict[str, Any]] = {}


def train_text_model(
    records: list[dict[str, Any]],
    *,
    seed: int = 42,
    holdout_ratio: float = 0.2,
    alpha: float = 0.6,
) -> tuple[dict[str, Any], dict[str, Any]]:
    usable = [record for record in records if _record_text(record) and record.get("emotion") and record.get("genre_label")]
    if len(usable) < 8:
        raise ValueError("Need at least 8 labeled training records.")

    rng = random.Random(seed)
    shuffled = list(usable)
    rng.shuffle(shuffled)
    holdout_size = max(1, int(round(len(shuffled) * holdout_ratio)))
    holdout = shuffled[:holdout_size]
    train = shuffled[holdout_size:] or shuffled

    holdout_model = _fit_model(train, seed=seed, alpha=alpha)
    report = evaluate_text_model(holdout_model, holdout)
    final_model = _fit_model(usable, seed=seed, alpha=alpha)
    final_model["training_report"] = report
    final_model["trained_on"] = {
        "record_count": len(usable),
        "holdout_count": len(holdout),
        "seed": seed,
        "alpha": alpha,
        "created_at": _now(),
        "sources": sorted({str(record.get("source") or "unknown") for record in usable}),
    }
    return final_model, report


def write_text_model(model: dict[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    _MODEL_CACHE.pop(str(output_path.resolve()), None)
    return output_path


def load_text_model(path: str | Path | None = None) -> tuple[dict[str, Any], Path] | tuple[None, None]:
    candidates = [Path(path)] if path else MODEL_CANDIDATES
    for candidate in candidates:
        if not candidate.exists():
            continue
        resolved = candidate.resolve()
        cached = _MODEL_CACHE.get(str(resolved))
        if cached is not None:
            return cached, resolved
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("schema_version") == MODEL_SCHEMA_VERSION and "classifiers" in data:
            _MODEL_CACHE[str(resolved)] = data
            return data, resolved
    return None, None


def predict_text_model(text: str, path: str | Path | None = None) -> TextModelPrediction | None:
    model, model_path = load_text_model(path)
    if model is None or model_path is None:
        return None
    features = _extract_features(text)
    emotion_scores = _predict_classifier(model["classifiers"]["emotion"], features)
    genre_scores = _predict_classifier(model["classifiers"]["genre"], features)
    emotion = max(emotion_scores.items(), key=lambda item: item[1])[0]
    raw_genre_label = max(genre_scores.items(), key=lambda item: item[1])[0]
    genre_label = _refine_genre_label(text, raw_genre_label)
    if genre_label != raw_genre_label:
        genre_scores = {label: score * 0.2 for label, score in genre_scores.items()}
        genre_scores[genre_label] = 0.82
        genre_scores = _normalize_probabilities(genre_scores)
    style_prompt = str(model.get("style_prompts", {}).get(genre_label) or style_prompt_for_genre(genre_label))
    return TextModelPrediction(
        emotion=emotion,
        emotion_confidence=round(emotion_scores[emotion], 4),
        emotion_scores={key: round(value, 4) for key, value in sorted(emotion_scores.items())},
        genre_label=genre_label,
        genre_confidence=round(genre_scores[genre_label], 4),
        genre_scores={key: round(value, 4) for key, value in sorted(genre_scores.items())},
        style_prompt=style_prompt,
        model_path=str(model_path),
        model_version=str(model.get("model_version") or "trained-text-model"),
    )


def trained_model_status(path: str | Path | None = None) -> dict[str, Any]:
    model, model_path = load_text_model(path)
    if model is None or model_path is None:
        return {
            "available": False,
            "checked_paths": [str(path)] if path else [str(candidate) for candidate in MODEL_CANDIDATES],
        }
    return {
        "available": True,
        "path": str(model_path),
        "model_version": model.get("model_version", ""),
        "trained_on": model.get("trained_on", {}),
        "training_report": model.get("training_report", {}),
        "labels": {
            "emotion": model.get("classifiers", {}).get("emotion", {}).get("labels", []),
            "genre": model.get("classifiers", {}).get("genre", {}).get("labels", []),
        },
    }


def evaluate_text_model(model: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"sample_count": 0, "emotion_accuracy": 0.0, "genre_accuracy": 0.0}
    emotion_hits = 0
    genre_hits = 0
    by_label: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "hit": 0})
    for record in records:
        features = _extract_features(_record_text(record))
        emotion_scores = _predict_classifier(model["classifiers"]["emotion"], features)
        genre_scores = _predict_classifier(model["classifiers"]["genre"], features)
        emotion = max(emotion_scores.items(), key=lambda item: item[1])[0]
        genre = max(genre_scores.items(), key=lambda item: item[1])[0]
        expected_emotion = str(record.get("emotion") or "")
        expected_genre = str(record.get("genre_label") or "")
        emotion_hits += int(emotion == expected_emotion)
        genre_hits += int(genre == expected_genre)
        by_label[expected_emotion]["total"] += 1
        by_label[expected_emotion]["hit"] += int(emotion == expected_emotion)
    return {
        "sample_count": len(records),
        "emotion_accuracy": round(emotion_hits / len(records), 4),
        "genre_accuracy": round(genre_hits / len(records), 4),
        "emotion_by_label": {
            label: round(values["hit"] / max(1, values["total"]), 4)
            for label, values in sorted(by_label.items())
        },
    }


def _fit_model(records: list[dict[str, Any]], *, seed: int, alpha: float) -> dict[str, Any]:
    emotion_classifier = _fit_classifier(records, "emotion", alpha=alpha)
    genre_classifier = _fit_classifier(records, "genre_label", alpha=alpha)
    style_prompts = {
        label: style_prompt_for_genre(label)
        for label in genre_classifier["labels"]
        if label in GENRE_SCENES
    }
    return {
        "schema_version": MODEL_SCHEMA_VERSION,
        "model_type": "multinomial_naive_bayes",
        "model_version": f"genmusic-text-nb-{_now_compact()}",
        "feature_extractor": "word_unigram_bigram_accentless_v1",
        "seed": seed,
        "classifiers": {
            "emotion": emotion_classifier,
            "genre": genre_classifier,
        },
        "style_prompts": style_prompts,
        "fallback_keywords": extract_keywords(" ".join(_record_text(record) for record in records))[:64],
    }


def _fit_classifier(records: list[dict[str, Any]], target_key: str, *, alpha: float) -> dict[str, Any]:
    labels = sorted({str(record[target_key]) for record in records if record.get(target_key)})
    vocabulary: set[str] = set()
    label_doc_counts: Counter[str] = Counter()
    label_feature_counts: dict[str, Counter[str]] = {label: Counter() for label in labels}
    label_total_features: Counter[str] = Counter()

    for record in records:
        label = str(record.get(target_key) or "")
        if label not in label_feature_counts:
            continue
        features = _extract_features(_record_text(record))
        label_doc_counts[label] += 1
        label_feature_counts[label].update(features)
        label_total_features[label] += sum(features.values())
        vocabulary.update(features)

    vocab_size = max(1, len(vocabulary))
    total_docs = sum(label_doc_counts.values())
    label_count = max(1, len(labels))
    classifier = {
        "target": target_key,
        "labels": labels,
        "alpha": alpha,
        "vocabulary_size": vocab_size,
        "priors": {},
        "default_log_probs": {},
        "feature_log_probs": {},
    }
    for label in labels:
        classifier["priors"][label] = math.log((label_doc_counts[label] + alpha) / (total_docs + alpha * label_count))
        denominator = label_total_features[label] + alpha * (vocab_size + 1)
        classifier["default_log_probs"][label] = math.log(alpha / denominator)
        classifier["feature_log_probs"][label] = {
            feature: round(math.log((count + alpha) / denominator), 8)
            for feature, count in sorted(label_feature_counts[label].items())
        }
    return classifier


def _predict_classifier(classifier: dict[str, Any], features: Counter[str]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for label in classifier["labels"]:
        total = float(classifier["priors"][label])
        feature_log_probs = classifier["feature_log_probs"].get(label, {})
        default = float(classifier["default_log_probs"][label])
        for feature, count in features.items():
            total += count * float(feature_log_probs.get(feature, default))
        scores[label] = total
    return _softmax(scores)


def _extract_features(text: str) -> Counter[str]:
    normalized = normalize_text(text).lower()
    tokens = tokenize_words(normalized)
    features: Counter[str] = Counter()
    for token in tokens:
        features[f"w:{token}"] += 1
        accentless = _strip_accents(token)
        if accentless != token:
            features[f"a:{accentless}"] += 1
    for left, right in zip(tokens, tokens[1:]):
        features[f"b:{left}_{right}"] += 1
        left_a = _strip_accents(left)
        right_a = _strip_accents(right)
        features[f"ba:{left_a}_{right_a}"] += 1
    for marker in ("r&b", "lo-fi", "edm", "trap", "bolero", "horror", "ambient", "orchestral"):
        if marker in normalized:
            features[f"m:{marker}"] += 2
    return features


def _softmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    max_score = max(scores.values())
    exps = {label: math.exp(score - max_score) for label, score in scores.items()}
    total = sum(exps.values()) or 1.0
    return {label: value / total for label, value in exps.items()}


def _normalize_probabilities(scores: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, value) for value in scores.values()) or 1.0
    return {label: max(0.0, value) / total for label, value in scores.items()}


def _refine_genre_label(text: str, predicted: str) -> str:
    normalized = normalize_text(text).lower()
    accentless = _strip_accents(normalized)
    has_rain_memory = any(marker in normalized for marker in ("mưa", "chiều mưa", "mưa rơi"))
    has_old_street = any(marker in normalized for marker in ("phố cũ", "phố quen", "ngày xưa", "lời hứa", "lời chưa nói", "nhớ"))
    if has_rain_memory and has_old_street:
        return "pop_ballad"
    if any(marker in accentless for marker in ("que nha", "huong lua", "dan bau", "sao truc", "bo tre")):
        return "folk"
    if any(marker in normalized for marker in ("808", "hi-hat", "rap", "trap")):
        return "trap"
    if any(marker in normalized for marker in ("edm", "drop", "synth", "lễ hội")):
        return "edm"
    if any(marker in normalized for marker in ("guitar điện", "trống live", "rock")):
        return "rock"
    if any(marker in normalized for marker in ("r&b", "snap drums", "groove")):
        return "rnb"
    if any(marker in normalized for marker in ("horror", "bóng tối", "sợ hãi", "bất an")):
        return "horror"
    return predicted


def _record_text(record: dict[str, Any]) -> str:
    return str(record.get("input_text") or record.get("text") or record.get("chorus") or "")


def _strip_accents(value: str) -> str:
    return "".join(
        char for char in unicodedata.normalize("NFD", value)
        if unicodedata.category(char) != "Mn"
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
