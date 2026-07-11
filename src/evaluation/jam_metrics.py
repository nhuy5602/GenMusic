"""Objective and subjective metrics for the self-authored music model."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]


def _require_numpy() -> Any:
    if np is None:
        raise RuntimeError("Cần numpy để tính metric âm thanh.")
    return np


def levenshtein(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, ref_token in enumerate(reference, start=1):
        current = [row]
        for column, hyp_token in enumerate(hypothesis, start=1):
            current.append(min(current[-1] + 1, previous[column] + 1, previous[column - 1] + (ref_token != hyp_token)))
        previous = current
    return previous[-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    reference_tokens = reference.casefold().split()
    hypothesis_tokens = hypothesis.casefold().split()
    if not reference_tokens:
        return 0.0 if not hypothesis_tokens else 1.0
    return levenshtein(reference_tokens, hypothesis_tokens) / len(reference_tokens)


def mel_cepstral_distortion(reference_audio: str | Path, generated_audio: str | Path) -> float | None:
    try:
        import librosa  # type: ignore
    except ImportError:
        return None
    reference, reference_rate = librosa.load(str(reference_audio), sr=None, mono=True)
    generated, generated_rate = librosa.load(str(generated_audio), sr=reference_rate, mono=True)
    reference_mfcc = librosa.feature.mfcc(y=reference, sr=reference_rate, n_mfcc=13)
    generated_mfcc = librosa.feature.mfcc(y=generated, sr=generated_rate, n_mfcc=13)
    frames = min(reference_mfcc.shape[1], generated_mfcc.shape[1])
    if frames == 0:
        return None
    distance = np.linalg.norm(reference_mfcc[:, :frames] - generated_mfcc[:, :frames], axis=0)
    return float(np.mean(distance))


def frame_error_rate(reference_f0: Iterable[float], generated_f0: Iterable[float], *, cents_threshold: float = 50.0) -> float | None:
    arrays = _require_numpy()
    reference = arrays.asarray(list(reference_f0), dtype=float)
    generated = arrays.asarray(list(generated_f0), dtype=float)
    frames = min(len(reference), len(generated))
    if frames == 0:
        return None
    reference = reference[:frames]
    generated = generated[:frames]
    voiced = (reference > 1e-5) & (generated > 1e-5)
    if not voiced.any():
        return None
    cents = 1200.0 * arrays.abs(arrays.log2(generated[voiced] / reference[voiced]))
    return float(arrays.mean(cents > cents_threshold))


def frechet_audio_distance(reference_embedding: Any, generated_embedding: Any) -> float:
    arrays = _require_numpy()
    reference = arrays.asarray(reference_embedding, dtype=float)
    generated = arrays.asarray(generated_embedding, dtype=float)
    if reference.ndim != 2 or generated.ndim != 2:
        raise ValueError("Embedding FAD phải có shape [samples, dimensions].")
    mean_difference = reference.mean(axis=0) - generated.mean(axis=0)
    reference_covariance = arrays.cov(reference, rowvar=False)
    generated_covariance = arrays.cov(generated, rowvar=False)
    try:
        from scipy.linalg import sqrtm  # type: ignore

        covariance_root = sqrtm(reference_covariance @ generated_covariance).real
    except ImportError:
        eigenvalues, eigenvectors = arrays.linalg.eigh(reference_covariance @ generated_covariance)
        covariance_root = (eigenvectors * arrays.sqrt(arrays.clip(eigenvalues, 0, None))) @ eigenvectors.T
    return float(mean_difference @ mean_difference + np.trace(reference_covariance + generated_covariance - 2 * covariance_root))


def objective_metrics(
    *,
    generated_audio: str | Path,
    reference_audio: str | Path | None = None,
    generated_transcript: str | None = None,
    reference_transcript: str | None = None,
    reference_f0: Iterable[float] | None = None,
    generated_f0: Iterable[float] | None = None,
    reference_embedding: Any | None = None,
    generated_embedding: Any | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "FAD": None,
        "MCD": mel_cepstral_distortion(reference_audio, generated_audio) if reference_audio else None,
        "FFE": frame_error_rate(reference_f0, generated_f0) if reference_f0 is not None and generated_f0 is not None else None,
        "WER": word_error_rate(reference_transcript, generated_transcript) if reference_transcript is not None and generated_transcript is not None else None,
        "availability": {},
    }
    if reference_embedding is not None and generated_embedding is not None:
        result["FAD"] = frechet_audio_distance(reference_embedding, generated_embedding)
    result["availability"] = {key: value is not None for key, value in result.items() if key.isupper()}
    result["generated_audio"] = str(Path(generated_audio).resolve())
    result["reference_audio"] = str(Path(reference_audio).resolve()) if reference_audio else None
    return result


def _read_votes(source: str | Path | Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(source, (str, Path)):
        return list(source)
    path = Path(source)
    if path.suffix.casefold() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def subjective_summary(source: str | Path | Iterable[dict[str, Any]]) -> dict[str, Any]:
    votes = _read_votes(source)
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for vote in votes:
        variant = str(vote.get("model_variant", vote.get("variant", "unknown")))
        by_variant.setdefault(variant, []).append(vote)
    variants: dict[str, Any] = {}
    for variant, rows in by_variant.items():
        musicality = [float(row["musicality"]) for row in rows if row.get("musicality") not in (None, "")]
        intelligibility = [float(row["intelligibility"]) for row in rows if row.get("intelligibility") not in (None, "")]
        variants[variant] = {
            "n": len(rows),
            "MOS_musicality": sum(musicality) / len(musicality) if musicality else None,
            "MOS_intelligibility": sum(intelligibility) / len(intelligibility) if intelligibility else None,
        }
    comparisons = [float(row["comparison"]) for row in votes if row.get("comparison") not in (None, "")]
    return {"vote_count": len(votes), "listener_count": len({str(row.get('listener_id', 'unknown')) for row in votes}), "variants": variants, "CMOS": sum(comparisons) / len(comparisons) if comparisons else None, "status": "measured" if votes else "no-votes"}


def write_metric_report(report: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "evaluation_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"report": str(path.resolve()), **report}
