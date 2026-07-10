"""Vietnamese tone-contour preference data and DPO alignment."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class PreferencePair:
    prompt_id: str
    preferred: str
    dispreferred: str
    preferred_score: float
    dispreferred_score: float
    reason: str
    text_ids: list[int] | None = None
    style_path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def tone_contour_agreement(music_f0: Iterable[float], speech_f0: Iterable[float]) -> float:
    music = np.asarray(list(music_f0), dtype=float)
    speech = np.asarray(list(speech_f0), dtype=float)
    count = min(len(music), len(speech))
    if count < 2:
        return 0.0
    music_delta = np.sign(np.diff(music[:count]))
    speech_delta = np.sign(np.diff(speech[:count]))
    return float(np.mean(music_delta == speech_delta))


def build_preference_pairs(scores: Iterable[dict[str, Any]], output_path: str | Path, *, threshold: float = 0.1) -> dict[str, Any]:
    pairs: list[PreferencePair] = []
    for item in scores:
        candidates = item.get("candidates") or [item]
        if len(candidates) < 2:
            continue
        ranked: list[tuple[float, dict[str, Any]]] = []
        for candidate in candidates:
            if "tone_score" in candidate:
                score = float(candidate["tone_score"])
            else:
                score = tone_contour_agreement(candidate.get("music_f0", []), candidate.get("speech_f0", []))
            ranked.append((score, candidate))
        ranked.sort(key=lambda value: value[0], reverse=True)
        winner_score, winner = ranked[0]
        loser_score, loser = ranked[-1]
        if winner_score - loser_score < threshold:
            continue
        pairs.append(
            PreferencePair(
                prompt_id=str(item.get("prompt_id", item.get("id", len(pairs)))),
                preferred=str(winner.get("latent_path", winner.get("id", ""))),
                dispreferred=str(loser.get("latent_path", loser.get("id", ""))),
                preferred_score=winner_score,
                dispreferred_score=loser_score,
                reason="music_f0 and speech_f0 contour direction agreement",
                text_ids=item.get("text_ids") or winner.get("text_ids"),
                style_path=winner.get("style_path") or item.get("style_path"),
            )
        )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair.as_dict(), ensure_ascii=False) + "\n")
    report = {"pair_count": len(pairs), "threshold": threshold, "output": str(destination.resolve()), "status": "created"}
    destination.with_name("dpo_pair_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def dpo_logistic_loss(preferred_loss: Any, dispreferred_loss: Any, *, beta: float = 0.1) -> Any:
    """Return the DPO penalty when lower diffusion loss means better audio."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Cần torch để train DPO.") from exc
    return -torch.nn.functional.logsigmoid(beta * (dispreferred_loss - preferred_loss)).mean()
