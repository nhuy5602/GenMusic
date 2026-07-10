"""Closed-loop local evaluation and conservative self-improvement runner."""

from __future__ import annotations

import json
import math
import re
import shutil
import unicodedata
import wave
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..data.vietnamese_text import normalize_vietnamese_lyrics
from ..integrations.kaggle_auto import run_local_generation
from ..training.self_diffusion import train_model


DEFAULT_SELF_IMPROVE_INPUTS = [
    ("Phố lên đèn, lòng em còn nhớ\nGió qua thềm, câu ca còn chờ.", "Vietnamese city pop, warm bass, mid-tempo drums"),
    ("Mưa rơi ngoài hiên, đêm nay nghe rất khác\nTin nhắn chưa gửi, nằm im trong màu mắt.", "Vietnamese synthwave ballad, rainy night, soft pulse"),
    ("Ta đi qua biển xanh, nghe mùa hè thức giấc\nCát vương trên bàn chân, tiếng cười vang rất thật.", "Vietnamese acoustic summer pop, bright guitar, carefree beat"),
    ("Ngày em rời xa, căn phòng bỗng rộng hơn\nAnh gom từng kỷ niệm, đặt cạnh khung cửa sổ.", "Vietnamese R&B soul, intimate piano, restrained groove"),
    ("Đường còn xa nhưng tim mình không ngại\nMỗi bước hôm nay gọi ngày mai trở lại.", "Vietnamese motivational rap pop, confident drums, clean bass"),
    ("Bên bờ sông cũ, khói lam bay chậm quá\nTiếng ru năm nào còn mắc giữa chiều xa.", "Vietnamese bolero, nylon guitar, nostalgic accordion"),
    ("Con diều no gió bay qua triền đê nhỏ\nLũ trẻ gọi nhau, chiều nghiêng màu hoa cỏ.", "Vietnamese folk pop, bamboo flute, playful hand percussion"),
    ("Trong vệt sao rơi, ta nghe thành phố ngủ\nMột giấc mơ xanh mở cửa giữa hư vô.", "Vietnamese cinematic dream pop, spacious synth, evolving strings"),
    ("Trang vở còn thơm, đèn khuya chưa muốn tắt\nMột câu chưa xong, ngày mới đã ghé sát.", "Vietnamese lo-fi study beat, mellow keys, soft vinyl texture"),
    ("Tiếng trống gọi tên, sân khấu bừng lên\nTa hát đến khi bình minh đứng bên.", "Vietnamese alternative rock, live drums, energetic guitar"),
]


def _words(text: str) -> list[str]:
    return re.findall(r"[\wÀ-ỹ]+", text.casefold(), flags=re.UNICODE)


def _rhyme_key(word: str) -> str:
    normalized = unicodedata.normalize("NFD", word)
    plain = "".join(char for char in normalized if unicodedata.category(char) != "Mn")
    plain = re.sub(r"[^a-zA-ZÀ-ỹ]", "", plain).casefold()
    vowels = "aeiouy"
    positions = [index for index, char in enumerate(plain) if char in vowels]
    if not positions:
        return plain[-2:]
    start = max(0, positions[-1] - 1)
    return plain[start:]


def rhyme_score(text: str) -> float:
    endings = [_rhyme_key(words[-1]) for line in text.splitlines() if (words := _words(line))]
    if len(endings) < 2:
        return 0.5
    pairs = list(zip(endings[::2], endings[1::2]))
    if not pairs:
        return 0.5
    matches = sum(left == right or left[-1:] == right[-1:] for left, right in pairs)
    return float(matches / len(pairs))


def _audio_features(audio_path: str | Path) -> dict[str, float]:
    with wave.open(str(audio_path), "rb") as stream:
        sample_rate = stream.getframerate()
        frames = stream.readframes(stream.getnframes())
        channels = stream.getnchannels()
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if audio.size == 0:
        return {"duration_seconds": 0.0, "rms": 0.0, "silence_ratio": 1.0, "clipping_ratio": 1.0, "clarity_score": 0.0, "estimated_bpm": 0.0}
    rms = float(np.sqrt(np.mean(audio * audio)))
    silence_ratio = float(np.mean(np.abs(audio) < 0.01))
    clipping_ratio = float(np.mean(np.abs(audio) > 0.985))
    clarity_score = float(np.clip(1.0 - silence_ratio * 0.65 - clipping_ratio * 4.0, 0.0, 1.0))
    estimated_bpm = 0.0
    try:
        import librosa  # type: ignore

        tempo, _ = librosa.beat.beat_track(y=audio, sr=sample_rate)
        estimated_bpm = float(np.asarray(tempo).reshape(-1)[0])
    except Exception:
        envelope = np.abs(audio[:: max(1, sample_rate // 100)])
        if envelope.size > 4:
            centered = envelope - envelope.mean()
            correlation = np.correlate(centered, centered, mode="full")[envelope.size - 1 :]
            low = max(1, int(100 * 60 / 180))
            high = min(len(correlation) - 1, int(100 * 60 / 50))
            if high > low:
                lag = low + int(np.argmax(correlation[low:high]))
                estimated_bpm = 60.0 * 100.0 / lag
    return {
        "duration_seconds": float(audio.size / max(1, sample_rate)),
        "rms": rms,
        "silence_ratio": silence_ratio,
        "clipping_ratio": clipping_ratio,
        "clarity_score": clarity_score,
        "estimated_bpm": estimated_bpm,
    }


def _target_bpm(style: str) -> float:
    lowered = style.casefold()
    if any(word in lowered for word in ("ballad", "bolero", "slow", "rainy")):
        return 72.0
    if any(word in lowered for word in ("rap", "rock", "energetic")):
        return 118.0
    if any(word in lowered for word in ("lo-fi", "dream", "soul")):
        return 82.0
    return 104.0


def evaluate_input(*, text: str, style: str, audio_path: str | Path, requested_duration: float) -> dict[str, Any]:
    features = _audio_features(audio_path)
    word_count = len(_words(text))
    words_per_second = word_count / max(1.0, requested_duration)
    flow_score = float(np.clip(1.0 - abs(words_per_second - 3.0) / 3.0, 0.0, 1.0))
    coverage_proxy = float(np.clip(1.0 - abs(words_per_second - 2.6) / 2.6, 0.0, 1.0))
    bpm = features["estimated_bpm"]
    mood_beat_score = float(np.clip(1.0 - abs(bpm - _target_bpm(style)) / 120.0, 0.0, 1.0)) if bpm else 0.0
    rhyme = rhyme_score(text)
    vocal_presence = None
    composite = float(np.mean([coverage_proxy, rhyme, mood_beat_score, features["clarity_score"], flow_score]))
    return {
        "text": text,
        "style": style,
        "audio": str(Path(audio_path).resolve()),
        "metrics": {
            "lyric_coverage_proxy": round(coverage_proxy, 4),
            "rhyme_score": round(rhyme, 4),
            "mood_beat_score": round(mood_beat_score, 4),
            "vocal_presence": vocal_presence,
            "clarity_score": round(features["clarity_score"], 4),
            "flow_score": round(flow_score, 4),
            "estimated_bpm": round(bpm, 2),
            "words_per_second": round(words_per_second, 3),
            **{key: round(value, 6) for key, value in features.items() if key not in {"clarity_score", "estimated_bpm"}},
        },
        "composite_score": round(composite, 6),
        "limitations": [
            "lyric_coverage_proxy không thay thế WER bằng ASR",
            "vocal_presence chưa đo được vì model hiện chưa có vocal stem/ASR riêng",
        ],
    }


def _dataset_texts(dataset_dir: str | Path) -> set[str]:
    path = Path(dataset_dir) / "records.jsonl"
    if not path.exists():
        raise ValueError(f"Thiếu records.jsonl trong dataset {path.parent}")
    return {normalize_vietnamese_lyrics(json.loads(line).get("text", "")).casefold() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def run_self_improve(*, dataset_dir: str | Path, output_root: str | Path, checkpoint: str | Path, rounds: int = 10, duration_seconds: float = 6.0, steps: int = 4, epochs: int = 1, batch_size: int = 4, max_records: int | None = 64, learning_rate: float = 2e-4, device: str | None = None, seed: int = 5602, inputs: Iterable[tuple[str, str]] | None = None) -> dict[str, Any]:
    dataset = Path(dataset_dir).resolve()
    destination = Path(output_root).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    current_checkpoint = Path(checkpoint).resolve()
    if not current_checkpoint.exists():
        initial = train_model(dataset, current_checkpoint, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, device=device, max_records=max_records)
    else:
        initial = {"status": "reused", "checkpoint": str(current_checkpoint)}
    dataset_texts = _dataset_texts(dataset)
    cases = list(inputs or DEFAULT_SELF_IMPROVE_INPUTS)
    cases = cases[: max(1, int(rounds))]
    normalized_cases = [normalize_vietnamese_lyrics(text).strip() for text, _ in cases]
    overlap = [text for text in normalized_cases if text.casefold() in dataset_texts]
    if overlap:
        raise ValueError(f"Input tự improve trùng dataset: {overlap[0]}")
    rounds_report: list[dict[str, Any]] = []
    accepted_count = 0
    for index, ((text, style), normalized_text) in enumerate(zip(cases, normalized_cases), start=1):
        round_dir = destination / f"round_{index:02d}"
        before_dir = round_dir / "before"
        after_dir = round_dir / "after"
        before = run_local_generation(text=normalized_text, style=style, output_dir=before_dir, duration_seconds=duration_seconds, checkpoint=current_checkpoint, steps=steps, seed=seed + index, device=device, mel_output=round_dir / "feedback_mel.pt")
        before_eval = evaluate_input(text=normalized_text, style=style, audio_path=before["audio_path"], requested_duration=duration_seconds)
        feedback_record = [{"id": f"feedback_{index:02d}", "text": normalized_text, "style": style, "mel_path": str(Path(before["mel_path"]).resolve()), "frames": int(torch_shape(before["mel_path"])[1])}]
        candidate_checkpoint = round_dir / "candidate.pt"
        training = train_model(dataset, candidate_checkpoint, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, device=device, max_records=max_records, additional_records=feedback_record)
        after = run_local_generation(text=normalized_text, style=style, output_dir=after_dir, duration_seconds=duration_seconds, checkpoint=candidate_checkpoint, steps=steps, seed=seed + index, device=device)
        after_eval = evaluate_input(text=normalized_text, style=style, audio_path=after["audio_path"], requested_duration=duration_seconds)
        delta = round(after_eval["composite_score"] - before_eval["composite_score"], 6)
        accepted = delta > 0.0
        if accepted:
            current_checkpoint = candidate_checkpoint.resolve()
            accepted_count += 1
        report = {"round": index, "input": {"text": normalized_text, "style": style}, "before": before_eval, "training": training, "after": after_eval, "score_delta": delta, "improved": accepted, "checkpoint_for_next_round": str(current_checkpoint)}
        (round_dir / "round_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        rounds_report.append(report)
    stable_checkpoint = destination / "final_checkpoint.pt"
    if current_checkpoint.resolve() != stable_checkpoint.resolve():
        shutil.copy2(current_checkpoint, stable_checkpoint)
    summary = {"status": "complete", "backend": "genmusic-vn-self-diffusion", "dataset": str(dataset), "initial": initial, "round_count": len(rounds_report), "accepted_rounds": accepted_count, "final_checkpoint": str(stable_checkpoint.resolve()), "rounds": rounds_report, "notes": ["Mỗi vòng train candidate trước khi đánh giá after; chỉ checkpoint có composite score tăng mới được dùng cho input kế tiếp."]}
    (destination / "self_improve_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def torch_shape(path: str | Path) -> tuple[int, int]:
    import torch

    value = torch.load(path, map_location="cpu", weights_only=True)
    mel = value["mel"] if isinstance(value, dict) else value
    return int(mel.shape[0]), int(mel.shape[1])
