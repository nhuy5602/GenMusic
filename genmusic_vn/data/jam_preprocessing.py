"""Preprocessing contracts for the JAM/DiffRhythm-style training pipeline."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

from .lyric_alignment import AlignedLine, align_wav_to_lyrics, write_lrc
from .vietnamese_g2p import vietnamese_g2p
from .vietnamese_text import normalize_vietnamese_lyrics


@dataclass(frozen=True)
class AudioPreprocessingSpec:
    sample_rate: int = 44_100
    channels: int = 2
    latent_rate: float = 21.5
    latent_dim: int = 64
    style_dim: int = 512
    vae_backend: str = "external-checkpoint-required"
    style_backend: str = "external-checkpoint-required"


class PreprocessingDependencyError(RuntimeError):
    """Raised when a production extractor is not configured."""


def _require_numpy() -> Any:
    if np is None:
        raise PreprocessingDependencyError("Cần numpy cho audio preprocessing.")
    return np


def _load_audio(path: str | Path, sample_rate: int) -> np.ndarray:
    arrays = _require_numpy()
    try:
        import librosa  # type: ignore

        audio, _ = librosa.load(str(path), sr=sample_rate, mono=False)
        if audio.ndim == 1:
            audio = audio[None, :]
        return arrays.asarray(audio, dtype=arrays.float32).T
    except ImportError:
        try:
            import soundfile as sf  # type: ignore

            audio, source_rate = sf.read(str(path), always_2d=True, dtype="float32")
            if source_rate != sample_rate:
                raise PreprocessingDependencyError("Cần librosa để resample audio về 44.1 kHz.")
            return arrays.asarray(audio, dtype=arrays.float32)
        except ImportError as exc:
            raise PreprocessingDependencyError("Cần librosa hoặc soundfile để đọc audio.") from exc


def _resample_frames(audio: np.ndarray, count: int) -> np.ndarray:
    arrays = _require_numpy()
    if len(audio) == count:
        return audio
    old_x = arrays.linspace(0.0, 1.0, max(1, len(audio)))
    new_x = arrays.linspace(0.0, 1.0, max(1, count))
    return arrays.stack([arrays.interp(new_x, old_x, audio[:, channel]) for channel in range(audio.shape[1])], axis=1)


def _proxy_latent(audio: np.ndarray, spec: AudioPreprocessingSpec) -> np.ndarray:
    """Create a deterministic shape-compatible proxy, never a production VAE."""

    arrays = _require_numpy()
    frames = max(1, int(math.ceil(len(audio) / spec.sample_rate * spec.latent_rate)))
    compressed = _resample_frames(audio, frames)
    mono = compressed.mean(axis=1)
    if len(mono) < frames:
        mono = arrays.pad(mono, (0, frames - len(mono)))
    values = arrays.stack(
        [
            mono,
            arrays.abs(mono),
            arrays.gradient(mono),
            arrays.sin(arrays.arange(frames) / 8.0),
            arrays.cos(arrays.arange(frames) / 8.0),
        ],
        axis=1,
    )
    repeats = int(math.ceil(spec.latent_dim / values.shape[1]))
    return arrays.tile(values, (1, repeats))[:, : spec.latent_dim].astype(arrays.float32)


def _proxy_style(audio: np.ndarray, spec: AudioPreprocessingSpec) -> np.ndarray:
    arrays = _require_numpy()
    mono = audio.mean(axis=1)
    spectrum = arrays.abs(arrays.fft.rfft(mono, n=max(512, 2 ** int(math.ceil(math.log2(max(2, len(mono))))))))
    bins = arrays.array_split(arrays.log1p(spectrum), 64)
    features = arrays.asarray([float(arrays.mean(item)) for item in bins], dtype=arrays.float32)
    features = (features - features.mean()) / (features.std() + 1e-6)
    repeats = int(math.ceil(spec.style_dim / len(features)))
    return arrays.tile(features, repeats)[: spec.style_dim].astype(arrays.float32)


class VaeLatentExtractor:
    """Adapter for the authorized DiffRhythm/Stable-Audio-compatible VAE."""

    def __init__(self, spec: AudioPreprocessingSpec, encoder: Callable[[np.ndarray, AudioPreprocessingSpec], np.ndarray] | None = None, *, allow_proxy: bool = False):
        self.spec = spec
        self.encoder = encoder
        self.allow_proxy = allow_proxy

    def extract(self, audio_path: str | Path) -> tuple[np.ndarray, str]:
        audio = _load_audio(audio_path, self.spec.sample_rate)
        if self.encoder is None:
            if not self.allow_proxy:
                raise PreprocessingDependencyError("Chưa cấu hình VAE checkpoint; truyền encoder hoặc bật --allow-proxy cho smoke test.")
            return _proxy_latent(audio, self.spec), "proxy-not-for-training"
        arrays = _require_numpy()
        latent = arrays.asarray(self.encoder(audio, self.spec), dtype=arrays.float32)
        if latent.ndim != 2 or latent.shape[1] != self.spec.latent_dim:
            raise ValueError(f"VAE latent phải có shape [frames, {self.spec.latent_dim}], nhận {latent.shape}.")
        return latent, "external-vae"


class StyleEmbeddingExtractor:
    """Adapter for the 512-dimensional style encoder used by the recipe."""

    def __init__(self, spec: AudioPreprocessingSpec, encoder: Callable[[np.ndarray, AudioPreprocessingSpec], np.ndarray] | None = None, *, allow_proxy: bool = False):
        self.spec = spec
        self.encoder = encoder
        self.allow_proxy = allow_proxy

    def extract(self, audio_path: str | Path) -> tuple[np.ndarray, str]:
        audio = _load_audio(audio_path, self.spec.sample_rate)
        if self.encoder is None:
            if not self.allow_proxy:
                raise PreprocessingDependencyError("Chưa cấu hình style encoder checkpoint; bật --allow-proxy chỉ cho smoke test.")
            return _proxy_style(audio, self.spec), "proxy-not-for-training"
        arrays = _require_numpy()
        embedding = arrays.asarray(self.encoder(audio, self.spec), dtype=arrays.float32).reshape(-1)
        if embedding.shape != (self.spec.style_dim,):
            raise ValueError(f"Style embedding phải có shape [{self.spec.style_dim}], nhận {embedding.shape}.")
        return embedding, "external-style-encoder"


def load_torchscript_encoder(checkpoint: str | Path, *, kind: str, spec: AudioPreprocessingSpec) -> Callable[[np.ndarray, AudioPreprocessingSpec], np.ndarray]:
    """Load a project-approved TorchScript encoder adapter.

    Native DiffRhythm/Stable-Audio checkpoints may need a repository-specific
    wrapper; this adapter is intentionally strict and accepts TorchScript only.
    """

    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise PreprocessingDependencyError("Cần torch để load TorchScript encoder.") from exc
    module = torch.jit.load(str(checkpoint), map_location="cpu").eval()

    def encode(audio: np.ndarray, current_spec: AudioPreprocessingSpec) -> np.ndarray:
        arrays = _require_numpy()
        waveform = torch.from_numpy(arrays.asarray(audio.T, dtype=arrays.float32)).unsqueeze(0)
        with torch.no_grad():
            value = module(waveform)
        if isinstance(value, dict):
            selected = None
            for key in (kind, "latent", "embedding"):
                if key in value and value[key] is not None:
                    selected = value[key]
                    break
            value = selected
        if isinstance(value, (tuple, list)):
            value = value[0]
        result = value.detach().cpu().numpy()
        if kind == "style_embedding":
            return arrays.asarray(result).reshape(-1)
        result = arrays.asarray(result)
        if result.ndim == 3:
            result = result[0]
        if result.ndim != 2:
            raise ValueError(f"TorchScript VAE output phải là 2D/3D, nhận {result.shape}.")
        if result.shape[-1] == current_spec.latent_dim:
            return result
        if result.shape[0] == current_spec.latent_dim:
            return result.T
        raise ValueError(f"TorchScript VAE output không có chiều latent {current_spec.latent_dim}: {result.shape}.")

    return encode


def _save_tensor(value: np.ndarray, path: Path, spec: AudioPreprocessingSpec, kind: str) -> None:
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise PreprocessingDependencyError("Cần torch để ghi latent/style theo format .pt.") from exc
    torch.save({"tensor": torch.from_numpy(value), "kind": kind, "spec": asdict(spec)}, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_source_records(path: str | Path, audio_root: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    raw = source.read_text(encoding="utf-8")
    if source.suffix.casefold() == ".jsonl":
        values = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        parsed = json.loads(raw)
        values = parsed if isinstance(parsed, list) else parsed.get("records", [])
    root = Path(audio_root)
    records: list[dict[str, Any]] = []
    for index, item in enumerate(values):
        audio = Path(str(item["audio"]))
        if not audio.is_absolute():
            audio = root / audio
        records.append({**item, "id": str(item.get("id", audio.stem or index)), "audio": str(audio)})
    return records


def _lyrics_value(item: dict[str, Any], base_dir: Path) -> str | None:
    if isinstance(item.get("lyrics"), str):
        try:
            candidate = Path(str(item["lyrics"]))
            if not candidate.is_absolute():
                candidate = base_dir / candidate
            if candidate.exists():
                return candidate.read_text(encoding="utf-8")
        except OSError:
            pass
        return str(item["lyrics"])
    return None


def _write_manifest(records: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in records:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def prepare_jam_dataset(
    source_manifest: str | Path,
    output_dir: str | Path,
    *,
    audio_root: str | Path = ".",
    spec: AudioPreprocessingSpec | None = None,
    vae_encoder: Callable[[np.ndarray, AudioPreprocessingSpec], np.ndarray] | None = None,
    style_encoder: Callable[[np.ndarray, AudioPreprocessingSpec], np.ndarray] | None = None,
    asr_model: str | None = None,
    allow_proxy: bool = False,
    max_files: int = 0,
) -> dict[str, Any]:
    """Create aligned lyric, phoneme, latent and style artifacts.

    The input manifest is the source of truth. A record may contain ``lyrics``
    as inline text or a path. Audio-only records require ``asr_model`` and are
    transcribed through the configured faster-whisper model.
    """

    spec = spec or AudioPreprocessingSpec()
    output = Path(output_dir)
    latent_dir = output / "latents"
    style_dir = output / "styles"
    lrc_dir = output / "lrc"
    for directory in (latent_dir, style_dir, lrc_dir):
        directory.mkdir(parents=True, exist_ok=True)
    source_records = _read_source_records(source_manifest, audio_root)
    if max_files:
        source_records = source_records[:max_files]
    latent_extractor = VaeLatentExtractor(spec, vae_encoder, allow_proxy=allow_proxy)
    style_extractor = StyleEmbeddingExtractor(spec, style_encoder, allow_proxy=allow_proxy)
    processed: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for item in source_records:
        audio_path = Path(item["audio"])
        try:
            lyrics = _lyrics_value(item, Path(item["audio"]).parent)
            asr_segments = None
            if not lyrics:
                if not asr_model:
                    raise ValueError("Thiếu lyrics; cần asr_model để bóc transcript từ audio.")
                try:
                    from faster_whisper import WhisperModel  # type: ignore
                except ImportError as exc:
                    raise PreprocessingDependencyError("Cần faster-whisper cho dataset chỉ có MP3.") from exc
                detector = WhisperModel(asr_model)
                detected, _ = detector.transcribe(str(audio_path), language="vi", word_timestamps=False)
                asr_segments = [{"start": part.start, "end": part.end, "text": part.text} for part in detected]
                lyrics = "\n".join(part["text"].strip() for part in asr_segments if part["text"].strip())
            assert lyrics is not None
            normalized = normalize_vietnamese_lyrics(lyrics)
            g2p = vietnamese_g2p(normalized)
            aligned = align_wav_to_lyrics(audio_path, normalized, segments=asr_segments, asr_model=None, allow_heuristic=allow_proxy)
            latent, latent_backend = latent_extractor.extract(audio_path)
            style, style_backend = style_extractor.extract(audio_path)
            record_id = str(item["id"])
            latent_path = latent_dir / f"{record_id}.pt"
            style_path = style_dir / f"{record_id}.pt"
            lrc_path = lrc_dir / f"{record_id}.lrc"
            _save_tensor(latent, latent_path, spec, "vae_latent")
            _save_tensor(style, style_path, spec, "style_embedding")
            write_lrc(aligned, lrc_path)
            processed.append(
                {
                    "id": record_id,
                    "audio_path": str(audio_path.resolve()),
                    "lyrics": normalized,
                    "phoneme_tokens": g2p.tokens,
                    "g2p_backend": g2p.backend,
                    "lrc_path": lrc_path.relative_to(output).as_posix(),
                    "latent_path": latent_path.relative_to(output).as_posix(),
                    "style_path": style_path.relative_to(output).as_posix(),
                    "latent_shape": list(latent.shape),
                    "style_shape": list(style.shape),
                    "latent_backend": latent_backend,
                    "style_backend": style_backend,
                    "audio_sha256": _sha256(audio_path),
                    "source": item.get("source", "user-supplied-manifest"),
                }
            )
        except Exception as exc:
            skipped.append({"id": str(item.get("id", audio_path.stem)), "error": str(exc)})
    manifest_path = output / "manifest.jsonl"
    _write_manifest(processed, manifest_path)
    report = {
        "spec": asdict(spec),
        "record_count": len(processed),
        "skipped_count": len(skipped),
        "skipped": skipped,
        "manifest": str(manifest_path.resolve()),
        "provenance": "input manifest plus explicitly configured extractors",
    }
    (output / "preprocessing_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
