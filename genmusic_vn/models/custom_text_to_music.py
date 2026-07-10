from __future__ import annotations

import json
import math
import re
import unicodedata
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


CUSTOM_MODEL_ID = "genmusic-vn/custom-text-to-music-v1"
CUSTOM_CHECKPOINT_FILENAME = "custom_text_to_music.pt"
SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>")


def _tokenize(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", str(text or "")).lower()
    return re.findall(r"[\wÀ-ỹ]+", normalized, flags=re.UNICODE)


@dataclass
class TextVocabulary:
    tokens: list[str]

    @classmethod
    def build(cls, texts: Iterable[str], max_size: int = 8192) -> "TextVocabulary":
        counts: dict[str, int] = {}
        for text in texts:
            for token in _tokenize(text):
                counts[token] = counts.get(token, 0) + 1
        ordered = sorted(counts, key=lambda token: (-counts[token], token))
        return cls(list(SPECIAL_TOKENS) + ordered[: max(1, max_size - len(SPECIAL_TOKENS))])

    @property
    def token_to_id(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.tokens)}

    def encode(self, text: str, max_length: int = 64) -> list[int]:
        mapping = self.token_to_id
        ids = [mapping["<bos>"]]
        ids.extend(mapping.get(token, mapping["<unk>"]) for token in _tokenize(text)[: max(1, max_length - 2)])
        ids.append(mapping["<eos>"])
        return ids


@dataclass(frozen=True)
class MusicFeatureCodec:
    """Low-rate learned audio representation used by the custom model.

    Each frame stores pitch class, energy, bass activity and spectral
    brightness. The Transformer learns to generate these feature frames from
    text; the renderer turns the generated sequence into audio.
    """

    bins: tuple[int, int, int, int] = (12, 8, 8, 8)
    frame_seconds: float = 0.25
    frames: int = 120

    def extract(self, audio_path: str | Path, *, seconds: int = 16) -> list[list[int]]:
        import librosa
        import numpy as np

        audio, sample_rate = librosa.load(str(audio_path), sr=22050, mono=True, duration=seconds)
        if audio.size < 64:
            raise ValueError(f"Audio rỗng hoặc quá ngắn: {audio_path}")
        frame_count = min(self.frames, max(8, int(seconds / self.frame_seconds)))
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=64,
            power=2.0,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate, hop_length=512)
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate, hop_length=512)[0]
        energy = np.mean(np.maximum(mel_db + 80.0, 0.0), axis=0)
        bass = np.mean(np.maximum(mel_db[:12] + 80.0, 0.0), axis=0)
        frame_indices = np.linspace(0, max(0, mel_db.shape[1] - 1), frame_count).astype(int)
        energy_scale = max(1e-6, float(np.percentile(energy, 95)))
        bass_scale = max(1e-6, float(np.percentile(bass, 95)))
        centroid_scale = max(1.0, float(np.percentile(centroid, 95)))
        features: list[list[int]] = []
        for index in frame_indices:
            chroma_index = min(chroma.shape[1] - 1, int(index))
            features.append(
                [
                    int(np.argmax(chroma[:, chroma_index])) % self.bins[0],
                    min(self.bins[1] - 1, int(float(energy[index]) / energy_scale * self.bins[1])),
                    min(self.bins[2] - 1, int(float(bass[index]) / bass_scale * self.bins[2])),
                    min(self.bins[3] - 1, int(float(centroid[index]) / centroid_scale * self.bins[3])),
                ]
            )
        return features


class CustomTextToMusicTransformer:
    """A project-owned text-conditioned autoregressive music feature model."""

    def __init__(self, vocab_size: int, *, d_model: int = 192, nhead: int = 6, layers: int = 4, max_text: int = 64):
        import torch
        from torch import nn

        class _Network(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.text_embedding = nn.Embedding(vocab_size, d_model)
                self.text_position = nn.Embedding(max_text, d_model)
                self.feature_embeddings = nn.ModuleList(nn.Embedding(size, d_model) for size in MusicFeatureCodec().bins)
                self.bos_frame = nn.Parameter(torch.zeros(1, 1, d_model))
                encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model * 4, batch_first=True, norm_first=True)
                decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=d_model * 4, batch_first=True, norm_first=True)
                self.text_encoder = nn.TransformerEncoder(encoder_layer, layers)
                self.music_decoder = nn.TransformerDecoder(decoder_layer, layers)
                self.output_heads = nn.ModuleList(nn.Linear(d_model, size) for size in MusicFeatureCodec().bins)

            def feature_embedding(self, features):
                value = 0.0
                for index, embedding in enumerate(self.feature_embeddings):
                    value = value + embedding(features[:, :, index])
                return value

            def forward(self, text_ids, music_features, text_padding_mask=None):
                positions = torch.arange(text_ids.shape[1], device=text_ids.device).unsqueeze(0)
                text_hidden = self.text_encoder(
                    self.text_embedding(text_ids) + self.text_position(positions),
                    src_key_padding_mask=text_padding_mask,
                )
                shifted = self.feature_embedding(music_features[:, :-1])
                shifted = torch.cat([self.bos_frame.expand(music_features.shape[0], -1, -1), shifted], dim=1)
                target_length = shifted.shape[1]
                causal_mask = torch.triu(
                    torch.ones(target_length, target_length, device=text_ids.device, dtype=torch.bool), diagonal=1
                )
                decoded = self.music_decoder(shifted, text_hidden, tgt_mask=causal_mask, memory_key_padding_mask=text_padding_mask)
                return [head(decoded) for head in self.output_heads]

        self.network = _Network()
        self.vocab_size = vocab_size
        self.config = {"d_model": d_model, "nhead": nhead, "layers": layers, "max_text": max_text}

    def parameters(self):
        return self.network.parameters()

    def to(self, *args, **kwargs):
        self.network.to(*args, **kwargs)
        return self

    def train(self, mode: bool = True):
        self.network.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def __call__(self, *args, **kwargs):
        return self.network(*args, **kwargs)

    def generate(self, text_ids, *, max_frames: int = 64, temperature: float = 0.9):
        import torch

        self.eval()
        generated = torch.zeros((text_ids.shape[0], 1, 4), dtype=torch.long, device=text_ids.device)
        for _ in range(max(1, max_frames)):
            logits = self(text_ids, generated)
            next_features = []
            for head in logits:
                probabilities = torch.softmax(head[:, -1, :] / max(0.05, temperature), dim=-1)
                next_features.append(torch.multinomial(probabilities, 1).squeeze(-1))
            generated = torch.cat([generated, torch.stack(next_features, dim=1).unsqueeze(1)], dim=1)
        return generated[:, 1:]


def create_custom_model(vocab_size: int, **kwargs) -> CustomTextToMusicTransformer:
    return CustomTextToMusicTransformer(vocab_size, **kwargs)


def save_custom_checkpoint(path: str | Path, model: CustomTextToMusicTransformer, vocabulary: TextVocabulary, codec: MusicFeatureCodec) -> Path:
    import torch

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_id": CUSTOM_MODEL_ID,
            "model_config": model.config,
            "vocabulary": vocabulary.tokens,
            "codec": {"bins": codec.bins, "frame_seconds": codec.frame_seconds, "frames": codec.frames},
            "state_dict": model.network.state_dict(),
        },
        destination,
    )
    return destination


def load_custom_checkpoint(path: str | Path, *, device: str = "cpu") -> tuple[CustomTextToMusicTransformer, TextVocabulary, MusicFeatureCodec]:
    import torch

    checkpoint = torch.load(path, map_location=device)
    vocabulary = TextVocabulary(list(checkpoint["vocabulary"]))
    config = dict(checkpoint.get("model_config") or {})
    model = CustomTextToMusicTransformer(len(vocabulary.tokens), **config).to(device)
    model.network.load_state_dict(checkpoint["state_dict"])
    codec_data = checkpoint.get("codec") or {}
    codec = MusicFeatureCodec(
        bins=tuple(codec_data.get("bins", (12, 8, 8, 8))),
        frame_seconds=float(codec_data.get("frame_seconds", 0.25)),
        frames=int(codec_data.get("frames", 120)),
    )
    return model, vocabulary, codec


def render_generated_features(features: list[list[int]], output_path: str | Path, *, sample_rate: int = 22050) -> Path:
    import numpy as np

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame_seconds = 0.25
    total_samples = max(sample_rate, int(len(features) * frame_seconds * sample_rate))
    audio = np.zeros(total_samples, dtype="float32")
    for frame_index, values in enumerate(features):
        if len(values) < 4:
            continue
        start = int(frame_index * frame_seconds * sample_rate)
        end = min(total_samples, int((frame_index + 1) * frame_seconds * sample_rate))
        if end <= start:
            continue
        energy = 0.08 + (int(values[1]) / 7.0) * 0.16
        root = 130.81 * (2.0 ** ((int(values[0]) % 12) / 12.0))
        bass = 55.0 * (2.0 ** ((int(values[2]) % 8) / 12.0))
        brightness = int(values[3]) / 7.0
        time_values = np.arange(end - start, dtype="float32") / sample_rate
        envelope = np.minimum(1.0, time_values / 0.025) * np.minimum(1.0, (end - start - time_values * sample_rate) / (0.08 * sample_rate))
        envelope = np.clip(envelope, 0.0, 1.0)
        pad = np.sin(2.0 * math.pi * root * time_values) + 0.22 * np.sin(4.0 * math.pi * root * time_values)
        pad += brightness * 0.12 * np.sin(6.0 * math.pi * root * time_values)
        low = 0.45 * np.sin(2.0 * math.pi * bass * time_values)
        audio[start:end] += (pad + low) * envelope * energy
        if frame_index % 4 == 0:
            kick_length = min(end - start, int(0.12 * sample_rate))
            kick_time = np.arange(kick_length, dtype="float32") / sample_rate
            audio[start : start + kick_length] += np.sin(2.0 * math.pi * (90.0 - 35.0 * kick_time) * kick_time) * np.exp(-kick_time * 24.0) * energy
    peak = max(1e-6, float(np.max(np.abs(audio))))
    audio = np.tanh(audio / peak * 1.15) * 0.82
    with wave.open(str(destination), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes((audio * 32767).astype("<i2").tobytes())
    return destination
