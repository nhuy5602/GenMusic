from __future__ import annotations

import inspect
import json
from pathlib import Path

import torch

from src.data import preprocess_aligned_vietnamese as aligned
from src.data.preprocess_raw_vietnamese import process_file


def test_process_file_exposes_compute_style_control() -> None:
    parameter = inspect.signature(process_file).parameters["compute_style"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is True


def test_aligned_raw_mode_is_forwarded_and_persisted(
    tmp_path: Path,
    monkeypatch,
) -> None:
    rows = [
        {
            "chunk_id": f"chunk{index:02d}",
            "song_id": f"song{index:02d}",
            "chunk_start_ms": 0,
            "chunk_end_ms": 10_000,
            "chunk_word_timestamps": [
                [
                    {"word": word, "start": 200 + position * 1_200, "end": 900 + position * 1_200}
                    for position, word in enumerate(
                        ("xin", "chào", "ngày", "mới", "bình", "yên")
                    )
                ]
            ],
            "audio": {"bytes": b"RIFF-not-decoded-by-test"},
        }
        for index in range(8)
    ]
    monkeypatch.setattr(aligned, "iter_parquet_rows", lambda _: iter(rows))
    monkeypatch.setattr(aligned, "run_demucs_batch", lambda *args, **kwargs: True)
    calls: list[bool] = []

    def fake_process_file(audio_path, output_root, *args, raw_audio=False, **kwargs):
        calls.append(bool(raw_audio))
        stem = Path(audio_path).stem
        waveforms = Path(output_root) / "waveforms"
        waveforms.mkdir(parents=True, exist_ok=True)
        torch.save(torch.zeros(48_000), waveforms / f"{stem}_vocal.pt")
        torch.save(torch.zeros(48_000), waveforms / f"{stem}_backing.pt")
        torch.save(torch.zeros(512), waveforms / f"{stem}_style.pt")
        return {
            "id": stem,
            "demucs_separated": True,
            "vocal_wav_path": f"waveforms/{stem}_vocal.pt",
            "backing_wav_path": f"waveforms/{stem}_backing.pt",
            "style_embed_path": f"waveforms/{stem}_style.pt",
            "frames": 48_000,
        }

    monkeypatch.setattr(aligned, "process_file", fake_process_file)
    output = tmp_path / "dataset"
    report = aligned.preprocess_aligned_parquet(
        tmp_path / "unused.parquet",
        output,
        max_records=8,
        max_chunks_per_song=1,
        batch_size=1,
        raw_audio=True,
    )
    assert calls == [True] * 8
    assert report["raw_audio_mode"] is True
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert config["raw_audio_mode"] is True
    record = json.loads(
        (output / "records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert record["exact_word_timestamps"] is True
    assert record["vocal_wav_path"].endswith("_vocal.pt")
    assert not (output / "incoming").exists()
