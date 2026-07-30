from __future__ import annotations

import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.request import Request, urlopen

import pytest

import server
from scripts.generation.run_kaggle_native_waveform import _kernel_code
from src.audio.native_waveform import target_words_from_text
from src.integrations.native_waveform_auto import (
    NATIVE_WAVEFORM_BACKEND,
    NativeWaveformJobConfig,
    stage_native_waveform_job,
)

LYRICS = "một chiều mưa tôi nhớ về con phố cũ"


def test_user_lyrics_become_line_aware_target_words() -> None:
    words = target_words_from_text("một chiều mưa tôi\nnhớ về phố cũ")
    assert len(words) == 8
    assert [word["segment_index"] for word in words] == [
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
    ]
    with pytest.raises(ValueError, match="at least eight"):
        target_words_from_text("quá ngắn")


def test_native_kernel_receives_text_genre_and_validated_duration() -> None:
    code = _kernel_code(
        patch_sha256="A" * 64,
        patch_tree_sha256="B" * 64,
        text=LYRICS,
        genre="Vietnamese pop",
        duration_seconds=16,
    )
    assert f"REQUEST_TEXT = {LYRICS!r}" in code
    assert "REQUEST_GENRE = 'Vietnamese pop'" in code
    assert "REQUEST_DURATION_SECONDS = 16.0" in code
    assert 'command.extend(["--text", REQUEST_TEXT])' in code
    assert "scripts/generation/generate_native.py" in code


def test_staged_web_job_uses_native_private_kaggle_request(
    tmp_path: Path,
) -> None:
    state = stage_native_waveform_job(
        text=LYRICS,
        output_root=tmp_path,
        duration_seconds=30,
        genre="Vietnamese pop",
        config=NativeWaveformJobConfig(submit=False),
    )
    assert state["status"] == "staged"
    assert state["backend"] == NATIVE_WAVEFORM_BACKEND
    assert state["duration_seconds"] == 16
    assert state["requested_duration_seconds"] == 30
    assert state["job_kind"] == "native_waveform_generation"
    metadata = json.loads(
        (Path(state["kernel_dir"]) / "kernel-metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["is_private"] == "true"
    assert metadata["kernel_sources"] == [state["raw_kernel_ref"]]
    kernel_code = (
        Path(state["kernel_dir"]) / metadata["code_file"]
    ).read_text(encoding="utf-8")
    assert f"REQUEST_TEXT = {LYRICS!r}" in kernel_code


def test_web_generate_route_calls_native_pipeline() -> None:
    response = {
        "status": "submitted",
        "run_id": "native-test",
        "backend": NATIVE_WAVEFORM_BACKEND,
        "model": "native-waveform-v80",
    }
    httpd = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        server.GenMusicHandler,
    )
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        with patch(
            "server.submit_native_waveform_job",
            return_value=response,
        ) as submit:
            request = Request(
                f"http://127.0.0.1:{httpd.server_port}/api/generate",
                data=json.dumps({
                    "text": LYRICS,
                    "duration_seconds": 16,
                    "genre": "Vietnamese pop",
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=5) as result:
                payload = json.loads(result.read().decode("utf-8"))
        assert payload == response
        assert submit.call_args.kwargs["text"] == LYRICS
        assert isinstance(
            submit.call_args.kwargs["config"],
            NativeWaveformJobConfig,
        )
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
