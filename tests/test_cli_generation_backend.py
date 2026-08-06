from __future__ import annotations

from unittest.mock import patch

import cli


LYRICS = "Một ngày mới bắt đầu trong nắng mai dịu dàng trên phố"


def test_generate_parser_defaults_to_v80_and_sixteen_seconds() -> None:
    args = cli.build_parser().parse_args(["generate", "--text", LYRICS])
    assert args.backend == "v80"
    assert args.duration == 16


def test_generate_default_calls_v80_not_research_stager(capsys) -> None:
    response = {
        "status": "staged",
        "backend": "genmusic-native-waveform-v80",
        "model": "native-waveform-v80",
    }
    with (
        patch("cli.submit_native_waveform_job", return_value=response) as native,
        patch("cli.submit_text_to_music_job") as research,
    ):
        assert cli.main([
            "generate",
            "--text",
            LYRICS,
            "--no-submit",
        ]) == 0
    assert native.call_count == 1
    assert research.call_count == 0
    assert native.call_args.kwargs["duration_seconds"] == 16
    assert "native-waveform-v80" in capsys.readouterr().out


def test_generate_research_backend_is_explicit_opt_in(capsys) -> None:
    response = {
        "status": "staged",
        "backend": "genmusic-vn-self-diffusion",
    }
    with (
        patch("cli.submit_native_waveform_job") as native,
        patch("cli.submit_text_to_music_job", return_value=response) as research,
    ):
        assert cli.main([
            "generate",
            "--backend",
            "cfm-research",
            "--text",
            LYRICS,
            "--no-submit",
        ]) == 0
    assert native.call_count == 0
    assert research.call_count == 1
    assert "self-diffusion" in capsys.readouterr().out
