from __future__ import annotations

import json
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path

from .emotion import analyze_emotion
from .generators.base import GeneratorInput, MusicGenerator
from .generators.guide_track import GuideTrackGenerator
from .lyric_writer import rewrite_lyrics
from .music_theory import build_harmony, build_melody_events
from .prompt_builder import build_music_prompt
from .schemas import GeneratedFile, MusicResult, to_plain_data
from .text_planner import build_text_plan
from .text_utils import normalize_text
from .vocal_planner import build_vocal_plan


def make_run_id(text: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    digest = sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"{timestamp}-{digest}"


def get_generator(backend: str) -> MusicGenerator:
    if backend == "guide":
        return GuideTrackGenerator()
    raise ValueError(f"Unsupported backend: {backend}")


def create_music_project(
    text: str,
    output_root: str | Path = "outputs",
    backend: str = "guide",
    duration_seconds: int = 30,
    genre: str | None = None,
    render_audio: bool = True,
) -> MusicResult:
    normalized = normalize_text(text)
    if not normalized:
        raise ValueError("Input text is empty.")
    duration_seconds = max(6, min(180, int(duration_seconds)))

    run_id = make_run_id(normalized)
    run_dir = Path(output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    text_plan = build_text_plan(normalized)
    generation_text = text_plan.condensed_text or normalized
    emotion = analyze_emotion(normalized)
    harmony = build_harmony(emotion, duration_seconds)
    melody = build_melody_events(generation_text, harmony, duration_seconds)
    lyrics = rewrite_lyrics(generation_text, emotion, harmony)
    vocal = build_vocal_plan(normalized, emotion, harmony)
    prompt, negative_prompt = build_music_prompt(emotion, harmony, lyrics, vocal, genre=genre)

    generator_input = GeneratorInput(
        run_id=run_id,
        text=normalized,
        prompt=prompt,
        negative_prompt=negative_prompt,
        emotion=emotion,
        harmony=harmony,
        lyrics=lyrics,
        vocal=vocal,
        melody=melody,
        duration_seconds=duration_seconds,
    )

    files: list[GeneratedFile] = []
    if render_audio:
        files.extend(get_generator(backend).generate(generator_input, run_dir))

    result = MusicResult(
        run_id=run_id,
        input_text=normalized,
        backend=backend,
        duration_seconds=duration_seconds,
        emotion=emotion,
        harmony=harmony,
        melody=melody,
        lyrics=lyrics,
        vocal=vocal,
        text_plan=text_plan,
        prompt=prompt,
        negative_prompt=negative_prompt,
        files=files,
    )

    prompt_pack_path = run_dir / "prompt_pack.json"
    prompt_pack_path.write_text(json.dumps(_prompt_pack(result), ensure_ascii=False, indent=2), encoding="utf-8")
    files.append(GeneratedFile(kind="prompt_pack", path=str(prompt_pack_path), description="Kaggle prompt pack"))

    report_path = run_dir / "report.json"
    report_path.write_text(json.dumps(to_plain_data(result), ensure_ascii=False, indent=2), encoding="utf-8")
    files.append(GeneratedFile(kind="report", path=str(report_path), description="Full pipeline report"))

    final_result = MusicResult(
        run_id=result.run_id,
        input_text=result.input_text,
        backend=result.backend,
        duration_seconds=result.duration_seconds,
        emotion=result.emotion,
        harmony=result.harmony,
        melody=result.melody,
        lyrics=result.lyrics,
        vocal=result.vocal,
        text_plan=result.text_plan,
        prompt=result.prompt,
        negative_prompt=result.negative_prompt,
        files=files,
    )
    report_path.write_text(json.dumps(to_plain_data(final_result), ensure_ascii=False, indent=2), encoding="utf-8")
    return final_result


def _prompt_pack(result: MusicResult) -> dict:
    return {
        "run_id": result.run_id,
        "duration_seconds": result.duration_seconds,
        "prompt": result.prompt,
        "negative_prompt": result.negative_prompt,
        "lyrics": to_plain_data(result.lyrics),
        "vocal": to_plain_data(result.vocal),
        "harmony": to_plain_data(result.harmony),
        "emotion": to_plain_data(result.emotion),
        "melody": to_plain_data(result.melody),
        "text_plan": to_plain_data(result.text_plan),
    }
