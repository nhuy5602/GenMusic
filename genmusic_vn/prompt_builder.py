from __future__ import annotations

from .schemas import EmotionProfile, HarmonyPlan, LyricDraft, VocalPlan
from .stylebank import get_emotion_music, match_genre_template, stylebank_prompt_context


MOOD_EN = {
    "joy": "bright, joyful, optimistic",
    "sadness": "melancholic, tender, intimate",
    "anger": "intense, dark, driving",
    "fear": "suspenseful, cold, uneasy",
    "calm": "peaceful, warm, minimal",
    "romantic": "romantic, soft, heartfelt",
    "hope": "hopeful, cinematic, forward moving",
    "nostalgic": "nostalgic, warm, slightly bittersweet",
}


def build_music_prompt(
    emotion: EmotionProfile,
    harmony: HarmonyPlan,
    lyrics: LyricDraft,
    vocal: VocalPlan,
    genre: str | None = None,
) -> tuple[str, str]:
    genre_text = genre or "Vietnamese cinematic pop text-to-song"
    chords = " - ".join(harmony.chord_progression)
    instruments = ", ".join(harmony.instruments)
    arrangement = ", ".join(harmony.arrangement)
    traits = ", ".join(harmony.music_traits)
    mood = MOOD_EN.get(emotion.label, "warm, expressive")
    lyric_hint = " / ".join(lyrics.chorus[:2])
    lyric_excerpt = _lyric_excerpt(lyrics)
    song_form = " -> ".join(lyrics.song_form) if lyrics.song_form else "verse -> chorus -> bridge"
    emotion_style = get_emotion_music(emotion.label)
    genre_style = match_genre_template(genre, emotion.label)
    genre_text = _vocalize_style_text(genre or genre_style.get("prompt_prefix") or genre_text)
    stylebank_context = _vocalize_style_text(stylebank_prompt_context(
        emotion.label,
        list(emotion_style.get("vietnamese_instruments", [])),
        genre,
    ))

    prompt = (
        f"{genre_text}; {mood}; {traits}; {harmony.bpm} BPM; "
        f"{harmony.time_signature}; key {harmony.key} {harmony.scale}; "
        f"chord progression {chords}; instruments: {instruments}; "
        f"arrangement: {arrangement}; clear melodic motif following Vietnamese speech-tone contours; "
        f"song form: {song_form}; "
        f"Vietnamese stylebank cues: {stylebank_context}; "
        f"lead vocal: {vocal.gender}, {vocal.register}, pitch center {vocal.pitch_center}, "
        f"comfortable range {vocal.range_low}-{vocal.range_high}, {vocal.delivery}, "
        f"{vocal.intensity} intensity; "
        f"sing in Vietnamese using the generated lyrics titled '{lyrics.title}' as the song frame; "
        f"lyric hook: '{lyric_hint}'; lyric lines: '{lyric_excerpt}'; "
        "original melody, clear vocal-forward mix, clean arrangement, natural ending"
    )
    negative = (
        "muddy mix, distorted clipping, harsh noise, wrong-language vocals, off-key melody, "
        "abrupt ending, copyrighted song imitation, low quality, robotic artifacts"
    )
    return prompt, negative


def _lyric_excerpt(lyrics: LyricDraft) -> str:
    lines = [
        line
        for line in lyrics.full_song
        if line.strip() and not line.startswith("[") and not line.startswith("[Title]")
    ]
    if not lines:
        lines = lyrics.verse + lyrics.chorus
    return " | ".join(lines[:8])


def _vocalize_style_text(text: str) -> str:
    replacements = {
        "without lead vocal": "with clear lead vocal",
        "no lead vocal": "with clear lead vocal",
        "background instrumental": "text-to-song production",
        "background music": "text-to-song production",
        "nostalgic instrumental": "nostalgic text-to-song",
        "pop instrumental": "pop text-to-song",
        "instrumental": "with lead vocal",
    }
    cleaned = text
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    return cleaned
