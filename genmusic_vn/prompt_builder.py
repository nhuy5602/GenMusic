from __future__ import annotations

from .schemas import EmotionProfile, HarmonyPlan, LyricDraft
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
    genre: str | None = None,
) -> tuple[str, str]:
    genre_text = genre or "Vietnamese cinematic pop background instrumental"
    chords = " - ".join(harmony.chord_progression)
    instruments = ", ".join(harmony.instruments)
    arrangement = ", ".join(harmony.arrangement)
    traits = ", ".join(harmony.music_traits)
    mood = MOOD_EN.get(emotion.label, "warm, expressive")
    lyric_hint = " / ".join(lyrics.chorus[:2])
    emotion_style = get_emotion_music(emotion.label)
    genre_style = match_genre_template(genre, emotion.label)
    genre_text = genre or genre_style.get("prompt_prefix") or genre_text
    stylebank_context = stylebank_prompt_context(
        emotion.label,
        list(emotion_style.get("vietnamese_instruments", [])),
        genre,
    )

    prompt = (
        f"{genre_text}; {mood}; {traits}; {harmony.bpm} BPM; "
        f"{harmony.time_signature}; key {harmony.key} {harmony.scale}; "
        f"chord progression {chords}; instruments: {instruments}; "
        f"arrangement: {arrangement}; clear melodic motif following Vietnamese speech-tone contours; "
        f"Vietnamese stylebank cues: {stylebank_context}; "
        f"background music for Vietnamese lyrics titled '{lyrics.title}', hook idea: '{lyric_hint}'; "
        "no lead vocal, no copyrighted melody, clean mix, loopable ending"
    )
    negative = (
        "muddy mix, distorted clipping, harsh noise, random vocals, off-key melody, "
        "abrupt ending, copyrighted song imitation, low quality"
    )
    return prompt, negative
