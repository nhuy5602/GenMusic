from __future__ import annotations

from .schemas import EmotionProfile, HarmonyPlan, LyricDraft, ScenePlan, VocalPlan
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
    scene: ScenePlan | None = None,
    source_keywords: list[str] | None = None,
    source_excerpt: str = "",
    genre: str | None = None,
) -> tuple[str, str]:
    genre_text = genre or "Vietnamese cinematic pop text-to-song"
    chords = " - ".join(harmony.chord_progression)
    instruments = ", ".join(harmony.instruments)
    arrangement = ", ".join(harmony.arrangement)
    traits = ", ".join(harmony.music_traits)
    mood = MOOD_EN.get(emotion.label, "warm, expressive")
    lyric_hint = " / ".join(lyrics.chorus[:2])
    song_form = " -> ".join(lyrics.song_form) if lyrics.song_form else "verse -> chorus -> bridge"
    scene_cues = ", ".join(scene.prompt_cues) if scene else ""
    scene_arrangement = ", ".join(scene.arrangement_cues) if scene else ""
    scene_mix = ", ".join(scene.mix_cues) if scene else ""
    keyword_hint = ", ".join((source_keywords or [])[:8])
    excerpt_hint = " ".join(source_excerpt.split())[:260]
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
        f"scene cues: {scene_cues}; "
        f"{harmony.time_signature}; key {harmony.key} {harmony.scale}; "
        f"chord progression {chords}; instruments: {instruments}; "
        f"arrangement: {arrangement}; {scene_arrangement}; clear melodic motif following Vietnamese speech-tone contours; "
        f"song form: {song_form}; "
        f"Vietnamese stylebank cues: {stylebank_context}; "
        f"source keywords: {keyword_hint}; "
        f"source text images: '{excerpt_hint}'; "
        f"vocal plan: {vocal.gender}, {vocal.register}, pitch center {vocal.pitch_center}, "
        f"comfortable range {vocal.range_low}-{vocal.range_high}, {vocal.delivery}, "
        f"{vocal.intensity} intensity; "
        "compose a singer-ready melody for the Vietnamese lyric sheet; "
        f"lyric hook reference: '{lyric_hint}'; "
        f"mix target: {scene_mix}, wide stereo, gentle reverb, vocal and backing start together; "
        "clean accompaniment, optional soft wordless humming only, no garbled sung words, "
        "no fast drums unless the emotion is energetic, no cheerful melody for sad text, "
        "original melody, clean arrangement, natural ending"
    )
    negative = (
        "muddy mix, distorted clipping, harsh noise, wrong-language vocals, off-key melody, "
        "garbled lyric singing, unintelligible words, harsh lead vocal, abrupt ending, "
        "mono narrow mix, vocal-only intro, backing enters late, cheerful melody over sad text, "
        "copyrighted song imitation, low quality, robotic artifacts"
    )
    return prompt, negative


def _vocalize_style_text(text: str) -> str:
    replacements = {
        "without lead vocal": "without garbled lead vocal",
        "no lead vocal": "without garbled lead vocal",
        "background instrumental": "song arrangement",
        "background music": "song arrangement",
        "nostalgic instrumental": "nostalgic song arrangement",
        "pop instrumental": "pop song arrangement",
        "instrumental": "song arrangement",
    }
    cleaned = text
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    return cleaned
