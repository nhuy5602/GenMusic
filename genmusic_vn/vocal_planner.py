from __future__ import annotations

from .schemas import EmotionProfile, HarmonyPlan, VocalPlan
from .text_utils import tokenize_words


FEMALE_WORD_CUES = {"em", "cô", "mẹ", "nàng", "chị"}
MALE_WORD_CUES = {"cha", "bố", "ông", "chàng"}
FEMALE_PHRASE_CUES = ("con gai", "con gái", "nguoi con gai", "người con gái")
MALE_PHRASE_CUES = ("con trai", "nguoi con trai", "người con trai")
LIGHT_AFTER_ANH = {"den", "đèn", "sang", "sáng", "mat", "mặt", "trang", "trăng"}


def build_vocal_plan(text: str, emotion: EmotionProfile, harmony: HarmonyPlan) -> VocalPlan:
    tokens = tokenize_words(text)
    lowered = text.lower()
    female_score = sum(1 for token in tokens if token in FEMALE_WORD_CUES)
    male_score = sum(1 for token in tokens if token in MALE_WORD_CUES)
    for index, token in enumerate(tokens):
        next_token = tokens[index + 1] if index + 1 < len(tokens) else ""
        if token == "anh" and next_token not in LIGHT_AFTER_ANH:
            male_score += 1
    female_score += sum(2 for phrase in FEMALE_PHRASE_CUES if phrase in lowered)
    male_score += sum(2 for phrase in MALE_PHRASE_CUES if phrase in lowered)

    rationale: list[str] = []
    if female_score and male_score:
        gender = "duet"
        rationale.append("Text has both female and male address cues.")
    elif female_score > male_score:
        gender = "female"
        rationale.append("Text leans toward a female narrative voice.")
    elif male_score > female_score:
        gender = "male"
        rationale.append("Text leans toward a male narrative voice.")
    else:
        gender = _emotion_default_gender(emotion)
        rationale.append(f"Gender chosen from emotion '{emotion.label}'.")

    if gender == "duet":
        register = "male tenor and female alto"
        range_low = "A2"
        range_high = "E5"
        pitch_center = _keyed_pitch(harmony, octave=4)
    elif gender == "female":
        register, range_low, range_high, octave = _female_register(emotion)
        pitch_center = _keyed_pitch(harmony, octave=octave)
    else:
        register, range_low, range_high, octave = _male_register(emotion)
        pitch_center = _keyed_pitch(harmony, octave=octave)

    delivery = _delivery(emotion, gender)
    intensity = _intensity(emotion)
    rationale.append(f"Pitch follows {harmony.key} {harmony.scale} at {harmony.bpm} BPM.")
    rationale.append(f"Delivery follows {emotion.label_vi} mood and energy {emotion.energy:.2f}.")

    return VocalPlan(
        gender=gender,
        register=register,
        pitch_center=pitch_center,
        range_low=range_low,
        range_high=range_high,
        delivery=delivery,
        intensity=intensity,
        rationale=rationale,
    )


def _emotion_default_gender(emotion: EmotionProfile) -> str:
    if emotion.label in {"romantic", "sadness", "nostalgic", "calm"}:
        return "female"
    if emotion.label == "anger" and emotion.energy >= 0.6:
        return "male"
    if emotion.label == "fear":
        return "female"
    if emotion.label in {"joy", "hope"} and emotion.valence >= 0.1:
        return "female"
    return "male"


def _female_register(emotion: EmotionProfile) -> tuple[str, str, str, int]:
    if emotion.label in {"sadness", "nostalgic", "calm", "fear"}:
        return "alto", "G3", "D5", 4
    if emotion.energy >= 0.7:
        return "mezzo-soprano", "A3", "F5", 4
    return "mezzo-soprano", "A3", "E5", 4


def _male_register(emotion: EmotionProfile) -> tuple[str, str, str, int]:
    if emotion.label in {"sadness", "nostalgic", "calm"}:
        return "baritone", "A2", "D4", 3
    if emotion.energy >= 0.65:
        return "tenor", "C3", "G4", 3
    return "warm tenor", "B2", "F4", 3


def _keyed_pitch(harmony: HarmonyPlan, octave: int) -> str:
    key = harmony.key.strip()
    if not key:
        key = "C"
    root = key[0].upper() + key[1:]
    if root.endswith("m"):
        root = root[:-1]
    if root not in {"A", "B", "C", "D", "E", "F", "G", "Ab", "Bb", "Db", "Eb", "Gb", "F#", "C#"}:
        root = "C"
    return f"{root}{octave}"


def _delivery(emotion: EmotionProfile, gender: str) -> str:
    voice = {
        "female": "female Vietnamese singer timbre",
        "male": "male Vietnamese singer timbre",
        "duet": "male and female Vietnamese duet",
    }.get(gender, "Vietnamese singer timbre")

    if emotion.label in {"sadness", "nostalgic"}:
        return f"soft, breathy, intimate {voice}"
    if emotion.label == "romantic":
        return f"warm, tender, close-mic {voice}"
    if emotion.label in {"joy", "hope"}:
        return f"bright, open, uplifting {voice}"
    if emotion.label == "anger":
        return f"firm, gritty, driving {voice}"
    if emotion.label == "fear":
        return f"thin, tense, whisper-like {voice}"
    return f"gentle, airy, natural {voice}"


def _intensity(emotion: EmotionProfile) -> str:
    if emotion.energy >= 0.68:
        return "high"
    if emotion.energy <= 0.38:
        return "low"
    return "medium"
