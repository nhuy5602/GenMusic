from __future__ import annotations

import math
import re

from .schemas import EmotionProfile, HarmonyPlan, NoteEvent
from .text_utils import tokenize_words


NOTE_TO_PC = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
}
PC_TO_NOTE = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
SCALE_INTERVALS = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
}
CHORD_INTERVALS = {
    "": [0, 4, 7],
    "m": [0, 3, 7],
    "maj7": [0, 4, 7, 11],
    "m7": [0, 3, 7, 10],
    "7": [0, 4, 7, 10],
    "dim": [0, 3, 6],
    "sus4": [0, 5, 7],
}

EMOTION_HARMONY = {
    "joy": {
        "key": "G",
        "scale": "major",
        "bpm": 122,
        "progression": ["G", "D", "Em", "C"],
        "register": "mid-high",
        "instruments": ["acoustic guitar", "bright piano", "warm bass", "light pop drums"],
        "arrangement": ["clean 4-bar intro", "uplifting chorus lift", "loopable ending"],
        "traits": ["bright", "optimistic", "clear rhythmic pulse"],
    },
    "sadness": {
        "key": "A",
        "scale": "minor",
        "bpm": 76,
        "progression": ["Am", "F", "C", "G"],
        "register": "mid-low",
        "instruments": ["soft piano", "warm strings", "subtle pad", "brush percussion"],
        "arrangement": ["sparse intro", "gentle build", "breathing pauses"],
        "traits": ["melancholic", "intimate", "tender"],
    },
    "anger": {
        "key": "E",
        "scale": "minor",
        "bpm": 138,
        "progression": ["Em", "C", "D", "Em"],
        "register": "mid",
        "instruments": ["distorted synth bass", "taiko-like hits", "aggressive drums", "dark strings"],
        "arrangement": ["sharp attack", "driving groove", "short breakdown"],
        "traits": ["intense", "dark", "percussive"],
    },
    "fear": {
        "key": "F#",
        "scale": "minor",
        "bpm": 88,
        "progression": ["F#m", "D", "Bm", "C#dim"],
        "register": "low-mid",
        "instruments": ["low drone", "prepared piano", "thin strings", "distant percussion"],
        "arrangement": ["slow tension", "uneven accents", "unresolved ending"],
        "traits": ["suspenseful", "cold", "uneasy"],
    },
    "calm": {
        "key": "D",
        "scale": "major",
        "bpm": 84,
        "progression": ["D", "Bm", "G", "A"],
        "register": "mid",
        "instruments": ["felt piano", "nylon guitar", "soft pad", "dan tranh texture"],
        "arrangement": ["soft intro", "steady pulse", "gentle loop"],
        "traits": ["peaceful", "warm", "minimal"],
    },
    "romantic": {
        "key": "A",
        "scale": "major",
        "bpm": 92,
        "progression": ["A", "C#m", "D", "E"],
        "register": "mid-high",
        "instruments": ["felt piano", "nylon guitar", "warm strings", "soft bass"],
        "arrangement": ["close intimate intro", "wide chorus", "delicate outro"],
        "traits": ["romantic", "soft", "heartfelt"],
    },
    "hope": {
        "key": "C",
        "scale": "major",
        "bpm": 108,
        "progression": ["C", "G", "Am", "F"],
        "register": "mid-high",
        "instruments": ["piano", "muted guitar", "light drums", "rising strings"],
        "arrangement": ["simple intro", "gradual lift", "open ending"],
        "traits": ["hopeful", "cinematic", "forward moving"],
    },
    "nostalgic": {
        "key": "F",
        "scale": "major",
        "bpm": 82,
        "progression": ["Fmaj7", "C", "Dm", "Bb"],
        "register": "mid",
        "instruments": ["upright piano", "vinyl-like texture", "soft strings", "muted guitar"],
        "arrangement": ["hazy intro", "gentle verse", "memory-like refrain"],
        "traits": ["nostalgic", "warm", "slightly bittersweet"],
    },
}

TONE_MARKS = {
    "sac": set("áắấéếíóốớúứý"),
    "huyen": set("àằầèềìòồờùừỳ"),
    "hoi": set("ảẳẩẻểỉỏổởủửỷ"),
    "nga": set("ãẵẫẽễĩõỗỡũữỹ"),
    "nang": set("ạặậẹệịọộợụựỵ"),
}
TONE_STEPS = {"ngang": 0, "sac": 2, "huyen": -2, "hoi": -1, "nga": 1, "nang": -3}


def note_name(pc: int) -> str:
    return PC_TO_NOTE[pc % 12]


def midi_to_note(midi: int) -> str:
    octave = midi // 12 - 1
    return f"{note_name(midi % 12)}{octave}"


def note_to_midi(note: str, octave: int = 4) -> int:
    return 12 * (octave + 1) + NOTE_TO_PC[note]


def parse_chord(chord: str) -> tuple[str, str]:
    match = re.match(r"^([A-G](?:#|b)?)(maj7|m7|dim|sus4|m|7)?$", chord)
    if not match:
        raise ValueError(f"Unsupported chord: {chord}")
    return match.group(1), match.group(2) or ""


def chord_notes(chord: str, octave: int = 4) -> list[str]:
    root, quality = parse_chord(chord)
    root_pc = NOTE_TO_PC[root]
    intervals = CHORD_INTERVALS[quality]
    return [f"{note_name(root_pc + interval)}{octave + ((root_pc + interval) // 12)}" for interval in intervals]


def chord_midis(chord: str, octave: int = 4) -> list[int]:
    root, quality = parse_chord(chord)
    root_midi = note_to_midi(note_name(NOTE_TO_PC[root]), octave)
    return [root_midi + interval for interval in CHORD_INTERVALS[quality]]


def scale_notes(key: str, scale: str, octave: int = 4) -> list[str]:
    root_pc = NOTE_TO_PC[key]
    return [f"{note_name(root_pc + interval)}{octave + ((root_pc + interval) // 12)}" for interval in SCALE_INTERVALS[scale]]


def build_harmony(emotion: EmotionProfile, duration_seconds: int = 30) -> HarmonyPlan:
    preset = EMOTION_HARMONY.get(emotion.label, EMOTION_HARMONY["calm"])
    bpm = preset["bpm"]
    if duration_seconds >= 45:
        bpm = max(64, bpm - 4)
    if emotion.energy > 0.75:
        bpm += 6
    elif emotion.energy < 0.3:
        bpm -= 4

    progression = list(preset["progression"])
    chord_map = {chord: chord_notes(chord) for chord in progression}
    return HarmonyPlan(
        key=preset["key"],
        scale=preset["scale"],
        bpm=int(bpm),
        time_signature="4/4",
        chord_progression=progression,
        chord_notes=chord_map,
        note_pool=scale_notes(preset["key"], preset["scale"]),
        melody_register=preset["register"],
        instruments=list(preset["instruments"]),
        arrangement=list(preset["arrangement"]),
        music_traits=list(preset["traits"]),
    )


def detect_vietnamese_tone(word: str) -> str:
    lowered = word.lower()
    for tone, chars in TONE_MARKS.items():
        if any(char in chars for char in lowered):
            return tone
    return "ngang"


def nearest_scale_midi(target: int, harmony: HarmonyPlan) -> int:
    root_pc = NOTE_TO_PC[harmony.key]
    pcs = {(root_pc + interval) % 12 for interval in SCALE_INTERVALS[harmony.scale]}
    candidates = [midi for midi in range(target - 12, target + 13) if midi % 12 in pcs]
    return min(candidates, key=lambda midi: abs(midi - target))


def build_melody_events(text: str, harmony: HarmonyPlan, duration_seconds: int) -> list[NoteEvent]:
    tokens = tokenize_words(text)
    if not tokens:
        tokens = ["la", "la", "la", "la"]

    beat_seconds = 60.0 / harmony.bpm
    root_midi = note_to_midi(harmony.key, 4)
    if "high" in harmony.melody_register:
        base = root_midi + 7
    elif "low" in harmony.melody_register:
        base = root_midi - 3
    else:
        base = root_midi + 2

    events: list[NoteEvent] = []
    current = nearest_scale_midi(base, harmony)
    start = 0.0
    max_start = max(1.0, duration_seconds - beat_seconds)

    for index, word in enumerate(tokens[:96]):
        tone = detect_vietnamese_tone(word)
        target = current + TONE_STEPS[tone]
        if index % 8 == 0:
            chord = harmony.chord_progression[(index // 8) % len(harmony.chord_progression)]
            chord_tones = chord_midis(chord, 4)
            target = min(chord_tones, key=lambda midi: abs(midi - target))
        current = nearest_scale_midi(target, harmony)
        current = max(root_midi - 7, min(root_midi + 19, current))

        duration = beat_seconds * (0.75 if len(word) > 4 else 0.5)
        velocity = 74 + min(18, len(word) * 2)
        events.append(
            NoteEvent(
                start=round(start, 3),
                duration=round(duration, 3),
                note=midi_to_note(current),
                midi=current,
                velocity=velocity,
                lyric=word,
            )
        )
        start += duration + beat_seconds * (0.08 if index % 4 else 0.16)
        if start >= max_start:
            break

    if not events:
        events.append(NoteEvent(start=0.0, duration=beat_seconds, note=midi_to_note(root_midi), midi=root_midi, velocity=80))
    return events


def midi_frequency(midi: int) -> float:
    return 440.0 * math.pow(2.0, (midi - 69) / 12.0)
