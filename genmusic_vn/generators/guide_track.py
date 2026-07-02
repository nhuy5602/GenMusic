from __future__ import annotations

import array
import math
import struct
import wave
from pathlib import Path

from genmusic_vn.music_theory import chord_midis, midi_frequency
from genmusic_vn.schemas import GeneratedFile

from .base import GeneratorInput, MusicGenerator


class GuideTrackGenerator(MusicGenerator):
    backend_name = "guide"

    def generate(self, data: GeneratorInput, output_dir: Path) -> list[GeneratedFile]:
        output_dir.mkdir(parents=True, exist_ok=True)
        wav_path = output_dir / "guide.wav"
        midi_path = output_dir / "guide.mid"
        render_wav(data, wav_path)
        render_midi(data, midi_path)
        return [
            GeneratedFile(kind="audio", path=str(wav_path), description="Local guide track WAV"),
            GeneratedFile(kind="midi", path=str(midi_path), description="Chord and melody MIDI sketch"),
        ]


def render_wav(data: GeneratorInput, path: Path, sample_rate: int = 44100) -> None:
    total_samples = max(sample_rate, int(data.duration_seconds * sample_rate))
    samples = [0.0] * total_samples
    beat = 60.0 / data.harmony.bpm
    bar = beat * 4.0

    chord_volume = 0.055 + max(0.0, data.emotion.energy) * 0.025
    melody_volume = 0.16
    drum_volume = 0.16 if data.emotion.energy >= 0.52 else 0.06

    for bar_index in range(max(1, math.ceil(data.duration_seconds / bar))):
        chord = data.harmony.chord_progression[bar_index % len(data.harmony.chord_progression)]
        start = bar_index * bar
        duration = min(bar, data.duration_seconds - start)
        if duration <= 0:
            break
        for midi in chord_midis(chord, 3):
            _add_tone(samples, sample_rate, start, duration, midi, chord_volume, harmonic=True)
        root = chord_midis(chord, 2)[0]
        _add_tone(samples, sample_rate, start, duration, root, chord_volume * 0.9, harmonic=False)

    for event in data.melody:
        _add_tone(samples, sample_rate, event.start, event.duration, event.midi, melody_volume, harmonic=True)

    for beat_index in range(int(data.duration_seconds / beat)):
        start = beat_index * beat
        if beat_index % 4 == 0 or data.emotion.energy > 0.68:
            _add_kick(samples, sample_rate, start, drum_volume)

    peak = max(0.001, max(abs(sample) for sample in samples))
    scale = min(0.95 / peak, 1.0)
    pcm = array.array("h", (int(max(-1.0, min(1.0, sample * scale)) * 32767) for sample in samples))

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def _add_tone(
    samples: list[float],
    sample_rate: int,
    start_seconds: float,
    duration_seconds: float,
    midi: int,
    volume: float,
    harmonic: bool,
) -> None:
    start = max(0, int(start_seconds * sample_rate))
    end = min(len(samples), int((start_seconds + duration_seconds) * sample_rate))
    if end <= start:
        return

    frequency = midi_frequency(midi)
    length = end - start
    attack = max(1, int(min(0.03, duration_seconds * 0.2) * sample_rate))
    release = max(1, int(min(0.12, duration_seconds * 0.25) * sample_rate))

    for offset, sample_index in enumerate(range(start, end)):
        time = offset / sample_rate
        envelope = 1.0
        if offset < attack:
            envelope = offset / attack
        elif offset > length - release:
            envelope = max(0.0, (length - offset) / release)

        value = math.sin(2.0 * math.pi * frequency * time)
        if harmonic:
            value += 0.22 * math.sin(2.0 * math.pi * frequency * 2.0 * time)
            value += 0.08 * math.sin(2.0 * math.pi * frequency * 3.0 * time)
        samples[sample_index] += value * envelope * volume


def _add_kick(samples: list[float], sample_rate: int, start_seconds: float, volume: float) -> None:
    start = int(start_seconds * sample_rate)
    length = int(0.12 * sample_rate)
    for offset in range(length):
        sample_index = start + offset
        if sample_index >= len(samples):
            break
        time = offset / sample_rate
        envelope = math.exp(-time * 28.0)
        frequency = 62.0 - 24.0 * min(1.0, time / 0.12)
        samples[sample_index] += math.sin(2.0 * math.pi * frequency * time) * envelope * volume


def render_midi(data: GeneratorInput, path: Path) -> None:
    ticks_per_quarter = 480
    beat = 60.0 / data.harmony.bpm
    bar_seconds = beat * 4.0

    tempo_track = bytearray()
    tempo = int(60_000_000 / data.harmony.bpm)
    tempo_track.extend(_vlq(0) + b"\xff\x51\x03" + tempo.to_bytes(3, "big"))
    tempo_track.extend(_vlq(0) + b"\xff\x58\x04\x04\x02\x18\x08")
    tempo_track.extend(_vlq(0) + b"\xff\x2f\x00")

    events: list[tuple[int, int, bytes]] = []
    events.append((0, 1, bytes([0xC0, 0])))   # Acoustic grand piano
    events.append((0, 1, bytes([0xC1, 48])))  # String ensemble

    for bar_index in range(max(1, math.ceil(data.duration_seconds / bar_seconds))):
        chord = data.harmony.chord_progression[bar_index % len(data.harmony.chord_progression)]
        start_tick = int(bar_index * 4 * ticks_per_quarter)
        end_tick = int((bar_index + 1) * 4 * ticks_per_quarter)
        for midi in chord_midis(chord, 3):
            events.append((start_tick, 2, bytes([0x91, midi, 52])))
            events.append((end_tick, 0, bytes([0x81, midi, 0])))

    for event in data.melody:
        start_tick = int((event.start / beat) * ticks_per_quarter)
        end_tick = int(((event.start + event.duration) / beat) * ticks_per_quarter)
        events.append((start_tick, 3, bytes([0x90, event.midi, event.velocity])))
        events.append((max(start_tick + 1, end_tick), 0, bytes([0x80, event.midi, 0])))

    events.sort(key=lambda item: (item[0], item[1]))
    music_track = bytearray()
    last_tick = 0
    for tick, _, payload in events:
        music_track.extend(_vlq(max(0, tick - last_tick)))
        music_track.extend(payload)
        last_tick = tick
    music_track.extend(_vlq(0) + b"\xff\x2f\x00")

    with path.open("wb") as midi:
        midi.write(b"MThd")
        midi.write(struct.pack(">IHHH", 6, 1, 2, ticks_per_quarter))
        _write_track(midi, tempo_track)
        _write_track(midi, music_track)


def _write_track(handle, data: bytearray) -> None:
    handle.write(b"MTrk")
    handle.write(struct.pack(">I", len(data)))
    handle.write(data)


def _vlq(value: int) -> bytes:
    value = max(0, int(value))
    buffer = value & 0x7F
    value >>= 7
    while value:
        buffer <<= 8
        buffer |= ((value & 0x7F) | 0x80)
        value >>= 7

    output = bytearray()
    while True:
        output.append(buffer & 0xFF)
        if buffer & 0x80:
            buffer >>= 8
        else:
            break
    return bytes(output)

