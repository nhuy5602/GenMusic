from __future__ import annotations

import array
import math
import shutil
import struct
import subprocess
import wave
from pathlib import Path

from genmusic_vn.core.music_theory import chord_midis, midi_frequency
from genmusic_vn.core.schemas import GeneratedFile

from .base import GeneratorInput, MusicGenerator


class GuideTrackGenerator(MusicGenerator):
    backend_name = "custom"

    def generate(self, data: GeneratorInput, output_dir: Path) -> list[GeneratedFile]:
        output_dir.mkdir(parents=True, exist_ok=True)
        wav_path = output_dir / "backing.wav"
        midi_path = output_dir / "song.mid"
        mp3_path = output_dir / "final.mp3"
        render_wav(data, wav_path)
        render_midi(data, midi_path)
        files = [
            GeneratedFile(kind="backing", path=str(wav_path), description="WAV nhạc nền từ custom composer"),
            GeneratedFile(kind="midi", path=str(midi_path), description="Phác thảo MIDI chord, bass, drum và melody từ custom composer"),
        ]
        if _convert_to_mp3(wav_path, mp3_path):
            files.insert(0, GeneratedFile(kind="audio", path=str(mp3_path), description="MP3 cuối từ custom composer"))
        return files


def _convert_to_mp3(wav_path: Path, mp3_path: Path) -> bool:
    ffmpeg = _ffmpeg_path()
    if not ffmpeg:
        return False
    try:
        subprocess.run(
            [ffmpeg, "-y", "-i", str(wav_path), "-codec:a", "libmp3lame", "-qscale:a", "2", str(mp3_path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return mp3_path.exists()


def _ffmpeg_path() -> str | None:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


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

    half_beat_count = int(data.duration_seconds / (beat / 2.0))
    for beat_index in range(int(data.duration_seconds / beat)):
        start = beat_index * beat
        if beat_index % 4 == 0 or data.emotion.energy > 0.68:
            _add_kick(samples, sample_rate, start, drum_volume)
        if beat_index % 4 == 2 and data.emotion.energy >= 0.42:
            _add_snare(samples, sample_rate, start, drum_volume * 0.72)
    for half_index in range(half_beat_count):
        if data.emotion.energy >= 0.52:
            _add_hat(samples, sample_rate, half_index * beat * 0.5, drum_volume * 0.35)

    peak = max(0.001, max(abs(sample) for sample in samples))
    scale = min(0.95 / peak, 1.0)
    stereo_values = []
    delay = int(0.006 * sample_rate)
    for index, sample in enumerate(samples):
        delayed = samples[index - delay] if index >= delay else 0.0
        left = max(-1.0, min(1.0, sample * scale))
        right = max(-1.0, min(1.0, (sample * 0.84 + delayed * 0.16) * scale))
        stereo_values.extend([int(left * 32767), int(right * 32767)])
    pcm = array.array("h", stereo_values)

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
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


def _add_snare(samples: list[float], sample_rate: int, start_seconds: float, volume: float) -> None:
    start = int(start_seconds * sample_rate)
    length = int(0.11 * sample_rate)
    for offset in range(length):
        sample_index = start + offset
        if sample_index >= len(samples):
            break
        time = offset / sample_rate
        envelope = math.exp(-time * 34.0)
        tonal = math.sin(2.0 * math.pi * 185.0 * time) * 0.35
        noise = ((offset * 1103515245 + 12345) & 0xFFFF) / 32768.0 - 1.0
        samples[sample_index] += (tonal + noise * 0.65) * envelope * volume


def _add_hat(samples: list[float], sample_rate: int, start_seconds: float, volume: float) -> None:
    start = int(start_seconds * sample_rate)
    length = int(0.045 * sample_rate)
    for offset in range(length):
        sample_index = start + offset
        if sample_index >= len(samples):
            break
        time = offset / sample_rate
        envelope = math.exp(-time * 85.0)
        noise = ((offset * 1664525 + 1013904223) & 0xFFFF) / 32768.0 - 1.0
        samples[sample_index] += noise * envelope * volume


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
    events.append((0, 1, bytes([0xC2, 33])))  # Fingered bass

    for bar_index in range(max(1, math.ceil(data.duration_seconds / bar_seconds))):
        chord = data.harmony.chord_progression[bar_index % len(data.harmony.chord_progression)]
        start_tick = int(bar_index * 4 * ticks_per_quarter)
        end_tick = int((bar_index + 1) * 4 * ticks_per_quarter)
        root = chord_midis(chord, 2)[0]
        events.append((start_tick, 2, bytes([0x92, root, 64])))
        events.append((end_tick, 0, bytes([0x82, root, 0])))
        for midi in chord_midis(chord, 3):
            events.append((start_tick, 2, bytes([0x91, midi, 52])))
            events.append((end_tick, 0, bytes([0x81, midi, 0])))

    beat_count = int(data.duration_seconds / beat)
    for beat_index in range(beat_count):
        tick = int(beat_index * ticks_per_quarter)
        if beat_index % 4 == 0 or data.emotion.energy > 0.68:
            events.append((tick, 4, bytes([0x99, 36, 78])))
            events.append((tick + int(ticks_per_quarter * 0.18), 0, bytes([0x89, 36, 0])))
        if beat_index % 4 == 2 and data.emotion.energy >= 0.42:
            events.append((tick, 4, bytes([0x99, 38, 58])))
            events.append((tick + int(ticks_per_quarter * 0.14), 0, bytes([0x89, 38, 0])))
        if data.emotion.energy >= 0.52:
            hat_tick = tick + int(ticks_per_quarter * 0.5)
            events.append((hat_tick, 4, bytes([0x99, 42, 42])))
            events.append((hat_tick + int(ticks_per_quarter * 0.08), 0, bytes([0x89, 42, 0])))

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
