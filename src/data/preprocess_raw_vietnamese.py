import argparse
import json
import os
import subprocess
import shutil
import sys
from pathlib import Path
import numpy as np
import librosa
import torch
try:
    import whisper
except ImportError:  # Optional for the fast full-mix preprocessing mode.
    whisper = None

# Configure FFMPEG path using imageio_ffmpeg so that whisper/demucs can find it on Windows
try:
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dir = os.path.dirname(ffmpeg_exe)
    target_ffmpeg = os.path.join(ffmpeg_dir, "ffmpeg.exe")
    if not os.path.exists(target_ffmpeg):
        import shutil
        shutil.copy2(ffmpeg_exe, target_ffmpeg)
    os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]
except ImportError:
    pass

# Default parameters
SAMPLE_RATE = 16000
N_MELS = 64
N_FFT = 512
HOP_LENGTH = 256

def run_demucs_separation(audio_path: Path, output_dir: Path, device: str = "auto") -> tuple[Path | None, Path | None]:
    """Separate vocals and backing using Demucs CLI."""
    model_name = "htdemucs"
    song_folder = output_dir / model_name / audio_path.stem
    vocals_file = song_folder / "vocals.wav"
    backing_file = song_folder / "no_vocals.wav"
    if vocals_file.exists() and backing_file.exists():
        print(f"-> Reusing separated stems for: {audio_path.name}", flush=True)
        return vocals_file, backing_file

    print(f"-> Separating vocal/backing stems for: {audio_path.name}...", flush=True)
    requested = (device or "auto").lower()
    devices = [requested] if requested != "auto" else (["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"])
    last_error = "unknown error"
    for selected_device in devices:
        try:
            command = [sys.executable, "-m", "demucs.separate", "--two-stems=vocals", "-o", str(output_dir)]
            if selected_device:
                command.extend(["-d", selected_device])
            command.append(str(audio_path))
            subprocess.run(command, check=True)
            if vocals_file.exists() and backing_file.exists():
                return vocals_file, backing_file
            last_error = f"Demucs completed on {selected_device} without producing both stems."
        except Exception as exc:
            last_error = str(exc)
            if selected_device != devices[-1]:
                print(f"[WARNING] Demucs {selected_device} failed; retrying on {devices[-1]}.", flush=True)

    print(f"[WARNING] Demucs separation failed: {last_error}. Falling back to raw mix.", flush=True)
    
    return None, None


def run_demucs_batch(audio_paths: list[Path], output_dir: Path, device: str = "auto") -> bool:
    """Separate a small batch so Demucs loads its model once per batch."""
    if not audio_paths:
        return True

    model_name = "htdemucs"
    pending = [
        path for path in audio_paths
        if not (output_dir / model_name / path.stem / "vocals.wav").exists()
        or not (output_dir / model_name / path.stem / "no_vocals.wav").exists()
    ]
    if not pending:
        print(f"-> Reusing separated stems for batch ({len(audio_paths)} files).", flush=True)
        return True

    requested = (device or "auto").lower()
    devices = [requested] if requested != "auto" else (['cuda', 'cpu'] if torch.cuda.is_available() else ['cpu'])
    for selected_device in devices:
        command = [
            sys.executable,
            "-m",
            "demucs.separate",
            "--two-stems=vocals",
            "-o",
            str(output_dir),
        ]
        if selected_device:
            command.extend(["-d", selected_device])
        command.extend(str(path) for path in pending)
        print(
            f"-> Demucs batch {len(pending)} file(s) on {selected_device}; "
            "progress will appear below...",
            flush=True,
        )
        try:
            subprocess.run(command, check=True)
            missing = [
                path for path in pending
                if not (output_dir / model_name / path.stem / "vocals.wav").exists()
                or not (output_dir / model_name / path.stem / "no_vocals.wav").exists()
            ]
            if not missing:
                return True
            print(f"[WARNING] Demucs batch missed {len(missing)} file(s).", flush=True)
        except Exception as exc:
            print(f"[WARNING] Demucs batch on {selected_device} failed: {exc}", flush=True)
            if selected_device != devices[-1]:
                print(f"[WARNING] Retrying Demucs batch on {devices[-1]}.", flush=True)

    print("[WARNING] Falling back to per-file Demucs attempts for this batch.", flush=True)
    return False

def process_file(
    audio_path: Path,
    output_dir: Path,
    whisper_model,
    keep_separated: bool = False,
    *,
    use_demucs: bool = True,
    transcribe: bool = True,
    sample_rate: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    mel_power: float = 2.0,
    demucs_device: str = "auto",
) -> dict:
    sample_id = audio_path.stem
    print(f"\n==================== PROCESSING {sample_id} ====================", flush=True)
    
    separated_dir = output_dir / "separated"
    mels_dir = output_dir / "mels"
    pitch_dir = output_dir / "pitch"
    
    for d in (separated_dir, mels_dir, pitch_dir):
        d.mkdir(parents=True, exist_ok=True)
        
    # 1. Stem separation
    vocals_wav, backing_wav = run_demucs_separation(audio_path, separated_dir, demucs_device) if use_demucs else (None, None)
    
    # 2. Backing processing
    if backing_wav and backing_wav.exists():
        y_backing, sr = librosa.load(backing_wav, sr=sample_rate)
    else:
        print("-> Using raw audio as backing track...", flush=True)
        y_backing, sr = librosa.load(audio_path, sr=sample_rate)
        vocals_wav = None
        
    mel_backing = librosa.feature.melspectrogram(
        y=y_backing, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=mel_power
    )
    log_mel_backing = np.clip(np.log(np.clip(mel_backing, 1e-5, None)), -5.0, 3.0)
    backing_tensor = torch.from_numpy(log_mel_backing).float()
    frames = backing_tensor.shape[1]
    
    backing_pt_path = mels_dir / f"{sample_id}_backing.pt"
    torch.save(backing_tensor, backing_pt_path)

    tempo, _ = librosa.beat.beat_track(y=y_backing, sr=sr)
    bpm = int(tempo[0]) if hasattr(tempo, "__len__") else int(tempo)

    # Keep this before cleanup: the separated wavs may be removed after tensors
    # are saved, but the record must still describe the actual training target.
    has_vocal = bool(vocals_wav and vocals_wav.exists())

    # 3. Vocal transcription & features extraction
    lyrics = f"Vietnamese music track {sample_id}."
    lyric_segments = []
    vocal_tensor = backing_tensor.clone()
    
    if has_vocal:
        y_vocal, _ = librosa.load(vocals_wav, sr=sample_rate)
        
        # Whisper Transcription
        if transcribe and whisper_model is not None:
            print("-> Transcribing lyrics using Whisper ASR...", flush=True)
            asr_res = whisper_model.transcribe(
                str(vocals_wav),
                language="vi",
                word_timestamps=True,
                fp16=(str(next(whisper_model.parameters()).device).startswith("cuda")),
            )
            lyrics = asr_res["text"].strip()
            lyric_segments = [
                {
                    "start": round(float(segment.get("start", 0.0)), 3),
                    "end": round(float(segment.get("end", 0.0)), 3),
                    "text": str(segment.get("text", "")).strip(),
                }
                for segment in asr_res.get("segments", [])
                if str(segment.get("text", "")).strip()
            ]
        
        mel_vocal = librosa.feature.melspectrogram(
            y=y_vocal[:frames * hop_length], sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=mel_power
        )
        if mel_vocal.shape[1] < frames:
            mel_vocal = np.pad(mel_vocal, ((0, 0), (0, frames - mel_vocal.shape[1])))
        elif mel_vocal.shape[1] > frames:
            mel_vocal = mel_vocal[:, :frames]
        log_mel_vocal = np.clip(np.log(np.clip(mel_vocal, 1e-5, None)), -5.0, 3.0)
        vocal_tensor = torch.from_numpy(log_mel_vocal).float()
        
    # Save outputs
    vocal_pt_path = mels_dir / f"{sample_id}_vocal.pt"
    torch.save(vocal_tensor, vocal_pt_path)

    # Clean up intermediate Demucs wav files to conserve disk space unless keeping it for post-evaluation
    if not keep_separated and vocals_wav and vocals_wav.exists():
        song_folder = vocals_wav.parent
        if song_folder.exists() and song_folder.name == sample_id:
            shutil.rmtree(song_folder, ignore_errors=True)

    return {
        "id": sample_id,
        "text": lyrics,
        "segments": lyric_segments,
        "style": f"Vietnamese music, {bpm} BPM, emotional melody",
        "bpm": bpm,
        "frames": frames,
        "has_vocal": has_vocal,
        "vocal_source": "demucs" if has_vocal else "raw_mix_fallback",
        "backing_mel_path": f"mels/{sample_id}_backing.pt",
        "vocal_mel_path": f"mels/{sample_id}_vocal.pt"
    }

def preprocess_raw_audio(
    input_path: str | Path,
    output_path: str | Path,
    whisper_model_name: str = "base",
    keep_separated_count: int = 10,
    max_files: int | None = None,
    *,
    use_demucs: bool = True,
    transcribe: bool = True,
    sample_rate: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    mel_power: float = 2.0,
    demucs_device: str = "auto",
    whisper_device: str = "auto",
) -> dict:
    raw_dir = Path(input_path)
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    raw_files = sorted(
        {path for pattern in ("*.wav", "*.mp3") for path in raw_dir.rglob(pattern)},
        key=lambda path: str(path).lower(),
    )
    if max_files is not None:
        raw_files = raw_files[:max_files]
        
    total_files = len(raw_files)
    print(f"Found {total_files} files to process.", flush=True)
    if not raw_files:
        print(f"[ERROR] No raw audio files found in {raw_dir.resolve()}. Please supply input wav/mp3 files.", flush=True)
        return {"status": "failed", "error": "No raw audio files found"}

    whisper_model = None
    if transcribe:
        if whisper is None:
            raise RuntimeError("Cần cài openai-whisper hoặc dùng --skip-asr.")
        print(f"Loading Whisper model ({whisper_model_name})...", flush=True)
        requested_device = (whisper_device or "auto").lower()
        whisper_devices = [requested_device] if requested_device != "auto" else (["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"])
        for selected_device in whisper_devices:
            try:
                whisper_model = whisper.load_model(whisper_model_name, device=selected_device)
                whisper_device = selected_device
                break
            except Exception as exc:
                if selected_device == whisper_devices[-1]:
                    raise
                print(f"[WARNING] Whisper {selected_device} failed; retrying on {whisper_devices[-1]}: {exc}", flush=True)
    
    records = []
    failures = []
    batch_size = 8 if use_demucs else total_files
    separated_dir = output_dir / "separated"
    for batch_start in range(0, total_files, batch_size):
        batch = raw_files[batch_start:batch_start + batch_size]
        if use_demucs:
            print(
                f"\n--- DEMUCS BATCH {batch_start + 1}-{batch_start + len(batch)} "
                f"of {total_files} ---",
                flush=True,
            )
            run_demucs_batch(batch, separated_dir, demucs_device)

        for local_index, f in enumerate(batch):
            idx = batch_start + local_index + 1
            try:
                print(f"\n-> [{idx}/{total_files}] Processing: {f.name}", flush=True)
                record = process_file(
                    f,
                    output_dir,
                    whisper_model,
                    keep_separated=(idx <= keep_separated_count),
                    use_demucs=use_demucs,
                    transcribe=transcribe,
                    sample_rate=sample_rate,
                    n_mels=n_mels,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    mel_power=mel_power,
                    demucs_device=demucs_device,
                )
                records.append(record)
            except Exception as e:
                print(f"[ERROR] processing [{idx}/{total_files}] {f.name}: {e}", flush=True)
                failures.append({"file": str(f), "error": str(e)})
            
    # Write metadata index
    records_jsonl_path = output_dir / "records.jsonl"
    with records_jsonl_path.open("w", encoding="utf-8") as out:
        for r in records:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    # Write dataset config
    config_data = {
        "sample_rate": sample_rate,
        "n_mels": n_mels,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "mel_power": mel_power,
    }
    (output_dir / "config.json").write_text(json.dumps(config_data, indent=2), encoding="utf-8")
    
    if not records:
        status = "failed"
    elif failures:
        status = "completed_with_warnings"
    else:
        status = "completed"
    print(f"\nPreprocessing {status}. Dataset generated at: {output_dir.resolve()}", flush=True)
    return {
        "status": status,
        "dataset_path": str(output_dir.resolve()),
        "records_count": len(records),
        "failed_count": len(failures),
        "failures": failures,
    }


def main():
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Dataset preprocessing and auto-labeling for DiffRhythm-style models.")
    parser.add_argument("--input", default="dataset/vietnamese_songs", help="Path to raw audio folder.")
    parser.add_argument("--output", default="dataset/diff_rhythm_dataset", help="Output dataset path.")
    parser.add_argument("--whisper-model", default="base", help="Size of Whisper model to use.")
    parser.add_argument("--skip-demucs", action="store_true", help="Bỏ tách vocal, dùng bản phối thật làm mục tiêu nhanh.")
    parser.add_argument("--skip-asr", action="store_true", help="Bỏ Whisper ASR và dùng nhãn text mặc định.")
    parser.add_argument("--vocos-compatible", action="store_true", help="Tạo mel đúng chuẩn Vocos 24 kHz/100 kênh.")
    parser.add_argument("--demucs-device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--whisper-device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()

    mel_config = {
        "sample_rate": 24000 if args.vocos_compatible else SAMPLE_RATE,
        "n_mels": 100 if args.vocos_compatible else N_MELS,
        "n_fft": 1024 if args.vocos_compatible else N_FFT,
        "hop_length": 256 if args.vocos_compatible else HOP_LENGTH,
        "mel_power": 1.0 if args.vocos_compatible else 2.0,
        "demucs_device": args.demucs_device,
        "whisper_device": args.whisper_device,
    }

    preprocess_raw_audio(
        args.input,
        args.output,
        args.whisper_model,
        use_demucs=not args.skip_demucs,
        transcribe=not args.skip_asr,
        **mel_config,
    )

if __name__ == "__main__":
    main()
