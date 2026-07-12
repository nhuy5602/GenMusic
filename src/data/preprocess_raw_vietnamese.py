import argparse
import json
import os
import subprocess
import shutil
from pathlib import Path
import numpy as np
import librosa
import torch
import whisper

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

def run_demucs_separation(audio_path: Path, output_dir: Path) -> tuple[Path, Path]:
    """Separate vocals and backing using Demucs CLI."""
    print(f"-> Separating vocal/backing stems for: {audio_path.name}...", flush=True)
    try:
        subprocess.run(
            ["demucs", "--two-stems=vocals", "-o", str(output_dir), str(audio_path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Demucs default structure: output_dir/htdemucs/song_stem_name/
        model_name = "htdemucs"
        song_folder = output_dir / model_name / audio_path.stem
        
        vocals_file = song_folder / "vocals.wav"
        backing_file = song_folder / "no_vocals.wav"
        
        if vocals_file.exists() and backing_file.exists():
            return vocals_file, backing_file
            
    except Exception as e:
        print(f"[WARNING] Demucs separation failed/not found: {e}. Falling back to treating audio as instrumental (no vocal split).", flush=True)
    
    return None, None

def process_file(audio_path: Path, output_dir: Path, whisper_model, keep_separated: bool = False) -> dict:
    sample_id = audio_path.stem
    print(f"\n==================== PROCESSING {sample_id} ====================", flush=True)
    
    separated_dir = output_dir / "separated"
    mels_dir = output_dir / "mels"
    pitch_dir = output_dir / "pitch"
    
    for d in (separated_dir, mels_dir, pitch_dir):
        d.mkdir(parents=True, exist_ok=True)
        
    # 1. Stem separation
    vocals_wav, backing_wav = run_demucs_separation(audio_path, separated_dir)
    
    # 2. Backing processing
    if backing_wav and backing_wav.exists():
        y_backing, sr = librosa.load(backing_wav, sr=SAMPLE_RATE)
    else:
        print("-> Using raw audio as backing track...", flush=True)
        y_backing, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        vocals_wav = None
        
    mel_backing = librosa.feature.melspectrogram(
        y=y_backing, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, power=2.0
    )
    log_mel_backing = np.clip(np.log(np.clip(mel_backing, 1e-5, None)), -5.0, 3.0)
    backing_tensor = torch.from_numpy(log_mel_backing).float()
    frames = backing_tensor.shape[1]
    
    backing_pt_path = mels_dir / f"{sample_id}_backing.pt"
    torch.save(backing_tensor, backing_pt_path)

    tempo, _ = librosa.beat.beat_track(y=y_backing, sr=sr)
    bpm = int(tempo[0]) if hasattr(tempo, "__len__") else int(tempo)

    # 3. Vocal transcription & F0 pitch extraction
    lyrics = "Instrumental track."
    f0_clean = np.zeros(frames)
    vocal_tensor = torch.zeros_like(backing_tensor)
    
    if vocals_wav and vocals_wav.exists():
        y_vocal, _ = librosa.load(vocals_wav, sr=SAMPLE_RATE)
        
        # Whisper Transcription
        print("-> Transcribing lyrics using Whisper ASR...", flush=True)
        asr_res = whisper_model.transcribe(str(vocals_wav), language="vi", word_timestamps=True)
        lyrics = asr_res["text"].strip()
        
        # Pitch tracking via pYIN
        print("-> Tracking F0 pitch melody contour...", flush=True)
        fmin = librosa.note_to_hz('C2')
        fmax = librosa.note_to_hz('C7')
        f0, voiced_flag, _ = librosa.pyin(y_vocal, fmin=fmin, fmax=fmax, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
        f0_clean = np.nan_to_num(f0)
        
        if len(f0_clean) > frames:
            f0_clean = f0_clean[:frames]
        elif len(f0_clean) < frames:
            f0_clean = np.pad(f0_clean, (0, frames - len(f0_clean)))
            
        mel_vocal = librosa.feature.melspectrogram(
            y=y_vocal[:frames * HOP_LENGTH], sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, power=2.0
        )
        log_mel_vocal = np.clip(np.log(np.clip(mel_vocal, 1e-5, None)), -5.0, 3.0)
        vocal_tensor = torch.from_numpy(log_mel_vocal).float()
        
    # Save outputs
    pitch_npy_path = pitch_dir / f"{sample_id}_f0.npy"
    np.save(pitch_npy_path, f0_clean)
    
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
        "style": f"Vietnamese music, {bpm} BPM, emotional melody",
        "bpm": bpm,
        "frames": frames,
        "backing_mel_path": f"mels/{sample_id}_backing.pt",
        "vocal_mel_path": f"mels/{sample_id}_vocal.pt",
        "f0_path": f"pitch/{sample_id}_f0.npy"
    }

def preprocess_raw_audio(input_path: str | Path, output_path: str | Path, whisper_model_name: str = "base", keep_separated_count: int = 10, max_files: int | None = None) -> dict:
    raw_dir = Path(input_path)
    output_dir = Path(output_path)
    
    raw_files = list(raw_dir.glob("*.wav")) + list(raw_dir.glob("*.mp3"))
    if max_files is not None:
        raw_files = raw_files[:max_files]
        
    total_files = len(raw_files)
    print(f"Found {total_files} files to process.", flush=True)
    if not raw_files:
        print(f"[ERROR] No raw audio files found in {raw_dir.resolve()}. Please supply input wav/mp3 files.", flush=True)
        return {"status": "failed", "error": "No raw audio files found"}

    print(f"Loading Whisper model ({whisper_model_name})...", flush=True)
    whisper_model = whisper.load_model(whisper_model_name)
    
    records = []
    for idx, f in enumerate(raw_files, start=1):
        try:
            print(f"\n-> [{idx}/{total_files}] Processing: {f.name}", flush=True)
            record = process_file(f, output_dir, whisper_model, keep_separated=(idx <= keep_separated_count))
            records.append(record)
        except Exception as e:
            print(f"[ERROR] processing [{idx}/{total_files}] {f.name}: {e}", flush=True)
            
    # Write metadata index
    records_jsonl_path = output_dir / "records.jsonl"
    with records_jsonl_path.open("w", encoding="utf-8") as out:
        for r in records:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    # Write dataset config
    config_data = {
        "sample_rate": SAMPLE_RATE,
        "n_mels": N_MELS,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH
    }
    (output_dir / "config.json").write_text(json.dumps(config_data, indent=2))
    
    print(f"\n🎉 Preprocessing completed! Dataset generated at: {output_dir.resolve()}", flush=True)
    return {"status": "completed", "dataset_path": str(output_dir.resolve()), "records_count": len(records)}


def main():
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Dataset preprocessing and auto-labeling for DiffRhythm-style models.")
    parser.add_argument("--input", default="dataset/vietnamese_songs", help="Path to raw audio folder.")
    parser.add_argument("--output", default="dataset/diff_rhythm_dataset", help="Output dataset path.")
    parser.add_argument("--whisper-model", default="base", help="Size of Whisper model to use.")
    args = parser.parse_args()

    preprocess_raw_audio(args.input, args.output, args.whisper_model)

if __name__ == "__main__":
    main()
