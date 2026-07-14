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
import whisper

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.models.text_to_music_diffusion import MusicDiffusionConfig, compute_mel_spectrogram

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

# Mel parameters MUST match Vocos's native "charactr/vocos-mel-24khz" feature
# extractor exactly (via compute_mel_spectrogram) so that training targets are
# directly decodable by Vocos at inference with no resampling. See
# docs/experiments/vocoder_fix.md for why the previous 16kHz/64-mel/log-power
# convention produced badly distorted audio.
_MEL_CONFIG = MusicDiffusionConfig()
SAMPLE_RATE = _MEL_CONFIG.sample_rate
N_MELS = _MEL_CONFIG.n_mels
N_FFT = _MEL_CONFIG.n_fft
HOP_LENGTH = _MEL_CONFIG.hop_length
STYLE_EMBED_DIM = 512  # MuQ-MuLan / DiffRhythm2 teacher cond_dim

_mulan_model = None


def _load_mulan(device: str = "cpu"):
    """Lazily load MuQ-MuLan (real contrastive audio-style embedding model used
    by the DiffRhythm2 teacher). Requires the `muq` package + internet on first
    use to fetch weights -- only available in the Kaggle preprocessing kernel.
    Returns None if unavailable so preprocessing degrades to a zero style vector
    instead of crashing (e.g. for local/offline smoke runs).
    """
    global _mulan_model
    if _mulan_model is None:
        try:
            from muq import MuQMuLan
            _mulan_model = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large").to(device).eval()
        except Exception as e:
            print(f"[WARNING] MuQ-MuLan unavailable ({e}); style embeddings will be zero vectors.", flush=True)
            _mulan_model = False
    return _mulan_model or None


def compute_style_embedding(waveform_24k: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """Real MuQ-MuLan style/genre embedding of a song, computed once at
    preprocess time and reused as the "Audio Style Anchor" for both the student
    and (during distillation) the teacher -- see docs/experiments/distillation_fix.md.
    """
    mulan = _load_mulan(device)
    if mulan is None:
        return torch.zeros(STYLE_EMBED_DIM)
    with torch.no_grad():
        clip = waveform_24k[: 24_000 * 10]  # MuLan is trained on ~10s clips
        wav = torch.tensor(clip, dtype=torch.float32, device=device).unsqueeze(0)
        embedding = mulan(wavs=wav)
    return embedding.squeeze(0).float().cpu()


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

def process_file(audio_path: Path, output_dir: Path, whisper_model, keep_separated: bool = False, device: str = "cpu") -> dict:
    sample_id = audio_path.stem
    print(f"\n==================== PROCESSING {sample_id} ====================", flush=True)

    separated_dir = output_dir / "separated"
    mels_dir = output_dir / "mels"

    for d in (separated_dir, mels_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 0. Style embedding (MuQ-MuLan) of the full original song -- the same
    # audio-style space the DiffRhythm2 teacher itself conditions on.
    print("-> Computing MuQ-MuLan style embedding...", flush=True)
    y_style, _ = librosa.load(audio_path, sr=SAMPLE_RATE, duration=10.0)
    style_embedding = compute_style_embedding(y_style, device=device)
    style_pt_path = mels_dir / f"{sample_id}_style.pt"
    torch.save(style_embedding, style_pt_path)

    # 1. Stem separation
    vocals_wav, backing_wav = run_demucs_separation(audio_path, separated_dir)

    # 2. Backing processing
    has_backing_stem = bool(backing_wav and backing_wav.exists())
    if has_backing_stem:
        y_backing, sr = librosa.load(backing_wav, sr=SAMPLE_RATE)
    else:
        print("-> Using raw audio as backing track...", flush=True)
        y_backing, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        vocals_wav = None

    backing_tensor = compute_mel_spectrogram(y_backing, _MEL_CONFIG)
    frames = backing_tensor.shape[1]

    backing_pt_path = mels_dir / f"{sample_id}_backing.pt"
    torch.save(backing_tensor, backing_pt_path)

    try:
        # librosa's beat_track calls scipy.signal.hann, which newer scipy releases
        # removed (moved to scipy.signal.windows.hann) -- depending on exactly which
        # scipy/librosa combo an environment resolves to (e.g. after installing a
        # third-party requirements.txt that repins scipy), this raises AttributeError.
        # BPM is cosmetic metadata (folded into the "style" text prompt), not required
        # for training, so fall back rather than losing the whole record over it.
        tempo, _ = librosa.beat.beat_track(y=y_backing, sr=sr)
        bpm = int(tempo[0]) if hasattr(tempo, "__len__") else int(tempo)
    except Exception as e:
        print(f"[WARNING] beat_track failed ({e}); defaulting bpm=120.", flush=True)
        bpm = 120

    # 3. Vocal transcription & features extraction
    lyrics = "Instrumental track."
    vocal_tensor = torch.zeros_like(backing_tensor)
    has_vocals = bool(vocals_wav and vocals_wav.exists())

    if has_vocals:
        y_vocal, _ = librosa.load(vocals_wav, sr=SAMPLE_RATE)

        # Whisper Transcription
        print("-> Transcribing lyrics using Whisper ASR...", flush=True)
        asr_res = whisper_model.transcribe(str(vocals_wav), language="vi", word_timestamps=True)
        lyrics = asr_res["text"].strip()

        vocal_tensor = compute_mel_spectrogram(y_vocal[: frames * HOP_LENGTH], _MEL_CONFIG)
        vocal_tensor = vocal_tensor[:, :frames]
        if vocal_tensor.shape[1] < frames:
            vocal_tensor = torch.nn.functional.pad(vocal_tensor, (0, frames - vocal_tensor.shape[1]))

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
        "style": f"Vietnamese music, {bpm} BPM, emotional melody",
        "bpm": bpm,
        "frames": frames,
        "has_vocals": has_vocals,
        "demucs_separated": has_backing_stem,
        "backing_mel_path": f"mels/{sample_id}_backing.pt",
        "vocal_mel_path": f"mels/{sample_id}_vocal.pt",
        "style_embed_path": f"mels/{sample_id}_style.pt"
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Whisper model ({whisper_model_name})...", flush=True)
    whisper_model = whisper.load_model(whisper_model_name, device=device)

    records = []
    failures = []
    for idx, f in enumerate(raw_files, start=1):
        try:
            print(f"\n-> [{idx}/{total_files}] Processing: {f.name}", flush=True)
            record = process_file(f, output_dir, whisper_model, keep_separated=(idx <= keep_separated_count), device=device)
            records.append(record)
        except Exception as e:
            import traceback

            tb = traceback.format_exc()
            print(f"[ERROR] processing [{idx}/{total_files}] {f.name}: {e}", flush=True)
            # Print() output is not reliably captured by Kaggle's log download for
            # scripts that shell out to a subprocess -- persist failures to the
            # returned report (written to disk) so they survive regardless.
            failures.append({"file": f.name, "error": str(e), "traceback": tb})

    # Write metadata index
    records_jsonl_path = output_dir / "records.jsonl"
    with records_jsonl_path.open("w", encoding="utf-8") as out:
        for r in records:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
    if failures:
        (output_dir / "preprocess_failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
            
    # Write dataset config
    config_data = {
        "sample_rate": SAMPLE_RATE,
        "n_mels": N_MELS,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH
    }
    (output_dir / "config.json").write_text(json.dumps(config_data, indent=2))
    
    print(f"\n🎉 Preprocessing completed! Dataset generated at: {output_dir.resolve()}", flush=True)
    return {
        "status": "completed" if records else "failed",
        "dataset_path": str(output_dir.resolve()),
        "records_count": len(records),
        "failed_count": len(failures),
        "failures": failures[:3],  # first few tracebacks inline for visibility without opening another file
    }


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
