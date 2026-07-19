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
except ImportError:  # Optional for the fast full-mix preprocessing mode (--skip-asr).
    whisper = None

try:
    from transformers import pipeline as hf_asr_pipeline
except ImportError:  # Only needed for --whisper-model <huggingface-repo-id> (see _is_hf_model_ref).
    hf_asr_pipeline = None

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.models.text_to_music_diffusion import MusicDiffusionConfig, compute_mel_spectrogram

# Configure FFMPEG path using imageio_ffmpeg so that whisper/demucs can find it on Windows
try:
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dir = os.path.dirname(ffmpeg_exe)
    target_ffmpeg = os.path.join(ffmpeg_dir, "ffmpeg.exe")
    if not os.path.exists(target_ffmpeg):
        shutil.copy2(ffmpeg_exe, target_ffmpeg)
    os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]
except ImportError:
    pass

# Mel parameters MUST match Vocos's native "charactr/vocos-mel-24khz" feature
# extractor exactly (via compute_mel_spectrogram) so that training targets are
# directly decodable by Vocos at inference with no resampling. See
# docs/experiments/vocoder_fix.md for why the previous 16kHz/64-mel/log-power
# convention produced badly distorted audio. Unlike an earlier iteration of
# this file, this is no longer an opt-in mode -- there is no non-Vocos-native
# mel format anymore, since it was the direct cause of that distortion.
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


def run_demucs_separation(audio_path: Path, output_dir: Path, device: str = "auto") -> tuple[Path | None, Path | None]:
    """Separate vocals and backing using Demucs CLI. Resumable (skips if stems
    already exist) and retries cuda -> cpu on failure."""
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
    """Separate a small batch so Demucs loads its model once per batch instead
    of once per file -- meaningfully faster for larger datasets."""
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
    devices = [requested] if requested != "auto" else (["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"])
    for selected_device in devices:
        command = [sys.executable, "-m", "demucs.separate", "--two-stems=vocals", "-o", str(output_dir)]
        if selected_device:
            command.extend(["-d", selected_device])
        command.extend(str(path) for path in pending)
        print(f"-> Demucs batch {len(pending)} file(s) on {selected_device}; progress will appear below...", flush=True)
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


def _is_hf_model_ref(whisper_model_name: str) -> bool:
    """A HuggingFace hub repo id ("owner/name") vs. an openai-whisper builtin
    size name ("base", "small", ...). Lets a Vietnamese-lyrics-specialized
    fine-tune (e.g. xyzDivergence/whisper-small-vietnamese-lyrics-transcription,
    fine-tuned on ~550h of zingmp3.vn songs, WER 30.7%) be selected via the
    same --whisper-model flag with no new CLI surface.
    """
    return "/" in whisper_model_name


def process_file(
    audio_path: Path,
    output_dir: Path,
    whisper_model,
    keep_separated: bool = False,
    *,
    use_demucs: bool = True,
    transcribe: bool = True,
    demucs_device: str = "auto",
    device: str = "cpu",
    whisper_backend: str = "openai",
) -> dict:
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
    vocals_wav, backing_wav = run_demucs_separation(audio_path, separated_dir, demucs_device) if use_demucs else (None, None)

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
    has_vocal = bool(vocals_wav and vocals_wav.exists())
    lyrics = f"Vietnamese music track {sample_id}."
    lyric_segments = []
    # Only claim the vocal target equals the backing mix when stem separation was
    # deliberately skipped for the whole dataset (--skip-demucs, a fast/approximate
    # mode). If Demucs was attempted but failed for just this one song, fall back to
    # silence instead -- claiming "vocal == whole mix" would train the model to treat
    # accompaniment as a vocal target, which is wrong when the rest of the dataset has
    # real separated vocals.
    vocal_tensor = backing_tensor.clone() if not use_demucs else torch.zeros_like(backing_tensor)

    if has_vocal:
        y_vocal, _ = librosa.load(vocals_wav, sr=SAMPLE_RATE)

        if transcribe and whisper_model is not None:
            print("-> Transcribing lyrics using Whisper ASR...", flush=True)
            if whisper_backend == "hf":
                # Whisper's long-form/chunked decoding can fall into a runaway
                # repetition loop (a token/phrase repeated for the whole
                # transcript) once one chunk hallucinates and later chunks
                # condition on that bad output -- condition_on_prev_tokens=False
                # stops that compounding; no_repeat_ngram_size/repetition_penalty
                # additionally suppress within-chunk repetition loops.
                asr_res = whisper_model(
                    str(vocals_wav), return_timestamps="word",
                    generate_kwargs={
                        "condition_on_prev_tokens": False,
                        "no_repeat_ngram_size": 3,
                        "repetition_penalty": 1.3,
                    },
                )
                lyrics = str(asr_res.get("text", "")).strip()
                words = []
                previous_end = 0.0
                for chunk in asr_res.get("chunks", []):
                    word_text = str(chunk.get("text", "")).strip()
                    if not word_text:
                        continue
                    timestamp = chunk.get("timestamp") or (None, None)
                    start = float(timestamp[0]) if timestamp[0] is not None else previous_end
                    end = float(timestamp[1]) if timestamp[1] is not None else start
                    words.append({"start": round(start, 3), "end": round(end, 3), "word": word_text})
                    previous_end = end
                lyric_segments = [
                    {
                        "start": words[0]["start"],
                        "end": words[-1]["end"],
                        "text": lyrics,
                        "words": words,
                    }
                ] if words else []
            else:
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
                        "words": [
                            {
                                "start": round(float(word.get("start", segment.get("start", 0.0))), 3),
                                "end": round(float(word.get("end", segment.get("end", 0.0))), 3),
                                "word": str(word.get("word", "")).strip(),
                            }
                            for word in segment.get("words", [])
                            if str(word.get("word", "")).strip()
                        ],
                    }
                    for segment in asr_res.get("segments", [])
                    if str(segment.get("text", "")).strip()
                ]

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
        "segments": lyric_segments,
        "style": f"Vietnamese music, {bpm} BPM, emotional melody",
        "bpm": bpm,
        "frames": frames,
        "has_vocal": has_vocal,
        "vocal_source": "demucs" if has_vocal else ("raw_mix_fallback" if not use_demucs else "silence_fallback"),
        "demucs_separated": has_backing_stem,
        "backing_mel_path": f"mels/{sample_id}_backing.pt",
        "vocal_mel_path": f"mels/{sample_id}_vocal.pt",
        "style_embed_path": f"mels/{sample_id}_style.pt",
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

    style_device = "cuda" if torch.cuda.is_available() else "cpu"

    whisper_model = None
    whisper_backend = "hf" if _is_hf_model_ref(whisper_model_name) else "openai"
    if transcribe:
        if whisper_backend == "hf":
            if hf_asr_pipeline is None:
                raise RuntimeError("Cần cài transformers để dùng model ASR HuggingFace (--whisper-model owner/name).")
            print(f"Loading HuggingFace ASR pipeline ({whisper_model_name})...", flush=True)
            requested_device = (whisper_device or "auto").lower()
            use_cuda = requested_device == "cuda" or (requested_device == "auto" and torch.cuda.is_available())
            whisper_model = hf_asr_pipeline(
                "automatic-speech-recognition",
                model=whisper_model_name,
                chunk_length_s=30,
                device=0 if use_cuda else -1,
            )
        else:
            if whisper is None:
                raise RuntimeError("Cần cài openai-whisper hoặc dùng --skip-asr.")
            print(f"Loading Whisper model ({whisper_model_name})...", flush=True)
            requested_device = (whisper_device or "auto").lower()
            whisper_devices = [requested_device] if requested_device != "auto" else (["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"])
            for selected_device in whisper_devices:
                try:
                    whisper_model = whisper.load_model(whisper_model_name, device=selected_device)
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
            print(f"\n--- DEMUCS BATCH {batch_start + 1}-{batch_start + len(batch)} of {total_files} ---", flush=True)
            run_demucs_batch(batch, separated_dir, demucs_device)

        for local_index, f in enumerate(batch):
            idx = batch_start + local_index + 1
            try:
                print(f"\n-> [{idx}/{total_files}] Processing: {f.name}", flush=True)
                record = process_file(
                    f, output_dir, whisper_model, keep_separated=(idx <= keep_separated_count),
                    use_demucs=use_demucs, transcribe=transcribe, demucs_device=demucs_device, device=style_device,
                    whisper_backend=whisper_backend,
                )
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
        "hop_length": HOP_LENGTH,
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
        "failures": failures[:3],  # first few tracebacks inline for visibility without opening another file
    }


def main():
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Dataset preprocessing and auto-labeling for DiffRhythm-style models.")
    parser.add_argument("--input", default="dataset/vietnamese_songs", help="Path to raw audio folder.")
    parser.add_argument("--output", default="dataset/diff_rhythm_dataset", help="Output dataset path.")
    parser.add_argument("--whisper-model", default="base", help="Openai-whisper size (base, small, ...) or a HuggingFace repo id (owner/name) for a fine-tuned ASR model, e.g. xyzDivergence/whisper-small-vietnamese-lyrics-transcription.")
    parser.add_argument("--skip-demucs", action="store_true", help="Bỏ tách vocal, dùng bản phối thật làm mục tiêu nhanh.")
    parser.add_argument("--skip-asr", action="store_true", help="Bỏ Whisper ASR và dùng nhãn text mặc định.")
    parser.add_argument("--demucs-device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--whisper-device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()

    preprocess_raw_audio(
        args.input, args.output, args.whisper_model,
        use_demucs=not args.skip_demucs, transcribe=not args.skip_asr,
        demucs_device=args.demucs_device, whisper_device=args.whisper_device,
    )


if __name__ == "__main__":
    main()
