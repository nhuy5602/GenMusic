# Vietnamese Audio Preprocessing

This package converts Vietnamese WAV/MP3 files into the structured dataset used
by the conditional diffusion model.

## Workflow

```mermaid
flowchart TD
    Raw[Raw WAV or MP3] --> Demucs[Demucs stem separation, batched+resumable]
    Raw --> MuLan[MuQ-MuLan style embedding]
    Demucs --> Vocals[vocals.wav]
    Demucs --> Backing[no_vocals.wav]
    Vocals --> Whisper[Whisper transcription + segment timestamps]
    Backing --> BPM[Librosa beat tracking]
    Vocals --> VocalMel[Vocal mel, Vocos-native format]
    Backing --> BackingMel[Backing mel, Vocos-native format]
    Whisper --> Records[records.jsonl]
    BPM --> Records
    VocalMel --> Records
    BackingMel --> Records
    MuLan --> Records
```

Demucs runs batched (loads its model once per batch of up to 8 files, not once
per file), is resumable (skips files whose stems already exist on disk), and
retries cuda→cpu on failure. If separation fails entirely for a song, or was
skipped dataset-wide via `--skip-demucs`, the record is marked with
`has_vocal: false` and `vocal_source` set to `"raw_mix_fallback"` (whole mix used
as backing, `--skip-demucs` mode) or `"silence_fallback"` (Demucs was attempted
but failed for just this song). Such records are useful for pipeline smoke
tests, but not for evaluating singing quality.

A MuQ-MuLan (`OpenMuQ/MuQ-MuLan-large`) style embedding is also computed once
per song, from the first 10s of the original mix — this is the real "Audio
Style Anchor" the model conditions on (see `docs/architecture.md`). If
the optional `muq` package isn't installed, this degrades to a zero vector
rather than failing the whole record.

## Usage

```powershell
uv run python cli.py preprocess-raw --input dataset/vietnamese_songs --output dataset/diff_rhythm_dataset --whisper-model base
```

The input directory is scanned recursively for `.wav` and `.mp3` files. Use
`--max-files` to limit a run, `--keep-separated-count` to keep selected Demucs
WAV files for inspection, `--skip-demucs`/`--skip-asr` to skip stem separation
or transcription for a fast approximate mode, and `--demucs-device`/
`--whisper-device` to force `cuda`/`cpu` instead of auto-detecting.

## Output Contract

```text
diff_rhythm_dataset/
  config.json
  records.jsonl
  mels/<song>_backing.pt
  mels/<song>_vocal.pt
  mels/<song>_style.pt
```

Each record contains `text` (full transcript), `segments` (word/segment-level
ASR timestamps, used to align cropped lyric text to cropped audio during
training), `style`, `bpm`, `frames`, `has_vocal`, `vocal_source`,
`demucs_separated`, `backing_mel_path`, `vocal_mel_path`, and
`style_embed_path`.

**Mel format matches Vocos's own native feature extractor exactly**
(`charactr/vocos-mel-24khz`: 100 mels, 24kHz, n_fft=1024, hop=256, magnitude mel
with `power=1`, natural log with a `1e-7` floor, **no** upper clip) — see
`compute_mel_spectrogram()` in `src/models/text_to_music_diffusion.py` and
`docs/project_history.md` §4.1 for why this specific format matters: an
earlier 64-mel/16kHz/log-power convention here was the root cause of severely
distorted generated audio, fixed this way and verified to restore >0.99
log-mel correlation on real audio.

## Optional: converting to latent-space (64-dim/5Hz)

The dataset above is mel-space, consumed directly by `train-self`. **This
conversion step is not optional for `train-distill`**: `KnowledgeDistillationTrainer`
now requires `config.latent_mode=True` unconditionally (raises immediately
otherwise, see `docs/architecture.md`) — there is no mel-space distillation
path anymore. To train the student inside DiffRhythm2's own compressed Music
VAE latent space, run `cli.py precompute-latent-dataset` on top of this
output — it re-decodes each record's mel through Vocos, re-encodes with a
trained `LatentAudioEncoder`, and writes a new dataset directory with the same
`records.jsonl`/`config.json` shape but `mels/*.pt` holding 64-dim/5Hz latents
instead of mel tensors, plus `config.json`'s `latent_mode: true`. See
`docs/usage.md` and `docs/project_history.md` §4.24 for the full
procedure (training the encoder first, its known collapse failure mode, and
the fix).

## Optional: `--raw-audio` (keep raw waveform, skip mel entirely)

`preprocess-raw --raw-audio` skips `compute_mel_spectrogram()` entirely and
saves the separated vocal/backing stems as raw 24kHz waveform tensors
instead — `waveforms/<song>_{vocal,backing}.pt` (shape `(samples,)`, not
`(n_mels, frames)`), plus `config.json`'s `raw_audio_mode: true`. Whisper
transcription and the MuQ-MuLan style embedding still run exactly as in the
default pipeline (neither depends on mel), so `records.jsonl`'s `text`/
`segments`/`style_embed_path` fields are unchanged; only the audio path keys
become `vocal_wav_path`/`backing_wav_path`.

This exists for `LatentAudioEncoder` (`src/models/latent_codec.py`), which
already takes raw waveform as input (`Conv1d(1, ...)`, not mel). Given a mel
dataset, `train-latent-encoder`/`precompute-latent-dataset` reconstruct an
approximation of the raw audio by decoding mel back through Vocos first.
Given a `--raw-audio` dataset (`config.json`'s `raw_audio_mode: true`), both
instead sum the already-separated `vocal_wav_path`/`backing_wav_path` tensors
directly and skip Vocos entirely — the encoder trains on the pristine
original recording, not a Vocos reconstruction of it.

```powershell
uv run python cli.py preprocess-raw --input dataset/vietnamese_songs --output dataset/raw_audio_dataset --whisper-model base --raw-audio
```

On Kaggle: `scripts/run_kaggle_preprocess_raw_audio.py` (same flags/flow as
`run_kaggle_preprocess_all.py`, minus the `vocos`/`encodec` dependencies,
which this mode never needs). To preprocess every part of the raw corpus
(`RAW_DATASETS`, `src/integrations/kaggle_dataset_refs.py`) instead of one at
a time, use `scripts/run_kaggle_multi_part_preprocess_raw_audio.py` — submits
up to `--max-new-jobs` kernels (Kaggle allows 2 concurrent batch GPU
sessions), tracks what's already been submitted in
`outputs/kaggle_datasets_preparation/submitted_state.json` so reruns don't
duplicate work, and `--wait-and-loop` chains through the rest of the parts
automatically.

`cli.py train-latent-encoder --dataset` accepts more than one of these
output directories at once (combined into a single training set, like
`train-self`) — `--max-records-per-dataset` caps each dir's contribution
*before* combining. `scripts/run_kaggle_latent_encoder.py` exposes both
through `--processed-kernel-ref`/`--max-records-per-dataset`, plus a
`--raw-audio-part 1 2 3 4 5 6` shortcut that looks the kernel refs up in
`PROCESSED_RAW_AUDIO_KERNELS` instead of pasting them by hand, e.g.
`--raw-audio-part 1 2 3 4 5 6 --max-records-per-dataset 1` for a 6-record
smoke test with one song from each part. `precompute-latent-dataset` does
not yet support multiple source dataset dirs.
