# GenMusic VN: Vietnamese Song Generator

GenMusic VN generates Vietnamese vocal audio from a lyric prompt and a style
description, via Conditional Flow Matching (CFM), distilled/inspired by
DiffRhythm2. Two student backbones are available
(`--architecture microdit|native_dit`) and two feature spaces (raw mel, or
DiffRhythm2's own compressed 64-dim/5Hz latent space) — see
[docs/architecture.md](docs/architecture.md) for the technical reference.

## Documentation

- [docs/architecture.md](docs/architecture.md) — system design: both student
  backbones, the CFM loss, distillation, the native latent encoder.
- [docs/data_preparation.md](docs/data_preparation.md) — the preprocessing
  pipeline (Demucs, Whisper, MuQ-MuLan) and dataset format.
- [docs/usage.md](docs/usage.md) — practical run instructions, local and on
  Kaggle.
- [docs/project_history.md](docs/project_history.md) — chronological record
  of experiments run, bugs found and fixed, and results (with real numbers).
- [scripts/README.md](scripts/README.md) — index of the Kaggle automation
  scripts.

## Project layout

```text
GenMusic/
├── src/                # Core Python package (data processing, model, training)
│   ├── data/           # Audio preprocessing, Whisper ASR, Demucs split, latent-dataset conversion
│   ├── models/         # MicroDiT, NativeDiTStudent, CFM loss, LatentAudioEncoder
│   ├── training/       # Training loops: self, distillation, latent-encoder
│   ├── evaluation/     # Objective evaluation metrics for audio spectrograms
│   └── integrations/   # Kaggle API cloud integrations and job submitters
├── scripts/            # Kaggle automation scripts (see scripts/README.md)
├── web/                # Interactive Web Client front-end UI (HTML, CSS, JS)
├── docs/               # Documentation (see above)
├── dataset/            # Local raw audio input folder (git-ignored)
├── outputs/            # Model checkpoints and generated audio waveforms
├── cli.py              # Main CLI entry point
└── server.py           # API web backend server
```

---

## Installation & Setup

### 1. Install Dependencies

* **Using `uv` (Recommended - extremely fast and secure):**
  ```powershell
  uv sync
  ```

* **Using Standard `pip`:**
  ```powershell
  pip install -e .
  ```

### 2. Setup Environment Variables (.env)
Create a `.env` file in the root directory based on the `.env.example` template:
```env
# Local Environment variables
RAW_AUDIO_INPUT_DIR=dataset/vietnamese_songs
PROCESSED_DATASET_DIR=dataset/diff_rhythm_dataset
MODEL_CHECKPOINT_PATH=outputs/my_trained_model.pt

# Kaggle API tokens (For scheduling training tasks to GPU Cloud)
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
KAGGLE_RAW_DATASET_REF=sonlest/vietnamese-music-dataset-version3-part6
# Set after a preprocess run (scripts/run_kaggle_preprocess_all.py) -- attaches the
# preprocess kernel's own output to downstream kernels via kernel_sources, so no
# Kaggle API key is ever embedded in the shared kernel code:
KAGGLE_PROCESSED_KERNEL_REF=your_kaggle_username/genmusic-prep-1234567890
# Legacy fallback: a pre-existing published Dataset, used only if the above is unset.
KAGGLE_PROCESSED_DATASET_REF=your_kaggle_username/vietnamese-music-processed-dataset
# Fixed training dataset ref used by the `generate` (Kaggle job staging) command:
GENMUSIC_KAGGLE_DATASET_REF=your_kaggle_username/genmusic-vn-self-diffusion-training
```
See `.env.example` for the full list, including optional per-run overrides.

---

## Quick start

The minimum path from raw audio to a generated song, run locally:

```powershell
uv run python cli.py preprocess-raw --input dataset/vietnamese_songs --output dataset/diff_rhythm_dataset --whisper-model tiny
uv run python cli.py train-self --dataset dataset/diff_rhythm_dataset --checkpoint outputs/my_dit_model.pt --epochs 30 --batch-size 4 --dim 256 --depth 4 --heads 4
uv run python cli.py generate-local --text "Đêm nay Hà Nội ngập tràn tiếng mưa rơi." --duration 8.0 --checkpoint outputs/my_dit_model.pt --out outputs/my_song
uv run python cli.py evaluate-self --generated outputs/my_song/final.wav --out outputs/evaluation_report
```

That's `train-self` with the default `MicroDiT` backbone on raw mel — the
fastest way to sanity-check the pipeline end-to-end. For everything else
(distillation, the native latent backbone, running on Kaggle, the full
automated experiment scripts), see **[docs/usage.md](docs/usage.md)**.

## Web demo

```powershell
uv run python server.py
```
Open `http://127.0.0.1:8000` to enter Vietnamese prompts and listen to
generated tracks.

## Unit testing

Run automated tests to verify model math, audio anchor slicing, and system stability:
```powershell
uv run python -m unittest discover -s tests -v
```
