# GenMusic VN: Background Music & Vietnamese Vocal Generator

GenMusic VN is a semi-autoregressive Conditional Flow Matching (CFM) diffusion transformer model inspired by DiffRhythm 2. It is engineered to generate high-fidelity Vietnamese vocals conditioned on both a text lyric prompt and an **Audio Style Anchor** extracted from the backing track.

Detailed technical documentations are located in the `docs/` folder:
- [Project Report](docs/PROJECT_REPORT.md) — related work, architecture, experiments, conclusion (start here)
- [Run Guide](docs/guides/run_full_pipeline.md) — practical step-by-step commands
- [System Architecture](docs/architecture.md)
- [Machine Learning Models](docs/model.md)
- [Training & Improvement Pipelines](docs/training.md)
- [Experiment write-ups](docs/experiments/) — specific bugs found and fixed (vocoder distortion, non-functional distillation, a Kaggle-quota-burning hang)

---

## 📂 Project Directory Structure

```text
GenMusic/
├── src/                # Core Python package (data processing, model, training)
│   ├── data/           # Audio preprocessing, Whisper ASR, Demucs split, Vietnamese G2P
│   ├── models/         # MicroDiT + Conditional Flow Matching (CFM) diffusion architecture
│   ├── training/       # PyTorch Dataset, DataLoader, Trainer, and Distillation loop
│   ├── evaluation/     # Objective evaluation metrics for audio spectrograms
│   └── integrations/   # Kaggle API cloud integrations and job submitters
├── scripts/            # Automated automation scripts (Kaggle training, batch prep)
├── web/                # Interactive Web Client front-end UI (HTML, CSS, JS)
├── docs/               # System architecture design documentations
├── dataset/            # Local raw audio input folder (git-ignored)
├── outputs/            # Model checkpoints and generated audio waveforms
├── cli.py              # Main CLI entry point
└── server.py           # API Web backend server
```

---

## 🛠️ Installation & Setup

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

## ⚡ Quick Start: Train & Run Inference

The minimum path from raw audio to a generated song, run locally with `cli.py`. Every command below is copy-pasteable; swap paths as needed.

**1. Preprocess raw songs into a training dataset**
```powershell
uv run python cli.py preprocess-raw --input dataset/vietnamese_songs --output dataset/diff_rhythm_dataset --whisper-model tiny
```
Splits vocal/backing stems (Demucs), transcribes lyrics (Whisper), computes the Audio Style Anchor (MuQ-MuLan), and writes mel-spectrograms in Vocos-native format. Produces `dataset/diff_rhythm_dataset/{config.json, records.jsonl, mels/}`.

**2. Train the student model (no teacher — fastest way to sanity-check the pipeline)**
```powershell
uv run python cli.py train-self --dataset dataset/diff_rhythm_dataset --checkpoint outputs/my_dit_model.pt --epochs 30 --batch-size 4 --dim 256 --depth 4 --heads 4
```

**3. (Optional) Knowledge-distill from the real DiffRhythm2 teacher**
Needs the [DiffRhythm2 repo](https://github.com/ASLP-lab/DiffRhythm2) checked out somewhere with its dependencies installed, and its path on `PYTHONPATH` — this works both on Kaggle (see `scripts/run_kaggle_distill.py`) and locally (clone it yourself, `uv pip install --no-deps <missing deps as they come up>`, `espeak-ng` installed as a system package for the lyric tokenizer). Without it, the command below still runs and trains ground-truth-only, but reports that honestly (`teacher_status`/`distillation_active` in the output) instead of silently faking distillation:
```powershell
$env:PYTHONPATH = "C:\path\to\DiffRhythm2"
uv run python cli.py train-distill --dataset dataset/diff_rhythm_dataset --student-checkpoint outputs/distilled_student.pt --epochs 30 --batch-size 4 --dim 256 --depth 4 --heads 4
```

**4. Generate a song (inference)**
```powershell
uv run python cli.py generate-local --text "Đêm nay Hà Nội ngập tràn tiếng mưa rơi." --duration 8.0 --vocoder vocos --checkpoint outputs/my_dit_model.pt --out outputs/my_song
```
Loads whichever checkpoint you point it at (from step 2 or 3), samples with CFM, and decodes to `outputs/my_song/final.wav` via the Vocos vocoder.

**5. Evaluate the result**
```powershell
uv run python cli.py evaluate-self --generated outputs/my_song/final.wav --out outputs/evaluation_report
```

For running the same pipeline on Kaggle GPUs instead (recommended for anything past a smoke test — CPU-only teacher forward passes are ~3s/sample), see the automated scripts below and [docs/guides/run_full_pipeline.md](docs/guides/run_full_pipeline.md).

---

## 🚀 Usage Guide for Automated Scripts

All key workflows are packaged into automated scripts in the `scripts/` directory:

### 0. Training-only / distillation with no manual environment setup (Kaggle)

These two are the ones to hand to a teammate — both fully automate their own GPU
environment (cloning DiffRhythm2, installing `espeak-ng` + Python deps, etc.) inside
the Kaggle kernel, so **no local dependency chasing is needed**:
```powershell
# Training-only (baseline, no teacher):
uv run python scripts/run_kaggle_training.py --epochs 5 --batch-size 4

# Knowledge distillation from the real DiffRhythm2 teacher:
uv run python scripts/run_kaggle_distill.py
```
Only two things are required in `.env` first — everything else is handled by the script:
1. `KAGGLE_USERNAME` / `KAGGLE_KEY`.
2. A processed-dataset reference **your own Kaggle account can access** —
   `KAGGLE_PROCESSED_KERNEL_REF` (preferred) or `KAGGLE_PROCESSED_DATASET_REF`. The
   scripts fall back to a hardcoded example ref if unset, but that example belongs to
   whoever produced it and is not guaranteed to be public/shared with your account —
   don't rely on it. If you don't have your own processed dataset yet, produce one
   first with `scripts/run_kaggle_preprocess_all.py`, or ask whoever already has one to
   make it accessible to you (public, or shared as a Kaggle dataset collaborator).

### 1. Run Complete Local Pipeline Test
Verify the whole end-to-end flow locally (Data scanning ➔ Prep ➔ Model training ➔ Sampling ➔ Evaluation):
```powershell
uv run python scripts/run_pipeline.py
```

### 2. Run Kaggle GPU Single-File Smoke Test
Upload source files to Kaggle, isolate a single audio track, run preprocessing and a 1-epoch training test on Kaggle's T4 GPU, then download the resulting `.pt` model checkpoint automatically:
```powershell
uv run python scripts/run_kaggle_training.py
```

### 3. Run Batch Preprocessing on Kaggle
Perform stem separation (Demucs) and transcription (Whisper) on raw audio tracks on Kaggle. Automatically cleans up intermediate heavy WAV files to prevent Disk OOM.
```powershell
uv run python scripts/run_kaggle_preprocess_all.py --max-files 40
```

### 4. Run the Full Pipeline in One Kaggle Kernel (recommended)
Preprocess → vocoder sanity check → baseline training → distillation attempt → generate → sanity stats, all in a single Kaggle kernel session (matters for GPU quota — see [docs/guides/run_full_pipeline.md](docs/guides/run_full_pipeline.md)):
```powershell
uv run python scripts/run_kaggle_full_experiment.py --max-files 40 --whisper-model tiny --baseline-epochs 60 --distill-epochs 30
```

### 5. Run the Distillation-vs-Baseline Comparison Experiment
Trains several configs (baseline; distillation at a few `alpha_feature` values; a smaller architecture variant) against one shared preprocessed dataset, for a finer-grained ablation than the direct 250-song comparison already run — see [docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md) §4.8/§4.9 for the answer to "does distillation actually help":
```powershell
uv run python scripts/run_kaggle_experiment_matrix.py --max-files 40 --whisper-model tiny --epochs 60
```

---

## 🎹 Advanced CLI Operations (`cli.py`)

### 1. Local Generation (Inference)
`vocos` (default, recommended) decodes with the pretrained Vocos neural vocoder; `griffinlim` is a real iterative-phase-estimation fallback if Vocos is unavailable. Both require the mel format to match Vocos's native convention exactly, which this project's default `MusicDiffusionConfig` and `preprocess-raw` output always do (see [docs/experiments/vocoder_fix.md](docs/experiments/vocoder_fix.md) for why this specific detail mattered a lot in practice):
```powershell
# Generate audio using the Vocos Vocoder for natural voice reconstruction:
uv run python cli.py generate-local --text "Đêm nay Hà Nội ngập tràn tiếng mưa rơi." --duration 8.0 --vocoder vocos --out outputs/my_song
```
Without `--reference-dataset`, generation conditions on a zero backing-track and falls back to a pooled-text style vector instead of a real MuQ-MuLan style anchor — a real train/inference mismatch, since the model is trained on real backing_mel + style_anchor conditioning. To condition generation the same way training did, extract both from an already-preprocessed dataset record instead of needing new raw audio input:
```powershell
uv run python cli.py generate-local --text "Đêm nay Hà Nội ngập tràn tiếng mưa rơi." --duration 8.0 --checkpoint outputs/my_dit_model.pt --reference-dataset dataset/diff_rhythm_dataset --reference-id <record_id> --out outputs/my_song
```
`--reference-id` defaults to the dataset's first record if omitted. See `load_reference_conditioning()` in `src/training/self_diffusion.py`.

### 2. Manual Preprocessing
Preprocess audio files with Demucs stem separation and Whisper lyric transcription. pYIN F0 extraction is removed to dramatically speed up data preparation:
```powershell
uv run python cli.py preprocess-raw --input dataset/vietnamese_songs --output dataset/diff_rhythm_dataset --whisper-model small --max-files 100 --keep-separated-count 100
```
- `--max-files`: Limits the maximum number of files to process.
- `--keep-separated-count`: Determines how many separated demuxed WAV files are kept in the final output directory for inspection/evaluation.

### 3. Model Training & Knowledge Distillation
You can train the diffusion denoiser from scratch or perform knowledge distillation from the pretrained DiffRhythm Teacher to your student MicroDiT model:

* **Train Model from Scratch:**
  ```powershell
  uv run python cli.py train-self --dataset dataset/diff_rhythm_dataset --checkpoint outputs/my_dit_model.pt --epochs 5 --batch-size 4 --dim 256 --depth 4 --heads 4
  ```
  MicroDiT (Transformer-based) with real MuQ-MuLan Audio Style Anchor conditioning is the only architecture. `--dim`/`--depth`/`--heads`/`--ff-mult` control its size (default: ~5.6M trainable params).

* **Knowledge Distillation:**
  Replicates the real DiffRhythm2 teacher's call contract (see [docs/experiments/distillation_fix.md](docs/experiments/distillation_fix.md)). Needs a clone of the [DiffRhythm2 repo](https://github.com/ASLP-lab/DiffRhythm2) on `PYTHONPATH` with its dependencies installed — done automatically on Kaggle (see `scripts/run_kaggle_distill.py`), or manually locally (see Quick Start step 3 above; verified working on Windows/CPU too, not just Kaggle). Without that clone, or without internet, `train-distill` **raises immediately** rather than silently completing as ground-truth-only training under the distillation name — never a silent fake teacher, and never a silent downgrade either. Use `train-self` if you want ground-truth-only training.
  - If `--teacher-checkpoint` is omitted, the script automatically downloads the latest model weights (`model.safetensors`) from the Hugging Face repo: `ASLP-lab/DiffRhythm2`.
  ```powershell
  uv run python cli.py train-distill --dataset dataset/diff_rhythm_dataset --student-checkpoint outputs/distilled_student.pt --epochs 5 --batch-size 4 --alpha-feature 0.5
  ```
  Verified at Kaggle scale (250 real songs, matched epochs/steps against `train-self`): distillation's `loss_gt` came in at roughly a third of the no-teacher baseline's, at ~49x the wall-clock GPU cost — see [docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md) §4.8/§4.9 and [docs/guides/run_full_pipeline.md](docs/guides/run_full_pipeline.md) for a finer-grained ablation if you want one.

### 4. Audio Quality Evaluation
```powershell
uv run python cli.py evaluate-self --generated outputs/my_song/final.wav --out outputs/evaluation_report
```

---

## 🖥️ Interactive Web UI Demo

Start the local standard-library backend server:
```powershell
uv run python server.py
```
Open your browser and navigate to `http://127.0.0.1:8000` to enter custom Vietnamese prompts and listen to generated musical tracks.

---

## 🧪 Unit Testing

Run automated tests to verify model math, audio anchor slicing, and system stability:
```powershell
uv run python -m unittest discover -s tests -v
```
