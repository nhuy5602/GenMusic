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
GENMUSIC_OUTPUT_DIR=outputs

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
```

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
uv run python scripts/run_kaggle_preprocess_all.py
```

### 4. Run the Full Pipeline in One Kaggle Kernel (recommended)
Preprocess → vocoder sanity check → baseline training → distillation attempt → generate → sanity stats, all in a single Kaggle kernel session (matters for GPU quota — see [docs/guides/run_full_pipeline.md](docs/guides/run_full_pipeline.md)):
```powershell
uv run python scripts/run_kaggle_full_experiment.py --max-files 40 --whisper-model tiny --baseline-epochs 60 --distill-epochs 30
```

### 5. Run the Distillation-vs-Baseline Comparison Experiment
Trains several configs (baseline; distillation at a few `alpha_feature` values; a smaller architecture variant) against one shared preprocessed dataset, to answer whether distillation actually helps this small model — see [docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md) §3.5:
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
  # Train standard Conv1D denoiser model (legacy/smoke-test baseline):
  uv run python cli.py train-self --dataset dataset/diff_rhythm_dataset --checkpoint outputs/my_model.pt --epochs 5 --batch-size 4
  
  # Train MicroDiT (Transformer-based) model with real MuQ-MuLan Audio Style Anchor conditioning (recommended):
  uv run python cli.py train-self --dataset dataset/diff_rhythm_dataset --checkpoint outputs/my_dit_model.pt --epochs 5 --batch-size 4 --dim 256 --depth 4 --heads 4
  ```
  `--dim`/`--depth`/`--heads`/`--ff-mult` control MicroDiT's architecture size (default: ~5.6M trainable params).

* **Knowledge Distillation:**
  Replicates the real DiffRhythm2 teacher's call contract (see [docs/experiments/distillation_fix.md](docs/experiments/distillation_fix.md)). Needs a clone of the [DiffRhythm2 repo](https://github.com/ASLP-lab/DiffRhythm2) on `PYTHONPATH` with its dependencies installed — done automatically on Kaggle (see `scripts/run_kaggle_distill.py`), or manually locally (see Quick Start step 3 above; verified working on Windows/CPU too, not just Kaggle). Running without that clone, or without internet, falls back to ground-truth-only training and reports this explicitly via `teacher_status`/`distillation_active` in the output — never a silent fake teacher.
  - If `--teacher-checkpoint` is omitted, the script automatically downloads the latest model weights (`model.safetensors`) from the Hugging Face repo: `ASLP-lab/DiffRhythm2`.
  ```powershell
  uv run python cli.py train-distill --dataset dataset/diff_rhythm_dataset --student-checkpoint outputs/distilled_student.pt --epochs 5 --batch-size 4 --alpha-feature 0.5
  ```
  Whether this actually improves quality over training from scratch has **not yet been verified at Kaggle scale** — see [docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md) §3.5/§4 and [docs/guides/run_full_pipeline.md](docs/guides/run_full_pipeline.md) for the comparison experiment designed to answer this.

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
