# GenMusic VN: Background Music & Vietnamese Vocal Generator

GenMusic VN is a self-authored conditional diffusion and flow matching model designed for generating Vietnamese vocals and background music.

Detailed technical documentations are located in the `docs/` folder:
- [System Architecture](docs/architecture.md)
- [Machine Learning Models](docs/model.md)
- [Training & Improvement Pipelines](docs/training.md)

---

## 📂 Project Directory Structure

```text
GenMusic/
├── src/                # Core Python package (data processing, model, training)
│   ├── data/           # Audio preprocessing, Whisper ASR, Demucs split, Vietnamese G2P
│   ├── models/         # Diffusion architectures (ResidualDenoiser, MicroDiT, CFM)
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
  pip install -r requirements.txt
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
KAGGLE_PROCESSED_DATASET_REF=your_kaggle_username/vietnamese-music-processed-dataset
```

---

## 🚀 Usage Guide for Automated Scripts

All key workflows are packaged into automated shell scripts in the `scripts/` directory:

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
Perform full stem separation (Demucs), transcribing (Whisper), and pitch contour tracking (pYIN) on all 250 raw audio tracks on Kaggle. Automatically uploads the result directly to your Kaggle profile as a clean dataset:
```powershell
uv run python scripts/run_kaggle_preprocess_all.py
```

---

## 🎹 Advanced CLI Operations (`cli.py`)

### 1. Local Generation (Inference)
You can choose between the fast mathematical `istft` decoder or the high-fidelity neural vocoder **Vocos** (`vocos` - highly recommended):
```powershell
# Generate audio using the Vocos Vocoder for natural voice reconstruction:
uv run python cli.py generate-local --text "Đêm nay Hà Nội ngập tràn tiếng mưa rơi." --duration 8.0 --vocoder vocos --out outputs/my_song
```

### 2. Manual Preprocessing
```powershell
uv run python cli.py preprocess-raw --input dataset/vietnamese_songs --output dataset/diff_rhythm_dataset --whisper-model small
```

### 3. Model Training & Knowledge Distillation
You can train the diffusion denoiser from scratch or perform knowledge distillation from the pretrained DiffRhythm Teacher to your student MicroDiT model:

* **Train Model from Scratch:**
  ```powershell
  # Train standard Conv1D denoiser model:
  uv run python cli.py train-self --dataset dataset/diff_rhythm_dataset --checkpoint outputs/my_model.pt --epochs 5 --batch-size 4
  
  # Train MicroDiT (Transformer-based) model:
  uv run python cli.py train-self --dataset dataset/diff_rhythm_dataset --checkpoint outputs/my_dit_model.pt --epochs 5 --batch-size 4 --model-type dit
  ```

* **Knowledge Distillation (Recommended for small datasets):**
  This maps predictions from a pretrained DiffRhythm Teacher to your student MicroDiT, bypassing the need for huge datasets:
  ```powershell
  uv run python cli.py train-distill --dataset dataset/diff_rhythm_dataset --student-checkpoint outputs/distilled_student.pt --teacher-checkpoint outputs/pretrained_teacher.pt --epochs 5 --batch-size 4
  ```

### 4. Audio Quality Evaluation
```powershell
uv run python cli.py evaluate-self --generated outputs/my_song/final.wav --out outputs/evaluation_report
```

---

## 🖥️ Interactive Web UI Demo

Start the FastAPI backend server:
```powershell
uv run python server.py
```
Open your browser and navigate to `http://127.0.0.1:8000` to enter custom Vietnamese prompts and listen to generated musical tracks.

---

## 🧪 Unit Testing

Run automated tests to verify model math and system stability:
```powershell
uv run python -m unittest discover -s tests -v
```
