# GenMusic VN: Background Music & Vietnamese Vocal Generator

GenMusic VN is a self-authored text-to-music conditional diffusion model.

Detailed technical documentations are located in the `docs/` folder:
- [System Architecture](docs/architecture.md)
- [Machine Learning Models](docs/model.md)
- [Training & Improvement Pipelines](docs/training.md)

---

## 📂 Project Directory Overview

```text
GenMusic/
├── genmusic_vn/        # Main Python package (pipeline, CLI, server)
├── data/               # Stylebanks and model checkpoints (formerly datasets)
├── frontend/           # Web client UI (HTML, CSS, JS) (formerly web)
├── docs/               # Detailed system documentations
├── kaggle_deploy/      # Kaggle deployment config (formerly kaggle)
├── outputs/            # Local and downloaded outputs
└── tests/              # Automated unit tests
```

---

## 🛠️ Installation & Setup

### Install Dependencies

* **Using `uv` (Recommended):**
  ```powershell
  uv sync
  # Or editable install:
  uv pip install -e ".[self]"
  ```

* **Using standard `pip`:**
  ```powershell
  pip install -e ".[self]"
  ```

### Setup Kaggle Credentials
Create a `.env` or `.env.local` file in your project root:
```env
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

---

## 🚀 Usage Guide

### 1. Create Dataset and Train
* **Using `uv`:**
  ```powershell
  uv run python -m genmusic_vn.cli make-random-dataset --out data/random_self_diffusion --count 16 --frames 128
  uv run python -m genmusic_vn.cli train-self --dataset data/random_self_diffusion --checkpoint outputs/self_music.pt --epochs 1 --batch-size 4
  ```

* **Without `uv`:**
  ```powershell
  python -m genmusic_vn.cli make-random-dataset --out data/random_self_diffusion --count 16 --frames 128
  python -m genmusic_vn.cli train-self --dataset data/random_self_diffusion --checkpoint outputs/self_music.pt --epochs 1 --batch-size 4
  ```

### 2. Local Generation
* **Using `uv`:**
  ```powershell
  uv run python -m genmusic_vn.cli generate-local --text "Mưa rơi nhẹ nhàng, em còn nhớ con đường xưa." --style "Vietnamese pop ballad, warm piano" --duration 4 --checkpoint outputs/self_music.pt --out outputs/local_self_music
  ```

* **Without `uv`:**
  ```powershell
  python -m genmusic_vn.cli generate-local --text "Mưa rơi nhẹ nhàng, em còn nhớ con đường xưa." --style "Vietnamese pop ballad, warm piano" --duration 4 --checkpoint outputs/self_music.pt --out outputs/local_self_music
  ```

### 3. Stage Kaggle Job
* **Using `uv`:**
  ```powershell
  uv run python -m genmusic_vn.cli generate --text "Một ngày mới bắt đầu." --duration 12 --no-submit
  ```

* **Without `uv`:**
  ```powershell
  python -m genmusic_vn.cli generate --text "Một ngày mới bắt đầu." --duration 12 --no-submit
  ```

### 4. Create and Upload Dataset (~1 GB)
* **Using `uv`:**
  ```powershell
  uv run python -m genmusic_vn.cli make-and-upload-dataset --out data/random_self_diffusion_training --target-gb 1
  ```

* **Without `uv`:**
  ```powershell
  python -m genmusic_vn.cli make-and-upload-dataset --out data/random_self_diffusion_training --target-gb 1
  ```
*(You can change the target size using `--target-gb 5` or change dataset reference using `--dataset-ref owner/slug`)*

### 5. Web Interface
* **Using `uv`:**
  ```powershell
  uv run python -m genmusic_vn.server --port 8000
  ```

* **Without `uv`:**
  ```powershell
  python -m genmusic_vn.server --port 8000
  ```
Open your browser at `http://127.0.0.1:8000`.

### 6. Evaluation
* **Using `uv`:**
  ```powershell
  uv run python -m genmusic_vn.cli evaluate-self --generated outputs/local_self_music/final.wav --out outputs/self_evaluation
  ```

* **Without `uv`:**
  ```powershell
  python -m genmusic_vn.cli evaluate-self --generated outputs/local_self_music/final.wav --out outputs/self_evaluation
  ```

> [!NOTE]
> The random model is only for smoke testing. To improve the generation quality, the model needs to be trained on a licensed audio/mel and lyric dataset.

---

## 🧪 Testing

Run the automated unit tests to verify system stability:

* **Using `uv`:**
  ```powershell
  uv run python -m unittest discover -s tests -v
  ```

* **Without `uv`:**
  ```powershell
  python -m unittest discover -s tests -v
  ```
