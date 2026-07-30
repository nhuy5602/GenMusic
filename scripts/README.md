# Scripts Directory Structure

The `scripts/` directory is organized into modular subdirectories by purpose to keep the codebase clean, maintainable, and clear.

```
scripts/
├── kaggle/          # Kaggle CLI, orchestration, status check, and subset downloading
├── data_prep/       # Dataset preprocessing, phonemization, and native materialization
├── training/        # Model training launchers, distillation, and experiment runners
├── generation/      # Waveform & music generation pipeline and demo scripts
└── evaluation/      # Audio quality evaluation, ASR metrics, and diversity diagnostics
```

---

## 1. `scripts/kaggle/` (Kaggle Integration & Infrastructure)

- **`cli.py`** — Wrapper for project-standard Kaggle CLI execution.
- **`phase_submit.py`** — Universal launcher and coordinator for Kaggle phases and multi-part batches.
- **`download_subset.py`** — Downloads subset outputs safely from Kaggle kernels.
- **`check_progress.py`** — Live progress monitoring for running Kaggle kernels.

---

## 2. `scripts/data_prep/` (Data Preprocessing & Phonemization)

- **`preprocess.py`** — Preprocesses raw audio (Demucs + Whisper) into raw/mel datasets.
- **`multi_part_preprocess.py`** — Multi-part raw audio preprocessing launcher across all parts.
- **`preprocess_all.py`** — Full preprocessing runner for multi-shard corpora.
- **`phonemize.py`** — Stage 2 G2P phonemizer (single part).
- **`multi_part_phonemize.py`** — Multi-part G2P phonemizer coordinator.
- **`materialize_native.py`** — Materializes aligned native waveform datasets.
- **`run_kaggle_native_waveform_dataset.py`** — Kaggle dataset materialization launcher.

---

## 3. `scripts/training/` (Model Training & Distillation)

- **`train.py`** — Main Kaggle training launcher for CFM models.
- **`multi_part_train.py`** — Preprocesses and trains across multiple dataset parts.
- **`train_distill.py`** — Student distillation training with teacher guidance.
- **`distill.py`** / **`latent_encoder.py`** / **`latent_pipeline.py`** / **`latent_resume.py`** — Latent encoder pretraining and pipeline runners.
- **`experiment_matrix.py`** / **`run_experiment_matrix.py`** / **`full_experiment.py`** — Experiment matrix & full pipeline execution.

---

## 4. `scripts/generation/` (Inference & Audio Generation)

- **`generate_native.py`** — Standalone native waveform audio generation CLI.
- **`run_kaggle_native_waveform.py`** — Web-facing Kaggle native waveform generation launcher.
- **`master_waveform_pipeline.py`** — Master waveform pipeline orchestrator.
- **`guidance_demo.py`** — Classifier-free guidance and style embedding demonstration.
- **`latent_generate_only.py`** — Fast spot-check generation from latent checkpoints.

---

## 5. `scripts/evaluation/` (Quality & Diversity Evaluation)

- **`evaluate.py`** — Quality evaluation implementation (WER, ASR, spectral metrics, voiced ratio).
- **`evaluate_kaggle.py`** — Kaggle runner for quality evaluation.
- **`check_latent_encoder.py`** / **`check_latent_encoder_kaggle.py`** — Sanity-checks `LatentAudioEncoder` checkpoints.
- **`check_teacher_diversity.py`** / **`check_latent_diversity.py`** / **`check_style_diversity.py`** — Diagnostic scripts for model diversity.

