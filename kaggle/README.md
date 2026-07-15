# Kaggle Integration

Kaggle is the required GPU backend for real preprocessing/training/distillation
work on this project (local execution is CPU-only, for smoke tests). **See
`docs/guides/run_full_pipeline.md` for the current recommended workflow** —
`scripts/run_kaggle_full_experiment.py` and
`scripts/run_kaggle_experiment_matrix.py` run the whole
preprocess→train→distill→generate sequence in one Kaggle kernel, which matters
for GPU quota (Kaggle gives ~30 GPU-hours/week; one consolidated kernel per
experiment burns far less than five separate round trips). This file documents
the older, narrower single-song "generate" job flow below, kept for staging a
one-off request against an already-trained checkpoint rather than a full
experiment run.

The local project stages a request dataset containing the lyric request,
LRC timing, source code, and links to the fixed training dataset.

## Dataset

The default reference is:

```text
<KAGGLE_USERNAME>/genmusic-vn-self-diffusion-training
```

Override it with `GENMUSIC_KAGGLE_DATASET_REF` or `--dataset-ref owner/slug`.
The job stops with an explicit error when that dataset does not exist.

Create and upload a configurable synthetic dataset:

```powershell
uv run python cli.py make-and-upload-dataset --out datasets/random_self_diffusion_1gb --target-gb 1
```

## Stage and Submit

```powershell
uv run python cli.py generate --text "Mot ngay moi bat dau." --duration 12 --no-submit
uv run python cli.py generate --text "Mot ngay moi bat dau." --duration 12 --wait
```

Kaggle requires `KAGGLE_USERNAME`, `KAGGLE_KEY`, and a working Kaggle CLI.
Synthetic data verifies the pipeline only; real vocal/backing Mel data is
required to assess singing quality.
