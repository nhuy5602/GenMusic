# Kaggle Integration

Kaggle is an optional GPU backend for the self-authored conditional diffusion
model. The local project stages a request dataset containing the lyric request,
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
