# Runtime Dataset Directory

This directory is reserved for local raw audio and generated preprocessing
outputs. Large files are ignored by Git.

Expected processed dataset layout:

```text
diff_rhythm_dataset/
  records.jsonl
  config.json
  mels/<id>_backing.pt
  mels/<id>_vocal.pt
  mels/<id>_style.pt   # MuQ-MuLan style embedding, see src/data/README.md
```

Create a synthetic smoke dataset with:

```powershell
uv run python cli.py make-random-dataset --out datasets/random_self_diffusion_1gb --target-gb 1
```

Preprocess real WAV/MP3 files with:

```powershell
uv run python cli.py preprocess-raw --input dataset/vietnamese_songs --output dataset/diff_rhythm_dataset --whisper-model base
```

Synthetic data checks the dataset contract and training loop. It does not
represent real singing quality.
