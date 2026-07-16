# Training and Evaluation

## 1. Prepare Audio Data

Raw WAV/MP3 files are discovered recursively. Demucs separates vocals and
backing, while Whisper supplies the lyric transcript:

```powershell
uv run python cli.py preprocess-raw --input dataset/vietnamese_songs --output dataset/diff_rhythm_dataset --whisper-model small --max-files 100 --keep-separated-count 10
```

If some files fail, the command returns `completed_with_warnings` and a non-zero
exit code. Inspect the printed failure list before training.

## 2. Create a Smoke Dataset

Use synthetic Mel tensors to verify the pipeline without downloading audio:

```powershell
uv run python cli.py make-random-dataset --out dataset/random_self_diffusion_training --count 16 --frames 128 --target-gb 1.0
uv run python cli.py validate-dataset --dataset dataset/random_self_diffusion_training
```

The synthetic dataset validates software behavior only; it is not a substitute
for real singing data.

## 3. Train the Self-authored Model

```powershell
uv run python cli.py train-self --dataset dataset/random_self_diffusion_training --checkpoint outputs/self_music_checkpoint.pt --epochs 2 --batch-size 4
```

For separated vocal/backing records, use the optional MicroDiT path:

```powershell
uv run python cli.py train-self --dataset dataset/diff_rhythm_dataset --checkpoint outputs/microdit.pt --epochs 2 --batch-size 2
```

The MicroDiT path may download its text encoder on first use and therefore
requires network access unless the encoder is already cached.

## 4. Distillation

```powershell
uv run python cli.py train-distill --dataset dataset/diff_rhythm_dataset --student-checkpoint outputs/distilled_student.pt --epochs 5 --batch-size 4 --alpha-feature 0.5 --dim 256 --depth 4 --heads 4
```

Without `--teacher-checkpoint`, this downloads the real teacher from
`ASLP-lab/DiffRhythm2` on HuggingFace. **Needs the DiffRhythm2 GitHub repo cloned
onto `PYTHONPATH` with its dependencies installed** — automated on Kaggle (see
`scripts/run_kaggle_distill.py` / `scripts/run_kaggle_experiment_matrix.py`), or
doable locally by cloning it yourself (verified working on Windows/CPU too, not
Kaggle-only). Without that clone, or without internet, `train-distill` **raises
immediately** rather than silently completing as ground-truth-only training
under the distillation name (use `train-self` for that) — never a silent fake
teacher, never a silent downgrade either. `--alpha-feature` blends
teacher-matching loss vs. ground-truth loss (`1.0` = ground-truth only).
See `docs/experiments/distillation_fix.md` for the real teacher call contract
this replicates, and `docs/PROJECT_REPORT.md` §4.8 for the comparison
experiment (`scripts/run_experiment_matrix.py`) meant to answer whether this
actually helps a small model — not yet completed at Kaggle scale as of this
writing.

## 5. Generate and Evaluate

```powershell
uv run python cli.py generate-local --text "Dem nay thanh pho ngu quen trong tieng mua." --style "soft Vietnamese ballad" --duration 8 --checkpoint outputs/self_music_checkpoint.pt --vocoder vocos --out outputs/demo
uv run python cli.py evaluate-self --generated outputs/demo/final.wav --out outputs/demo/evaluation
uv run python cli.py project-report --source outputs --out outputs/project_report
```

`--vocoder vocos` (the default) requires the mel format to match Vocos's native
convention exactly, which `preprocess-raw`'s output always does now; `griffinlim`
is the fallback if Vocos is unavailable. There is no `istft` option anymore —
that path fabricated a fake phase spectrum and produced badly distorted audio
regardless of model quality (see `docs/experiments/vocoder_fix.md`).

## 6. Consolidated Kaggle experiment scripts (recommended over the steps above)

Rather than running preprocess/train/distill/generate as separate Kaggle round
trips, `scripts/run_kaggle_full_experiment.py` and
`scripts/run_kaggle_experiment_matrix.py` run the whole sequence in one Kaggle
kernel (one GPU-quota session, not several). See
`docs/guides/run_full_pipeline.md` for exact commands and how to read the
results, and `docs/experiments/kaggle_runs.md` for the run log this project
has already accumulated using them (including a real ~11-hour hang and its
fix — worth reading before submitting a new large run).

Evaluation writes objective metrics and plots. MOS/CMOS remain skipped unless a
human listening survey is supplied.
