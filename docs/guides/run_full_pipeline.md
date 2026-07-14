# Guide: running the full distillation pipeline

This is a practical, step-by-step guide for running preprocessing → training →
distillation → generation, either as a quick local correctness check or as a real
Kaggle GPU run. Read `docs/PROJECT_REPORT.md` and `docs/experiments/*.md` first if you
want the *why* behind these steps — this doc is just the *how*.

## TL;DR: the 4 commands

Run `uv sync` once first. Everything below assumes you're in the project root.

**1. Preprocess raw songs → training dataset:**
```powershell
uv run python cli.py preprocess-raw --input dataset/vietnamese_songs --output dataset/diff_rhythm_dataset --whisper-model tiny
```

**2. Train the student model (baseline, no teacher):**
```powershell
uv run python cli.py train-self --dataset dataset/diff_rhythm_dataset --checkpoint outputs/my_model.pt --model-type dit --epochs 60 --dim 256 --depth 4 --heads 4
```

**3. Distill from the real DiffRhythm2 teacher instead of step 2:**
```powershell
$env:PYTHONPATH = "C:\path\to\DiffRhythm2"   # clone from github.com/ASLP-lab/DiffRhythm2 first
uv run python cli.py train-distill --dataset dataset/diff_rhythm_dataset --student-checkpoint outputs/distilled_model.pt --epochs 60 --dim 256 --depth 4 --heads 4
```

**4. Download the checkpoint (if trained on Kaggle) and run inference:**
```powershell
uv run python -m kaggle kernels output <kernel_ref> -p outputs/downloaded -o    # skip if trained locally
uv run python cli.py generate-local --text "Đêm nay Hà Nội ngập tràn tiếng mưa rơi." --checkpoint outputs/my_model.pt --model-type dit --vocoder vocos --duration 8 --out outputs/my_song
```
Output lands at `outputs/my_song/final.wav`.

Everything past this point is detail and troubleshooting for running the same 4 steps
at Kaggle GPU scale — read on if something above doesn't work, or you need real
training scale rather than a quick check.

## 0. One-time setup

```powershell
uv sync
```

Fill in `.env` (copy from `.env.example`) with `KAGGLE_USERNAME` and `KAGGLE_KEY` (from
your Kaggle account → Settings → API → Create New Token). `KAGGLE_RAW_DATASET_REF`
should already point at the raw Vietnamese song corpus
(`sonlest/vietnamese-music-dataset-version3-part6`).

**Check your Kaggle GPU quota before starting anything big.** Kaggle gives ~30 GPU-hours
per week (resets weekly); a hung/misbehaving kernel silently burns this even though it
looks identical to a slow-but-working one (see the ~11-hour hang in
`docs/experiments/kaggle_runs.md` — caught only because someone happened to check
elapsed wall-clock time). If quota is exhausted, everything below still works locally
(§1), just slower and at much smaller scale.

## 1. Quick local smoke test (no Kaggle, ~2-3 minutes on CPU)

Confirms your environment is set up correctly before spending any Kaggle GPU time. Uses
the 2 real songs already in `dataset/vietnamese_songs/`.

```powershell
uv run python scripts/run_full_experiment.py --raw-dataset dataset/vietnamese_songs --output-root outputs/local_smoke_test --max-files 2 --whisper-model tiny --baseline-epochs 5 --distill-epochs 2 --batch-size 2
```

Check `outputs/local_smoke_test/summary.json` afterward. Expect:
- `preprocess.status: "completed"`, `records_count: 2`.
- `vocoder_roundtrip.logmel_corr` > 0.95 (this is the regression test for the vocoder
  distortion bug — if it drops well below this, something broke the mel/vocoder match).
- `distillation.distillation_active: false` locally is **expected and correct** — the
  real DiffRhythm2 teacher isn't cloned locally, so the honest fallback should kick in
  (`teacher_status` will say so explicitly). Seeing `distillation_active: true` here
  would actually be suspicious.
- `generation.baseline.sanity_stats` and `generation.distilled.sanity_stats`: both
  `has_nan_or_inf: false`, `silence_ratio` well under 1%.

If any of these look wrong, don't proceed to a real Kaggle run yet — something in the
environment or the code changed in a way that needs fixing first.

## 2. Real Kaggle run: preprocessing at scale

This is the expensive step (Demucs + Whisper + MuQ-MuLan over however many songs you
choose). Start small (`--max-files`) to sanity-check before committing to the full
corpus.

```powershell
uv run python scripts/run_kaggle_full_experiment.py --max-files 40 --whisper-model tiny --baseline-epochs 60 --distill-epochs 30
```

This one script runs the *entire* pipeline (preprocess → vocoder check → baseline train
→ distillation attempt → generate → sanity stats) in a **single Kaggle kernel**, which
matters for quota: one kernel session, not five separate round trips. It prints a
kernel URL (`https://www.kaggle.com/code/<ref>`) — watch it there, or poll:

```powershell
uv run python -m kaggle kernels status <kernel_ref>
```

When it shows `KernelWorkerStatus.COMPLETE` (or `ERROR`), download results:

```powershell
uv run python -m kaggle kernels output <kernel_ref> -p outputs/kaggle_full_experiment/<run>/downloaded_output -o
```

Check `summary.json` in the downloaded output the same way as step 1, but now on real
Kaggle-scale data. **After downloading, delete the redundant echoed source
code/DiffRhythm2 clone/teacher checkpoint** from the download (`downloaded_output/GenMusic/`)
— it's fully reproducible and was responsible for an 8.8GB→510MB local disk bloat once
already this session. Keep `processed_dataset/`, the checkpoints, and the generated
audio.

**If a kernel seems stuck at "RUNNING" for far longer than expected** (preprocessing 12
songs + training took ~14 minutes in this session's successful runs — if you're well
past an hour with no sign of progress, something is wrong, not just slow): there is no
`kaggle kernels stop` command. Kill it with:

```powershell
uv run python -m kaggle kernels delete <kernel_ref>
```

then check `docs/experiments/kaggle_runs.md`'s "Run 3" entry for the specific bug that
caused this once already (an unrelated import triggering a CUDA JIT compile) — if a new
hang doesn't match that story, treat it as a new bug and investigate before just
resubmitting blindly (which would burn more quota on the same problem).

## 3. The comparison experiment: does distillation actually help?

This is the experiment that answers "is distillation worth it for this small model,"
which was scoped but **not completed** in this session (Kaggle quota ran out first —
see `docs/PROJECT_REPORT.md` §3.5/§4). It trains 6 configs (baseline; distillation at
`alpha_feature` = 0.2/0.5/0.8; a smaller architecture variant × baseline/distillation)
against one shared preprocessed dataset, so preprocessing only happens once.

```powershell
uv run python scripts/run_kaggle_experiment_matrix.py --max-files 40 --whisper-model tiny --epochs 60 --batch-size 4
```

**How to read the results** (`summary.json` → `configs.<name>.training.loss_curve`):
compare `loss_gt` (not the blended `loss`) across configs at the same epoch — this is
the ground-truth CFM loss, directly comparable whether or not a teacher was used (see
`docs/PROJECT_REPORT.md` §2.2 for why). If a `distill_*` config's `loss_gt` trends lower
than `baseline_no_distill` at the same epoch for the same architecture size, that's
evidence distillation is helping this student converge faster/better. Also check
`configs.<name>.training.distillation_active` — if it's `false` for a `distill_*`
config, the teacher didn't actually load on Kaggle and that config's numbers are really
just another ground-truth-only baseline, not a real distillation result (compare against
`teacher_status` for why).

With only 40 songs, don't expect statistically clean signal — this is still a
small-data validation, not a rigorous ablation. Scaling `--max-files` up (the full raw
corpus) and `--epochs` up is the natural next step once this smaller run confirms the
machinery works.

## 4. Listening to results

Generated audio lands as both `.wav` and `.mp3` under
`outputs/<run>/generated_<config>/final.{wav,mp3}` (Kaggle) or
`outputs/local_smoke_test/generated_<config>/final.{wav,mp3}` (local). `sanity_stats` in
`summary.json` (peak amplitude, RMS, silence ratio, NaN/Inf check) catches
crashes/degenerate output automatically, but **only a human listening can judge musical
quality** — sanity stats passing is a necessary, not sufficient, signal.

`outputs/demo_audio/` has reference examples from this session: files 0–3 demonstrate
the vocoder distortion bug and its fix (before/after on the same real song content —
listen to these first, they're the clearest illustration), files 4–8 are actual
model-generated output at various (small) training scales, clearly labeled
`UNDERTRAINED`/`TINY_*SONGS` where applicable.

## 5. Once you have a config you like

Train it standalone (not inside the matrix/experiment scripts) for a real training run:

```powershell
# Baseline (no distillation):
uv run python cli.py train-self --dataset <processed_dataset_dir> --checkpoint outputs/final_model.pt --model-type dit --epochs 200 --dim 256 --depth 4 --heads 4

# With distillation:
uv run python cli.py train-distill --dataset <processed_dataset_dir> --student-checkpoint outputs/final_model.pt --epochs 200 --alpha-feature 0.5 --dim 256 --depth 4 --heads 4
```

Then generate:

```powershell
uv run python cli.py generate-local --text "<Vietnamese lyrics>" --checkpoint outputs/final_model.pt --model-type dit --vocoder vocos --duration 12 --out outputs/final_song
```

Both of these need to run wherever the checkpoint is (Kaggle kernel, or download the
checkpoint and run `generate-local` locally — generation itself is cheap enough to run
on CPU, unlike preprocessing/training).
