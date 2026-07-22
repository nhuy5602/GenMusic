# Usage guide

Practical run instructions — preprocessing, training, generation, evaluation,
and the Kaggle automation around all of it. Read `docs/architecture.md` first
if you want the *why* behind these steps; this file is just the *how*. Run
`uv sync` once before anything below.

**All heavy compute (Demucs+Whisper preprocessing, model training,
distillation) should run on Kaggle**, not locally — local execution is
CPU-only and fine for smoke tests, but a real training run needs a GPU.
Kaggle gives a limited number of GPU-hours per week; a hung/misbehaving
kernel silently burns this even though it looks identical to a
slow-but-working one — see "Kaggle infrastructure notes" below before
launching anything long-running.

## 1. One-time setup

Fill in `.env` (copy from `.env.example`) with `KAGGLE_USERNAME` and
`KAGGLE_KEY` (Kaggle account → Settings → API → Create New Token).
`KAGGLE_RAW_DATASET_REF` should point at the raw Vietnamese song corpus.

## 2. Preprocess raw songs into a training dataset

```powershell
uv run python cli.py preprocess-raw --input dataset/vietnamese_songs --output dataset/diff_rhythm_dataset --whisper-model small --max-files 100 --keep-separated-count 10
```

Splits vocal/backing stems (Demucs), transcribes lyrics (Whisper), computes
the Audio Style Anchor (MuQ-MuLan), writes mel tensors in Vocos-native
format. Produces `dataset/diff_rhythm_dataset/{config.json, records.jsonl,
mels/}`. If some files fail, the command returns `completed_with_warnings`
and a non-zero exit code — inspect the printed failure list before training.

For a quick software-only check without downloading audio:
```powershell
uv run python cli.py make-random-dataset --out dataset/random_self_diffusion_training --count 16 --frames 128 --target-gb 1.0
uv run python cli.py validate-dataset --dataset dataset/random_self_diffusion_training
```

On Kaggle (recommended for anything past a smoke test):
```powershell
uv run python scripts/run_kaggle_preprocess_all.py --max-files 40
```

## 3. Train the student

Two independent choices: which **architecture** (`--architecture
microdit|native_dit`, default `microdit`), and which **feature space** (raw
mel, the default; or the native latent space, §3b below). Ground-truth-only
training (no teacher):
```powershell
uv run python cli.py train-self --dataset dataset/diff_rhythm_dataset --checkpoint outputs/my_model.pt --epochs 60 --dim 256 --depth 4 --heads 4
```
Pass a large `--epochs` cap and let early stopping decide when to actually
stop (see `docs/architecture.md`'s training-loop section) rather than
tuning the epoch count by hand.

### 3a. Distillation from the real DiffRhythm2 teacher

```powershell
$env:PYTHONPATH = "C:\path\to\DiffRhythm2"   # clone github.com/ASLP-lab/DiffRhythm2 first
uv run python cli.py train-distill --dataset dataset/diff_rhythm_dataset --student-checkpoint outputs/distilled_student.pt --epochs 60 --dim 256 --depth 4 --heads 4 --alpha-feature 0.8
```
Needs the DiffRhythm2 repo on `PYTHONPATH` with its dependencies installed —
automated on Kaggle (`scripts/run_kaggle_distill.py`), or done manually
locally (clone it yourself, install missing deps as they come up,
`espeak-ng` as a system package for the lyric tokenizer; verified working on
Windows/CPU too). Without that clone, or without internet, `train-distill`
**raises immediately** rather than silently completing as ground-truth-only
training under the distillation name — use `train-self` for that.
`--alpha-feature≈0.8` is a verified-good default (see
`docs/project_history.md` §4.14), not `0.5`.

On Kaggle:
```powershell
uv run python scripts/run_kaggle_training.py --epochs 5 --batch-size 4     # train-self
uv run python scripts/run_kaggle_distill.py                               # train-distill
```
Both fully automate their own GPU environment (cloning DiffRhythm2,
installing `espeak-ng` + Python deps) inside the kernel. Two things are
needed in `.env` first: `KAGGLE_USERNAME`/`KAGGLE_KEY`, and a
processed-dataset reference your own account can access
(`KAGGLE_PROCESSED_KERNEL_REF` preferred, or `KAGGLE_PROCESSED_DATASET_REF`)
— produce one first with `scripts/run_kaggle_preprocess_all.py` if you don't
have one yet.

### 3b. Native latent backbone (optional — gives the student the teacher's own 64-dim/5Hz space)

Three steps on top of an existing mel-space dataset (see
`docs/architecture.md`'s "Native latent backbone" section for why, and
`docs/project_history.md` §4.24 for the bugs found/fixed along the way):

```powershell
# 1. Pretrain a small encoder against the real, frozen BigVGAN decoder (reconstruction loss only)
uv run python cli.py train-latent-encoder --dataset dataset/diff_rhythm_dataset --checkpoint outputs/latent_encoder.pt --epochs 40 --batch-size 4

# 2. Convert the mel dataset into a latent one (64-dim/5Hz) using that encoder
uv run python cli.py precompute-latent-dataset --source-dataset dataset/diff_rhythm_dataset --encoder-checkpoint outputs/latent_encoder.pt --out dataset/latent_dataset

# 3. Train the CFM student inside that latent space
uv run python cli.py train-self --dataset dataset/latent_dataset --checkpoint outputs/latent_cfm_model.pt --architecture native_dit --lambda-vocal 0 --epochs 300 --batch-size 8

# 4. Generate -- decodes via the real frozen BigVGAN decoder automatically (config.latent_mode=True), not Vocos
uv run python cli.py generate-local --text "..." --style "..." --checkpoint outputs/latent_cfm_model.pt --out outputs/latent_demo
```

`--architecture native_dit` requires the DiffRhythm2 repo cloned onto
`PYTHONPATH` (same requirement as distillation above), since `bigvgan` is not
a pip package — needed by both `train-latent-encoder` and step 3's
`train-self`. **Before trusting a freshly (re)trained encoder**, verify it
didn't collapse (a real failure mode hit once already — flat/oscillating
loss curve, near-zero `pitch_std_semitones` when ground-truth latents are
decoded directly, bypassing the CFM student — see
`scripts/evaluate_generation_quality.py` and `docs/project_history.md`
§4.24's before/after numbers).

On Kaggle, in order (see `scripts/README.md` for the full list):
```powershell
uv run python scripts/run_kaggle_latent_encoder.py --epochs 40 --batch-size 4
uv run python scripts/run_kaggle_latent_pipeline.py --encoder-checkpoint outputs/.../latent_encoder.pt --cfm-epochs 300
```
If a CFM training run gets cut off partway (Kaggle sessions have a
wall-clock limit), use `scripts/run_kaggle_latent_resume.py` with a small,
bounded epoch count per round trip rather than restarting from scratch or
re-launching with an unbounded epoch cap — see that script's docstring.

## 4. Generate

```powershell
uv run python cli.py generate-local --text "Đêm nay Hà Nội ngập tràn tiếng mưa rơi." --duration 8.0 --vocoder vocos --checkpoint outputs/my_dit_model.pt --out outputs/my_song
```
`vocos` (default) decodes with the pretrained Vocos neural vocoder;
`griffinlim` is a real iterative-phase-estimation fallback if Vocos is
unavailable. Both require the mel format to match Vocos's native convention
exactly, which this project's default config and `preprocess-raw` output
always do (see `docs/architecture.md`'s "Mel and vocoder" section).

Without `--reference-dataset`, generation falls back to a pooled-text style
vector instead of a real MuQ-MuLan style anchor. To condition the same way
training did:
```powershell
uv run python cli.py generate-local --text "..." --duration 8.0 --checkpoint outputs/my_dit_model.pt --reference-dataset dataset/diff_rhythm_dataset --reference-id <record_id> --out outputs/my_song
```
`--reference-id` defaults to the dataset's first record if omitted. See
`load_reference_conditioning()` in `src/training/self_diffusion.py`.

## 5. Evaluate

```powershell
uv run python cli.py evaluate-self --generated outputs/my_song/final.wav --out outputs/evaluation_report
```
Or the fuller objective-metrics script (spectral flatness, voiced ratio,
pitch-std semitones — see `docs/architecture.md`'s evaluation-boundary note
for what these do and don't tell you):
```powershell
uv run python scripts/evaluate_generation_quality.py outputs/my_dit_model.pt
uv run python scripts/run_kaggle_evaluate.py   # same, on Kaggle
```

## 6. Consolidated Kaggle experiment scripts (recommended over running each stage separately)

Rather than preprocess/train/distill/generate as separate Kaggle round
trips (5 separate GPU-quota sessions), run the whole sequence in one kernel:

```powershell
uv run python scripts/run_kaggle_full_experiment.py --max-files 40 --whisper-model tiny --baseline-epochs 60 --distill-epochs 30
```
Preprocess → vocoder sanity check → baseline train → distillation attempt →
generate → sanity stats, all in one kernel session. It prints a kernel URL
(`https://www.kaggle.com/code/<ref>`) — watch it there, or poll with
`uv run python -m kaggle kernels status <kernel_ref>`. When it shows
`COMPLETE`/`ERROR`, download results with
`uv run python -m kaggle kernels output <kernel_ref> -p outputs/.../downloaded -o`.
**After downloading, delete the redundant echoed source code/DiffRhythm2
clone/teacher checkpoint** from the download — it's fully reproducible and
just bloats local disk. Keep the processed dataset, checkpoints, and
generated audio.

For a finer-grained ablation across `alpha_feature` values and architecture
sizes against one shared preprocessed dataset (the core "does distillation
help" question already has a real answer from a direct 250-song comparison
— see `docs/project_history.md` §4.8/§4.9; this is for follow-up questions):
```powershell
uv run python scripts/run_kaggle_experiment_matrix.py --max-files 40 --whisper-model tiny --epochs 60
```

For training across the full multi-part raw corpus rather than a single
dataset part:
```powershell
uv run python scripts/run_kaggle_all_parts.py
uv run python scripts/run_kaggle_multi_part_training.py
```

Before any of the above, a quick local smoke test (no Kaggle, ~2-3 minutes
on CPU) confirms your environment is set up correctly:
```powershell
uv run python scripts/run_full_experiment.py --raw-dataset dataset/vietnamese_songs --output-root outputs/local_smoke_test --max-files 2 --whisper-model tiny --baseline-epochs 5 --distill-epochs 2 --batch-size 2
```
Check `outputs/local_smoke_test/summary.json` — expect
`preprocess.status: "completed"`, `vocoder_roundtrip.logmel_corr > 0.95`,
and (with no `PYTHONPATH` set) `distillation.status: "failed"` with an error
about the DiffRhythm2 package not being importable — that's expected and
correct, not a bug (see §3a above).

## Kaggle infrastructure notes

- **`kaggle kernels output` only returns files once a kernel finishes** —
  there is no way to inspect a still-running kernel's working directory. Use
  `scripts/check_kernel_progress.py` (reads Kaggle's live SSE log-stream, with
  a read timeout so it doesn't hang forever if nothing new is printed) to
  confirm a job is *actually progressing* (epoch/step increasing), not just
  sitting at status `RUNNING`.
- **If a kernel's own launcher script buffers a training subprocess's output**
  (`subprocess.run(..., capture_output=True)`) instead of streaming it, the
  Kaggle web UI shows nothing new for the entire duration of that subprocess
  — a real multi-hour stall becomes indistinguishable from normal progress
  even by watching the kernel live. The current latent-pipeline launcher
  scripts stream output via `Popen` for exactly this reason; if you write a
  new launcher, do the same.
- **Launch long training runs in small, bounded epoch increments** rather
  than one open-ended "run until early-stopping" call, especially on a
  feature/architecture combination that hasn't been run start-to-finish
  before. A bounded run either completes (proving health) or fails fast;
  an unbounded one can silently consume most of a GPU-quota budget before
  anyone notices something is wrong — this happened for real, twice, in
  this project's history (`docs/project_history.md` §4.5, §4.24).
- **There is no `kaggle kernels stop` command** — recovering from a truly
  stuck kernel means `uv run python -m kaggle kernels delete <kernel_ref>`.
- Kaggle sometimes assigns a P100 GPU (compute capability sm_60) incompatible
  with the preinstalled PyTorch build (`CUDA error: no kernel image is
  available`). The launcher scripts under `scripts/run_kaggle_latent_*.py`
  probe for this and force-reinstall a compatible `torch`/`torchaudio`/
  `torchvision` triple if needed — copy that pattern into any new launcher
  that trains on GPU.

## Staging a one-off Kaggle generation job (`cli.py generate`)

A narrower, older flow than everything above: stages a single lyric request
(with LRC timing, source code, and a link to a fixed training dataset) as a
Kaggle job, rather than running a full experiment. Mostly superseded by
`generate-local` (§4) for anyone with a downloaded checkpoint — use this only
if you specifically want the generation itself to run on Kaggle rather than
locally.
```powershell
uv run python cli.py make-and-upload-dataset --out datasets/random_self_diffusion_1gb --target-gb 1   # one-time: stage a synthetic dataset
uv run python cli.py generate --text "Mot ngay moi bat dau." --duration 12 --wait
```
The dataset reference defaults to `<KAGGLE_USERNAME>/genmusic-vn-self-diffusion-training`
(override via `GENMUSIC_KAGGLE_DATASET_REF` or `--dataset-ref`); the job stops
with an explicit error if that dataset doesn't exist. Synthetic data verifies
the pipeline only — real vocal/backing mel data is required to assess
singing quality.

## Interactive web demo

```powershell
uv run python server.py
```
Open `http://127.0.0.1:8000` to enter Vietnamese prompts and listen to
generated tracks.

## Unit tests

```powershell
uv run python -m unittest discover -s tests -v
```
