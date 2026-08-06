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
`KAGGLE_API_TOKEN` (or the legacy `KAGGLE_KEY`). No owner/slug from a
developer account is committed.

For the default V80 demo, bootstrap the raw corpus once in the new account:

```powershell
uv run python scripts/run_kaggle_native_waveform_dataset.py --wait
```

The command writes the submitted corpus ref to ignored
`.genmusic/kaggle.json`; CLI and web generation resolve it automatically.

## 2. Preprocess raw songs into a training dataset

```powershell
uv run python cli.py preprocess-raw --input dataset/vietnamese_songs --output dataset/diff_rhythm_dataset --whisper-model small --max-files 100 --keep-separated-count 10
```

Splits vocal/backing stems (Demucs), transcribes lyrics (Whisper), computes
the Audio Style Anchor (MuQ-MuLan), writes mel tensors in Vocos-native
format. Produces `dataset/diff_rhythm_dataset/{config.json, records.jsonl,
mels/}`. If some files fail, the command returns `completed_with_warnings`
and a non-zero exit code — inspect the printed failure list before training.
The report-aligned default also fails if the real MuQ-MuLan model cannot
produce a finite 512-dimensional anchor. `--allow-zero-style` is only an
explicit debug escape hatch and must not be used to reproduce the report.

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

One independent choice: which **feature space** (raw mel, the default; or
the native latent space, §3b below) — the student backbone (`MicroDiT`) is
the same either way. Ground-truth-only training (no teacher):
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
prior measurements), not `0.5`.

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

### 3b. Native latent space (optional — gives the student the teacher's own 64-dim/5Hz space)

Three steps on top of an existing mel-space dataset (see
`docs/architecture.md`'s "Native latent backbone and encoder" section for
why, and prior measurements for the bugs found/fixed along the
way). Same `MicroDiT` backbone as mel-space, no architecture flag needed:

```powershell
# 1. Pretrain a small encoder against the real, frozen BigVGAN decoder
#    (reconstruction loss + KL divergence -- a real VAE bottleneck, see
#    docs/architecture.md's "sec:vae_bottleneck" section for why this matters)
uv run python cli.py train-latent-encoder --dataset dataset/diff_rhythm_dataset --checkpoint outputs/latent_encoder.pt --epochs 40 --batch-size 4 --kl-weight 0.15 --num-workers 4

# 2. Sanity-check the encoder BEFORE trusting it downstream (see note below)
uv run python scripts/check_latent_encoder_quality.py --encoder-checkpoint outputs/latent_encoder.pt --dataset dataset/diff_rhythm_dataset

# 3. Convert the mel dataset into a latent one (64-dim/5Hz) using that encoder
uv run python cli.py precompute-latent-dataset --source-dataset dataset/diff_rhythm_dataset --encoder-checkpoint outputs/latent_encoder.pt --out dataset/latent_dataset

# 4. Train the CFM student inside that latent space
uv run python cli.py train-self --dataset dataset/latent_dataset --checkpoint outputs/latent_cfm_model.pt --lambda-vocal 0 --epochs 300 --batch-size 8

# 5. Generate -- decodes via the real frozen BigVGAN decoder automatically (config.latent_mode=True), not Vocos
uv run python cli.py generate-local --text "..." --style "..." --checkpoint outputs/latent_cfm_model.pt --out outputs/latent_demo
```

Steps 1, 2, and 5 require the DiffRhythm2 repo cloned onto `PYTHONPATH` (same
requirement as distillation above), since `bigvgan` is not a pip package —
needed to load the real frozen decoder they all call. **Before trusting a
freshly (re)trained encoder**, always run step 2: a real failure mode hit
twice already at scale is a collapsed encoder (flat/oscillating loss curve,
near-zero `pitch_std_semitones` when ground-truth latents are decoded
directly, bypassing the CFM student). Root cause was a missing probabilistic
bottleneck; fixed by adding `mu`/`logvar` + reparameterization + KL loss
(the `--kl-weight` flag in step 1) — see prior measurements
for the before/after numbers. If audio decoded through the encoder sounds
crackly despite a healthy `pitch_std_semitones`, that's a separate issue in
BigVGAN's chunked `decode_audio` overlap parameter, not the encoder itself —
see §4.31. Distillation (`train-distill`, §3a) also works directly on a
`latent_dataset`: since the student's latent is already at the teacher's own
64-dim/5Hz rate, the mel-bin/frame-rate bridging described in §3a is
automatically skipped.

On Kaggle, in order (see `scripts/README.md` for the full list):
```powershell
uv run python scripts/run_kaggle_latent_encoder.py --epochs 40 --batch-size 4 --kl-weight 0.15
uv run python scripts/run_kaggle_check_latent_encoder_quality.py --encoder-checkpoint outputs/.../latent_encoder.pt
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
— see prior measurements; this is for follow-up questions):
```powershell
uv run python scripts/run_kaggle_experiment_matrix.py --max-files 40 --whisper-model tiny --epochs 60
```

For training across the full multi-part raw corpus rather than a single
dataset part:
```powershell
uv run python scripts/run_kaggle_multi_part_training.py
```
(there is currently no mel-space multi-part *preprocessing* script -- the
`--raw-audio` corpus has one, `scripts/run_kaggle_multi_part_preprocess_raw_audio.py`,
see `docs/data_preparation.md`'s "`--raw-audio`" section, but it produces a
raw-waveform dataset, not the mel dataset `run_kaggle_multi_part_training.py`
expects.)

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
  feature-space/size combination that hasn't been run start-to-finish
  before. A bounded run either completes (proving health) or fails fast;
  an unbounded one can silently consume most of a GPU-quota budget before
  anyone notices something is wrong — this happened for real, twice, in
  this project's history (prior measurements).
- **There is no `kaggle kernels stop` command** — recovering from a truly
  stuck kernel means `uv run python -m kaggle kernels delete <kernel_ref>`.
- Kaggle sometimes assigns a P100 GPU (compute capability sm_60) incompatible
  with the preinstalled PyTorch build (`CUDA error: no kernel image is
  available`). The launcher scripts under `scripts/run_kaggle_latent_*.py`
  probe for this and force-reinstall a compatible `torch`/`torchaudio`/
  `torchvision` triple if needed — copy that pattern into any new launcher
  that trains on GPU.

## Staging a one-off Kaggle generation job (`cli.py generate`)

The default product route is V80, matching the web app. It creates a private
Kaggle request and uses the validated 16-second waveform pipeline:

```powershell
uv run python cli.py generate --text "Một ngày mới bắt đầu trong nắng mai dịu dàng" --duration 16 --wait
```

To stage the CFM research flow instead, opt in explicitly:

```powershell
uv run python cli.py generate --backend cfm-research --text "..." --duration 12 --wait
```

`--model` and `--dataset-ref` apply only to `cfm-research`. This explicit
switch prevents a V80 demo from being mislabeled as a MicroDiT result.

## Interactive web demo

```powershell
uv run python server.py
```
Open `http://127.0.0.1:8000` to enter Vietnamese prompts and listen to
V80-generated tracks. Check `http://127.0.0.1:8000/api/health` before a demo;
it must report `native-waveform-v80`.

## Unit tests

```powershell
uv run --with pytest python -m pytest -q
```
