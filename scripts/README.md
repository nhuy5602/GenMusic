# Scripts index

Every script here is a local orchestrator: it zips this repo as a Kaggle
dataset, pushes a kernel that installs dependencies and runs the real work,
then downloads the results. None of them contain model logic themselves —
that all lives in `src/` and is reached via `cli.py`. See
`docs/usage.md` for the full walkthrough; this file is just "which script do
I run for X."

## Mel-space pipeline (the default feature space)

- **`run_kaggle_preprocess_all.py`** — batch preprocess raw audio (Demucs +
  Whisper + MuQ-MuLan) into a training dataset.
- **`run_kaggle_training.py`** — train the student with `train-self` (no
  teacher).
- **`run_kaggle_distill.py`** — train the student with `train-distill` (real
  DiffRhythm2 teacher).
- **`run_kaggle_evaluate.py`** — run `evaluate_generation_quality.py`
  (spectral flatness / voiced ratio / pitch-std) against a checkpoint on
  Kaggle.
- **`run_kaggle_full_experiment.py`** (→ remotely runs `run_full_experiment.py`)
  — preprocess → vocoder check → baseline train → distill → generate →
  sanity stats, all in one kernel. The recommended way to run a full
  mel-space experiment; see `docs/usage.md`.
- **`run_kaggle_experiment_matrix.py`** (→ remotely runs
  `run_experiment_matrix.py`) — baseline vs. several `alpha_feature` values
  vs. a smaller architecture, against one shared preprocessed dataset.
- **`run_kaggle_all_parts.py`** / **`run_kaggle_multi_part_training.py`** —
  preprocess and train across multiple dataset parts (for scaling past a
  single-part corpus), a deliberately separate workflow from the
  single-dataset scripts above.
- **`run_pipeline.py`** — local (no Kaggle) end-to-end smoke test:
  preprocess → train → sample, for verifying your environment before
  spending any Kaggle GPU time.
- **`run_kaggle_preprocess_raw_audio.py`** — same preprocessing, but
  `--raw-audio` (skips mel, keeps `waveforms/*.pt` raw 24kHz tensors instead)
  — see `docs/data_preparation.md`'s "`--raw-audio`" section. Not yet
  consumable by `train-self`/`train-latent-encoder` (still expect a mel
  dataset); this is for a planned follow-up letting `LatentAudioEncoder`
  train on the pristine original recording instead of a Vocos reconstruction.

## Native latent pipeline (same `MicroDiT` backbone, `latent_mode` dataset)

Gives the student DiffRhythm2's own compressed 64-dim/5Hz Music VAE latent
space instead of raw mel — see `docs/architecture.md`'s "Native latent
backbone and encoder" section for why, and `docs/project_history.md` §4.24
for what went wrong the first time (a collapsed encoder) and how it was
fixed. Run in this order:

1. **`run_kaggle_latent_encoder.py`** — pretrain `LatentAudioEncoder` against
   the real, frozen BigVGAN decoder (reconstruction loss only). Sanity-check
   the result before proceeding (see `docs/architecture.md`) — a flat/
   oscillating loss curve or near-zero `pitch_std_semitones` on decoded
   ground-truth latents means retrain with more epochs, not move on.
2. **`run_kaggle_latent_pipeline.py`** — precompute the latent dataset with
   that encoder, train the CFM student, generate one sample.
3. **`run_kaggle_latent_resume.py`** — if step 2 gets cut off partway (Kaggle
   sessions have a wall-clock limit), resume CFM training from the
   downloaded checkpoint instead of restarting from scratch. Launch with a
   small, bounded epoch count per round trip — see the script's own
   docstring for why.
4. **`run_kaggle_latent_generate_only.py`** — cheapest way to spot-check any
   existing checkpoint: generates one sample, no training, no dataset
   (~10 minutes). Use this between training rounds instead of a full
   pipeline/resume run just to listen to where a checkpoint currently is.

## Utilities

- **`evaluate_generation_quality.py`** — the actual metric implementation
  (spectral flatness, voiced ratio, pitch-std semitones) used by
  `run_kaggle_evaluate.py` and referenced throughout `docs/project_history.md`.
  Can also be run standalone against any local checkpoint/wav.
- **`check_kernel_progress.py`** — tails a *running* Kaggle kernel's log via
  the SSE log-stream endpoint (`kaggle kernels output` only returns files
  once a kernel finishes, so this is the only way to see live progress).
