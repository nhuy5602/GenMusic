# Scripts index

Model and training logic lives in `src/`; scripts are thin entrypoints around
that code. Most `run_kaggle_*.py` files package a bounded Kaggle job. See
`docs/usage.md` for the full walkthrough.

## Portable V80 demo

- **`run_kaggle_native_waveform_dataset.py`** — one-time bootstrap for a new
  Kaggle account. It packages the current clone, builds the 2,048-song raw
  vocal/backing corpus, and saves the resulting ref in ignored local config.
- **`run_kaggle_native_waveform.py`** — submits one V80 generation using that
  completed corpus; `cli.py generate` and the web app call the same path.
- **`check_portability.py`** — checks submission-visible files for local paths,
  credentials, and report-only artifacts.

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
- **`run_kaggle_multi_part_training.py`** — preprocess and train across
  multiple dataset parts (for scaling past a single-part corpus), a
  deliberately separate workflow from the single-dataset scripts above.
- **`run_kaggle_preprocess_raw_audio.py`** — same preprocessing, but
  `--raw-audio` (skips mel, keeps `waveforms/*.pt` raw 24kHz tensors instead)
  — see `docs/data_preparation.md`'s "`--raw-audio`" section. Consumable by
  `train-latent-encoder`/`precompute-latent-dataset` (both detect
  `raw_audio_mode: true` and sum the vocal/backing waveform tensors directly,
  skipping Vocos); still NOT consumable by `train-self` (that path still
  expects a mel dataset).
- **`run_kaggle_multi_part_preprocess_raw_audio.py`** — the `--raw-audio`
  analogue of `run_kaggle_multi_part_training.py`: submits
  `run_kaggle_preprocess_raw_audio.py`'s kernel across every part in
  `RAW_DATASETS` (`src/integrations/kaggle_dataset_refs.py`), respecting
  Kaggle's 2-concurrent-batch-GPU-session limit, with `--wait-and-loop` to
  auto-chain through the rest and a `submitted_state.json` dedup tracker so
  reruns skip already-submitted parts.

## Native latent pipeline (same `MicroDiT` backbone, `latent_mode` dataset)

Gives the student DiffRhythm2's own compressed 64-dim/5Hz Music VAE latent
space instead of raw mel — see `docs/architecture.md`'s "Native latent
backbone and encoder" section for why, and prior measurements
for what went wrong the first time (a collapsed encoder) and how it was
fixed. Run in this order:

1. **`run_kaggle_latent_encoder.py`** — pretrain `LatentAudioEncoder` against
   the real, frozen BigVGAN decoder with reconstruction plus cyclical KL
   (report-aligned default maximum beta `0.15`). Sanity-check
   the result before proceeding (see `docs/architecture.md`) — a flat/
   oscillating loss curve or near-zero `pitch_std_semitones` on decoded
   ground-truth latents means retrain with more epochs, not move on. Accepts
   several `--processed-kernel-ref`s at once (combined into one training set,
   `N` usable records from each via `--max-records-per-dataset`) — or the
   shortcut `--raw-audio-part 1 2 3 4 5 6` when the corresponding ignored
   `GENMUSIC_PROCESSED_RAW_KERNEL_PART_N` environment variables are set.
2. **`run_kaggle_latent_pipeline.py`** — precompute the latent dataset with
   that encoder, train the CFM student, generate one sample. Also has the
   `--raw-audio-part` shortcut, but only a single one (`precompute-latent-dataset`
   doesn't yet combine multiple source datasets the way `train-latent-encoder` does).
3. **`run_kaggle_latent_resume.py`** — if step 2 gets cut off partway (Kaggle
   sessions have a wall-clock limit), resume CFM training from the
   downloaded checkpoint instead of restarting from scratch. Launch with a
   small, bounded epoch count per round trip — see the script's own
   docstring for why.
4. **`run_kaggle_latent_generate_only.py`** — cheapest way to spot-check any
   existing checkpoint: generates one sample, no training, no dataset
   (~10 minutes). Use this between training rounds instead of a full
   pipeline/resume run just to listen to where a checkpoint currently is.
- **`run_kaggle_check_latent_encoder_quality.py`** — Kaggle launcher for
  `check_latent_encoder_quality.py` (see Utilities below): uploads an
  encoder checkpoint and runs the ground-truth encode/decode sanity check on
  Kaggle (needs the real `bigvgan` decoder). Run this after step 1 and before
  trusting the encoder for step 2.

## Utilities

- **`evaluate_generation_quality.py`** — the actual metric implementation
  (spectral flatness, voiced ratio, pitch-std semitones) used by
  `run_kaggle_evaluate.py` and referenced throughout prior measurements.
  Can also be run standalone against any local checkpoint/wav.
- **`check_latent_encoder_quality.py`** — sanity-check a `LatentAudioEncoder`
  checkpoint (from `run_kaggle_latent_encoder.py`) BEFORE trusting any
  downstream CFM training on its latents: encodes real ground-truth audio
  (no CFM involved), decodes through the real frozen decoder, and reports
  `pitch_std_semitones` via `evaluate_generation_quality.py`'s `wav_metrics`
  — catches the collapsed-encoder failure mode from prior measurements without a one-off script each time. Needs the real decoder (`bigvgan`
  on PYTHONPATH), so only runs on Kaggle.
- **`check_kernel_progress.py`** — tails a *running* Kaggle kernel's log via
  the SSE log-stream endpoint (`kaggle kernels output` only returns files
  once a kernel finishes, so this is the only way to see live progress).
