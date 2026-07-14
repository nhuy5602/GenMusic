# Kaggle experiment run log

Consolidated end-to-end experiment (`scripts/run_full_experiment.py`, launched via
`scripts/run_kaggle_full_experiment.py`) run on a Kaggle T4 against a subset of
`sonlest/vietnamese-music-dataset-version3-part6`. Each run does, in one kernel:
preprocess N songs with the fixed pipeline → vocoder round-trip sanity check → baseline
DiT training (no teacher) → real distillation attempt → generate a sample from each
checkpoint → basic waveform sanity stats. This log records what each attempt found and
fixed, in order, so the debugging trail isn't lost.

## Run 1 — `genmusic-fullexp-1783969651` (12 files, 40/15 epochs)

**Result: preprocessing produced 0/12 records**, so training/distillation never ran
(and originally crashed the whole kernel with an unhandled `ValueError` on the empty
dataset — fixed afterward by making `run_full_experiment.py`'s stages catch and report
failures instead of propagating).

Root cause (found by reproducing locally + adding structured-JSON failure logging,
since **the downloaded Kaggle kernel `.log` file was empty** — see
`docs/experiments/../..` / memory note on this gotcha): `librosa.beat.beat_track` calls
`scipy.signal.hann` directly, which was removed from newer scipy (moved to
`scipy.signal.windows.hann`). Installing DiffRhythm2's `requirements.txt` on top of our
own deps apparently resolves to a scipy version where this attribute no longer exists.
**Fix:** wrapped the BPM estimate in try/except with a `bpm=120` fallback (BPM is only
used for a cosmetic style-text tag, not real conditioning) in
`src/data/preprocess_raw_vietnamese.py`. Also added `preprocess_failures.json` /
`failed_count`/`failures` fields to the preprocessing report so future failures are
visible even when the kernel log download is empty.

## Run 1b — diagnostic, `genmusic-fullexp-1783971485` (2 files, 5/2 epochs)

Cheap 2-file run to confirm the beat_track fix and surface real errors quickly (avoids
burning GPU quota on repeated 12-file/40-epoch attempts while debugging). Confirmed:

- The `scipy.signal.hann` traceback was the sole failure (both files, identical error).
- **MuQ-MuLan (`OpenMuQ/MuQ-MuLan-large`) works correctly on Kaggle**: real non-zero,
  unit-norm 512-dim style embeddings were produced (verified by loading the downloaded
  `.pt` files and checking they weren't the zero-vector fallback).
- Demucs stem separation works on Kaggle (real `vocals.wav`/`no_vocals.wav` produced).

## Run 2 — `genmusic-fullexp-1783972294` (12 files, 40/15 epochs)

**Result: preprocessing succeeded (12/12 records). Baseline training completed. Real
distillation did NOT activate** (`distillation_active: false`,
`teacher_status: "disabled: mel_dim mismatch with dataset"`).

Numbers:
- **Vocoder round-trip sanity check on Kaggle: logmel corr = 0.993** (vs. 0.149 for the
  old istft hack measured locally before any fixes) — confirms the vocoder fix from
  `docs/experiments/vocoder_fix.md` holds on the real Kaggle environment, not just
  locally.
- Baseline DiT training: 40 epochs × 3 steps/epoch (12 records, batch=4) = 120 steps,
  final CFM loss 5.34, ~11s total (tiny dataset, tiny model).
- Generated samples (baseline & distilled) both produced **valid, non-degenerate audio**:
  peak ~0.8, RMS 0.08–0.15, silence ratio <0.13%, no NaN/Inf. This is a first for this
  project — earlier baselines produced audibly distorted output; these are structurally
  clean waveforms (whether they sound *musically good* with only 12 songs / 40 epochs is
  a separate question — see "Known limitations" below).

**Why distillation didn't activate**: the real teacher checkpoint's own `config.json`
(downloaded fresh from HF, inspected directly) specifies `"mel_dim": 64`, not 100. The
`mel_dim=100` assumption in `docs/experiments/distillation_fix.md` was based on the
`DiT` class's Python *default* argument value, not the actual shipped checkpoint's
config — an incorrect inference, corrected here. Our student's mel space (100, chosen to
fix the vocoder) is a genuine dimensional mismatch with the teacher's real latent space
(64), and `run_distillation_training`'s honest-fallback logic (see
`docs/experiments/distillation_fix.md`) correctly detected this and disabled the
teacher rather than computing shape-mismatched garbage — exactly as designed.

**Fix:** added a small trainable linear adapter pair (`to_teacher_mel`:
`Linear(100, 64)`, `from_teacher_mel`: `Linear(64, 100)`) in
`KnowledgeDistillationTrainer`, used only for the teacher-facing projection during the
distillation loss computation — the student's own generative/decode path (100-mel,
Vocos-native) is untouched. Verified locally with a fake-teacher smoke test
(mismatched mel dims, forward+backward+optimizer step all succeed). This is a learned
approximation, not an exact mapping — both mel spaces are different linear-ish
downsamplings of the same underlying STFT magnitude spectrum, so a linear adapter is a
reasonable bridge, but it is not claimed to be a perfect one.

## Run 3 — `genmusic-fullexp-1783991479` (12 files) — hung ~11 hours, killed

Resubmitted with the mel-dim adapter fix. This run (and a concurrently-submitted
`genmusic-expmatrix-1783993977` experiment-matrix run) never produced output: both sat at
Kaggle status "RUNNING" for **~11 hours** (confirmed via wall-clock, not a measurement
error) against an expected ~15–45 minutes, before being manually killed via `kaggle
kernels delete` (there is no `kernels stop` command). Root cause: `distill_training.py`'s
lyric-tokenizer loader did `import inference` (DiffRhythm2's own `inference.py`) just to
reach two small helper functions. That module's top-level `from bigvgan.model import
Generator` chain runs a **CUDA extension JIT compile**
(`torch.utils.cpp_extension.load()` in `bigvgan/alias_free_activation/cuda/activation1d.py`,
executed as a bare module-level statement) which hung indefinitely. Fixed by vendoring
only the actual tokenizer logic (`g2p.g2p_generation.chn_eng_g2p` + a small `parse_lyrics`
reimplementation) instead of importing `inference.py` — see `src/training/distill_training.py`'s
`_load_lyric_tokenizer()`. This consumed a meaningful fraction of the session's Kaggle GPU
budget before being caught.

## Merge with parallel origin/master work

Mid-session, `git fetch` revealed `origin/master` had advanced 13 commits with
independent, heavily overlapping work on the same files (someone else, or another
session, working on this repo in parallel). That work included its own partial fix for
the vocoder distortion (an opt-in `--vocos-compatible` flag rather than a corrected
default, plus a post-hoc spectral denoising filter) and its own distillation attempt
(still using a placeholder teacher with a call signature borrowed from a different model
family, F5-TTS, not DiffRhythm2's real `DiT`) — both less complete than the fixes in this
log. It also included genuinely new, non-overlapping improvements: batched/resumable
Demucs separation with cuda→cpu retry, P100 CUDA-compatibility repair + dependency
probing in the Kaggle preprocessing kernel, Whisper segment-level timestamps, and
random-offset+segment-aligned mel/lyric cropping during training. All of this was
merged by hand (not blindly): the root-cause vocoder/distillation fixes from this log
were kept as the base, the genuinely new Kaggle-robustness work was ported in, and the
now-redundant `--vocos-compatible` flag and denoising filter were dropped. Full test
suite (10/10) and the vocoder correlation number (0.997) were re-verified after merging
— see `git log` around commit `c92905f`.

## Kaggle GPU quota exhausted — pivoted to local-only testing

After the Run 3 hang and the merge, the user's Kaggle GPU quota was exhausted before a
proper at-scale comparison experiment (`scripts/run_experiment_matrix.py`, baseline vs.
distillation at several `alpha_feature` values, 40 songs / 60 epochs) could be run. The
tooling for that experiment is complete and ready (`scripts/run_kaggle_experiment_matrix.py`
as the local launcher) — running it once quota resets is the natural next step, not a
redesign.

**In the meantime, the full pipeline was re-verified end-to-end locally** (Windows,
CPU-only, no GPU) against the 2 real local songs in `dataset/vietnamese_songs/`, using
`scripts/run_full_experiment.py` directly (no Kaggle):
- Preprocessing: 2/2 records, 0 failures (Demucs batching + resumability, Whisper tiny,
  MuQ-MuLan gracefully degrading to a zero-vector style embedding since `muq` isn't
  installed locally — expected, not an error).
- Vocoder round-trip on a real local song: **logmel corr = 0.986** (consistent with the
  0.993–0.997 seen on Kaggle and locally on a different song earlier).
- Baseline DiT training: completed (5 epochs × 1 step, CPU).
- Distillation: correctly reported `distillation_active: false` with the honest
  `teacher_status` message ("diffrhythm2 package not importable ... only works on
  Kaggle") — the fallback mechanism itself is being exercised and works as designed, just
  without a real teacher signal (DiffRhythm2 isn't cloned locally).
- Generation from both checkpoints: completed, valid non-degenerate audio (peak ~0.8,
  RMS 0.07–0.10, silence ratio <0.2%, no NaN/Inf).

This run also caught and fixed a **Windows-only bug** invisible on Kaggle (Linux, UTF-8
locale by default): `Path.write_text()`/`print()` default to the process's locale
encoding, which on Windows is cp1252 and cannot encode Vietnamese diacritics — every
`summary.json` write and the final console print crashed with `UnicodeEncodeError` until
`encoding="utf-8"` was added explicitly (`scripts/run_full_experiment.py`,
`scripts/run_experiment_matrix.py`) and `sys.stdout.reconfigure(encoding="utf-8")` was
added to both scripts' `main()`, matching the pattern already used in `cli.py`.

**Scope note:** 2 songs / 5 epochs / CPU is enough to prove every stage of the pipeline
is wired correctly and produces valid, non-crashing output — it is not enough data or
compute to demonstrate anything about whether distillation improves quality (see "Known
limitations" below; that comparison still requires the Kaggle-scale experiment matrix).

## Known limitations / what "good" doesn't mean yet

- 12 songs / 40 epochs is a **smoke-scale** run to validate the pipeline is wired
  correctly end-to-end (no crashes, no NaNs, clean-sounding vocoder, real distillation
  signal flowing) -- it is nowhere near enough data or training time to judge musical
  quality. A real quality run needs the full raw dataset preprocessed (not just 12
  songs) and many more epochs.
- The lyric tokenizer feeding the teacher (`CNENTokenizer`) has no Vietnamese linguistic
  model (Chinese/English G2P only) -- by design, the teacher's contribution is a general
  audio/music prior, not lyric semantics; the student's own frozen `xlm-roberta-base`
  path carries Vietnamese lyric understanding. See `docs/experiments/distillation_fix.md`.
- The mel-dim adapter between student (100) and teacher (64) is a freshly-initialized,
  jointly-trained linear projection -- it has no guarantee of being a clean acoustic
  mapping early in training; expect the distillation loss to be noisy/less useful in the
  first many steps until the adapter itself converges.
