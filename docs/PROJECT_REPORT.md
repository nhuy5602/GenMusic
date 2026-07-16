# GenMusic VN — Project Report

Vietnamese text- and audio-conditioned music generation via knowledge distillation from
DiffRhythm2. This report documents the architecture, the related work it builds on, the
experiments run to validate and improve it, and what was concluded. It is kept in sync
with the codebase — see the note at the end of each section for where to look if the
code has moved on since this was written.

**Status as of 2026-07-14**: session complete. Every pipeline stage (preprocess → train
→ distill-attempt → generate) is verified working end-to-end, on Kaggle at moderate scale
and locally at small scale. The one experiment *not* completed is a proper at-scale
comparison of distillation vs. no-distillation — Kaggle GPU quota ran out mid-session
before it could run; §3.6 and §4 explain exactly what's needed to finish it.

---

## 1. Related Work

### 1.1 DiffRhythm2 (teacher)

[ASLP-lab/DiffRhythm2](https://github.com/ASLP-lab/DiffRhythm2) is the teacher model this
project distills from. It is a latent conditional-flow-matching (CFM) model for
full-song music generation, conditioned on:
- **Lyrics**, tokenized by a custom `CNENTokenizer` (a Chinese/English G2P frontend —
  no Vietnamese linguistic model), fed into the transformer as ordinary sequence
  positions (not cross-attention).
- **Style**, a single embedding from **MuQ-MuLan** (`OpenMuQ/MuQ-MuLan-large`), a
  contrastive audio-text/audio-audio embedding model in the CLAP/MuLan family — the
  same 512-dim space is added to every position via the input embedding and to the
  final adaLN modulation.

Its backbone (`diffrhythm2/backbones/dit.py`) is a stack of Llama-style decoder blocks
("`LlamaNARDecoderLayer`" — non-autoregressive, full bidirectional attention within a
generation block) with rotary position embeddings, generating in causally-cached
*blocks* (`sample_block_cache` in `diffrhythm2/cfm.py`) rather than the whole song at
once — a streaming/chunked generation strategy. Its own real shipped checkpoint
(`config.json` on HuggingFace) uses `dim=2048, depth=16, heads=16, mel_dim=64` — i.e. a
genuinely large model (on the order of hundreds of millions of parameters in the
backbone alone) operating on a 64-bin mel latent, decoded to audio by a dedicated
BigVGAN-family vocoder trained specifically for that latent.

### 1.2 Conditional Flow Matching / rectified flow

CFM (Lipman et al., 2022; used by DiffRhythm2, and by this project's student) trains a
velocity field `v_θ(x_t, t, cond)` to match `x_1 - x_0` along a straight-line
interpolation `x_t = (1-t)x_0 + t x_1` between Gaussian noise `x_0` and data `x_1`, then
generates by integrating an ODE from `t=0` to `t=1`. Compared to classical DDPM, this
gives a well-posed regression target at every `t` and typically needs far fewer sampling
steps for comparable quality, at the cost of losing the explicit noise-schedule/SNR
framing DDPM offers. This project uses plain fixed-step Euler integration
(`src/models/cfm_flow.py`) — no adaptive step-size control, no higher-order solver — the
simplest correct option, sufficient for now given the model is far from converged; an
adaptive/higher-order solver is listed as a legitimate future optimization once model
quality is no longer the binding constraint.

### 1.3 Knowledge distillation for generative audio models

The classical KD setup (Hinton et al., 2015) matches a small student's *output
distribution* to a large teacher's, generally for classification. For a continuous
generative field like CFM's velocity, the analogous move is **velocity/feature
matching**: at a shared `(x_t, t)`, penalize `‖v_student - v_teacher‖²` in addition to
(or blended with) the ground-truth CFM loss `‖v_student - (x_1 - x_0)‖²`. This project's
`alpha_feature` blend follows that pattern (see `src/training/distill_training.py`).
The interesting engineering problem specific to this project — not really covered by the
KD literature, which usually assumes matching output spaces — is that the teacher's own
mel latent (64-dim) and the format the student needs to be *decodable* in (100-dim, to
match the only available high-quality open vocoder for held-out use, Vocos) are
different. Section 3 covers the adapter used to bridge this.

### 1.4 Neural vocoders

Two vocoder families appear in this project's history: the teacher's own BigVGAN-family
decoder (trained specifically for DiffRhythm2's 64-mel latent, not reusable for a
different mel convention without retraining it) and **Vocos**
(`charactr/vocos-mel-24khz`, [Siuzdak, 2023](https://arxiv.org/abs/2306.00814)), a
GAN-based vocoder that predicts STFT coefficients directly rather than upsampling in the
time domain, chosen here because it is a generic, pretrained, drop-in decoder for a
*standard* 100-mel/24kHz representation this project can align its student to exactly,
with no extra training required. Section 3/4 covers why this specific choice mattered a
great deal in practice.

### 1.5 Collaborative development note

Partway through this session, `origin/master` was found to have advanced 13 commits via
independent, unrelated work on the same files (see
[docs/experiments/kaggle_runs.md](experiments/kaggle_runs.md) for the merge log). That
work is itself informative as a point of comparison: it reached for very similar fixes
(a Vocos-format flag, a distillation attempt) via less complete means (an opt-in flag
instead of a corrected default; a placeholder teacher with the wrong call signature),
which is one data point suggesting these particular pitfalls (silent format mismatches,
guessing a black-box teacher's interface instead of reading its source) are natural ones
to fall into on this kind of integration, not one-off mistakes.

*(To extend: add citations for Whisper (ASR), Demucs (source separation),
xlm-roberta-base (text encoder) if the report needs a full bibliography — currently just
named where used.)*

---

## 2. Architecture

*(Filled in from the current codebase — see `src/models/`, `src/training/`,
`src/data/`.)*

### 2.1 Student model — MicroDiT (`src/models/dit_transformer.py`)

A small Diffusion-Transformer-style CFM velocity predictor:

- **Text conditioning**: frozen `xlm-roberta-base` (`PretrainedRobertaEncoder`, ~278M
  params, `requires_grad=False`) projected to the model's hidden dim by a small trainable
  2-layer MLP. Chosen specifically because it is genuinely multilingual (unlike the
  teacher's Chinese/English-only lyric tokenizer) — this is the component that actually
  carries Vietnamese lyric semantics into the model.
- **Style conditioning ("Audio Style Anchor")**: a single 512-dim MuQ-MuLan embedding,
  computed once per song at preprocessing time (`compute_style_embedding` in
  `src/data/preprocess_raw_vietnamese.py`), projected into the model's conditioning
  space by `AudioStyleEncoder` (a 2-layer MLP). This is the **same embedding space** the
  real DiffRhythm2 teacher conditions on, so distillation and the student's own
  generation share one consistent notion of "style."
- **Backbone**: `depth` HuggingFace `LlamaDecoderLayer` blocks (rotary embeddings, SDPA
  attention, no causal mask — full bidirectional attention over the mel sequence),
  `dim`/`heads`/`ff_mult` configurable (CLI: `--dim`/`--depth`/`--heads`/`--ff-mult` on
  `train-self`/`train-distill`). Default `dim=256, depth=4, heads=4, ff_mult=4` — on the
  order of a few million trainable parameters, tiny relative to the teacher's
  `dim=2048, depth=16, heads=16`.
- **Mel I/O**: predicts a velocity field over `(seq_len, n_mels=100)` frames, at
  24kHz/n_fft=1024/hop=256 — chosen to exactly match Vocos's native mel format (see
  §2.3) rather than any dimension DiffRhythm2 itself uses.

### 2.2 Teacher integration (`src/training/distill_training.py`)

The teacher (`diffrhythm2.backbones.dit.DiT`, instantiated with its *own* downloaded
`config.json` dimensions, not guessed ones) and student are trained with a shared
CFM/rectified-flow recipe: same `x_t`, same `t`, same style embedding. The teacher's
lyric tokens and the noisy mel latent are concatenated into one sequence
(`text_embed(tokens)` at `time=-1` sentinel positions, `latent_embed(x_t)` at
`time=t` positions) and run through one non-cached forward pass — mathematically
equivalent to the teacher's own streaming block-cache inference path, just without the
caching optimization (see `docs/experiments/distillation_fix.md` for the
reverse-engineering this is based on).

**Mel-dim adapter**: because the teacher's real checkpoint uses `mel_dim=64` and the
student's mel space is 100-dim (a hard requirement from the vocoder choice, §2.3), a
pair bridges the two spaces solely for the distillation loss computation — the
student's own generative path never touches these. `to_teacher_mel` (student→teacher)
is a **fixed deterministic mel-bin interpolation**, not a trainable layer: its output
feeds directly into the frozen teacher's `torch.no_grad()`-scoped forward pass, so a
trainable layer there would (and, until this was found and fixed, did) never receive
a gradient — see `docs/experiments/distillation_fix.md`'s "Mel-dim adapter gradient
bug" section. `from_teacher_mel` (teacher→student) has no such constraint and is a
real trainable `Linear(64→100)`, verified with a real backward pass
(`tests/test_self_diffusion.py::test_mel_adapter_gradient_flow`).

**Loss**: `loss = (1 - alpha_feature) * MSE(v_student, v_teacher) + alpha_feature * MSE(v_student, x_1 - x_0)`,
i.e. a blend of teacher-matching and ground-truth CFM loss, with `alpha_feature`
exposed via `--alpha-feature`. `run_distillation_training()` (called by `train-distill`)
requires a real teacher and a real lyric tokenizer to actually be loaded — if either
fails (no internet, package not vendored), it raises immediately rather than either
(a) silently substituting a fake teacher, or (b) silently downgrading to ground-truth-only
training under the `train-distill` name. Ground-truth-only training is what `train-self`
is for; `train-distill` completing successfully always means a real teacher was used.
(`_load_teacher()`/`_load_lyric_tokenizer()` themselves stay non-raising — they return
a status string so diagnostics/status checks can inspect availability without a training
run — the raise happens one level up, in `run_distillation_training()`.)

### 2.3 Mel representation & vocoder (`src/models/text_to_music_diffusion.py`)

Both training targets and generated output use `compute_mel_spectrogram()`: 100 mels,
24kHz, n_fft=1024, hop=256, magnitude mel (`power=1`), natural log with a `1e-7` floor —
verified bit-identical to `vocos.feature_extractors.MelSpectrogramFeatures`. This choice
means the pretrained Vocos vocoder (`charactr/vocos-mel-24khz`) decodes the model's mel
output with **no resampling step** — see §4 for why this specific decision was the
single highest-impact fix in this project's history. A Griffin-Lim fallback
(`vocoder_type="griffinlim"`, real iterative phase estimation via
`librosa.feature.inverse.mel_to_audio`) exists for when Vocos is unavailable.

### 2.4 Data pipeline (`src/data/preprocess_raw_vietnamese.py`)

Per song: Demucs (`htdemucs`, two-stem) separates vocals/backing, batched (loads the
Demucs model once per batch of up to 8 files rather than once per file), resumable
(skips files whose stems already exist on disk), and retries cuda→cpu on failure —
falling back to treating the whole mix as backing if separation fails entirely, flagged
via `demucs_separated`/`vocal_source` fields rather than silently degrading. Whisper
(`tiny`/`small`, configurable, with cuda→cpu retry) transcribes the vocal stem with
`language="vi"`, keeping word/segment-level timestamps (`segments` field) so training
crops can align lyric text to the actual audio window rather than the whole-song
transcript. MuQ-MuLan computes one style embedding per song on the first 10s of the
original mix. Both mel channels are computed with the same `compute_mel_spectrogram`.
Output: `records.jsonl` (one record per song: lyric text + timestamped segments, style
tag, BPM, paths to backing/vocal mel + style embedding tensors) plus `config.json` (the
mel format, consumed by training to reconstruct `MusicDiffusionConfig` exactly).

**Training-time augmentation** (`MusicDiffusionDataset.__getitem__` in
`src/training/self_diffusion.py`): for songs longer than one training chunk, a random
offset is chosen each epoch and applied identically to both the vocal and backing mel
(keeping them temporally aligned) — different epochs see different windows of longer
songs — and the lyric text used for that item is trimmed to just the segments whose
timestamps fall inside the cropped window, using the ASR segment timestamps above.

### 2.5 What's *not* wired in (by design, for now)

- **Tone-aware Vietnamese G2P** (`src/data/vietnamese_g2p.py`) and **ASR-lyric alignment**
  (`src/data/lyric_alignment.py`) exist as standalone, tested utilities but are not
  consumed by the training pipeline — the model conditions on raw lyric text through
  frozen `xlm-roberta-base`, not phonemes. Flagged as a real quality lever, not yet
  pulled (see Conclusion).
- **Pitch/F0 conditioning** was present in an earlier version (`librosa.pyin`) and was
  removed when the Audio Style Anchor was introduced; not restored, since a proper
  reintroduction would need to fit into the current mel/style pipeline coherently rather
  than being bolted back on as a separate signal.

---

## 3. Experiments

All heavy compute (preprocessing, training) runs on Kaggle T4 GPUs via the project's
Kaggle-API automation (`scripts/run_kaggle_*.py`) — see
[docs/experiments/kaggle_runs.md](experiments/kaggle_runs.md) for the full run-by-run
log, this section summarizes.

### 3.1 Vocoder distortion (root cause + fix)

**Symptom**: generated songs were badly distorted compared to real reference audio.
**Root cause**: the default renderer fabricated a fixed linear phase spectrum instead of
real phase reconstruction (measured **0.149 log-mel correlation** with ground truth on a
real reference song — near-noise), and the alternate path fed a real neural vocoder
(Vocos) a resampled mel from an incompatible format. **Fix**: made the model's native
mel format bit-identical to Vocos's own. **Result**: 0.997 correlation locally, 0.993 on
Kaggle's real environment. Full write-up: `docs/experiments/vocoder_fix.md`.

### 3.2 Distillation contract (root cause + fix)

**Symptom (found by code audit, not yet reported by the user)**: the distillation code
had a fake teacher fallback with no error surfaced, guessed architecture dimensions, the
wrong attention-mask convention for the teacher's custom layer, lyric tokens never
actually reaching the teacher, and a fabricated style embedding. **Fix**: reverse
engineered the real call contract from the DiffRhythm2 GitHub source and replicated it
exactly, with an honest fallback (reports `teacher_status`/`distillation_active`
explicitly) instead of a silent fake teacher. Full write-up:
`docs/experiments/distillation_fix.md`.

### 3.3 Mel-dim mismatch (found *by* the honest-fallback mechanism working correctly)

The teacher's real checkpoint uses `mel_dim=64`; the student's Vocos-aligned mel space
is 100-dim. The honest fallback (§3.2) correctly detected this and disabled the teacher
rather than compute shape-mismatched garbage. **Fix**: a small trainable linear adapter
pair bridging the two mel spaces for the distillation loss only. Verified locally with a
fake-teacher unit test (mismatched dims, forward+backward+optimizer step all succeed).

### 3.4 Checkpoint bloat

Checkpoints were 1.1GB each because `save_checkpoint` was saving the frozen (never
trained) RoBERTa text encoder every time. Fixed to exclude it (loaded fresh from
HuggingFace on load instead); `load_checkpoint` now uses `strict=False` to accommodate
this. Verified: checkpoint size ~67MB for the default architecture (down from ~1.1GB),
generation from a loaded checkpoint still works correctly.

### 3.5 Does distillation actually help? (comparison experiment — completed at real scale, §3.9)

The originally planned comparison (`scripts/run_experiment_matrix.py`, a 40-song
multi-variant matrix) never got GPU time and was superseded by a simpler direct
comparison at a larger, real scale (250 songs, matched epochs/batch/steps) — see §3.9 for
the actual result. The matrix script itself is still unrun and would give a finer-grained
`alpha_feature`/architecture-size ablation if useful later, but the core question this
section used to leave open — "does the teacher signal actually help vs. training from
scratch on the same data budget" — now has a real, measured answer.

### 3.6 End-to-end pipeline validation

**On Kaggle** (`genmusic-fullexp-1783972294`, 12 real songs, T4 GPU): preprocessing
12/12 succeeded, vocoder round-trip scored 0.993 log-mel correlation, baseline DiT
training completed (120 steps), and both baseline and (honest-fallback) generation
produced valid non-degenerate audio (peak ~0.8, RMS 0.08–0.15, silence ratio <0.13%, no
NaN/Inf).

**A second Kaggle attempt** (`genmusic-fullexp-1783991479`) with the mel-dim adapter
applied hung for ~11 hours before being killed — see
[docs/experiments/kaggle_runs.md](experiments/kaggle_runs.md) for the root cause (an
unrelated top-level import triggering a CUDA extension JIT compile) and fix. This
consumed a meaningful fraction of the session's Kaggle GPU budget and, combined with
further usage, exhausted it before a re-run could confirm `distillation_active: true`
at Kaggle scale.

**Locally** (Windows, CPU-only, no GPU, 2 real songs from `dataset/vietnamese_songs/`,
after the fix + a large merge with parallel work — see
[docs/experiments/kaggle_runs.md](experiments/kaggle_runs.md)): every stage ran
end-to-end with real data — preprocessing (2/2 records), vocoder round-trip (0.986
correlation), baseline training, distillation's honest fallback (`distillation_active:
false`, teacher correctly reported as unavailable rather than faked), and generation
from both checkpoints (valid non-degenerate audio, peak ~0.8, RMS 0.07–0.10, silence
ratio <0.2%, no NaN/Inf). Full test suite: 10/10 passing. This run also surfaced and
fixed a Windows-only `UnicodeEncodeError` (cp1252 console/file encoding vs. Vietnamese
diacritics) invisible on Kaggle's Linux/UTF-8 environment.

**Net result**: the pipeline is verified correct end-to-end, twice, in two different
environments. What's *not* verified is whether real distillation (`distillation_active:
true`, i.e. the real DiffRhythm2 teacher successfully loaded and contributed signal)
completes a full training run without error at Kaggle scale — the adapter path was
verified in isolation (a unit test with a fake teacher of mismatched mel_dim, forward +
backward + optimizer step all succeed) but never end-to-end against the real teacher
before quota ran out.

### 3.7 Real-teacher distillation confirmed locally (`distillation_active: true`, first time end-to-end)

After §3.6, Kaggle quota ran out entirely, so this was verified on CPU instead: a fresh
shallow clone of `github.com/ASLP-lab/DiffRhythm2` was patched locally to work around
transformers version skew (`StaticCache`/`FlashAttentionKwargs` import paths moved,
`LlamaConfig.rope_theta` routing changed) and a chain of ~20 missing Python packages plus
three Windows cp1252-vs-UTF-8 file-encoding bugs in the *vendored DiffRhythm2 code itself*
(not this project's code). With `espeak-ng` installed as a system package and the patched
clone on `PYTHONPATH`, both `_load_teacher()` (`teacher_status: "ok"`) and
`_load_lyric_tokenizer()` (`tokenizer_status: "ok"`) now load the real teacher and real
lyric tokenizer on a plain Windows/CPU machine — no Kaggle required for this part.

A real 30-epoch distillation run (2 real songs, batch_size=2, `dim=128, depth=2`) then
completed with `distillation_active: true` for the first time against the actual
1,136,249,664-param DiffRhythm2 teacher (vs. the student's 745,188 trainable params —
teacher is ~1,525× larger, which is the whole point of distillation). Isolated timing:
one teacher forward pass at batch_size=1 costs ≈3s on CPU (inference-only, no backward);
the dominant cost in every distillation step.

**Comparing against a baseline (no-teacher) run on the same 2 songs/30 epochs**:
final `loss_gt` ≈17.9 (distilled) vs. ≈15.7 (baseline) — statistically indistinguishable.
This is *not* evidence against distillation; it's an artifact of scale: with only 2 songs
and `batch_size=2` there is exactly 1 gradient step per epoch, and CFM's random per-step
timestep sampling makes loss swing from ~3.5 to ~229 within the *same run* independent of
learning progress. 30 such steps isn't enough to average that variance out. Answering
"does distillation help" still requires either more songs, more steps per epoch, or both
— this local run's contribution is proving the *mechanism* is real and correct end-to-end
(mel-dim adapter, teacher call contract, lyric tokenization all genuinely exercised), not
answering the quality question, which remains exactly as open as §3.5/§4 already said.

**Full-corpus feasibility** (`sonlest/vietnamese-music-dataset-version3-part6` = 201
songs, 2.52GB, confirmed via the Kaggle API): preprocessing extrapolates to ≈3.9 GPU-hours
(measured 12-song Kaggle rate ≈70s/song), comfortably within a weekly quota. Baseline
training is trivial at any scale (≈5 min extrapolated, tiny student). Real-teacher
distillation's GPU cost is *not* measured, only extrapolated from the CPU numbers above
with a generic CPU→T4 speedup assumption (≈15–50×, unverified) — landing around 30
minutes to a few hours for a full run. No Kaggle GPU distillation run has ever completed
even once (every attempt hit a bug or ran out of quota first), so this number carries
real uncertainty; a small (~12-song) real-teacher GPU smoke test is the next step to
replace the estimate with a measurement, before committing quota to the full corpus.

**Behavior change**: `run_distillation_training()` (i.e. `train-distill`) now raises
immediately if the real teacher or its lyric tokenizer fails to load, rather than
silently downgrading to ground-truth-only training under the `train-distill` name (which
is what it did through all of §3.1–§3.6 above, reported honestly via
`distillation_active: false` but still a "successful" run that didn't do what was asked).
`_load_teacher()`/`_load_lyric_tokenizer()` themselves are unchanged and still
non-raising, for callers that just want a status check. Ground-truth-only training is
what `train-self` is for; a `train-distill` run that completes now always means a real
teacher was used, full stop.

### 3.8 Two independent bugs behind a coworker's "almost entirely noise" report

A teammate reported that generation from a real run (~250 songs, 60 epochs,
`train-distill`) came out nearly all noise. Investigating found two separate, unrelated
bugs, both now fixed:

**Bug 1 — generation was always zero-conditioned.** `generate_audio()`'s call to
`sample_cfm` never passed `backing_mel`/`style_prompt`, so every `generate-local` run
(and every automated generation stage in `run_full_experiment.py`/
`run_experiment_matrix.py`) conditioned on a zero backing-track and a pooled-text
stand-in instead of a real MuQ-MuLan style anchor — a real train/inference mismatch,
worse at real scale (250 songs, mostly real non-zero backing_mel from successful Demucs
separation) than in this session's earlier 2-song tests (where zero-fallback was common
anyway). Fixed by `load_reference_conditioning()` (`src/training/self_diffusion.py`),
which extracts a real `backing_mel`/`style_anchor` pair from an existing preprocessed
dataset record; `generate-local --reference-dataset --reference-id` wires it into the
CLI, and both experiment scripts now pass it automatically to their own generation
stage. Verified end-to-end: `generation.baseline.reference_conditioning` now reports
`{"has_backing_mel": true, "has_style_anchor": true}` instead of not existing at all.

**Bug 2 — the mel-dim adapter's gradient was cut for real distillation runs.** See §2.2
and `docs/experiments/distillation_fix.md`'s "Mel-dim adapter gradient bug" section —
`to_teacher_mel`/`from_teacher_mel` were both computed inside `torch.no_grad()`, so
neither ever received a gradient despite being registered as trainable and handed to
the optimizer, for every `train-distill` run where `mel_adapter_used: true` (i.e. every
real run against the actual teacher, whose `mel_dim=64` never matches the student's
`100`). Fixed: `from_teacher_mel` now receives a real gradient (verified with an actual
`.backward()` call); `to_teacher_mel` was replaced with a fixed deterministic mel-bin
interpolation instead of a nominally-trainable-but-structurally-untrainable layer,
since fully fixing it would require backward through the ~1.14B-param teacher every
step.

**Which one explains the coworker's report?** Unknown — the report was from a
`train-distill` checkpoint, so both bugs applied simultaneously; this hasn't been
isolated to one cause via a controlled re-run (train the same 250-song dataset again
post-fix, listen to the result) which is the natural next step. Bug 1 alone applies to
`train-self` (baseline) too; Bug 2 is `train-distill`-specific.

### 3.9 First completed real-scale run: 250 songs, distillation vs. self-diffusion, with objective quality metrics

The controlled re-run §3.8 called for: full 250-song dataset (`sonlest/vietnamese-music-dataset-version3-part6`,
preprocessed with `--whisper-model base` after tiny was found to hallucinate/loop on real
lyrics), then `train-distill` and `train-self` on identical data with matched
`epochs=25, batch_size=4` so their `loss_gt` curves are directly comparable.

**Two real bugs found and fixed before this run produced anything usable** (both now
committed):
- Kaggle preprocessing's subprocess timeout was `1800s` whenever `--max-files` was set at
  all, conflating "small smoke test" with "capped at the full dataset size" — it silently
  killed a real 250-file run at the 30-minute mark while still healthy (song 46/250), losing
  that GPU time with nothing salvageable (`records.jsonl` is written once at the end, not
  incrementally). Fixed by scaling the timeout with file count.
- `train-distill` OOM'd on epoch 3 of 25 (batch_size=8) — `CUDA out of memory`, ~11.6GiB
  allocated, 3.57GiB reserved-but-unallocated. Epochs 1–2 completed fine with the same batch
  size, so this was allocator fragmentation building up across steps (each batch has a
  different lyric-token length, so the CUDA caching allocator accumulates many
  differently-sized blocks), not a logic bug or an undersized GPU. Fixed with
  `batch_size=4`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, and
  `torch.cuda.empty_cache()` between epochs; the re-run completed all 25 epochs without OOM.

**Training result** (`ddvnam05/genmusic-distill-1784166040` vs. `ddvnam05/genmusic-train-1784173776`,
same dataset, same `epochs=25`/`batch_size=4`/1575 steps):

| | loss_gt @ epoch 1 | loss_gt @ epoch 25 | final_loss_gt (last 10 steps) | wall-clock |
|---|---|---|---|---|
| `train-distill` (real teacher) | 8.86 | 3.28 | 2.57 | 6527s |
| `train-self` (no teacher) | 11.24 | 8.06 | 7.15 | 134s |

Both runs saw the identical 250 songs in the identical order for the identical number of
steps. Distillation's ground-truth loss dropped to roughly a third of the no-teacher
baseline's, and did so far more consistently (the no-teacher curve oscillates between ~7
and ~13 with no clear downward trend after epoch ~10, e.g. epoch 20 jumps back to 11.02).
This is the first real evidence (not just "the mechanism works in isolation", §3.7) that
the teacher signal measurably helps on this project's own data at this scale. It costs
~49x the wall-clock GPU time for that improvement, which is the real trade-off, not a
free win — see the quota note below.

**Objective audio quality check** (`scripts/evaluate_generation_quality.py`, new this
session): since no one was available to listen, this scores generated audio without a
human — spectral flatness (0 = tonal/harmonic, 1 = white noise; this is the direct proxy
for "toàn nhiễu" / all-noise complaints), clip ratio, silence ratio, and RMS, computed on
7 samples spread across the dataset, each compared against that same song's *real* vocal
track rendered through the identical Vocos vocoder (isolating model quality from vocoder
artifacts) and against a synthesized white-noise clip as a fixed sanity anchor:

| | mean spectral flatness | mean RMS | clip ratio |
|---|---|---|---|
| white noise (sanity anchor) | 0.562 | 0.577 | 2.0% |
| `train-distill` output | 0.086 | 0.078–0.131 | 0.0% |
| `train-self` output | 0.075 | 0.037–0.055 | 0.0% |
| real vocal (same vocoder) | 0.053 | 0.080–0.109 | 0.0% |

Neither checkpoint's output is remotely close to the white-noise anchor on flatness — both
are clearly tonal/structured, not noise, and neither clips. Flatness alone reads as
close between the two models (distill's is even slightly *higher*/less-tonal than
self-diffusion's on this narrow metric), which on its own would be a misleading way to
compare them: `train-self`'s RMS (0.037–0.055) sits well below the real-vocal range
(0.080–0.109), while `train-distill`'s (0.078–0.131) tracks it closely. Combined with the
loss_gt gap above, the more complete picture is that distillation output is closer to the
real target on the axes that actually distinguish the two models, and flatness alone is a
good "is this noise" filter but not a full quality signal — it can't tell a correct quiet
passage from an undertrained quiet-and-wrong one.

**Quota spent this run**: preprocessing ≈2h (one wasted 30min run + one successful
~2h run), `train-distill` ≈1.8h, `train-self` ≈2min, local evaluation ≈free (runs on the
requester's machine, not Kaggle). Total Kaggle GPU time ≈4h of this project's weekly
budget.

**Conclusion for this run**: no model-size increase was warranted — the "is distillation
broken" and "is the model too small" hypotheses this section was set up to test were both
falsified before reaching model size: the bugs were infrastructure (timeout, OOM), not
distillation logic or capacity, and once fixed, distillation converged as expected on
`loss_gt` and produced clearly non-noise, real-audio-comparable output at the current
small size (`dim=256, depth=4, heads=4`). A model-size ablation remains a reasonable
follow-up if a *later* run shows the gains plateauing, but there is no evidence for that yet.

---

## 4. Conclusion

- The single highest-leverage fix in this project's history was **not** a model or
  training change at all — it was the audio rendering path. A model can only be judged
  once its output can be faithfully turned into sound; before this session, it could
  not be (0.15 log-mel correlation with ground truth, i.e. the default output path was
  producing structured noise regardless of model quality).
- Getting distillation to do anything real required treating the teacher as a black box
  whose interface had to be *read from its actual source code*, not inferred from
  variable names or class defaults — the `mel_dim=100` default vs. the real checkpoint's
  `mel_dim=64` is the clearest example of this trap, and it recurred independently in
  the parallel `origin/master` work this session merged with (§1.5), which is some
  evidence it's a natural mistake for this kind of integration, not a one-off.
- **The distillation-helps-or-not question now has a real answer: yes, measurably, at
  250-song/25-epoch/matched-steps scale (§3.9).** `train-distill`'s `loss_gt` (2.57 final)
  came in at roughly a third of `train-self`'s (7.15 final) on identical data, and an
  objective (non-listening) audio-quality check found both checkpoints clearly non-noise
  and non-clipping, with `train-distill`'s output amplitude tracking real vocal audio more
  closely. The cost side of that trade-off is real too: distillation took ~49x the
  wall-clock GPU time of the no-teacher baseline for that improvement. Two infrastructure
  bugs (a preprocessing timeout that killed a healthy run, and a CUDA OOM from allocator
  fragmentation across epochs) were found and fixed in the process of getting this result
  — neither was a distillation-logic or model-capacity problem, so no model-size increase
  was warranted this round (§3.9).
- `train-distill` now raises immediately if the real teacher/tokenizer can't be loaded,
  instead of silently completing as ground-truth-only training under the distillation
  name (§3.7). This closes a real gap: every §3.1–§3.6 "successful" distillation run
  before §3.7 was, by construction, either a real distillation or a same-named
  ground-truth-only run depending on environment — correctly *reported* via
  `distillation_active`, but easy to miss if you didn't check that field.
- Small-model architecture choice (`dim=256, depth=4, heads=4`, a few million trainable
  parameters vs. the teacher's few-hundred-million) was carried through as the working
  default throughout, with `--dim`/`--depth`/`--heads`/`--ff-mult` exposed for the size
  ablation the comparison experiment was designed to include — this part of "small model,
  good quality" is set up correctly even though the "good quality" half is unverified.
- Honest scope limits: everything reported here is either a wiring/correctness result
  (pipeline runs, produces valid non-degenerate audio, honest fallbacks work as designed)
  or a data point from a tiny (2–12 song) dataset run for a handful of epochs — none of it
  is a claim about musical quality. See `docs/guides/run_full_pipeline.md` for exactly
  what to run next once Kaggle GPU quota is available again.
