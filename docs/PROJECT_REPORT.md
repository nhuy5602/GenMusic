# GenMusic VN — Project Report

Vietnamese text- and audio-conditioned music generation via knowledge distillation from
DiffRhythm2. This report documents the architecture, the related work it builds on, the
experiments run to validate and improve it, and what was concluded. It is kept in sync
with the codebase — see the note at the end of each section for where to look if the
code has moved on since this was written.

**Status as of 2026-07-14**: this report is being written *during* an active debugging
and experimentation session; the Experiment section is updated as runs complete.

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
small trainable `Linear(100→64)` / `Linear(64→100)` pair bridges the two spaces solely
for the distillation loss computation — the student's own generative path never touches
these adapters. See `docs/experiments/distillation_fix.md`.

**Loss**: `loss = (1 - alpha_feature) * MSE(v_student, v_teacher) + alpha_feature * MSE(v_student, x_1 - x_0)`,
i.e. a blend of teacher-matching and ground-truth CFM loss, with `alpha_feature`
exposed via `--alpha-feature`. If the teacher or its lyric tokenizer cannot be loaded
(no internet, package not vendored), training falls back to `alpha_feature=1.0`
(ground-truth only) and reports this plainly (`teacher_status`, `distillation_active`
fields) rather than silently substituting a fake teacher.

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

Per song: Demucs (`htdemucs`, two-stem) separates vocals/backing (falls back to treating
the whole mix as backing if separation fails, now flagged via a `demucs_separated`
field rather than silently degrading); Whisper (`tiny`/`small`, configurable)
transcribes the vocal stem with `language="vi"` for the lyric text; MuQ-MuLan computes
the style embedding on the first 10s of the original mix; both mel channels are computed
with the same `compute_mel_spectrogram`. Output: `records.jsonl` (one record per song:
lyric text, style tag, BPM, paths to backing/vocal mel + style embedding tensors) plus
`config.json` (the mel format, consumed by training to reconstruct `MusicDiffusionConfig`
exactly).

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

### 3.5 Does distillation actually help? (comparison experiment)

*(This subsection is being filled in as the comparison-matrix Kaggle run
(`scripts/run_experiment_matrix.py`, run id `expmatrix-1783993939`) completes. It trains
baseline-no-teacher vs. distillation-with-adapter at several `alpha_feature` values, and
a smaller architecture variant, all against the same 40-song preprocessed dataset for
the same epoch budget, tracking ground-truth CFM loss (`loss_gt`) as the common
comparison axis — see §2.2 for why this is the fair metric to compare on, since the
baseline's loss *is* `loss_gt` and the distilled runs' blended loss is not directly
comparable without decomposing it.)*

**[PLACEHOLDER — results pending]**

### 3.6 End-to-end pipeline validation

*(Filled in once the final chosen configuration is re-verified end-to-end: preprocess →
train (baseline and/or distill) → generate → basic sanity checks, on Kaggle, with no
crashes and non-degenerate audio output.)*

**[PLACEHOLDER — results pending]**

---

## 4. Conclusion

*(To be finalized once §3.5/3.6 land — drafted now with what's already solid.)*

- The single highest-leverage fix in this project's history was **not** a model or
  training change at all — it was the audio rendering path. A model can only be judged
  once its output can be faithfully turned into sound; before this session, it could
  not be.
- Getting distillation to do anything real required treating the teacher as a black box
  whose interface had to be *read from its actual source code*, not inferred from
  variable names or class defaults (the `mel_dim=100` default vs. the real checkpoint's
  `mel_dim=64` is the clearest example of this trap).
- **[Pending final numbers]**: whether the measured comparison supports the thesis that
  distillation helps this specific small student converge faster/better than
  ground-truth-only training at equal compute, and what config (architecture size,
  alpha_feature) is recommended as "the good direction" for further scaling.
- Honest scope limits: this remains a data- and compute-constrained validation, not a
  quality result. The recommended next steps for anyone continuing this work are listed
  in `docs/guides/run_full_pipeline.md`.
