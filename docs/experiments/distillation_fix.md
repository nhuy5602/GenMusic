# Experiment: fixing the DiffRhythm2 → MicroDiT knowledge distillation contract

**Date:** 2026-07-14

## What was wrong

A code audit of `src/training/distill_training.py` (pre-fix) found the "distillation"
step was very unlikely to transfer any real signal from the teacher:

1. **Guessed architecture.** The teacher DiT was instantiated with hardcoded
   `dim=512, depth=8, heads=8`, "reference standard" values, not the dimensions in the
   teacher's own downloaded `config.json`. If these didn't match the checkpoint,
   `load_state_dict(..., strict=False)` would silently skip most tensors.
2. **Fake fallback with no error surfaced.** If `diffrhythm2` wasn't importable (true on
   any environment without manually cloning the DiffRhythm2 GitHub repo onto
   `PYTHONPATH`), the code silently substituted `DummyTeacher`, a single randomly
   initialized `nn.Linear`. Training against this produces a loss that pulls the student
   toward noise, with no warning in the report that this happened.
3. **Wrong call contract.** The real teacher's forward signature (confirmed by reading
   `diffrhythm2/backbones/dit.py` and `diffrhythm2/cfm.py` from the official repo) is:
   ```
   DiT.forward(x, time, position_ids, style_prompt, attn_mask, use_cache=False, past_key_value=None)
   ```
   with `attn_mask` a **boolean** `[B,1,Q,KV]` tensor (True = attend). The old code built
   a float `-inf`-masked tensor (the convention for HF's plain `LlamaDecoderLayer`, used
   by our *own* student, but not by the teacher's custom `LlamaNARDecoderLayer`). It also
   never encoded the lyric tokens into the sequence at all — the teacher's `text_embed`
   was never called — so even a correctly-loaded teacher would have been generating
   music velocity with no lyric conditioning, i.e. an unconditional prior at best.
4. **Fabricated style prompt.** The 512-dim `style_prompt` the teacher expects (a
   MuQ-MuLan contrastive audio embedding) was approximated by interpolating the *mean* of
   the backing-track mel to 512 dims — unrelated to the real embedding space the teacher
   was trained with.
5. **Dead `temperature` parameter** — accepted by the constructor, never referenced in
   the loss. Leftover from a classification-style KD template that doesn't apply to
   velocity regression.
6. **Student's own "Audio Style Anchor" was also crude**: a random 64-frame crop of the
   backing mel, average-pooled through an untrained Conv1D. Re-cropped every batch, so
   it's not even a stable representation of "this song's style" — and unrelated to
   whatever the teacher was conditioned on, so student and teacher couldn't share a
   style-conditioning space even if everything else were fixed.

## What the real contract actually is

Read directly from the official ASLP-lab/DiffRhythm2 GitHub repo
(`diffrhythm2/backbones/dit.py`, `diffrhythm2/cfm.py`, `inference.py`):

- `DiT(dim, depth=8, heads=8, ff_mult=4, mel_dim=100, text_num_embeds=256, ...)`, always
  with an internal `cond_dim=512` regardless of `dim`. **`mel_dim` defaults to 100** —
  confirming the mel-format fix in `docs/experiments/vocoder_fix.md` also aligns the
  student with the teacher's own latent space.
- Lyric tokens and the noisy mel latent are **one shared sequence**: `text_embed(tokens)`
  (shape `[B, T_text, 512]`, `time=-1` sentinel for those positions) and
  `latent_embed(xt)` (shape `[B, T_noisy, 512]`, `time=t` for those positions) are
  concatenated and run through the same `input_embed` + Llama-style blocks. There is no
  cross-attention block; conditioning is purely via concatenation + the shared adaptive
  `style_prompt`/`time` terms added at `input_embed` and the final adaLN.
- `style_prompt` is a single 512-dim MuQ-MuLan (`OpenMuQ/MuQ-MuLan-large`) embedding,
  added at every sequence position — not per-frame, not backing-track-derived.
- The teacher's own KV-cache (`sample_block_cache`) is a **streaming inference
  optimization only** — mathematically it is equivalent to running one full,
  non-cached forward pass over `[text_tokens; noisy_frames]` with a fully-visible
  (non-causal) attention mask. This is what we replicate for distillation (no caching
  needed since we're not doing multi-block autoregressive generation).
- Lyric tokenization uses a custom `CNENTokenizer` (Chinese/English G2P frontend). It has
  no linguistic model of Vietnamese, but it does not crash on arbitrary Unicode text
  either — it just tokenizes deterministically without Vietnamese-specific meaning. This
  is an accepted limitation: the teacher's role in distillation is to transfer its
  **general audio/music generation prior** (rhythm, timbre, dynamics — captured mostly
  through the real MuQ-MuLan style embedding and its trained CFM velocity field), while
  the **student's own** text conditioning (frozen `xlm-roberta-base`, genuinely
  multilingual) is what actually carries Vietnamese lyric semantics. The two text paths
  serve different purposes and don't need to agree.

## The fix

- `_load_teacher()`: downloads `config.json` from the HF repo and instantiates
  `DiT(**model_config)` (matching `scripts/test_teacher_inference.py`'s known-good
  pattern) instead of guessing dimensions. Strips the `transformer.` prefix from the
  full `CFM` checkpoint's state dict before loading into the raw `DiT` module (we don't
  need `CFM`'s `sample_block_cache`/`odeint` machinery for one-shot velocity
  distillation).
- `_load_lyric_tokenizer()`: imports the real `inference.CNENTokenizer` /
  `parse_lyrics` from the cloned DiffRhythm2 repo, patching the required
  `inference.lrc_tokenizer` module global exactly as the official scripts do.
- `KnowledgeDistillationTrainer._teacher_velocity()`: builds the real
  `[text_emb; noisy_latent]` combined sequence, boolean full-attention mask (with padded
  lyric-token positions masked out per-item), and calls `teacher(...)` with the exact
  argument names/shapes above. No fabricated inputs.
- **Style anchor unification**: preprocessing (`src/data/preprocess_raw_vietnamese.py`)
  now computes a real MuQ-MuLan embedding once per song (`compute_style_embedding`,
  10s clip) and stores it as `{id}_style.pt`. `AudioStyleEncoder`
  (`src/models/dit_transformer.py`) now projects this precomputed 512-dim vector
  directly (`Linear(512, dim)` stack) instead of pooling a raw mel crop. The exact same
  vector is now used as `style_prompt` for **both** the student and the teacher during
  distillation — previously these were unrelated fabrications on both sides.
- **Honest fallback**: if the teacher or tokenizer can't be loaded (no internet, repo not
  cloned, weight mismatch), `alpha_feature` is forced to `1.0` (train on ground-truth CFM
  loss only) and the returned report includes explicit `teacher_status` /
  `tokenizer_status` / `distillation_active` fields — no more silent `DummyTeacher`.
- Removed the dead `temperature` parameter.

## Known residual limitations

- The one-shot (non-cached) equivalence to the teacher's block-cached streaming
  inference is believed correct by construction (KV-caching is a performance
  optimization over the same attention computation) but has not been verified against a
  literal side-by-side numerical comparison of cached vs. non-cached teacher outputs —
  worth doing if distilled quality is suspiciously low.
- Position IDs for the noisy segment restart at 0 (matching the *first inference block*
  case in `cfm.py`, since our training chunks are analogous to a single block) rather
  than continuing after the text segment's position IDs — this matches official
  first-block behavior but means our replication is only exactly correct if the trained
  chunk length matches the teacher's own `block_size` (from `config.json`); a mismatch
  here would still give a *usable* signal, just not a bit-exact replication.
- CNENTokenizer's Chinese/English-only G2P is a real ceiling on how much lyric-level
  signal the teacher can contribute; this is a design tradeoff (see above), not a bug.
