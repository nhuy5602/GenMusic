# Model Details

See `docs/PROJECT_REPORT.md` for the full architecture writeup with citations and
rationale; this file is a shorter reference. `docs/experiments/vocoder_fix.md`
and `docs/experiments/distillation_fix.md` cover the specific bugs found and
fixed in the areas below — read those before changing this code again, several
non-obvious pitfalls (wrong mel format, wrong teacher call signature, guessed
architecture dims) are documented there in detail.

## MicroDiT and CFM

MicroDiT is the only model architecture in this project (the earlier Conv1D
`ResidualDenoiser` smoke-test baseline was removed once MicroDiT/CFM proved out
— see git history before this cleanup if you need it for reference).

`MicroDiT` (`src/models/dit_transformer.py`) conditions on: (1) lyric phonemes via
a frozen `vinai/xphonebert-base` encoder using the G2P parser `text2phonemesequence`
(pre-trained specifically for multilingual phoneme-level representations including Vietnamese);
(2) a single 512-dim
**MuQ-MuLan style embedding**, precomputed once per song at preprocessing time
(`AudioStyleEncoder`, a small 2-layer MLP adapter — *not* an audio encoder
itself, see `docs/PROJECT_REPORT.md` §3.2). Training target is Conditional Flow
Matching (`src/models/cfm_flow.py`): a noisy mel state is interpolated between
Gaussian noise and the clean vocal target, and the network learns the velocity
field, integrated at sample time with fixed-step Euler ODE integration.

Unlike the previous version which concatenated all features along the channel dimension, or the intermediate version which added lyric embeddings directly, the model now implements **unified sequence concatenation** matching DiffRhythm2 exactly:
- **Sequence Concatenation**: The text phoneme embeddings (from XPhoneBERT) and the projected vocal mel-spectrogram embeddings are concatenated along the **sequence dimension** (`dim=1`), forming a single unified sequence of length `text_len + seq_len` passed to the Transformer.
- **Additive Conditioning**: Style embeddings (from MuQ-MuLan) and 2D time embeddings (with `-1.0` sentinels for text tokens and timestep `t` for vocal frames) are added directly to the concatenated sequence representation.
- **Attention**: The model uses bidirectional non-causal attention across the entire text-audio sequence (excluding padding tokens), allowing the model to learn soft G2P-audio alignment implicitly.
- The backing track mel-spectrogram conditioning (`backing_mel`/`cond`) has been completely removed to align the student's architecture with the teacher's.

**Generation conditioning**: `generate_audio()` accepts a `style_prompt`/`style_anchor` tensor; without them, generation falls back to using the pooled-text representation as a stand-in style vector.
`load_reference_conditioning()` (`src/training/self_diffusion.py`) extracts the style anchor from a preprocessed dataset record, and `generate-local --reference-dataset --reference-id` wires it into the CLI — see README's Local Generation section.

Architecture size is configurable: `--dim`/`--depth`/`--heads`/`--ff-mult` on
both `train-self` and `train-distill`. Default
`dim=256, depth=4, heads=4` is ~5.6M trainable parameters (plus 135M frozen
XPhoneBERT weights, never trained, re-downloaded fresh from HuggingFace on
checkpoint load rather than saved).

## Distillation

`src/training/distill_training.py` replicates the *real* DiffRhythm2 teacher's
call contract (reverse-engineered from its actual GitHub source — not guessed
from class defaults, which was the previous approach and was wrong: the
teacher's real checkpoint uses `mel_dim=64`, not the `100` its Python default
suggested). A small trainable linear adapter bridges the teacher's 64-mel space
and the student's 100-mel space for the distillation loss only. If the teacher
(or its lyric tokenizer) can't be loaded — no internet, DiffRhythm2 repo not
on `PYTHONPATH` with its dependencies installed (works on Kaggle via
`scripts/run_kaggle_distill.py`, or locally by cloning it yourself and
installing deps as they come up) — `train-distill` **raises immediately**
rather than either silently substituting a fake teacher or silently
downgrading to ground-truth-only training under the distillation name. Use
`train-self` for ground-truth-only training; a `train-distill` run that
completes always means a real teacher was used.

**`distillation_active: true` has been confirmed locally** (CPU, real
DiffRhythm2 teacher + real lyric tokenizer, 2-song dataset, 30 epochs) — the
mechanism is genuinely correct end-to-end. **Not yet confirmed**: whether
distillation actually improves quality over training from scratch — at this
tiny 2-song/1-step-per-epoch scale, CFM's random timestep sampling makes
per-step loss too noisy (swings 3.5-229 in the same run) to distinguish signal
from noise, and it has also never completed a full run on Kaggle GPU (every
attempt there hit a bug or ran out of quota first — see
`docs/experiments/kaggle_runs.md`). Both require a larger dataset/more steps
per epoch to answer.

## Important Evaluation Boundary

Random mel data can verify tensor shapes, optimization, checkpoint loading, and
audio rendering. It cannot demonstrate natural singing, Vietnamese
intelligibility, rhyme quality, or vocal pacing. Those claims require real
audio with a valid vocal stem and lyric/alignment metadata, and — even then —
a human listening to the output; automated sanity stats (peak amplitude, RMS,
silence ratio, NaN/Inf checks) catch crashes and degenerate output, not
musical quality.
