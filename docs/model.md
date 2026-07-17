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
itself, see `docs/PROJECT_REPORT.md` §3.2) — this replaced an earlier version
that average-pooled a raw mel crop through an untrained Conv1D, which had no
learned notion of musical style at all. Training target is Conditional Flow
Matching (`src/models/cfm_flow.py`): a noisy mel state is interpolated between
Gaussian noise and the clean vocal target, and the network learns the velocity
field, integrated at sample time with fixed-step Euler ODE integration.

The student's mel representation (100 mels, 24kHz, n_fft=1024, hop=256) is
chosen to exactly match the pretrained Vocos vocoder's own native format, not
any dimension DiffRhythm2 itself uses — this was the single highest-impact fix
in this project's history (see `docs/experiments/vocoder_fix.md`).

**Generation conditioning**: `generate_audio()` accepts `backing_mel`/`style_anchor`
tensors; without them (the default), generation uses a zero backing-track and a
pooled-text stand-in for the style vector instead of a real MuQ-MuLan anchor — a
genuine train/inference mismatch, not a deliberate simplification, since training
always conditions on real `backing_mel` + a real (or zero-fallback) style anchor.
`load_reference_conditioning()` (`src/training/self_diffusion.py`) extracts both
from an existing preprocessed dataset record, and `generate-local --reference-dataset
--reference-id` wires it into the CLI — see README's Local Generation section.
Long lyrics spanning multiple flow-matching chunks get the *matching* time window of
the reference backing track per chunk (not always frame 0), wrapping around if the
reference is shorter than the requested duration.

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
