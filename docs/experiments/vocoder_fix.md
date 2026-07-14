# Experiment: root-causing and fixing distorted audio output

**Date:** 2026-07-14
**Reported symptom:** user ran a "very simple baseline" (`cli.py generate-local`) and the
generated song was badly distorted / low quality compared to real reference songs in
`dataset/vietnamese_songs/`.

## Root cause

`generate-local` defaulted to `--vocoder istft`, which called
`render_mel_to_wav(..., vocoder_type="istft")` in
`src/models/text_to_music_diffusion.py`. That function did **not** implement Griffin-Lim
or any real phase reconstruction. It fabricated a phase spectrum from a deterministic
linear ramp (`phase = 2*pi*f*t + linspace(0, pi, n_freq)`) and combined it with the
magnitude recovered via mel pseudo-inverse. This is not a phase estimator — it produces a
fixed, content-independent phase pattern that has no relationship to the actual signal.

The secondary path, `--vocoder vocos`, used a real pretrained neural vocoder
(`charactr/vocos-mel-24khz`) but fed it a **bilinear-interpolated** mel: the project's own
mel convention (64 mels, 16kHz, n_fft=512, hop=256, power-mel, log-clipped to [-5,3]) is
completely different from what Vocos was trained on (100 mels, 24kHz, n_fft=1024,
hop=256, magnitude-mel with `power=1`, natural log, no clipping). Bilinearly resizing a
64-bin power-mel image into a 100-bin magnitude-mel image does not perform the correct
psychoacoustic frequency-scale conversion, and the value/scale mismatch pushes Vocos far
outside its training distribution.

Either path explains "distorted, bad quality" independent of how good the trained model's
mel predictions are — even a perfect mel prediction would decode to garbage.

## Diagnosis method

Round-tripped a real reference song (`dataset/vietnamese_songs/-B2Zao5CRB0.mp3`) through
each candidate reconstruction path and measured log-mel correlation/RMSE between the
original audio's mel and the mel of the reconstructed audio (same content in and out, so
this isolates vocoder fidelity from model quality):

| Path | logmel corr | logmel RMSE |
|---|---|---|
| Old istft hack (fabricated phase) | **0.149** | 4.37 |
| Old vocos path (bilinear-interpolated mel) | 0.686 | 2.92 |
| Griffin-Lim (proper, `librosa.feature.inverse.mel_to_audio`, old 64-mel/16kHz config) | 0.495 | 8.12 |
| Vocos decoding a mel computed with Vocos's *own* native feature extractor | **0.896** | 3.11 |

A correlation of 0.149 is close to noise. This confirms the istft path was the primary
cause of the reported distortion, and that decoding with Vocos using its *native* mel
format (no resampling) is dramatically better.

## Fix

1. **`MusicDiffusionConfig`** (`src/models/text_to_music_diffusion.py`) defaults changed
   from `(sample_rate=16000, n_mels=64, n_fft=512, hop=256)` to
   `(sample_rate=24000, n_mels=100, n_fft=1024, hop=256)` — bit-for-bit matching Vocos's
   `MelSpectrogramFeatures` (`torchaudio.transforms.MelSpectrogram(power=1)` +
   `log(clip(x, 1e-7))`, no upper clip).
2. Added `compute_mel_spectrogram()` implementing that exact transform, used by both
   preprocessing (`src/data/preprocess_raw_vietnamese.py`) and any inference/eval code
   that needs to compute a mel — verified bit-identical (`max abs diff == 0.0`) against
   `vocos.feature_extractors.MelSpectrogramFeatures` directly.
3. Rewrote `render_mel_to_wav`: `vocoder_type="vocos"` (now the default) decodes the mel
   **unmodified** — no resampling needed since the representation already matches.
   `vocoder_type="griffinlim"` (fallback if Vocos unavailable or config doesn't match)
   uses real iterative Griffin-Lim (`librosa.feature.inverse.mel_to_audio`, 64 iterations)
   instead of the fabricated-phase hack, which has been deleted entirely.
4. `cli.py generate-local --vocoder` default changed from `istft` to `vocos`;
   `--vocoder istft` choice removed (replaced by `griffinlim`, the correct name for what
   the fallback actually does).
5. Verified end-to-end with the fixed code on the same reference song:

   | Path (after fix) | logmel corr | logmel RMSE |
   |---|---|---|
   | Fixed Vocos (native mel, no resampling) | **0.997** | 1.78 |
   | Fixed Griffin-Lim fallback (native mel) | 0.960 | 2.85 |

   Both are now in the "should sound clean" range; Vocos remains the recommended default.

## A second-order consequence (not a bug, a discovery)

DiffRhythm2's own teacher DiT defaults to `mel_dim=100` (confirmed by reading
`diffrhythm2/backbones/dit.py`, see `docs/experiments/distillation_fix.md`). Choosing
Vocos's 100-mel/24kHz convention as the student's native representation, therefore,
doesn't just fix the vocoder — it also happens to match the *teacher's own* mel space,
which is necessary groundwork for distillation to be meaningful at all (previously the
student trained in a 64-mel/16kHz space that the teacher's velocity predictions,
mel_dim=100, couldn't even be shape-compatible with).

## Follow-up / residual risk

- The legacy `conv1d`/`ResidualDenoiser` DDPM path (`train-self` without `--model-type
  dit`) still exists as a toy baseline; its sampling clamp range was widened
  (`-5..3` &rarr; `-12..4`) to stay roughly sane under the new mel convention, but it was
  not otherwise redesigned — it is not the recommended path for quality work.
- Changing the mel convention invalidates any previously preprocessed dataset / trained
  checkpoint (old 64-mel tensors are shape-incompatible with the new 100-mel model). All
  datasets must be regenerated with `preprocess-raw` after this fix; see
  `docs/experiments/kaggle_runs.md` for the regenerated dataset run.
