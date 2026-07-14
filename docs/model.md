# Model Details

## Conv1D Diffusion Model

The default model is a self-authored conditional diffusion denoiser. It predicts
the denoising direction for a log-Mel spectrogram using a text/style condition
and a continuous diffusion timestep. It is implemented in
`src/models/text_to_music_diffusion.py` and trained by
`src/training/self_diffusion.py`.

The model is intentionally small enough for a local smoke run. Generated Mel
features can be rendered with the deterministic ISTFT path or the optional Vocos
neural vocoder.

## MicroDiT and CFM

The optional `MicroDiT` path uses a frozen pretrained text encoder, a compact
Transformer backbone, and an audio-style anchor encoder. Its training target is
Conditional Flow Matching: a noisy Mel state is interpolated between Gaussian
noise and a clean vocal target, and the network learns the velocity field.

`src/models/cfm_flow.py` contains the loss and Euler sampler. Separated datasets
provide the backing Mel as the audio condition and a cropped backing segment as
the style anchor. Legacy and synthetic records retain a zero backing fallback,
so they are useful for smoke tests but are not evidence of vocal quality.

## Distillation

`src/training/distill_training.py` contains the optional teacher-to-MicroDiT
training loop. A local teacher checkpoint is preferred; if the external teacher
package/checkpoint is unavailable, the code uses its initialized fallback only
to keep the training path executable. That fallback is a smoke-test teacher, not
a quality baseline.

## Important Evaluation Boundary

Random Mel data can verify tensor shapes, optimization, checkpoint loading, and
audio rendering. It cannot demonstrate natural singing, Vietnamese intelligibility,
rhyme quality, or vocal pacing. Those claims require real audio with a valid vocal
stem and lyric/alignment metadata.
