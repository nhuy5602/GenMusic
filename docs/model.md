# Machine Learning Model Details

GenMusic VN utilizes three distinct model classes to power the generative audio pipe:

---

## 1. Trained Text Classifier (Multinomial Naive Bayes)

- **Purpose:** Analyzes Vietnamese prose to detect emotions and musical genre cues.
- **Algorithm:** Built from scratch using multinomial distributions of word unigrams and bigrams.
- **Diacritic Normalization:** Extracts accentless tokens (e.g. `mua roi` from `mưa rơi`) to handle varying input typing conventions.
- **Priors and Inference:** Calculates class probabilities:
  $$\text{Score}(c) = \log P(c) + \sum_{i} \text{count}(f_i) \cdot \log P(f_i | c)$$
  Uses a softmax layer to normalize confidence outputs.

---

## 2. Self-authored Music Diffusion Model

Designed to replace massive third-party packages, this model is a self-contained PyTorch conditional diffusion model tailored for spectrogram synthesis.

### Network Components
1. **Text Conditioner:**
   - Embeds text tokens from the input lyric and style descriptions.
   - Passes embeddings through a positional embedding layer and a 2-layer `TransformerEncoder` with Multi-Head Attention.
   - Pools character-level features to build a single condition vector.
2. **Residual Conv1D Denoiser:**
   - Processes Mel-spectrogram grids using 1D convolutional residual blocks.
   - Injects sinusoidal time step embeddings along with the text conditioner embedding at each residual layer.
   - Employs Group Normalization (`GroupNorm`) and `SiLU` activations.

### Loss Function
Computes the Mean Squared Error (MSE) loss between the true Gaussian noise $\epsilon$ and the predicted noise $\epsilon_\theta$ at random timesteps $t$:
$$\mathcal{L} = \mathbb{E}_{x_0, \epsilon, t} \left[ \| \epsilon - \epsilon_\theta(x_t, t, \text{condition}) \|^2 \right]$$

### Spectrogram to Audio Rendering
Mel-spectrogram outputs are reconstructed using Librosa:
- Applies a pseudo-inverse operation on the Mel filterbank matrix to project Mel bands back to linear-frequency magnitude bins.
- Estimates phase characteristics using a deterministic frequency-time projection phase matrix.
- Runs Inverse Short-Time Fourier Transform (`ISTFT`) to obtain time-domain waveform signals.

---

## 3. Custom Music Composer (Rule-Based Synth)
- **Melody/MIDI Engine:** Translates note sequences into standard MIDI events.
- **Waveform Synthesis:** Performs additive synthesis by summing fundamental sine waves and fractional harmonics to generate keyboard, bass, and string waveforms.
- **Percussion Modeling:**
  - **Kick:** Logarithmically decays frequency from 62Hz down to 38Hz.
  - **Snare:** Blends low-mid sine wave tones with high-pass filtered white noise.
  - **Hi-hat:** Rapidly decays high-pass filtered white noise.

---

## 4. Singing Voice Synthesis (TTS)
- **F5-TTS Vietnamese:** Acoustic model trained on Vietnamese voices (`hynt/F5-TTS-Vietnamese-ViVoice`). Takes a 15-second reference audio slice to capture natural timbre, singing tone, and pitch variation.
- **MMS-TTS:** Meta's Massively Multilingual Speech model (`facebook/mms-tts-vie`), serving as a fast CPU-safe fallback.
