# GenMusic VN

Big assignment project for Generative AI.

Final architecture:

```text
Local web/CLI
  input: Vietnamese text, from 1-2 sentences to dozens of sentences
  output: MP3 file link

Kaggle GPU
  long-text planning and condensation
  Vietnamese emotion analysis
  Vietnamese music stylebank lookup
  full song lyric rewriting
  key / scale / chord / melody planning
  MusicGen inference
  WAV -> MP3 conversion
```

Local does not run AI models and does not generate music. It only creates a Kaggle job from raw text, polls the job, downloads the generated `.mp3`, and serves the file.

## Vietnamese Music Stylebank Dataset

The project includes a structured knowledge dataset:

```text
datasets/vn_music_stylebank/
  emotion_to_music.json
  vietnamese_instruments.json
  genre_templates.json
  chord_presets.json
  lyric_patterns.json
```

This dataset is uploaded to Kaggle inside `genmusic_vn_source.zip`. The Kaggle kernel uses it before MusicGen inference to choose:

- mood-specific BPM, key, scale and chord progression
- Vietnamese instrument colors such as dan tranh, dan bau, sao truc and trong com
- genre prompt templates
- lyric imagery and chorus/bridge patterns
- MusicGen prompt keywords for Vietnamese emotional context

## Long Text Handling

The input can be a short prompt, 1-2 sentences, or a longer Vietnamese passage with dozens of sentences.

For long text, the Kaggle-side pipeline creates a `TextPlan`:

- counts sentences and words
- extracts keywords and motifs
- keeps representative sentences from the opening, development and ending
- builds a condensed text for melody, lyric and prompt generation
- rewrites the content into a complete song structure:
  `Verse 1 -> Pre-Chorus -> Chorus -> Verse 2 -> Bridge -> Final Chorus -> Outro`

The original text is still preserved in `request.json`; only the music-generation prompt is condensed.

## Why MusicGen Only

This is a non-commercial course project, so MusicGen is a good fit:

- It is simple to explain as text-to-music generative AI.
- It runs well in a Kaggle GPU notebook/kernel.
- The local app stays lightweight.
- We avoid maintaining two model backends in one assignment demo.

## Quick Start

Run the local client:

```powershell
python -m genmusic_vn.server --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

Submit Vietnamese text. The UI creates a Kaggle MusicGen job. When the job completes, the UI shows an MP3 player and a download link.

## Kaggle Setup

Install and configure Kaggle API once:

```powershell
pip install -U kaggle
mkdir $HOME\.kaggle
```

Download `kaggle.json` from Kaggle Account Settings and put it here:

```text
$HOME\.kaggle\kaggle.json
```

Or set environment variables:

```powershell
$env:KAGGLE_USERNAME="your_username"
$env:KAGGLE_KEY="your_api_key"
```

## CLI Demo

Stage only, no Kaggle submit:

```powershell
python -m genmusic_vn.cli generate --text "Mot chieu mua, toi nho ve nhung con pho cu." --duration 30 --no-submit
```

Submit to Kaggle and wait for MP3:

```powershell
python -m genmusic_vn.cli generate --text "Mot chieu mua, toi nho ve nhung con pho cu." --duration 30 --wait
```

The downloaded MP3 is stored under:

```text
outputs/<run_id>/kaggle_job/downloaded_output/
```

## What Gets Uploaded To Kaggle

For each request, local creates:

```text
outputs/<run_id>/
  request.json
  kaggle_job/
    dataset/
      request.json
      genmusic_vn_source.zip
      dataset-metadata.json
    kernel/
      run_genmusic_vn.py
      kernel-metadata.json
    run_commands.ps1
```

The Kaggle kernel unzips `genmusic_vn_source.zip`, runs the Vietnamese analysis pipeline, builds the MusicGen prompt, generates audio, converts it to MP3, and writes:

```text
/kaggle/working/genmusic_vn/<run_id>.mp3
/kaggle/working/genmusic_vn/kaggle_result.json
```

## Project Structure

```text
genmusic_vn/
  server.py           # local web/API client
  cli.py              # local CLI client
  kaggle_auto.py      # Kaggle dataset/kernel automation
  emotion.py          # Vietnamese emotion analysis, executed on Kaggle
  text_planner.py     # long input planning and condensation, executed on Kaggle
  music_theory.py     # key, scale, chord, melody planning, executed on Kaggle
  lyric_writer.py     # complete song lyric rewrite, executed on Kaggle
  prompt_builder.py   # MusicGen prompt builder, executed on Kaggle
  pipeline.py         # Kaggle-side orchestration
datasets/
  vn_music_stylebank/ # structured Vietnamese music knowledge dataset
web/
  index.html
  app.css
  app.js
tests/
  test_pipeline.py
```

## Verification

```powershell
python -m unittest discover -s tests -v
```

## References

- AudioCraft / MusicGen: https://github.com/facebookresearch/audiocraft
- MusicGen docs: https://raw.githubusercontent.com/facebookresearch/audiocraft/main/docs/MUSICGEN.md
- Kaggle API docs: https://www.kaggle.com/docs/api
