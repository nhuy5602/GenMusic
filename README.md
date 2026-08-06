# GenMusic VN

GenMusic VN is a Vietnamese lyric-to-music research project with two explicit
runtime paths:

- `native-waveform-v80`: the default 16-second demo path. It retrieves and
  connects real Vietnamese vocal units, then mixes a cross-song backing track.
- `cfm-research`: the self-authored Conditional Flow Matching/MicroDiT path
  used for model research and ablations.

The two paths are intentionally labelled separately. A V80 output is not
reported as a diffusion result.

## Fresh clone → fresh Kaggle account

The repository contains no developer Kaggle owner/slug, credential, local
checkpoint, PDF, or defense artifact. A new clone builds its native corpus in
the Kaggle account supplied by the user.

### Prerequisites

- Git
- [uv](https://docs.astral.sh/uv/)
- A Kaggle account with GPU access and an API token
- Internet access from the Kaggle kernel (for the documented Hugging Face
  dataset and Demucs model)

### 1. Install from the lock file

```powershell
git clone <repository-url> GenMusic
cd GenMusic
uv sync --locked
Copy-Item .env.example .env
```

Fill only your own credentials in `.env`:

```env
KAGGLE_USERNAME=your_account
KAGGLE_API_TOKEN=your_token
```

`KAGGLE_KEY` remains supported for legacy Kaggle credentials. `.env` is
ignored and must never be committed.

### 2. Bootstrap the V80 corpus in that account

```powershell
uv run python scripts/run_kaggle_native_waveform_dataset.py --wait
```

This command:

1. packages the current clone as a token-free private source asset;
2. submits a private GPU kernel under the current Kaggle username;
3. materializes the exact-timestamp, song-disjoint 2,048-record vocal/backing
   corpus from the configured public Hugging Face dataset;
4. stores the resulting `owner/kernel` ref in
   `.genmusic/kaggle.json` after submission.

`.genmusic/`, datasets, outputs, caches, and local notes are ignored by Git.
No ref from the original developer account is required. The bootstrap is
expensive and runs once per account; later generations reuse that completed
kernel output.

If you do not want to wait in the terminal, omit `--wait`, monitor the printed
Kaggle URL, and generate only after the corpus kernel is `COMPLETE`.

### 3. Generate an MP3/WAV candidate

Use 8–32 Vietnamese words. V80 stays at its validated 16-second duration.

```powershell
uv run python cli.py generate `
  --text "Một chiều mưa tôi nhớ về những con phố cũ và lời hẹn trong tim" `
  --wait
```

The generated state and downloaded audio are written below `outputs/`.
Kaggle scheduling and upstream hosting can introduce small runtime differences,
but the shard list, corpus-selection rules, duration policy, and V80 synthesis
policy are carried by the repository rather than a personal account.

### 4. Run the web app

```powershell
uv run python server.py
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). The web app uses the
same ignored `.genmusic/kaggle.json` and the same V80 submitter as the CLI.

## Research CFM path

The smallest local research loop is:

```powershell
uv run python cli.py preprocess-raw `
  --input dataset/vietnamese_songs `
  --output dataset/diff_rhythm_dataset `
  --whisper-model tiny

uv run python cli.py train-self `
  --dataset dataset/diff_rhythm_dataset `
  --checkpoint outputs/cfm_student.pt `
  --epochs 30 --batch-size 4 --dim 256 --depth 4 --heads 4

uv run python cli.py generate-local `
  --text "Đêm nay Hà Nội ngập tràn tiếng mưa rơi" `
  --checkpoint outputs/cfm_student.pt `
  --out outputs/cfm_demo
```

For preprocessing contracts, latent-codec experiments, distillation, and
Kaggle research launchers, read:

- [Architecture](docs/architecture.md)
- [Data preparation](docs/data_preparation.md)
- [Usage](docs/usage.md)
- [Scripts index](scripts/README.md)

## Repository layout

```text
src/            model, training, audio, data and Kaggle integration code
scripts/        reproducible local/Kaggle entry points
tests/          unit and integration-contract tests
web/            browser UI
docs/           technical documentation
cli.py          command-line interface
server.py       lightweight HTTP server
```

Report/PDF/defense notes are not part of this tree. Local copies may be kept
under ignored `local_notes/` without entering a commit or Kaggle source asset.

## Verification before submission

```powershell
uv run python scripts/check_portability.py
uv run --with pytest python -m pytest -q
```

The portability audit checks Git-visible files for local-machine paths,
embedded Kaggle tokens, and report/defense-only artifacts. Tests also verify
that a native corpus ref is loaded from environment/local config rather than a
committed personal default.
