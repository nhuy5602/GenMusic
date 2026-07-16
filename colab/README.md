# Google Colab backend

Google Colab is an additional runtime for GenMusic. Existing Kaggle launchers,
datasets, authentication, and guides remain unchanged.

The notebook:

- mounts Google Drive for resumable checkpoints and generated audio;
- reads `KAGGLE_API_TOKEN` from Colab Secrets or a hidden runtime prompt;
- downloads the six public Kaggle preprocessing outputs;
- merges exactly 1,843 processed records;
- trains the improved model for 40 total epochs;
- resumes from the last completed epoch after a Colab disconnect;
- generates a conditioned MP3 using a real backing mel and MuQ style anchor.

Open the shared notebook:

<https://colab.research.google.com/drive/1zGT80eSQdyUjP6rMxY0WWsAo-xEVd8GD?usp=sharing>

To regenerate the local notebook:

```powershell
uv run python scripts/create_colab_notebook.py
```

Processed mel caching on Drive is disabled by default because the full dataset
can exceed a free Drive account. Enable it only when the Drive has enough space:

```powershell
uv run python scripts/create_colab_notebook.py --cache-data-on-drive
```
