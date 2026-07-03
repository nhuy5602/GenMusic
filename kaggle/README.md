# Kaggle Workflow

The current project uses only MusicGen on Kaggle.

Normal demo flow is automated by `genmusic_vn.kaggle_auto`:

1. Local receives raw Vietnamese text.
2. Local uploads `request.json` and `genmusic_vn_source.zip` as a private Kaggle Dataset.
3. Local pushes a private Kaggle Kernel with GPU enabled.
4. Kaggle unzips source code plus `datasets/vn_music_stylebank`.
5. Kaggle runs the full AI pipeline and MusicGen inference.
6. Kaggle converts the generated WAV to MP3.
7. Local downloads the MP3 from the kernel output.

The generated kernel starts from raw text. It does not require a prebuilt prompt pack.
