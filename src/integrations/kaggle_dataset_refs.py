"""Single source of truth for the Kaggle dataset/kernel refs this project's
scripts point at -- committed here instead of scattered across .env values or
gitignored outputs/ state files, so they're visible and easy to update.
"""

from __future__ import annotations

# Raw Vietnamese song corpus (sonlest/vietnamese-music-dataset-version3),
# split across 6 Kaggle dataset parts. Input to preprocess-raw.
RAW_DATASETS = [
    "https://www.kaggle.com/datasets/sonlest/vietnamese-music-dataset-version3-part1",
    "https://www.kaggle.com/datasets/sonlest/vietnamese-music-dataset-version3-part2",
    "https://www.kaggle.com/datasets/sonlest/vietnamese-music-dataset-version3-part3",
    "https://www.kaggle.com/datasets/sonlest/vietnamese-music-dataset-version3-part4",
    "https://www.kaggle.com/datasets/sonlest/vietnamese-music-dataset-version3-part5",
    "https://www.kaggle.com/datasets/sonlest/vietnamese-music-dataset-version3-part6",
]

# Kernel refs holding the `--raw-audio` preprocessing output (config.json's
# raw_audio_mode: true -- waveforms/*.pt, not mels/*.pt) for each RAW_DATASETS
# part, produced by scripts/run_kaggle_multi_part_preprocess_raw_audio.py. Confirmed accessible
# on 2026-07-24 (1843 records total). Point run_kaggle_latent_encoder.py /
# run_kaggle_latent_pipeline.py's `--raw-audio-part` at one of these keys
# instead of pasting the kernel ref by hand.
PROCESSED_RAW_AUDIO_KERNELS = {
    1: "quynhvu03/genmusic-data-prep-p1-1784830145",
    2: "quynhvu03/genmusic-data-prep-p2-1784830148",
    3: "quynhvu03/genmusic-data-prep-p3-1784843636",
    4: "quynhvu03/genmusic-data-prep-p4-1784843640",
    5: "quynhvu03/genmusic-data-prep-p5-1784859349",
    6: "quynhvu03/genmusic-data-prep-p6-1784859353",
}
