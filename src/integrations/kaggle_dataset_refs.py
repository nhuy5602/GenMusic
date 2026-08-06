"""Single source of truth for the Kaggle dataset/kernel refs this project's
scripts point at -- committed here instead of scattered across .env values or
gitignored outputs/ state files, so they're visible and easy to update.
"""

from __future__ import annotations

import os

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

# Optional shortcuts for outputs created in the current Kaggle account. They
# are intentionally supplied through local environment variables rather than
# committed owner/slug refs. Every runner also accepts explicit kernel refs.
PROCESSED_RAW_AUDIO_KERNELS = {
    part: value
    for part in range(1, 7)
    if (
        value := os.environ.get(
            f"GENMUSIC_PROCESSED_RAW_KERNEL_PART_{part}", ""
        ).strip()
    )
}
