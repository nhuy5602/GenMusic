"""Shared constants for the retained native waveform pipeline."""

from __future__ import annotations


# Stable source bundle used by the Kaggle launchers before they overlay the
# versioned waveform-pipeline patch.  Keeping this here avoids importing an
# obsolete experiment runner solely for one constant.
BASE_SOURCE_DATASET_REF = (
    "ngochuy5602/genmusic-source-master-segment-quality-1785099667"
)
