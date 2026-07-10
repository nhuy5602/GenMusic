"""Model integration is delegated to the official DiffRhythm checkout."""

from ..integrations.diffrhythm_official import DiffRhythmConfig, DiffRhythmError, ensure_official_checkout, run_official_inference

__all__ = ["DiffRhythmConfig", "DiffRhythmError", "ensure_official_checkout", "run_official_inference"]
