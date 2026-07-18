"""Submit the training/resume phase without evaluation or report plots.

All training options are provided by ``run_kaggle_iterative_self.py``. This
small entry point makes the three independent Kaggle phases explicit next to
``run_kaggle_quality_phase.py`` and ``run_kaggle_report_phase.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.run_kaggle_iterative_self import main


if __name__ == "__main__":
    if "--train-only" not in sys.argv:
        sys.argv.append("--train-only")
    main()
