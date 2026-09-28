"""基本面因子消融对比 CLI。

用法:
  python scripts/research/train_fundamentals_compare.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from asharerank.forecasting.experimental.fundamentals_compare import run_fundamentals_compare

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")


if __name__ == "__main__":
    run_fundamentals_compare()
