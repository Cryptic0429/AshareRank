"""XGBoost 对比基线（独立于生产 LightGBM 管线，不写入模型注册表）。

用法:
  python scripts/research/train_xgboost.py
  python scripts/research/train_xgboost.py --max-stocks 60
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from asharerank.forecasting.experimental.xgboost_baseline import run_xgboost_baseline

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-stocks", type=int, default=None)
    args = p.parse_args()
    from asharerank.research_record import record_arguments
    record_arguments(args)
    run_xgboost_baseline(max_stocks=args.max_stocks)


if __name__ == "__main__":
    from asharerank.research_record import run_recorded
    run_recorded(main)
