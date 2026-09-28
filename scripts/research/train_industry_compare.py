"""对比全市场标签 vs 行业中性化标签的 RankIC（全市场口径 + 行业内口径）。

用法:
  python scripts/research/train_industry_compare.py
  python scripts/research/train_industry_compare.py --max-stocks 200
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from asharerank.forecasting.experimental.industry_compare import run_industry_compare

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-stocks", type=int, default=None)
    args = p.parse_args()
    from asharerank.research_record import record_arguments
    record_arguments(args)
    run_industry_compare(max_stocks=args.max_stocks)


if __name__ == "__main__":
    from asharerank.research_record import run_recorded
    run_recorded(main)
