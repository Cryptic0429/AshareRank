"""LSTM 对比基线（独立于生产管线，不写入模型注册表）。

用法:
  python scripts/research/train_lstm.py
  python scripts/research/train_lstm.py --max-stocks 60 --window 20
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from asharerank.forecasting.experimental.lstm.config import LSTMConfig
from asharerank.forecasting.experimental.lstm.train import run_lstm_baseline

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-stocks", type=int, default=None)
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--epochs-per-fold", type=int, default=8)
    args = p.parse_args()
    from asharerank.research_record import record_arguments
    record_arguments(args)

    cfg = LSTMConfig(window=args.window, hidden_size=args.hidden_size,
                     epochs_per_fold=args.epochs_per_fold)
    run_lstm_baseline(max_stocks=args.max_stocks, cfg=cfg)


if __name__ == "__main__":
    from asharerank.research_record import run_recorded
    run_recorded(main)
