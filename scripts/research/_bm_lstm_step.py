"""Benchmark 子进程：训练对照 LSTM + 打分（内部脚本，勿单独使用）。

由 `scripts/research/run_benchmark.py` 通过 `subprocess` 调用，刻意跑在独立进程里
（本模块 import torch），原因见 `asharerank/predict/benchmark_stockmixer_step.py`
顶部注释。
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from asharerank.predict.benchmark_lstm_step import run_lstm_step

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--codes-file", required=True, help="股票代码列表 json 文件路径")
    p.add_argument("--train-start", required=True)
    p.add_argument("--train-end", required=True)
    p.add_argument("--val-start", required=True)
    p.add_argument("--val-end", required=True)
    p.add_argument("--test-start", required=True)
    p.add_argument("--test-end", required=True)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--window", type=int, default=16, help="序列回看窗口长度，默认对齐 StockMixer 的 --lookback")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None, choices=[None, "mps", "cuda", "cpu"])
    p.add_argument("--out-frame", required=True, help="预测长表输出路径（parquet）")
    p.add_argument("--out-meta", required=True, help="切分与 purge 元信息输出路径（json）")
    args = p.parse_args()

    codes = json.loads(Path(args.codes_file).read_text())
    frame = run_lstm_step(
        codes,
        train_start=args.train_start, train_end=args.train_end,
        val_start=args.val_start, val_end=args.val_end,
        test_start=args.test_start, test_end=args.test_end,
        horizon=args.horizon, window=args.window,
        epochs=args.epochs, patience=args.patience, seed=args.seed, device=args.device,
    )
    frame.to_parquet(args.out_frame, index=False)
    Path(args.out_meta).write_text(json.dumps(frame.attrs.get("split_metadata", {}), ensure_ascii=False, indent=2))
    print(f"LSTM_STEP_DONE frame={args.out_frame}")


if __name__ == "__main__":
    main()
