"""内部脚本：只做 StockMixer 数据准备（不训练），导出对齐用的股票池 + 时间边界。

由 `scripts/research/train_lgbm_aligned.py` 通过 `subprocess` 调用，跑在独立进程里
（本脚本 import torch），原因见 `asharerank/predict/benchmark_stockmixer_step.py`
顶部注释（PyTorch + LightGBM 同进程在 macOS 上会 OpenMP 冲突 segfault）。
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

from asharerank.predict.benchmark_stockmixer_step import get_universe_and_bounds
from asharerank.predict.experimental.stockmixer.config import StockMixerConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", default="csi300")
    p.add_argument("--max-stocks", type=int, default=None)
    p.add_argument("--lookback", type=int, default=16)
    p.add_argument("--label-horizon", type=int, default=5)
    p.add_argument("--out", required=True, help="输出 json 路径")
    args = p.parse_args()

    cfg = StockMixerConfig(
        pool=args.pool, max_stocks=args.max_stocks,
        lookback=args.lookback, label_horizon=args.label_horizon,
    )
    info = get_universe_and_bounds(cfg)
    Path(args.out).write_text(json.dumps(info, ensure_ascii=False, indent=2))
    print(f"UNIVERSE_STEP_DONE out={args.out} n_codes={len(info['codes'])}")


if __name__ == "__main__":
    main()
