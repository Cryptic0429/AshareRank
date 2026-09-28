"""Benchmark 子进程 2/2：训练对照 LightGBM + 打分（内部脚本，勿单独使用）。

由 `scripts/research/run_benchmark.py` 通过 `subprocess` 调用，刻意跑在独立进程里，
不 import torch，避免和 StockMixer 那一步同进程导致 OpenMP 冲突 segfault。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from asharerank.forecasting.benchmark_lgbm_step import run_lgbm_step

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--codes-file", required=True, help="股票代码列表 json 文件路径")
    p.add_argument("--test-start", required=True)
    p.add_argument("--test-end", required=True)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--valid-months", type=int, default=3)
    p.add_argument("--train-start", default=None, help="对齐模式：train 起点（不传则走宽松模式）")
    p.add_argument("--train-end", default=None, help="对齐模式：train 终点")
    p.add_argument("--val-start", default=None, help="对齐模式：早停验证集起点")
    p.add_argument("--val-end", default=None, help="对齐模式：早停验证集终点")
    p.add_argument("--out-frame", required=True, help="预测长表输出路径（parquet）")
    p.add_argument("--out-meta", required=True, help="切分与 purge 元信息输出路径（json）")
    args = p.parse_args()
    from asharerank.research_record import record_arguments
    record_arguments(args)

    codes = json.loads(Path(args.codes_file).read_text())
    frame = run_lgbm_step(codes, args.test_start, args.test_end,
                          horizon=args.horizon, valid_months=args.valid_months,
                          train_start=args.train_start, train_end=args.train_end,
                          val_start=args.val_start, val_end=args.val_end)
    frame.to_parquet(args.out_frame, index=False)
    Path(args.out_meta).write_text(json.dumps(frame.attrs.get("split_metadata", {}), ensure_ascii=False, indent=2))
    print(f"LGBM_STEP_DONE frame={args.out_frame}")


if __name__ == "__main__":
    from asharerank.research_record import run_recorded
    run_recorded(main)
