"""Benchmark 子进程 1/2：StockMixer 训练 + 导出测试集预测（内部脚本，勿单独使用）。

由 `scripts/research/run_benchmark.py` 通过 `subprocess` 调用，刻意跑在独立进程里，
避免和 LightGBM 那一步同进程导致 OpenMP 冲突 segfault（详见
`asharerank/forecasting/benchmark_stockmixer_step.py` 顶部注释）。
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

from asharerank.forecasting.benchmark_stockmixer_step import run_stockmixer_step, run_stockmixer_step_multiseed
from asharerank.forecasting.experimental.stockmixer.config import StockMixerConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", default="csi300")
    p.add_argument("--max-stocks", type=int, default=None)
    p.add_argument("--lookback", type=int, default=16)
    p.add_argument("--label-horizon", type=int, default=5)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--rank-loss-weight", type=float, default=0.5)
    p.add_argument("--use-fundamentals", action="store_true")
    p.add_argument("--use-ema", action="store_true", help="启用权重 EMA（同 train_stockmixer_multiseed.py）")
    p.add_argument("--ema-decay", type=float, default=0.9)
    p.add_argument("--rankic-smooth-window", type=int, default=1,
                  help="早停判据用最近 N 轮 val_RankIC 滑动平均；1=裸单轮值")
    p.add_argument("--market-layerscale-init", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42, help="仅单 seed 模式生效（未传 --seeds/--n-seeds>1 时）")
    p.add_argument("--seeds", type=str, default=None,
                  help="逗号分隔的随机种子列表，例如 '42,43,44,45,46'；传了就走多 seed 等权集成，"
                       "覆盖 --seed/--n-seeds/--seed-start")
    p.add_argument("--n-seeds", type=int, default=1,
                  help="未传 --seeds 时，生成的随机种子个数；>1 时走多 seed 等权集成（默认 1=沿用旧的单 seed 行为）")
    p.add_argument("--seed-start", type=int, default=42, help="未传 --seeds 时，随机种子的起始值（依次+1）")
    p.add_argument("--device", default=None, choices=[None, "mps", "cuda", "cpu"])
    p.add_argument("--out-frame", required=True, help="预测长表输出路径（parquet）")
    p.add_argument("--out-meta", required=True, help="元信息输出路径（json）")
    args = p.parse_args()

    if args.seeds:
        seeds = tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip())
    else:
        seeds = tuple(args.seed_start + i for i in range(args.n_seeds))

    cfg = StockMixerConfig(
        pool=args.pool, max_stocks=args.max_stocks, lookback=args.lookback,
        label_horizon=args.label_horizon, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, dropout=args.dropout, weight_decay=args.weight_decay, patience=args.patience,
        rank_loss_weight=args.rank_loss_weight, use_fundamentals=args.use_fundamentals,
        use_ema=args.use_ema, ema_decay=args.ema_decay,
        rankic_smooth_window=args.rankic_smooth_window,
        market_layerscale_init=args.market_layerscale_init,
        seed=seeds[0], device=args.device,
    )
    if len(seeds) > 1:
        frame, meta = run_stockmixer_step_multiseed(cfg, seeds=seeds)
    else:
        frame, meta = run_stockmixer_step(cfg, seed=seeds[0])
    frame.to_parquet(args.out_frame, index=False)
    Path(args.out_meta).write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str))
    print(f"STOCKMIXER_STEP_DONE frame={args.out_frame} meta={args.out_meta}")


if __name__ == "__main__":
    main()
