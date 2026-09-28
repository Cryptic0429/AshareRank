"""训练 StockMixer（AAAI-24）沪深300 基线模型，并在测试集上打印 IC/RIC/Precision@K/Sharpe。

用法:
  python scripts/research/train_stockmixer.py
  python scripts/research/train_stockmixer.py --max-stocks 60 --epochs 30   # 调试用小样本
  python scripts/research/train_stockmixer.py --device cpu                  # 强制用 CPU
"""
from __future__ import annotations

# 必须在 import torch（间接经由下方模块导入）之前设置，使 MPS 不支持的算子自动
# fallback 到 CPU，而不是抛异常中断训练。
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from asharerank.forecasting.experimental.stockmixer.config import StockMixerConfig
from asharerank.forecasting.experimental.stockmixer.train import run_stockmixer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    p = argparse.ArgumentParser(description="StockMixer (AAAI-24) 沪深300 训练/评估")
    p.add_argument("--pool", default="csi300", choices=["csi300", "csi500", "csi800"])
    p.add_argument("--max-stocks", type=int, default=None, help="调试用：限制股票数量")
    p.add_argument("--train-years", type=float, default=3.0,
                  help="训练历史长度（年）。val/test 仍固定为近 6+6 月，只有 train_start 往前推")
    p.add_argument("--min-listed-days", type=int, default=250,
                  help="训练起点前最少历史交易日（默认250≈1年）")
    p.add_argument("--freeze-codes-from-ckpt", type=str, default=None,
                  help="从已有 .pt 读取 codes 并冻结股票池")
    p.add_argument("--lookback", type=int, default=16, help="滑动窗口长度 T")
    p.add_argument("--label-horizon", type=int, default=5,
                  help="预测目标前瞻天数（默认5，与当前多 seed/线上口径对齐；1日噪声过大）")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=8, help="一个 batch 内的交易日数")
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--dropout", type=float, default=0.1, help="Mixer block dropout（调大可缓解过拟合）")
    p.add_argument("--weight-decay", type=float, default=1e-5, help="Adam weight_decay（调大可缓解过拟合）")
    p.add_argument("--patience", type=int, default=10, help="早停耐心轮数（val_RankIC 连续多少轮不改善就停）")
    p.add_argument("--market-hidden", type=int, default=10, help="Market-Aware Stock Mixing 隐层维度 m")
    p.add_argument("--rank-loss-weight", type=float, default=0.5,
                  help="组合损失中 Rank Loss 权重 λ（论文默认 0.1，A股实测调高到 0.5 效果更好）")
    p.add_argument("--no-cross-sectional-norm", action="store_true",
                  help="关闭逐日横截面标准化（默认开启，P0 优化项）")
    p.add_argument("--use-fundamentals", action="store_true",
                  help="开启基本面因子（ROE/净利率/估值比率等，默认关闭：2026-07-23实测覆盖率"
                       "不足+加剧过拟合导致 RankIC 下降，见 config.py 注释）")
    p.add_argument("--fund-fields", type=str, default=None,
                  help="配合 --use-fundamentals：逗号分隔字段，覆盖默认 7 维")
    p.add_argument("--no-industry-neutralize", action="store_true",
                  help="配合 --use-fundamentals：关闭基本面因子的行业内截面标准化（默认开启）")
    p.add_argument("--fund-missing-mask", action="store_true",
                  help="配合 --use-fundamentals：附加 fund_missing 缺测掩码维（默认关闭；"
                       "覆盖率不足时实测伤 RankIC，等覆盖 90%+ 再开）")
    p.add_argument("--use-ema", action="store_true",
                  help="启用权重 EMA（修复版：warmup 衰减 + 验证集直接评 EMA 权重）。默认关闭=基线行为")
    p.add_argument("--ema-decay", type=float, default=0.9, help="EMA 衰减上限（配合 --use-ema）")
    p.add_argument("--rankic-smooth-window", type=int, default=1,
                  help="早停判据用最近 N 轮 val_RankIC 滑动平均；1=裸单轮值（基线行为）")
    p.add_argument("--market-layerscale-init", type=float, default=1.0,
                  help="MarketAwareStockMixing 残差门控初始值；1.0=与原始结构等价（基线行为）")
    p.add_argument("--activation", default="hardswish", choices=["relu", "hardswish"])
    p.add_argument("--top-k", type=int, default=10, help="Precision@K / Sharpe 组合的 K")
    p.add_argument("--device", default=None, choices=[None, "mps", "cuda", "cpu"],
                  help="留空自动检测（优先 mps）")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    from asharerank.research_record import record_arguments
    record_arguments(args)

    freeze_codes = None
    if args.freeze_codes_from_ckpt:
        import torch
        ckpt = torch.load(args.freeze_codes_from_ckpt, map_location="cpu", weights_only=False)
        freeze_codes = tuple(str(c) for c in ckpt["codes"])

    fund_fields = None
    if args.fund_fields:
        fund_fields = tuple(s.strip() for s in args.fund_fields.split(",") if s.strip())

    cfg = StockMixerConfig(
        pool=args.pool,
        max_stocks=args.max_stocks,
        train_years=args.train_years,
        min_listed_days=args.min_listed_days,
        freeze_codes=freeze_codes,
        lookback=args.lookback,
        label_horizon=args.label_horizon,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        patience=args.patience,
        market_hidden=args.market_hidden,
        rank_loss_weight=args.rank_loss_weight,
        cross_sectional_norm=not args.no_cross_sectional_norm,
        use_fundamentals=args.use_fundamentals,
        fundamental_fields=fund_fields,
        industry_neutralize_fundamentals=not args.no_industry_neutralize,
        fund_missing_mask=args.fund_missing_mask,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
        rankic_smooth_window=args.rankic_smooth_window,
        market_layerscale_init=args.market_layerscale_init,
        activation=args.activation,
        top_k=args.top_k,
        device=args.device,
        seed=args.seed,
    )
    run_stockmixer(cfg)


if __name__ == "__main__":
    from asharerank.research_record import run_recorded
    run_recorded(main)
