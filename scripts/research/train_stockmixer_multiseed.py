"""多 seed 复现 StockMixer（AAAI-24）沪深300 基线模型，检验单次结果是否可信。

同一份数据只准备一次，用不同随机种子重复"模型初始化 + 训练 + 早停 + 测试集评估"，
最后汇总各 seed 的 IC/RankIC/ICIR/Precision@K/Sharpe 的均值与标准差。

背景：单次跑出 RankIC=0.097 之后，需要回答"这是稳定的信号，还是这次随机种子
恰好运气好"——如果多个 seed 的 RankIC 方差很小、且都是正的，说明信号可信；
如果方差很大甚至符号不稳定，说明还不能过度解读单次结果。

用法:
  python scripts/research/train_stockmixer_multiseed.py
  python scripts/research/train_stockmixer_multiseed.py --seeds 42,43,44,45,46
  python scripts/research/train_stockmixer_multiseed.py --n-seeds 5 --seed-start 100
  python scripts/research/train_stockmixer_multiseed.py --max-stocks 60 --epochs 30   # 调试用小样本
  python scripts/research/train_stockmixer_multiseed.py --train-years 5 --min-listed-days 120
  python scripts/research/train_stockmixer_multiseed.py --train-years 5 \\
      --freeze-codes-from-ckpt runs/experiments/stockmixer_multiseed_best.pt
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

import torch

from asharerank.forecasting.experimental.stockmixer.config import StockMixerConfig
from asharerank.forecasting.experimental.stockmixer.train import run_stockmixer_multiseed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _load_freeze_codes(ckpt_path: str) -> tuple[str, ...]:
    path = Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(f"冻结股票池 checkpoint 不存在: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    codes = ckpt.get("codes")
    if not codes:
        raise ValueError(f"checkpoint 中没有 codes 字段: {path}")
    return tuple(str(c) for c in codes)


def main():
    p = argparse.ArgumentParser(description="StockMixer (AAAI-24) 沪深300 多 seed 复现")
    p.add_argument("--pool", default="csi300", choices=["csi300", "csi500", "csi800"])
    p.add_argument("--max-stocks", type=int, default=None, help="调试用：限制股票数量")
    p.add_argument("--train-years", type=float, default=3.0,
                  help="训练历史长度（年）。val/test 仍固定为近 6+6 月，只有 train_start 往前推；"
                       "例如 --train-years 5 可与默认 3 年公平对比同一段验证/测试")
    p.add_argument("--min-listed-days", type=int, default=250,
                  help="训练起点前最少历史交易日（默认250≈1年）；降到120可放宽新股过滤")
    p.add_argument("--freeze-codes-from-ckpt", type=str, default=None,
                  help="从已有 .pt 读取 codes 并冻结股票池（跳过上市年限过滤，仍做覆盖率过滤）；"
                       "用于「固定 276 只、只改 train_years」控变量消融")
    p.add_argument("--lookback", type=int, default=16, help="滑动窗口长度 T")
    p.add_argument("--label-horizon", type=int, default=5,
                  help="预测目标前瞻天数（默认5，与当前最佳结果对齐；1日噪声过大不推荐）")
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
                  help="配合 --use-fundamentals：逗号分隔字段，覆盖默认 7 维。"
                       "可选 roe,net_margin,debt_ratio,rev_growth,profit_growth,ep_ratio,bp_ratio。"
                       "例：--fund-fields ep_ratio,bp_ratio 或 --fund-fields roe")
    p.add_argument("--no-industry-neutralize", action="store_true",
                  help="配合 --use-fundamentals：关闭基本面因子的行业内截面标准化（默认开启）")
    p.add_argument("--fund-missing-mask", action="store_true",
                  help="配合 --use-fundamentals：附加 fund_missing 缺测掩码维（默认关闭；"
                       "2026-07-25 在覆盖率≈70% 时实测伤 RankIC，等覆盖 90%+ 再开）")
    p.add_argument("--use-ema", action="store_true",
                  help="启用权重 EMA（修复版：warmup 衰减 + 验证集直接评 EMA 权重）。"
                       "默认关闭=基线行为，单独开启做消融")
    p.add_argument("--ema-decay", type=float, default=0.9, help="EMA 衰减上限（配合 --use-ema）")
    p.add_argument("--rankic-smooth-window", type=int, default=1,
                  help="早停判据用最近 N 轮 val_RankIC 滑动平均；1=裸单轮值（基线行为）")
    p.add_argument("--market-layerscale-init", type=float, default=1.0,
                  help="MarketAwareStockMixing 残差门控初始值；1.0=与原始结构等价（基线行为），"
                       "调小（如 0.1）单独消融")
    p.add_argument("--activation", default="hardswish", choices=["relu", "hardswish"])
    p.add_argument("--top-k", type=int, default=10, help="Precision@K / Sharpe 组合的 K")
    p.add_argument("--device", default=None, choices=[None, "mps", "cuda", "cpu"],
                  help="留空自动检测（优先 mps）")
    p.add_argument("--seeds", type=str, default=None,
                  help="逗号分隔的随机种子列表，例如 '42,43,44,45,46'；不传则用 --n-seeds/--seed-start 生成")
    p.add_argument("--n-seeds", type=int, default=5, help="未指定 --seeds 时，生成的随机种子个数")
    p.add_argument("--seed-start", type=int, default=42, help="未指定 --seeds 时，随机种子的起始值（依次+1）")
    args = p.parse_args()

    if args.seeds:
        seeds = tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip())
    else:
        seeds = tuple(args.seed_start + i for i in range(args.n_seeds))

    freeze_codes = None
    if args.freeze_codes_from_ckpt:
        freeze_codes = _load_freeze_codes(args.freeze_codes_from_ckpt)
        logging.info("已从 %s 冻结股票池: %d 只", args.freeze_codes_from_ckpt, len(freeze_codes))

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
    )
    run_stockmixer_multiseed(cfg, seeds=seeds)


if __name__ == "__main__":
    main()
