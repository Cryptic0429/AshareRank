"""Benchmark 子步骤 2：训练一个"训练截止日期严格早于测试窗口"的 LightGBM 对照模型。

**必须在独立进程里运行**，本模块不 import torch / StockMixer 任何代码
（原因见 `benchmark_stockmixer_step.py` 顶部注释：PyTorch + LightGBM 同进程
在 macOS 上会因 OpenMP 运行时冲突 segfault）。

与生产 `asharerank/forecasting/train.py::run_training` 用的是同一套
`build_train_frame` + `train_booster`，唯一区别：训练 + 早停验证集都严格
早于测试窗口起点（生产 `lgbm_v3` 终训时早停用的是最新一段数据，与"最近半年"
这类测试窗口有重叠，不能直接拿来做 OOS 对比）。不注册进
`runs/models/registry.json`，不影响生产模型。
"""
from __future__ import annotations

import logging

import pandas as pd

from asharerank.settings import PREDICT_HORIZON
from asharerank.forecasting.booster import train_booster
from asharerank.forecasting.dataset import build_train_frame
from asharerank.data.splits import purge_tail_dates
from asharerank.forecasting.evaluate import calc_ic_series, summarize_ic, topk_backtest
from asharerank.forecasting.registry import save_model
from asharerank.forecasting.universe import load_panel

logger = logging.getLogger(__name__)


def run_lgbm_step(codes: list[str], test_start: str, test_end: str,
                  horizon: int = PREDICT_HORIZON, valid_months: int = 3, *,
                  train_start: str | None = None, train_end: str | None = None,
                  val_start: str | None = None, val_end: str | None = None) -> pd.DataFrame:
    """训练对照 LightGBM 并在 (test_start, test_end] 窗口打分，返回
    `(date, ts_code, pred, fwd_ret)` 长表。

    两种切分模式：
      - **对齐模式**（传了 `train_start`/`val_start`）：train/val 边界直接复用
        StockMixer 的切分结果（同一段训练历史、同一个早停验证窗口），是目前
        推荐的默认对齐口径，由 `scripts/research/run_benchmark.py` 使用。
      - **宽松模式**（不传，仅给 `valid_months`）：只对齐了测试窗口起点，训练集
        用"测试窗口起点往前 valid_months 个月"之前的全部历史当早停验证集、
        更早的全部历史当训练集——保留给需要更长训练历史、不严格对齐 StockMixer
        训练窗口的场景（如单独调用本函数做快速对照）。
    """
    test_start = pd.Timestamp(test_start)
    test_end = pd.Timestamp(test_end)

    panel = load_panel(codes=codes)
    df = build_train_frame(panel, horizon=horizon)
    feature_cols = df.attrs["feature_cols"]

    if train_start is not None and val_start is not None:
        train_start_ts, train_end_ts = pd.Timestamp(train_start), pd.Timestamp(train_end)
        val_start_ts, val_end_ts = pd.Timestamp(val_start), pd.Timestamp(val_end)
        train_df = df[(df["date"] > train_start_ts) & (df["date"] <= train_end_ts)]
        valid_df = df[(df["date"] > val_start_ts) & (df["date"] <= val_end_ts)]
        logger.info("[benchmark/LightGBM] 对齐模式：train (%s, %s]，val (%s, %s]（与 StockMixer 完全一致）",
                    train_start_ts.date(), train_end_ts.date(), val_start_ts.date(), val_end_ts.date())
    else:
        valid_start = test_start - pd.DateOffset(months=valid_months)
        train_df = df[df["date"] < valid_start]
        valid_df = df[(df["date"] >= valid_start) & (df["date"] < test_start)]
        logger.info("[benchmark/LightGBM] 宽松模式：训练截止=%s（严格早于测试窗口）", valid_start.date())

    train_df, purged_train = purge_tail_dates(train_df, horizon)
    valid_df, purged_valid = purge_tail_dates(valid_df, horizon)
    logger.info("[benchmark/LightGBM] purge=%d dates (train=%d, valid=%d)",
                horizon, len(purged_train), len(purged_valid))

    if len(train_df) < 1000 or len(valid_df) < 200:
        raise ValueError(
            f"benchmark LightGBM 训练/早停样本不足（train={len(train_df)}, valid={len(valid_df)}），"
            f"请检查测试窗口起点是否过早，或历史数据是否够长"
        )
    booster = train_booster(
        train_df[feature_cols].to_numpy(), train_df["label"].to_numpy(),
        valid_df[feature_cols].to_numpy(), valid_df["label"].to_numpy(), feature_cols,
    )
    logger.info("[benchmark/LightGBM] 训练完成：train=%d 条，早停验证=%d 条", len(train_df), len(valid_df))

    test_df = df[(df["date"] > test_start) & (df["date"] <= test_end)].copy()
    if test_df.empty:
        raise ValueError("LightGBM 测试窗口内没有样本，请检查日期范围/股票池是否与 StockMixer 对齐")
    test_df["pred"] = booster.predict(test_df[feature_cols].to_numpy())
    result = test_df[["date", "ts_code", "pred", "fwd_ret"]].reset_index(drop=True)
    result.attrs["split_metadata"] = {
        "horizon": horizon,
        "alignment": "stockmixer" if train_start is not None and val_start is not None else "loose",
        "purged_train_dates": [str(value.date()) for value in purged_train],
        "purged_val_dates": [str(value.date()) for value in purged_valid],
        "n_train": len(train_df),
        "n_validation": len(valid_df),
    }
    return result


def train_aligned_candidate(codes: list[str], *,
                            train_start: str, train_end: str,
                            val_start: str, val_end: str,
                            test_start: str, test_end: str,
                            horizon: int = PREDICT_HORIZON,
                            model_id: str = "lgbm_v4",
                            top_k: int = 30,
                            status: str = "候选") -> dict:
    """训练一个"训练数据（股票池 + train/val/test 时间边界）与 StockMixer 完全对齐"
    的 LightGBM 候选模型，并注册进 registry（默认 `status="候选"`，不影响当前
    已部署的生产模型，`get_active_model_id()` 不会选中它）。

    与生产 `run_training`（walk-forward 滚动 + 全样本终训）的关键区别：
    这里是固定单次切分，train/val/test 三段边界直接复用 StockMixer 的
    `date_bounds`（`benchmark_stockmixer_step.get_universe_and_bounds` 的输出），
    确保两个模型看到的是同一批股票、同一段训练历史、同一个早停验证窗口、
    同一个最终测试窗口——一次比 `benchmark_lgbm_step.run_lgbm_step`（只对齐了
    测试窗口）更严格的对齐。
    """
    train_start, train_end = pd.Timestamp(train_start), pd.Timestamp(train_end)
    val_start, val_end = pd.Timestamp(val_start), pd.Timestamp(val_end)
    test_start, test_end = pd.Timestamp(test_start), pd.Timestamp(test_end)

    panel = load_panel(codes=codes)
    df = build_train_frame(panel, horizon=horizon)
    feature_cols = df.attrs["feature_cols"]

    train_df = df[(df["date"] > train_start) & (df["date"] <= train_end)]
    valid_df = df[(df["date"] > val_start) & (df["date"] <= val_end)]
    test_df = df[(df["date"] > test_start) & (df["date"] <= test_end)].copy()
    train_df, purged_train = purge_tail_dates(train_df, horizon)
    valid_df, purged_valid = purge_tail_dates(valid_df, horizon)
    logger.info("[%s] purge=%d dates (train=%d, valid=%d)",
                model_id, horizon, len(purged_train), len(purged_valid))
    if len(train_df) < 1000 or len(valid_df) < 200 or test_df.empty:
        raise ValueError(
            f"对齐后样本不足（train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}），"
            f"请检查股票池/日期边界是否正确"
        )

    booster = train_booster(
        train_df[feature_cols].to_numpy(), train_df["label"].to_numpy(),
        valid_df[feature_cols].to_numpy(), valid_df["label"].to_numpy(), feature_cols,
    )
    logger.info("[%s] 训练完成：train=%d 条 (%s~%s)，早停验证=%d 条 (%s~%s)",
                model_id, len(train_df), train_start.date(), train_end.date(),
                len(valid_df), val_start.date(), val_end.date())

    test_df["pred"] = booster.predict(test_df[feature_cols].to_numpy())
    ic_summary = summarize_ic(calc_ic_series(test_df, label_col="fwd_ret"))
    bt = topk_backtest(test_df, top_k=top_k, hold_days=horizon)
    metrics = {
        **ic_summary,
        "annual_return": bt["annual_return"],
        "sharpe": bt["sharpe"],
        "max_drawdown": bt["max_drawdown"],
    }
    meta = save_model(
        model_id, booster, feature_cols, metrics,
        extra={
            "n_stocks": len(codes),
            "n_samples": len(train_df) + len(valid_df),
            "horizon": horizon,
            "train_window": [str(train_start.date()), str(train_end.date())],
            "valid_window": [str(val_start.date()), str(val_end.date())],
            "test_window": [str(test_start.date()), str(test_end.date())],
            "aligned_with": "stockmixer",
        },
        status=status,
    )
    logger.info("[%s] 测试窗口 OOS：RankIC=%.4f ICIR=%.3f Sharpe=%.3f 年化收益=%.2f%%",
                model_id, metrics["rank_ic_mean"], metrics["icir"],
                metrics["sharpe"], metrics["annual_return"] * 100)
    return {"model_id": model_id, "metrics": metrics, "meta": meta}
