"""Benchmark 子步骤：训练一个"训练/验证/测试边界与 StockMixer 对齐"的 XGBoost 对照模型。

与 `benchmark_lgbm_step.py` 是同构实现（同一套 `build_train_frame` + 对齐边界），
唯一区别是换成 XGBoost 后端；复用 `experimental/xgboost_baseline.py::_fit_one_fold`
的超参数，确保和树模型 walk-forward 基线实验（`xgboost_report.json`）用的是
同一组超参数，只是数据切分方式不同（这里固定单次切分对齐 StockMixer，baseline
是多年滚动窗口 walk-forward）。

不注册进 `runs/models/registry.json`，不影响生产模型。可以和 LightGBM
共用同一个子进程（两者都依赖 libomp，互相不冲突，`train_ensemble.py` 曾经在
同一进程里跑过两者），但 `run_benchmark.py` 仍让每个模型各起一个独立子进程，
保持"一个模型一个子进程"的一致模式，方便日后单独扩展/调试（多一次 Python
解释器启动开销，可忽略）。
"""
from __future__ import annotations

import logging

import pandas as pd

from asharerank.settings import PREDICT_HORIZON
from asharerank.data.splits import purge_tail_dates
from asharerank.predict.dataset import build_train_frame
from asharerank.predict.experimental.xgboost_baseline import DEFAULT_PARAMS, _fit_one_fold
from asharerank.predict.universe import load_panel

logger = logging.getLogger(__name__)


def run_xgboost_step(codes: list[str], test_start: str, test_end: str,
                     horizon: int = PREDICT_HORIZON, valid_months: int = 3, *,
                     train_start: str | None = None, train_end: str | None = None,
                     val_start: str | None = None, val_end: str | None = None) -> pd.DataFrame:
    """训练对照 XGBoost 并在 (test_start, test_end] 窗口打分，返回
    `(date, ts_code, pred, fwd_ret)` 长表。切分模式说明见 `benchmark_lgbm_step.run_lgbm_step`。
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
        logger.info("[benchmark/XGBoost] 对齐模式：train (%s, %s]，val (%s, %s]（与 StockMixer 完全一致）",
                    train_start_ts.date(), train_end_ts.date(), val_start_ts.date(), val_end_ts.date())
    else:
        valid_start = test_start - pd.DateOffset(months=valid_months)
        train_df = df[df["date"] < valid_start]
        valid_df = df[(df["date"] >= valid_start) & (df["date"] < test_start)]
        logger.info("[benchmark/XGBoost] 宽松模式：训练截止=%s（严格早于测试窗口）", valid_start.date())

    train_df, purged_train = purge_tail_dates(train_df, horizon)
    valid_df, purged_valid = purge_tail_dates(valid_df, horizon)
    logger.info("[benchmark/XGBoost] purge=%d dates (train=%d, valid=%d)",
                horizon, len(purged_train), len(purged_valid))

    if len(train_df) < 1000 or len(valid_df) < 200:
        raise ValueError(
            f"benchmark XGBoost 训练/早停样本不足（train={len(train_df)}, valid={len(valid_df)}），"
            f"请检查测试窗口起点是否过早，或历史数据是否够长"
        )
    model = _fit_one_fold(
        train_df[feature_cols].to_numpy(), train_df["label"].to_numpy(),
        valid_df[feature_cols].to_numpy(), valid_df["label"].to_numpy(), DEFAULT_PARAMS,
    )
    logger.info("[benchmark/XGBoost] 训练完成：train=%d 条，早停验证=%d 条", len(train_df), len(valid_df))

    test_df = df[(df["date"] > test_start) & (df["date"] <= test_end)].copy()
    if test_df.empty:
        raise ValueError("XGBoost 测试窗口内没有样本，请检查日期范围/股票池是否与 StockMixer 对齐")
    test_df["pred"] = model.predict(test_df[feature_cols].to_numpy())
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
