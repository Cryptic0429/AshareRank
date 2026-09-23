"""统一未来收益、超额收益和横截面排名标签。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def cross_sectional_rank_score(values: pd.Series) -> pd.Series:
    """把当日横截面百分位映射到 [-1, 1]。"""
    valid = values.notna()
    result = pd.Series(np.nan, index=values.index, dtype=float)
    if valid.any():
        result.loc[valid] = 2.0 * values.loc[valid].rank(pct=True, method="average") - 1.0
    return result


def make_forward_labels(
    stock_panel: pd.DataFrame,
    horizon: int = 5,
    target: str = "excess_return",
) -> pd.DataFrame:
    """按股票计算未来收益，并生成每天的横截面排名训练标签。

    当没有单独传入基准时，超额收益基准为当日股票池未来收益的等权平均值。
    该基准只用于构造标签，不会进入任何输入特征。
    """
    if horizon < 1:
        raise ValueError("horizon 必须为正整数")
    if target not in {"return", "excess_return"}:
        raise ValueError("target 只能是 return 或 excess_return")
    data = stock_panel[["date", "ts_code", "close"]].copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(["ts_code", "date"])
    data["future_return"] = data.groupby("ts_code")["close"].transform(
        lambda values: values.shift(-horizon) / values - 1.0
    )
    daily_market_return = data.groupby("date")["future_return"].transform("mean")
    data["future_excess_return"] = data["future_return"] - daily_market_return
    target_column = "future_excess_return" if target == "excess_return" else "future_return"
    data["label"] = data.groupby("date", group_keys=False)[target_column].transform(
        cross_sectional_rank_score
    )
    return data[[
        "date",
        "ts_code",
        "future_return",
        "future_excess_return",
        "label",
    ]].sort_values(["date", "ts_code"]).reset_index(drop=True)
