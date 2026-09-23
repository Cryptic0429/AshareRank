"""IC、RankIC、稳定性和正向比例。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _correlation(left: pd.Series, right: pd.Series, method: str) -> float:
    valid = left.notna() & right.notna()
    left = left.loc[valid]
    right = right.loc[valid]
    if len(left) < 5 or left.nunique() < 2 or right.nunique() < 2:
        return np.nan
    if method == "spearman":
        return float(stats.spearmanr(left, right).statistic)
    return float(stats.pearsonr(left, right).statistic)


def calculate_daily_ic(
    predictions: pd.DataFrame,
    target_column: str = "future_excess_return",
) -> pd.DataFrame:
    required = {"date", "raw_score", target_column}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"IC 输入缺少字段: {sorted(missing)}")
    rows = []
    group_columns = ["date"]
    if "model_name" in predictions.columns:
        group_columns.insert(0, "model_name")
    for keys, group in predictions.groupby(group_columns, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row["ic"] = _correlation(group["raw_score"], group[target_column], "pearson")
        row["rank_ic"] = _correlation(group["raw_score"], group[target_column], "spearman")
        rows.append(row)
    return pd.DataFrame(rows).dropna(subset=["ic", "rank_ic"], how="all")


def _safe_ratio(mean: float, std: float) -> float:
    return float(mean / std) if np.isfinite(std) and std > 0 else 0.0


def summarise_ic(daily: pd.DataFrame) -> dict[str, float | int]:
    if daily.empty:
        return {
            "ic": 0.0,
            "ic_std": 0.0,
            "icir": 0.0,
            "ic_positive_ratio": 0.0,
            "rank_ic": 0.0,
            "rank_ic_std": 0.0,
            "rank_icir": 0.0,
            "rank_ic_positive_ratio": 0.0,
            "n_dates": 0,
        }
    ic = daily["ic"].dropna()
    rank_ic = daily["rank_ic"].dropna()
    ic_mean = float(ic.mean()) if len(ic) else 0.0
    rank_mean = float(rank_ic.mean()) if len(rank_ic) else 0.0
    ic_std = float(ic.std(ddof=0)) if len(ic) else 0.0
    rank_std = float(rank_ic.std(ddof=0)) if len(rank_ic) else 0.0
    return {
        "ic": ic_mean,
        "ic_std": ic_std,
        "icir": _safe_ratio(ic_mean, ic_std),
        "ic_positive_ratio": float((ic > 0).mean()) if len(ic) else 0.0,
        "rank_ic": rank_mean,
        "rank_ic_std": rank_std,
        "rank_icir": _safe_ratio(rank_mean, rank_std),
        "rank_ic_positive_ratio": float((rank_ic > 0).mean()) if len(rank_ic) else 0.0,
        "n_dates": int(daily["date"].nunique()),
    }
