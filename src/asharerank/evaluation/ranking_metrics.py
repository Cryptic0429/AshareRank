"""Top-K 命中、排序质量和头尾收益差指标。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score


def _daily_metrics(group: pd.DataFrame, target: str, k: int) -> dict[str, float]:
    size = len(group)
    effective_k = min(k, size)
    predicted_top = group.nlargest(effective_k, "raw_score")
    actual_top = group.nlargest(effective_k, target)
    predicted_bottom = group.nsmallest(effective_k, "raw_score")
    overlap = len(set(predicted_top["ts_code"]) & set(actual_top["ts_code"]))

    truth = group[target].rank(pct=True, method="average").to_numpy(dtype=float)
    scores = group["raw_score"].to_numpy(dtype=float)
    ndcg = float(ndcg_score(truth.reshape(1, -1), scores.reshape(1, -1), k=effective_k))
    top_return = float(predicted_top[target].mean())
    bottom_return = float(predicted_bottom[target].mean())
    return {
        "precision_at_k": overlap / effective_k,
        "ndcg_at_k": ndcg,
        "top_k_mean_return": top_return,
        "bottom_k_mean_return": bottom_return,
        "top_bottom_spread": top_return - bottom_return,
        "top_k_hit": float(top_return > 0),
    }


def _stability(predictions: pd.DataFrame, k: int) -> dict[str, float]:
    top_sets: list[set[str]] = []
    rank_maps: list[pd.Series] = []
    for _, group in predictions.sort_values("date").groupby("date", sort=True):
        top_sets.append(set(group.nlargest(min(k, len(group)), "raw_score")["ts_code"]))
        rank_maps.append(group.set_index("ts_code")["rank_pct"])
    overlaps: list[float] = []
    rank_changes: list[float] = []
    for previous, current in zip(top_sets, top_sets[1:]):
        union = previous | current
        overlaps.append(len(previous & current) / len(union) if union else 1.0)
    for previous, current in zip(rank_maps, rank_maps[1:]):
        common = previous.index.intersection(current.index)
        if len(common):
            rank_changes.append(float((current.loc[common] - previous.loc[common]).abs().mean()))
    return {
        "top_k_overlap": float(np.mean(overlaps)) if overlaps else 0.0,
        "ranking_turnover": float(np.mean(rank_changes)) if rank_changes else 0.0,
    }


def calculate_ranking_metrics(
    predictions: pd.DataFrame,
    target_column: str = "future_excess_return",
    k: int = 10,
) -> dict[str, float | int]:
    required = {"date", "ts_code", "raw_score", "rank_pct", target_column}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"排名评价输入缺少字段: {sorted(missing)}")
    if k < 1:
        raise ValueError("k 必须为正整数")
    daily = [
        _daily_metrics(group, target_column, k)
        for _, group in predictions.groupby("date", sort=True)
        if len(group) >= 2
    ]
    if not daily:
        return {
            "precision_at_k": 0.0,
            "ndcg_at_k": 0.0,
            "top_k_mean_return": 0.0,
            "bottom_k_mean_return": 0.0,
            "top_bottom_spread": 0.0,
            "top_k_hit_rate": 0.0,
            "top_k_overlap": 0.0,
            "ranking_turnover": 0.0,
        }
    frame = pd.DataFrame(daily)
    result = {
        "precision_at_k": float(frame["precision_at_k"].mean()),
        "ndcg_at_k": float(frame["ndcg_at_k"].mean()),
        "top_k_mean_return": float(frame["top_k_mean_return"].mean()),
        "bottom_k_mean_return": float(frame["bottom_k_mean_return"].mean()),
        "top_bottom_spread": float(frame["top_bottom_spread"].mean()),
        "top_k_hit_rate": float(frame["top_k_hit"].mean()),
    }
    result.update(_stability(predictions, k))
    return result
