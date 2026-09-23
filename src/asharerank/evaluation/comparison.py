"""汇总单模型指标并生成五模型横向对比表。"""
from __future__ import annotations

import pandas as pd

from asharerank.evaluation.ic_metrics import calculate_daily_ic, summarise_ic
from asharerank.evaluation.ranking_metrics import calculate_ranking_metrics


def evaluate_predictions(
    predictions: pd.DataFrame,
    target_column: str = "future_excess_return",
    k: int = 10,
) -> dict[str, float | int | str]:
    models = predictions["model_name"].dropna().unique()
    if len(models) != 1:
        raise ValueError("evaluate_predictions 每次只评价一个模型")
    daily = calculate_daily_ic(predictions, target_column=target_column)
    result: dict[str, float | int | str] = {"model_name": str(models[0])}
    result.update(summarise_ic(daily))
    result.update(calculate_ranking_metrics(predictions, target_column=target_column, k=k))
    return result


def compare_prediction_frames(
    frames: list[pd.DataFrame],
    target_column: str = "future_excess_return",
    k: int = 10,
) -> pd.DataFrame:
    """只比较模型，不组合或融合任何模型分数。"""
    rows = [evaluate_predictions(frame, target_column=target_column, k=k) for frame in frames]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["rank_icir", "rank_ic"], ascending=False
    ).reset_index(drop=True)
