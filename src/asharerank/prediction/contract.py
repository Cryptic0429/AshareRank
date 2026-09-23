"""所有模型共同遵守的预测结果协议。"""
from __future__ import annotations

import numpy as np
import pandas as pd

PREDICTION_COLUMNS = (
    "date",
    "ts_code",
    "model_name",
    "raw_score",
    "rank_pct",
    "rank_score",
    "rank",
)


def build_prediction_frame(
    metadata: pd.DataFrame,
    scores,
    model_name: str,
) -> pd.DataFrame:
    if not {"date", "ts_code"}.issubset(metadata.columns):
        raise ValueError("metadata 必须包含 date 和 ts_code")
    values = np.asarray(scores, dtype=float).reshape(-1)
    if len(values) != len(metadata):
        raise ValueError("预测分数与 metadata 样本数不一致")
    result = metadata.reset_index(drop=True).copy()
    result["date"] = pd.to_datetime(result["date"])
    result["model_name"] = model_name
    result["raw_score"] = values
    return result


def validate_predictions(frame: pd.DataFrame) -> None:
    missing = set(PREDICTION_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"预测结果缺少字段: {sorted(missing)}")
    if frame.empty:
        raise ValueError("预测结果为空")
    if frame.duplicated(["date", "ts_code", "model_name"]).any():
        raise ValueError("预测结果存在重复主键")
    if not np.isfinite(frame["raw_score"].to_numpy(dtype=float)).all():
        raise ValueError("raw_score 包含非有限值")
    if not frame["rank_pct"].between(0.0, 1.0).all():
        raise ValueError("rank_pct 必须位于 [0, 1]")
    if not frame["rank_score"].between(-1.0, 1.0).all():
        raise ValueError("rank_score 必须位于 [-1, 1]")
