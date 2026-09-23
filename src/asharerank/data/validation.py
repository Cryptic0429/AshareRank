"""统一数据集的结构、时间与数值检查。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def validate_common_frame(frame: pd.DataFrame, feature_columns: list[str]) -> None:
    required = {
        "date",
        "ts_code",
        "future_return",
        "future_excess_return",
        "label",
        *feature_columns,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"统一数据集缺少字段: {sorted(missing)}")
    if frame.empty:
        raise ValueError("统一数据集为空")
    if frame.duplicated(["date", "ts_code"]).any():
        raise ValueError("统一数据集存在重复的 date + ts_code")
    if not frame["label"].dropna().between(-1.0, 1.0).all():
        raise ValueError("label 必须位于 [-1, 1]")
    values = frame[feature_columns].to_numpy(dtype=float, copy=True)
    if np.isinf(values).any():
        raise ValueError("统一特征中仍存在正负无穷")


def assert_disjoint_dates(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame) -> None:
    sets = [set(pd.to_datetime(part["date"]).unique()) for part in (train, valid, test)]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError("训练、验证和测试集合包含重叠日期")
    if max(sets[0]) >= min(sets[1]) or max(sets[1]) >= min(sets[2]):
        raise ValueError("训练、验证和测试集合没有严格按时间排序")
