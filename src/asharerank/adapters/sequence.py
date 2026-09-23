"""从统一长表构造严格向后看的股票时序样本。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SequenceSamples:
    values: np.ndarray
    target: np.ndarray
    metadata: pd.DataFrame


def make_sequence_samples(
    frame: pd.DataFrame,
    feature_columns: list[str] | tuple[str, ...],
    lookback: int,
) -> SequenceSamples:
    """为每只股票生成以当前预测日结尾的长度为 lookback 的窗口。"""
    if lookback < 1:
        raise ValueError("lookback 必须为正整数")
    required = {"date", "ts_code", "label", *feature_columns}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"构造序列缺少字段: {sorted(missing)}")

    arrays: list[np.ndarray] = []
    targets: list[float] = []
    metadata: list[dict] = []
    extra_columns = [
        column for column in ("future_return", "future_excess_return", "volatility_20")
        if column in frame.columns
    ]
    for ticker, group in frame.groupby("ts_code", sort=False):
        ordered = group.sort_values("date").reset_index(drop=True)
        values = ordered[list(feature_columns)].to_numpy(dtype=float)
        for end in range(lookback - 1, len(ordered)):
            target = ordered.loc[end, "label"]
            if pd.isna(target):
                continue
            arrays.append(values[end - lookback + 1:end + 1])
            targets.append(float(target))
            row = {
                "date": pd.Timestamp(ordered.loc[end, "date"]),
                "ts_code": ticker,
            }
            for column in extra_columns:
                row[column] = float(ordered.loc[end, column])
            metadata.append(row)
    if not arrays:
        raise ValueError("没有足够数据构造时序样本")
    return SequenceSamples(
        values=np.stack(arrays).astype(np.float32, copy=False),
        target=np.asarray(targets, dtype=np.float32),
        metadata=pd.DataFrame(metadata),
    )
