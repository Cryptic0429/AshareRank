"""在每个模型、每个交易日内部独立生成股票排名。"""
from __future__ import annotations

import pandas as pd

from asharerank.prediction.contract import validate_predictions


def rank_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "ts_code", "model_name", "raw_score"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"排名输入缺少字段: {sorted(missing)}")
    result = frame.copy()
    groups = result.groupby(["model_name", "date"])["raw_score"]
    result["rank_pct"] = groups.rank(pct=True, method="average")
    result["rank_score"] = 2.0 * result["rank_pct"] - 1.0
    result = result.sort_values(
        ["model_name", "date", "raw_score", "ts_code"],
        ascending=[True, True, False, True],
    )
    result["rank"] = result.groupby(["model_name", "date"]).cumcount() + 1
    result = result.reset_index(drop=True)
    validate_predictions(result)
    return result
