"""构建五模型共享的标准样本表。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from asharerank.config import ExperimentConfig, MARKET_FEATURES, STOCK_FEATURES
from asharerank.data.features import build_common_features
from asharerank.data.labels import make_forward_labels
from asharerank.data.splits import DateSplit, chronological_split
from asharerank.data.validation import validate_common_frame


@dataclass(frozen=True)
class CommonDataset:
    frame: pd.DataFrame
    split: DateSplit
    stock_features: tuple[str, ...] = STOCK_FEATURES
    market_features: tuple[str, ...] = MARKET_FEATURES

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return (*self.stock_features, *self.market_features)

    def partition(self, name: str) -> pd.DataFrame:
        if name not in {"train", "valid", "test"}:
            raise ValueError("name 必须是 train、valid 或 test")
        return self.frame.loc[self.split.mask(self.frame["date"], name)].copy()


def build_common_dataset(
    stock_panel: pd.DataFrame,
    market: pd.DataFrame | None = None,
    config: ExperimentConfig | None = None,
) -> CommonDataset:
    """从 OHLCV 长表生成统一特征、标签和时间切分。"""
    config = config or ExperimentConfig()
    features = build_common_features(stock_panel, market=market)
    labels = make_forward_labels(
        stock_panel,
        horizon=config.horizon,
        target=config.target,
    )
    frame = features.merge(labels, on=["date", "ts_code"], how="inner", validate="one_to_one")
    feature_columns = [*STOCK_FEATURES, *MARKET_FEATURES]
    frame[feature_columns] = frame[feature_columns].replace([np.inf, -np.inf], np.nan)
    # 至少保留标签有效的样本；特征缺失由 Adapter 仅基于训练集统计量处理。
    frame = frame.dropna(subset=["label", "future_return", "future_excess_return"])
    valid_counts = frame.groupby("date")["ts_code"].transform("nunique")
    frame = frame.loc[valid_counts >= config.min_cross_section]
    frame = frame.sort_values(["date", "ts_code"]).reset_index(drop=True)
    validate_common_frame(frame, feature_columns=feature_columns)
    split = chronological_split(
        frame["date"],
        valid_fraction=config.valid_fraction,
        test_fraction=config.test_fraction,
        purge_days=config.horizon,
    )
    return CommonDataset(frame=frame, split=split)
