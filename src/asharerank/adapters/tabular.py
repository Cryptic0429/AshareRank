"""Ridge、Random Forest 和 LightGBM 的统一表格 Adapter。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from asharerank.adapters.base import AdaptedData, BaseAdapter
from asharerank.adapters.sequence import make_sequence_samples
from asharerank.config import MARKET_FEATURES, STOCK_FEATURES


class TabularAdapter(BaseAdapter):
    """将 8 日 × 26 特征窗口展开成一维向量。"""

    def __init__(self, lookback: int = 8, scale: bool = True) -> None:
        self.lookback = lookback
        self.scale = scale
        self.feature_columns = (*STOCK_FEATURES, *MARKET_FEATURES)
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        self.scaler = StandardScaler() if scale else None
        self._fitted = False

    @property
    def output_feature_names(self) -> tuple[str, ...]:
        return tuple(
            f"lag_{lag}_{feature}"
            for lag in range(self.lookback - 1, -1, -1)
            for feature in self.feature_columns
        )

    def _raw(self, frame: pd.DataFrame):
        samples = make_sequence_samples(frame, self.feature_columns, self.lookback)
        flat = samples.values.reshape(len(samples.values), -1)
        return samples, flat

    def fit(self, frame: pd.DataFrame) -> "TabularAdapter":
        _, flat = self._raw(frame)
        transformed = self.imputer.fit_transform(flat)
        if self.scaler is not None:
            self.scaler.fit(transformed)
        self._fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> AdaptedData:
        if not self._fitted:
            raise RuntimeError("TabularAdapter 必须先在训练集上 fit")
        samples, flat = self._raw(frame)
        values = self.imputer.transform(flat)
        if self.scaler is not None:
            values = self.scaler.transform(values)
        return AdaptedData(
            X=np.asarray(values, dtype=np.float32),
            y=samples.target,
            metadata=samples.metadata,
            feature_names=self.output_feature_names,
        )


class RankingTabularAdapter(TabularAdapter):
    """表格特征外附带交易日，供按日分组的学习排序模型使用。"""

    def __init__(self, lookback: int = 8) -> None:
        super().__init__(lookback=lookback, scale=False)

    def transform(self, frame: pd.DataFrame) -> AdaptedData:
        adapted = super().transform(frame)
        return AdaptedData(
            X={
                "values": adapted.X,
                "dates": pd.to_datetime(adapted.metadata["date"]).to_numpy(),
            },
            y=adapted.y,
            metadata=adapted.metadata,
            feature_names=adapted.feature_names,
        )
