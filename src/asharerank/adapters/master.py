"""MASTER-Lite 的股票序列、市场序列 Adapter。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from asharerank.adapters.base import AdaptedData, BaseAdapter
from asharerank.adapters.sequence import make_sequence_samples
from asharerank.config import MARKET_FEATURES, STOCK_FEATURES


class MasterAdapter(BaseAdapter):
    def __init__(self, lookback: int = 8) -> None:
        self.lookback = lookback
        self.stock_features = tuple(STOCK_FEATURES)
        self.market_features = tuple(MARKET_FEATURES)
        self.feature_columns = (*self.stock_features, *self.market_features)
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        self.scaler = StandardScaler()
        self._fitted = False

    def _raw(self, frame: pd.DataFrame):
        return make_sequence_samples(frame, self.feature_columns, self.lookback)

    @staticmethod
    def _flatten(values: np.ndarray) -> np.ndarray:
        return values.reshape(-1, values.shape[-1])

    def fit(self, frame: pd.DataFrame) -> "MasterAdapter":
        samples = self._raw(frame)
        flattened = self._flatten(samples.values)
        imputed = self.imputer.fit_transform(flattened)
        self.scaler.fit(imputed)
        self._fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> AdaptedData:
        if not self._fitted:
            raise RuntimeError("MasterAdapter 必须先在训练集上 fit")
        samples = self._raw(frame)
        shape = samples.values.shape
        flattened = self._flatten(samples.values)
        scaled = self.scaler.transform(self.imputer.transform(flattened)).reshape(shape)
        n_stock = len(self.stock_features)
        inputs = {
            "stock": scaled[:, :, :n_stock].astype(np.float32, copy=False),
            "market": scaled[:, :, n_stock:].astype(np.float32, copy=False),
            "dates": pd.to_datetime(samples.metadata["date"]).to_numpy(),
        }
        return AdaptedData(
            X=inputs,
            y=samples.target,
            metadata=samples.metadata,
            feature_names=self.feature_columns,
        )
