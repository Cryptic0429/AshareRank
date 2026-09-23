"""所有股票预测模型遵守的最小接口。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import numpy as np


class BaseStockModel(ABC):
    name: str

    @abstractmethod
    def fit(
        self,
        X: Any,
        y: np.ndarray,
        X_valid: Any | None = None,
        y_valid: np.ndarray | None = None,
    ) -> "BaseStockModel":
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: Any) -> np.ndarray:
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)

    @classmethod
    def load(cls, path: str | Path) -> "BaseStockModel":
        model = joblib.load(Path(path))
        if not isinstance(model, BaseStockModel):
            raise TypeError("模型文件不符合 BaseStockModel 协议")
        return model
