"""Ridge 线性基线。"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from asharerank.models.base import BaseStockModel


class RidgeModel(BaseStockModel):
    name = "ridge"

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.estimator = Ridge(alpha=alpha)

    def fit(self, X, y, X_valid=None, y_valid=None) -> "RidgeModel":
        self.estimator.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return np.asarray(self.estimator.predict(X), dtype=float)
