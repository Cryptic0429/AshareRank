"""Random Forest 非线性 Bagging 基线。"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from asharerank.models.base import BaseStockModel


class RandomForestModel(BaseStockModel):
    name = "random_forest"

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int | None = 12,
        min_samples_leaf: int = 10,
        max_features: str | float = "sqrt",
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        self.estimator = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def fit(self, X, y, X_valid=None, y_valid=None) -> "RandomForestModel":
        self.estimator.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return np.asarray(self.estimator.predict(X), dtype=float)
