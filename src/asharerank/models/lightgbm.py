"""LightGBM 强表格基线，缺少 LightGBM 时使用 sklearn 回退。"""
from __future__ import annotations

import inspect

import numpy as np

from asharerank.models.base import BaseStockModel


class LightGBMModel(BaseStockModel):
    name = "lightgbm"

    def __init__(
        self,
        learning_rate: float = 0.05,
        n_estimators: int = 500,
        num_leaves: int = 31,
        max_depth: int = 6,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        self.params = {
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "random_state": random_state,
            "n_jobs": n_jobs,
        }
        self.backend = "uninitialised"
        self.estimator = None

    def fit(self, X, y, X_valid=None, y_valid=None) -> "LightGBMModel":
        try:
            import lightgbm as lgb

            self.backend = "lightgbm"
            self.estimator = lgb.LGBMRegressor(
                objective="regression",
                verbosity=-1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=1.0,
                reg_lambda=5.0,
                min_child_samples=30,
                **self.params,
            )
            fit_kwargs = {}
            if X_valid is not None and y_valid is not None and len(y_valid):
                callbacks = [
                    lgb.early_stopping(40, verbose=False),
                    lgb.log_evaluation(period=0),
                ]
                if "eval_X" in inspect.signature(self.estimator.fit).parameters:
                    fit_kwargs = {
                        "eval_X": X_valid,
                        "eval_y": y_valid,
                        "callbacks": callbacks,
                    }
                else:
                    fit_kwargs = {
                        "eval_set": [(X_valid, y_valid)],
                        "callbacks": callbacks,
                    }
            self.estimator.fit(X, y, **fit_kwargs)
        except (ImportError, OSError):
            from sklearn.ensemble import HistGradientBoostingRegressor

            self.backend = "sklearn_hgb"
            self.estimator = HistGradientBoostingRegressor(
                learning_rate=self.params["learning_rate"],
                max_iter=self.params["n_estimators"],
                max_depth=self.params["max_depth"],
                l2_regularization=5.0,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=self.params["random_state"],
            )
            self.estimator.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        if self.estimator is None:
            raise RuntimeError("LightGBMModel 尚未训练")
        return np.asarray(self.estimator.predict(X), dtype=float)
