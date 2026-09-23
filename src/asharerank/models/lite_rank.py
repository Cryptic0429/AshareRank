"""LiteRank：面向股票横截面排序的轻量级 LightGBM Ranker。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from asharerank.models.base import BaseStockModel


def _unpack_grouped_input(X) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(X, dict) or not {"values", "dates"}.issubset(X):
        raise ValueError("LiteRank 输入必须包含 values 和 dates")
    values = np.asarray(X["values"], dtype=np.float32)
    dates = pd.to_datetime(np.asarray(X["dates"]))
    if len(values) != len(dates):
        raise ValueError("LiteRank 特征与交易日数量不一致")
    return values, dates.to_numpy()


def _integer_relevance(y: np.ndarray, levels: int = 16) -> np.ndarray:
    """把统一的 [-1, 1] 横截面标签映射为 LambdaRank 整数相关度。"""
    values = np.asarray(y, dtype=float)
    scaled = np.rint((np.clip(values, -1.0, 1.0) + 1.0) * (levels - 1) / 2.0)
    return scaled.astype(np.int32)


def _date_groups(dates: np.ndarray) -> np.ndarray:
    return (
        pd.Series(pd.to_datetime(dates))
        .groupby(pd.to_datetime(dates), sort=False)
        .size()
        .to_numpy(dtype=np.int32)
    )


def _mean_daily_rank_ic(dates: np.ndarray, scores: np.ndarray, y: np.ndarray) -> float:
    frame = pd.DataFrame({"date": pd.to_datetime(dates), "score": scores, "label": y})
    correlations = frame.groupby("date", sort=False).apply(
        lambda group: group["score"].corr(group["label"], method="spearman"),
        include_groups=False,
    )
    finite = correlations.replace([np.inf, -np.inf], np.nan).dropna()
    return float(finite.mean()) if not finite.empty else 0.0


class LiteRankModel(BaseStockModel):
    """以交易日为 query group，直接优化股票头部排序。"""

    name = "lite_rank"

    def __init__(
        self,
        learning_rate: float = 0.035,
        n_estimators: int = 500,
        num_leaves: int = 15,
        max_depth: int = 4,
        min_child_samples: int = 45,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        self.params = {
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "min_child_samples": min_child_samples,
            "random_state": random_state,
            "n_jobs": n_jobs,
        }
        self.estimator = None
        self.orientation = 1.0
        self.validation_rank_ic = 0.0
        self.best_iteration = 0

    def fit(self, X, y, X_valid=None, y_valid=None) -> "LiteRankModel":
        try:
            import lightgbm as lgb
        except (ImportError, OSError) as exc:
            raise RuntimeError("LiteRank 需要安装 LightGBM") from exc

        values, dates = _unpack_grouped_input(X)
        order = np.argsort(dates, kind="stable")
        train_values = values[order]
        train_dates = dates[order]
        train_y = _integer_relevance(np.asarray(y)[order])

        self.estimator = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            label_gain=list(range(16)),
            verbosity=-1,
            subsample=0.82,
            subsample_freq=1,
            colsample_bytree=0.78,
            reg_alpha=1.0,
            reg_lambda=8.0,
            max_bin=127,
            **self.params,
        )
        fit_kwargs = {
            "group": _date_groups(train_dates),
            "eval_at": (5, 10),
            "callbacks": [lgb.log_evaluation(period=0)],
        }

        valid_values = valid_dates = valid_labels = None
        if X_valid is not None and y_valid is not None and len(y_valid):
            values_v, dates_v = _unpack_grouped_input(X_valid)
            valid_order = np.argsort(dates_v, kind="stable")
            valid_values = values_v[valid_order]
            valid_dates = dates_v[valid_order]
            valid_labels = np.asarray(y_valid)[valid_order]
            fit_kwargs.update({
                "eval_X": valid_values,
                "eval_y": _integer_relevance(valid_labels),
                "eval_group": [_date_groups(valid_dates)],
                "callbacks": [
                    lgb.early_stopping(60, first_metric_only=True, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            })

        self.estimator.fit(train_values, train_y, **fit_kwargs)
        self.best_iteration = int(self.estimator.best_iteration_ or self.params["n_estimators"])

        # 方向只在验证集校准一次，测试集不会参与模型或符号选择。
        if valid_values is not None and valid_dates is not None and valid_labels is not None:
            valid_scores = np.asarray(self.estimator.predict(valid_values), dtype=float)
            correlation = _mean_daily_rank_ic(valid_dates, valid_scores, valid_labels)
            self.orientation = 1.0 if correlation >= 0 else -1.0
            self.validation_rank_ic = abs(correlation)
        return self

    def predict(self, X) -> np.ndarray:
        if self.estimator is None:
            raise RuntimeError("LiteRankModel 尚未训练")
        values, _ = _unpack_grouped_input(X)
        scores = self.estimator.predict(values, num_iteration=self.best_iteration or None)
        return self.orientation * np.asarray(scores, dtype=float)
