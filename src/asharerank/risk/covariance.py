"""协方差矩阵估计：样本协方差 / EWMA / Ledoit-Wolf 收缩。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from asharerank.settings import EWMA_LAMBDA


def sample_cov(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.cov()


def ewma_cov(returns: pd.DataFrame, lam: float = EWMA_LAMBDA) -> pd.DataFrame:
    """RiskMetrics 指数加权协方差：Sigma_t = (1-λ) Σ λ^k r_{t-k} r_{t-k}'。"""
    x = returns.values - returns.values.mean(axis=0)
    n = len(x)
    weights = (1 - lam) * lam ** np.arange(n - 1, -1, -1)
    weights /= weights.sum()
    cov = np.einsum("t,ti,tj->ij", weights, x, x)
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)


def ledoit_wolf_cov(returns: pd.DataFrame) -> pd.DataFrame:
    lw = LedoitWolf().fit(returns.values)
    return pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)


ESTIMATORS = {
    "sample": sample_cov,
    "ewma": ewma_cov,
    "ledoit_wolf": ledoit_wolf_cov,
}


def estimate_cov(returns: pd.DataFrame, method: str = "ledoit_wolf") -> pd.DataFrame:
    if method not in ESTIMATORS:
        raise ValueError(f"未知协方差估计方法: {method}")
    return ESTIMATORS[method](returns)


def condition_number(cov: pd.DataFrame) -> float:
    """协方差矩阵条件数（最大/最小特征值比）。数值越大，矩阵越接近病态，
    权重/因子分解等下游计算对估计误差越敏感，是协方差估计可信度的诊断指标。"""
    return float(np.linalg.cond(cov.values))
