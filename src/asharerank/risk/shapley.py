"""Shapley 值风险归因。

将组合波动率 σ(S)（子集 S 内资产按原权重构成的子组合波动率）作为特征函数，
对每个资产 i 计算 Shapley 值：
    φ_i = Σ_{S ⊆ N\\{i}} |S|!(n-|S|-1)!/n! · [σ(S∪{i}) - σ(S)]
满足有效性：Σ φ_i = σ(N)。

n ≤ 15 时精确枚举（2^15 = 32768 个子集可接受）；更大时用蒙特卡洛排列采样。
"""
from __future__ import annotations

from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd


def _subset_vol(cov: np.ndarray, weights: np.ndarray, idx: tuple[int, ...]) -> float:
    if not idx:
        return 0.0
    w = weights[list(idx)]
    sub = cov[np.ix_(list(idx), list(idx))]
    return float(np.sqrt(w @ sub @ w))


def shapley_exact(cov: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    n = len(weights)
    covv = cov.values
    fact = [factorial(k) for k in range(n + 1)]
    phi = np.zeros(n)
    others = list(range(n))
    for i in range(n):
        rest = [j for j in others if j != i]
        for size in range(n):
            coef = fact[size] * fact[n - size - 1] / fact[n]
            for S in combinations(rest, size):
                phi[i] += coef * (_subset_vol(covv, weights, S + (i,)) -
                                  _subset_vol(covv, weights, S))
    return pd.Series(phi, index=cov.index)


def shapley_monte_carlo(cov: pd.DataFrame, weights: np.ndarray,
                        n_perm: int = 2000, seed: int = 42) -> pd.Series:
    """排列采样近似：对随机排列依次加入资产，累积边际贡献。"""
    n = len(weights)
    covv = cov.values
    rng = np.random.default_rng(seed)
    phi = np.zeros(n)
    for _ in range(n_perm):
        perm = rng.permutation(n)
        prev = 0.0
        current: list[int] = []
        for i in perm:
            current.append(i)
            v = _subset_vol(covv, weights, tuple(current))
            phi[i] += v - prev
            prev = v
    return pd.Series(phi / n_perm, index=cov.index)


def shapley_risk_attribution(cov: pd.DataFrame, weights: np.ndarray) -> pd.DataFrame:
    """返回每只股票的 Shapley 风险贡献（日波动率口径）及占比。"""
    n = len(weights)
    phi = shapley_exact(cov, weights) if n <= 15 else shapley_monte_carlo(cov, weights)
    total = phi.sum()
    return pd.DataFrame({
        "shapley_vol": phi,
        "share": phi / total if total > 0 else phi,
    }).sort_values("shapley_vol", ascending=False)
