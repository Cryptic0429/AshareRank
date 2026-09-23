"""市场组合与 beta 估计：以沪深300 为市场基准，Vasicek/Blume 收缩。"""
from __future__ import annotations

import pandas as pd

BLUME_SHRINK = (0.67, 0.33)  # beta_adj = 0.67*beta_raw + 0.33*1


def align_market(returns: pd.DataFrame, market: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """按交易日对齐个股收益与指数收益。"""
    common = returns.index.intersection(market.index)
    return returns.loc[common], market.loc[common]


def estimate_betas(returns: pd.DataFrame, market: pd.Series,
                   shrink: bool = True) -> pd.Series:
    """个股对市场指数的回归 beta。shrink=True 时做 Blume 收缩（向 1 收缩），
    缓解样本 beta 的估计噪声。"""
    rets, mkt = align_market(returns, market)
    var_m = mkt.var()
    beta_raw = rets.apply(lambda s: s.cov(mkt) / var_m)
    if shrink:
        a, b = BLUME_SHRINK
        return a * beta_raw + b
    return beta_raw
