"""压力测试。

一、假设情景：市场整体冲击，经个股收缩 beta 传导到组合。
二、历史情景：A 股历次极端行情重演。组合在情景期并不存在，规范做法为
   beta 映射法：组合损失 ≈ Σ w_i · beta_i · R_index(情景期)。
   对情景期内已上市的持仓，同时给出真实个股收益回放口径及其覆盖率。
三、组合自身历史最差区间（滚动窗口）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from asharerank.risk.market import align_market, estimate_betas

# 假设情景：市场指数收益冲击
HYPOTHETICAL_SCENARIOS = {
    "市场下跌 5%": -0.05,
    "市场下跌 10%": -0.10,
    "市场下跌 20%": -0.20,
    "市场上涨 5%": 0.05,
}

# 历史情景库：情景名 -> (起始日, 结束日)
SCENARIO_LIBRARY = {
    "2008 全球金融危机": ("2008-01-14", "2008-11-04"),
    "2015 股灾": ("2015-06-12", "2015-08-26"),
    "2016 熔断": ("2016-01-01", "2016-01-28"),
    "2018 贸易战": ("2018-01-24", "2018-12-28"),
    "2020 新冠疫情冲击": ("2020-01-20", "2020-03-23"),
    "2022 年初调整": ("2021-12-13", "2022-04-26"),
}

# 组合自身历史最差滚动区间
HISTORICAL_WINDOWS = {
    "自身历史最差单日": 1,
    "自身历史最差 5 日": 5,
    "自身历史最差 20 日": 20,
}


def hypothetical_stress(returns: pd.DataFrame, weights: np.ndarray,
                        market: pd.Series) -> pd.DataFrame:
    """市场冲击情景：资产 i 损失 = 收缩 beta_i × 指数冲击。"""
    betas = estimate_betas(returns, market, shrink=True).values
    rows = []
    for name, shock in HYPOTHETICAL_SCENARIOS.items():
        port_pnl = float(weights @ (betas * shock))
        rows.append({"scenario": name, "type": "假设情景",
                     "portfolio_pnl": port_pnl, "index_return": shock,
                     "replay_pnl": None, "coverage": None, "period": ""})
    return pd.DataFrame(rows)


def _window_cum_return(series: pd.Series, start: str, end: str) -> float | None:
    """情景窗口内的累计收益；窗口内数据不足（未上市/停牌过半）返回 None。"""
    seg = series.loc[start:end].dropna()
    if len(seg) < 2:
        return None
    return float(seg.iloc[-1] / seg.iloc[0] - 1)


def historical_scenario_stress(returns: pd.DataFrame, weights: np.ndarray,
                               market: pd.Series,
                               index_prices_full: pd.Series,
                               stock_prices_full: pd.DataFrame) -> pd.DataFrame:
    """历史情景重演。

    - beta 映射口径：组合损失 = Σ w_i · beta_i · R_index
    - 真实回放口径：情景期已上市个股用真实收益，缺失个股回退 beta 映射；
      coverage 为按权重计的真实数据覆盖率。
    """
    betas = pd.Series(estimate_betas(returns, market, shrink=True).values,
                      index=returns.columns)
    w = pd.Series(weights, index=returns.columns)
    rows = []
    for name, (start, end) in SCENARIO_LIBRARY.items():
        r_index = _window_cum_return(index_prices_full, start, end)
        if r_index is None:
            continue
        beta_pnl = float((w * betas * r_index).sum())

        replay_pnl, covered = 0.0, 0.0
        for code in returns.columns:
            r_stock = _window_cum_return(stock_prices_full[code], start, end) \
                if code in stock_prices_full else None
            if r_stock is not None:
                replay_pnl += w[code] * r_stock
                covered += w[code]
            else:
                replay_pnl += w[code] * betas[code] * r_index
        rows.append({
            "scenario": name, "type": "历史情景",
            "portfolio_pnl": beta_pnl,
            "index_return": r_index,
            "replay_pnl": float(replay_pnl),
            "coverage": float(covered),
            "period": f"{start} ~ {end}",
        })
    return pd.DataFrame(rows)


def own_history_stress(returns: pd.DataFrame, weights: np.ndarray) -> pd.DataFrame:
    """组合自身历史最差滚动区间。"""
    port = pd.Series(returns.values @ weights, index=returns.index)
    rows = []
    for name, window in HISTORICAL_WINDOWS.items():
        rolling = (1 + port).rolling(window).apply(np.prod, raw=True) - 1
        worst = rolling.min()
        worst_end = rolling.idxmin()
        rows.append({"scenario": name, "type": "自身回放",
                     "portfolio_pnl": float(worst),
                     "index_return": None, "replay_pnl": None, "coverage": None,
                     "period": str(worst_end.date()) if pd.notna(worst_end) else ""})
    return pd.DataFrame(rows)


def run_stress_tests(returns: pd.DataFrame, weights: np.ndarray,
                     market: pd.Series,
                     index_prices_full: pd.Series,
                     stock_prices_full: pd.DataFrame,
                     nav: float) -> pd.DataFrame:
    returns_a, market_a = align_market(returns, market)
    df = pd.concat([
        hypothetical_stress(returns_a, weights, market_a),
        historical_scenario_stress(returns_a, weights, market_a,
                                   index_prices_full, stock_prices_full),
        own_history_stress(returns, weights),
    ], ignore_index=True)
    df["pnl_amount"] = df["portfolio_pnl"] * nav
    return df
