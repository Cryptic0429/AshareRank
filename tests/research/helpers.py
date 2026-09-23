"""Panel 测试数据工厂。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_panel(n_stocks: int = 8, n_days: int = 100, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    rows = []
    market_shocks = rng.normal(0.0003, 0.007, n_days)
    for stock_index in range(n_stocks):
        idiosyncratic = rng.normal(stock_index * 0.00004, 0.01, n_days)
        close = (20 + stock_index) * np.cumprod(1 + market_shocks + idiosyncratic)
        open_ = close * (1 + rng.normal(0, 0.002, n_days))
        high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, n_days))
        low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, n_days))
        volume = rng.integers(500_000, 2_000_000, n_days).astype(float)
        amount = close * volume
        ticker = f"{stock_index:06d}.SZ"
        rows.extend({
            "date": date,
            "ts_code": ticker,
            "open": open_value,
            "high": high_value,
            "low": low_value,
            "close": close_value,
            "volume": volume_value,
            "amount": amount_value,
        } for date, open_value, high_value, low_value, close_value, volume_value, amount_value in zip(
            dates, open_, high, low, close, volume, amount
        ))
    return pd.DataFrame(rows).sort_values(["date", "ts_code"]).reset_index(drop=True)
