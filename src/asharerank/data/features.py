"""为全部候选模型构建相同的精简股票特征和市场特征。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from asharerank.config import MARKET_FEATURES, STOCK_FEATURES

_REQUIRED_STOCK_COLUMNS = {"date", "ts_code", "open", "high", "low", "close"}


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.astype(float) / denominator.astype(float).replace(0, np.nan)


def _normalise_market_frame(market: pd.DataFrame) -> pd.DataFrame:
    frame = market.copy()
    if "trade_date" in frame.columns and "date" not in frame.columns:
        frame = frame.rename(columns={"trade_date": "date"})
    missing = {"date", "close"} - set(frame.columns)
    if missing:
        raise ValueError(f"市场数据缺少字段: {sorted(missing)}")
    frame["date"] = pd.to_datetime(frame["date"])
    for col in ("volume", "amount"):
        if col not in frame.columns:
            frame[col] = np.nan
    return frame.sort_values("date").drop_duplicates("date", keep="last")


def derive_market_data(stock_panel: pd.DataFrame) -> pd.DataFrame:
    """没有独立指数行情时，用当日股票横截面构造可复现的市场代理。"""
    panel = stock_panel.sort_values(["ts_code", "date"]).copy()
    daily_returns = panel.groupby("ts_code")["close"].pct_change()
    panel["_ret"] = daily_returns
    aggregations: dict[str, tuple[str, str]] = {
        "market_ret": ("_ret", "mean"),
        "close": ("close", "mean"),
    }
    if "volume" in panel.columns:
        aggregations["volume"] = ("volume", "sum")
    if "amount" in panel.columns:
        aggregations["amount"] = ("amount", "sum")
    market = panel.groupby("date", as_index=False).agg(**aggregations)
    if "volume" not in market:
        market["volume"] = np.nan
    if "amount" not in market:
        market["amount"] = np.nan
    # 用累计横截面平均收益生成代理指数，避免平均价格受成分价格尺度影响。
    market["close"] = (1.0 + market["market_ret"].fillna(0.0)).cumprod()
    return market[["date", "close", "volume", "amount"]]


def compute_stock_features(stock: pd.DataFrame) -> pd.DataFrame:
    """计算单只股票的 18 个统一特征。"""
    data = stock.sort_values("date").copy()
    close = data["close"].astype(float)
    open_ = data["open"].astype(float)
    high = data["high"].astype(float)
    low = data["low"].astype(float)
    volume = data.get("volume", pd.Series(np.nan, index=data.index)).astype(float)
    if not volume.notna().any():
        volume = pd.Series(1.0, index=data.index)
    amount = data.get("amount", pd.Series(np.nan, index=data.index)).astype(float)
    if not amount.notna().any():
        amount = close * volume

    if amount.notna().any() and volume.notna().any():
        vwap = _safe_div(amount, volume)
        # A 股部分数据源的成交量单位是“手”；若量价比例明显异常则修正为股。
        ratio = (vwap / close.replace(0, np.nan)).median(skipna=True)
        if pd.notna(ratio) and ratio > 20:
            vwap = _safe_div(amount, volume * 100.0)
    else:
        vwap = (high + low + close) / 3.0

    result = pd.DataFrame({"date": data["date"], "ts_code": data["ts_code"]})
    for window in (1, 5, 10, 20):
        result[f"ret_{window}"] = close.pct_change(window)
    result["open_close_ratio"] = _safe_div(open_, close) - 1.0
    result["high_close_ratio"] = _safe_div(high, close) - 1.0
    result["low_close_ratio"] = _safe_div(low, close) - 1.0
    result["vwap_close_ratio"] = _safe_div(vwap, close) - 1.0
    for window in (5, 10, 20):
        result[f"close_ma{window}_ratio"] = _safe_div(
            close, close.rolling(window).mean()
        ) - 1.0
        result[f"volatility_{window}"] = close.pct_change().rolling(window).std(ddof=0)
    for window in (5, 20):
        result[f"volume_ratio_{window}"] = _safe_div(
            volume, volume.rolling(window).mean()
        ) - 1.0
        result[f"amount_ratio_{window}"] = _safe_div(
            amount, amount.rolling(window).mean()
        ) - 1.0
    return result[["date", "ts_code", *STOCK_FEATURES]]


def compute_market_features(market: pd.DataFrame) -> pd.DataFrame:
    """计算 8 个统一市场特征。"""
    data = _normalise_market_frame(market)
    close = data["close"].astype(float)
    amount = data["amount"].astype(float)
    if not amount.notna().any():
        volume = data["volume"].astype(float)
        amount = close * (volume if volume.notna().any() else 1.0)
    returns = close.pct_change()
    result = pd.DataFrame({"date": data["date"]})
    for window in (1, 5, 20):
        result[f"market_ret_{window}"] = close.pct_change(window)
    for window in (5, 20):
        result[f"market_volatility_{window}"] = returns.rolling(window).std(ddof=0)
        result[f"market_amount_ratio_{window}"] = _safe_div(
            amount, amount.rolling(window).mean()
        ) - 1.0
    rolling_peak = close.rolling(20, min_periods=1).max()
    result["market_drawdown_20"] = _safe_div(close, rolling_peak) - 1.0
    return result[["date", *MARKET_FEATURES]]


def build_common_features(
    stock_panel: pd.DataFrame,
    market: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """构建包含 18 个股票特征和 8 个市场特征的统一长表。"""
    missing = _REQUIRED_STOCK_COLUMNS - set(stock_panel.columns)
    if missing:
        raise ValueError(f"股票数据缺少字段: {sorted(missing)}")
    panel = stock_panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    parts = [compute_stock_features(group) for _, group in panel.groupby("ts_code", sort=False)]
    stock_features = pd.concat(parts, ignore_index=True)
    market_data = derive_market_data(panel) if market is None else market
    market_features = compute_market_features(market_data)
    result = stock_features.merge(market_features, on="date", how="left", validate="many_to_one")
    return result.sort_values(["date", "ts_code"]).reset_index(drop=True)
