"""量价特征工程（Alpha158 核心子集，约 60+ 特征，纯 OHLCV）。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_div(a, b):
    return a / b.replace(0, np.nan)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """对单只股票 OHLCV（按日期升序）计算特征。返回带 date/ts_code 的特征表。"""
    d = df.sort_values("date").copy()
    o, h, l, c = d["open"], d["high"], d["low"], d["close"]
    vol = d["volume"].astype(float) if "volume" in d else pd.Series(np.nan, index=d.index)
    ret = c.pct_change()

    feats: dict[str, pd.Series] = {}

    # K 线形态
    feats["KMID"] = _safe_div(c - o, o)
    feats["KLEN"] = _safe_div(h - l, o)
    feats["KUP"] = _safe_div(h - np.maximum(o, c), o)
    feats["KLOW"] = _safe_div(np.minimum(o, c) - l, o)
    feats["KSFT"] = _safe_div(2 * c - h - l, o)

    # 多窗口收益 / 波动 / 量价
    for w in (5, 10, 20, 30, 60):
        feats[f"ROC{w}"] = c.pct_change(w)
        feats[f"MA{w}"] = _safe_div(c, c.rolling(w).mean()) - 1
        feats[f"STD{w}"] = ret.rolling(w).std()
        feats[f"MAX{w}"] = _safe_div(c, c.rolling(w).max()) - 1
        feats[f"MIN{w}"] = _safe_div(c, c.rolling(w).min()) - 1
        feats[f"QTLU{w}"] = _safe_div(c, c.rolling(w).quantile(0.8)) - 1
        feats[f"QTLD{w}"] = _safe_div(c, c.rolling(w).quantile(0.2)) - 1
        # 用收盘价在窗口内的分位近似 RANK（避免 rolling.apply 过慢）
        feats[f"RANK{w}"] = (c - c.rolling(w).min()) / (
            c.rolling(w).max() - c.rolling(w).min()).replace(0, np.nan)
        feats[f"RSV{w}"] = _safe_div(c - l.rolling(w).min(),
                                     h.rolling(w).max() - l.rolling(w).min())
        if not vol.isna().all():
            feats[f"VMA{w}"] = _safe_div(vol, vol.rolling(w).mean()) - 1
            feats[f"VSTD{w}"] = vol.pct_change().rolling(w).std()
            feats[f"WVMA{w}"] = (ret.abs() * vol).rolling(w).std() / (
                (ret.abs() * vol).rolling(w).mean().replace(0, np.nan))

    # 量价相关
    for w in (5, 10, 20, 60):
        feats[f"CORR{w}"] = ret.rolling(w).corr(vol.pct_change())
        feats[f"CORD{w}"] = ret.rolling(w).corr(vol)

    # 动量 / RSI 风格
    up = ret.clip(lower=0)
    down = (-ret).clip(lower=0)
    for w in (6, 12, 24):
        rs = up.rolling(w).mean() / down.rolling(w).mean().replace(0, np.nan)
        feats[f"RSI{w}"] = 100 - 100 / (1 + rs)

    out = pd.DataFrame(feats, index=d.index)
    out["date"] = d["date"].values
    out["ts_code"] = d["ts_code"].values
    return out


def build_feature_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """对全市场长表逐股算特征后拼接。"""
    parts = []
    for code, g in panel.groupby("ts_code", sort=False):
        parts.append(compute_features(g))
    feat = pd.concat(parts, ignore_index=True)
    return feat
