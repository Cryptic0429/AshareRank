"""标签构造与训练面板：前瞻收益 + 横截面 CSRankNorm。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from asharerank.settings import FEATURE_WARMUP, PREDICT_HORIZON
from asharerank.predict.features import build_feature_panel


def cs_rank_norm(s: pd.Series) -> pd.Series:
    """横截面百分位秩映射到 [-1, 1]（近似均匀）。"""
    r = s.rank(pct=True, method="average")
    return (r - 0.5) * 2


def cs_zscore(s: pd.Series) -> pd.Series:
    mu, sigma = s.mean(), s.std(ddof=0)
    if sigma == 0 or np.isnan(sigma):
        return s * 0.0
    return (s - mu) / sigma


def make_labels(panel: pd.DataFrame, horizon: int = PREDICT_HORIZON,
               industry_map: "pd.Series | dict | None" = None) -> pd.DataFrame:
    """前瞻 horizon 日收益，并做横截面 rank 归一化。

    Args:
        industry_map: ts_code → 行业名 的映射。传入时按 (date, industry) 分组做
            行业内排名（行业中性化标签，去掉行业 beta 对排名的干扰）；
            不传则保持原行为——全市场按日排名。
    """
    d = panel.sort_values(["ts_code", "date"]).copy()
    d["fwd_ret"] = d.groupby("ts_code")["close"].transform(
        lambda x: x.shift(-horizon) / x - 1)
    if industry_map is not None:
        d["_industry"] = d["ts_code"].map(industry_map).fillna("未分类")
        d["label"] = d.groupby(["date", "_industry"])["fwd_ret"].transform(cs_rank_norm)
        d = d.drop(columns="_industry")
    else:
        d["label"] = d.groupby("date")["fwd_ret"].transform(cs_rank_norm)
    return d[["date", "ts_code", "fwd_ret", "label"]]


def build_train_frame(panel: pd.DataFrame,
                      horizon: int = PREDICT_HORIZON,
                      industry_map: "pd.Series | dict | None" = None,
                      extra_features: pd.DataFrame | None = None) -> pd.DataFrame:
    """特征 + 标签合并，特征做横截面 z-score，丢掉预热与标签缺失行。

    Args:
        industry_map: 见 make_labels，传入则做行业中性化标签。
        extra_features: 额外特征表（如基本面因子），需含 date/ts_code + 特征列，
            按 (date, ts_code) 左连接合并，同样做横截面 z-score 后计入 feature_cols。
    """
    feats = build_feature_panel(panel)
    labels = make_labels(panel, horizon=horizon, industry_map=industry_map)
    df = feats.merge(labels, on=["date", "ts_code"], how="inner")

    if extra_features is not None:
        df = df.merge(extra_features, on=["date", "ts_code"], how="left")

    feature_cols = [c for c in df.columns
                    if c not in ("date", "ts_code", "fwd_ret", "label")]
    # 横截面标准化特征（按日）
    for col in feature_cols:
        df[col] = df.groupby("date")[col].transform(cs_zscore)

    # 丢掉特征预热期与末尾无标签期
    dates = sorted(df["date"].unique())
    if len(dates) <= FEATURE_WARMUP + horizon:
        raise ValueError("样本交易日不足，请扩大股票池或拉更长历史")
    valid_dates = set(dates[FEATURE_WARMUP: -horizon] if horizon > 0 else dates[FEATURE_WARMUP:])
    df = df[df["date"].isin(valid_dates)].dropna(subset=["label"] + feature_cols)
    df.attrs["feature_cols"] = feature_cols
    return df.reset_index(drop=True)


@dataclass(frozen=True)
class RollingSplit:
    """一个按交易日对齐、记录边界清除日期的滚动训练窗口。

    迭代和下标访问保留旧四元组接口，便于已有调用方逐步迁移。
    """

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp
    purged_train_dates: tuple[pd.Timestamp, ...] = ()
    purged_valid_dates: tuple[pd.Timestamp, ...] = ()

    def __iter__(self):
        return iter((self.train_start, self.train_end, self.valid_start, self.valid_end))

    def __getitem__(self, index):
        return (self.train_start, self.train_end, self.valid_start, self.valid_end)[index]

    def as_dict(self) -> dict[str, object]:
        return {
            "train": {"start": str(self.train_start.date()), "end": str(self.train_end.date())},
            "valid": {"start": str(self.valid_start.date()), "end": str(self.valid_end.date())},
            "purge_days": len(self.purged_train_dates),
            "purged_train_dates": [str(value.date()) for value in self.purged_train_dates],
            "purged_valid_dates": [str(value.date()) for value in self.purged_valid_dates],
        }


def rolling_splits(dates: list, train_years: int = 2, valid_months: int = 3,
                   step_months: int = 3, *, purge_days: int = PREDICT_HORIZON) -> list[RollingSplit]:
    """生成滚动窗口，并从训练和验证窗口末尾清除指定交易日数。

    ``purge_days`` 表示交易日数量，不是日历天数。每个窗口按传入数据里的
    实际日期清除，且四元组边界只覆盖清除后保留的日期。
    """
    dates = pd.to_datetime(sorted(dates))
    if purge_days < 0:
        raise ValueError("purge_days 不能为负数")
    if len(dates) == 0:
        return []
    train_delta = pd.DateOffset(years=train_years)
    valid_delta = pd.DateOffset(months=valid_months)
    step_delta = pd.DateOffset(months=step_months)

    splits: list[RollingSplit] = []
    cursor = dates[0] + train_delta
    end = dates[-1]
    while cursor + valid_delta <= end:
        train_start = cursor - train_delta
        train_end = cursor - pd.Timedelta(days=1)
        valid_start = cursor
        valid_end = cursor + valid_delta - pd.Timedelta(days=1)
        # 对齐到实际交易日范围
        if train_start <= dates[0]:
            train_start = dates[0]
        train_dates = dates[(dates >= train_start) & (dates <= train_end)]
        valid_dates = dates[(dates >= valid_start) & (dates <= valid_end)]
        if len(train_dates) <= purge_days or len(valid_dates) <= purge_days:
            cursor = cursor + step_delta
            continue
        purged_train = tuple(pd.Timestamp(value) for value in train_dates[-purge_days:]) if purge_days else ()
        purged_valid = tuple(pd.Timestamp(value) for value in valid_dates[-purge_days:]) if purge_days else ()
        kept_train = train_dates[:-purge_days] if purge_days else train_dates
        kept_valid = valid_dates[:-purge_days] if purge_days else valid_dates
        splits.append(RollingSplit(
            train_start=pd.Timestamp(kept_train[0]),
            train_end=pd.Timestamp(kept_train[-1]),
            valid_start=pd.Timestamp(kept_valid[0]),
            valid_end=pd.Timestamp(kept_valid[-1]),
            purged_train_dates=purged_train,
            purged_valid_dates=purged_valid,
        ))
        cursor = cursor + step_delta
    return splits
