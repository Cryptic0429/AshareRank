"""按交易日切分训练集、验证集和测试集。"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


def purge_tail_dates(
    frame: pd.DataFrame,
    horizon: int,
    *,
    date_column: str = "date",
) -> tuple[pd.DataFrame, tuple[pd.Timestamp, ...]]:
    """Remove the final ``horizon`` dates so their forward labels stay in-fold."""
    if horizon < 0:
        raise ValueError("horizon 不能为负数")
    if horizon == 0 or frame.empty:
        return frame.copy(), ()
    dates = tuple(sorted(pd.Timestamp(value) for value in frame[date_column].dropna().unique()))
    removed = dates[-horizon:]
    if not removed:
        return frame.copy(), ()
    kept = frame.loc[~pd.to_datetime(frame[date_column]).isin(removed)].copy()
    return kept, tuple(removed)


@dataclass(frozen=True)
class DateSplit:
    train_dates: tuple[pd.Timestamp, ...]
    valid_dates: tuple[pd.Timestamp, ...]
    test_dates: tuple[pd.Timestamp, ...]
    purge_days: int = 0
    purged_dates: tuple[pd.Timestamp, ...] = ()

    def mask(self, dates: pd.Series, part: str) -> pd.Series:
        selected = getattr(self, f"{part}_dates")
        return pd.to_datetime(dates).isin(selected)

    def as_dict(self) -> dict[str, object]:
        def describe(values: tuple[pd.Timestamp, ...]) -> dict[str, str | int]:
            return {
                "start": str(values[0].date()),
                "end": str(values[-1].date()),
                "n_dates": len(values),
            }

        return {
            "train": describe(self.train_dates),
            "valid": describe(self.valid_dates),
            "test": describe(self.test_dates),
            "purge_days": self.purge_days,
            "purged_dates": [str(value.date()) for value in self.purged_dates],
        }


def chronological_split(
    dates,
    valid_fraction: float = 0.15,
    test_fraction: float = 0.15,
    purge_days: int = 0,
) -> DateSplit:
    """按日期切分并在边界清除可能重叠的 forward-label 日期。"""
    unique_dates = tuple(pd.Timestamp(value) for value in sorted(pd.unique(pd.to_datetime(dates))))
    if len(unique_dates) < 10:
        raise ValueError("至少需要 10 个有效交易日才能进行时间切分")
    if not 0 < valid_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("valid_fraction 和 test_fraction 必须位于 (0, 1)")
    if purge_days < 0:
        raise ValueError("purge_days 不能为负数")
    n_valid = max(1, int(round(len(unique_dates) * valid_fraction)))
    n_test = max(1, int(round(len(unique_dates) * test_fraction)))
    n_train = len(unique_dates) - n_valid - n_test
    if n_train < 1:
        raise ValueError("训练日期不足，请降低验证集或测试集比例")
    train_block = unique_dates[:n_train]
    valid_block = unique_dates[n_train:n_train + n_valid]
    test_block = unique_dates[n_train + n_valid:]
    if len(train_block) <= purge_days or len(valid_block) <= purge_days:
        raise ValueError("purge_days 过大，训练集或验证集将为空")
    purged = (train_block[-purge_days:] if purge_days else ()) + (
        valid_block[-purge_days:] if purge_days else ()
    )
    return DateSplit(
        train_dates=train_block[:-purge_days] if purge_days else train_block,
        valid_dates=valid_block[:-purge_days] if purge_days else valid_block,
        test_dates=test_block,
        purge_days=purge_days,
        purged_dates=tuple(purged),
    )


def walk_forward_splits(
    dates,
    train_days: int,
    valid_days: int,
    test_days: int,
    step_days: int | None = None,
    purge_days: int = 0,
) -> list[DateSplit]:
    """生成固定长度的滚动样本外窗口。"""
    values = tuple(pd.Timestamp(value) for value in sorted(pd.unique(pd.to_datetime(dates))))
    step = step_days or test_days
    if min(train_days, valid_days, test_days, step) < 1 or purge_days < 0:
        raise ValueError("窗口长度必须为正整数")
    result: list[DateSplit] = []
    stop = train_days + valid_days + test_days
    for end in range(stop, len(values) + 1, step):
        start = end - stop
        train_block = values[start:start + train_days]
        valid_block = values[start + train_days:start + train_days + valid_days]
        test_block = values[start + train_days + valid_days:end]
        if len(train_block) <= purge_days or len(valid_block) <= purge_days:
            continue
        purged = (train_block[-purge_days:] if purge_days else ()) + (
            valid_block[-purge_days:] if purge_days else ()
        )
        result.append(DateSplit(
            train_dates=train_block[:-purge_days] if purge_days else train_block,
            valid_dates=valid_block[:-purge_days] if purge_days else valid_block,
            test_dates=test_block,
            purge_days=purge_days,
            purged_dates=tuple(purged),
        ))
    return result
