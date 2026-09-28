"""按股票拼滑动窗口：复用 build_train_frame 的截面因子特征，不重造特征工程。

每只股票内部按日期排序后，用位置连续窗口（而非严格按日历连续）截取过去
window 天特征 → 预测窗口末日的截面排名标签。个股偶发缺口对结果影响很小，
换来的是实现简单、内存可控。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


@dataclass
class SequencePanel:
    """整个面板一次性构建；不同 fold 复用，只按日期筛选 index 子集。"""

    feature_cols: list[str]
    window: int
    per_stock: dict[str, dict[str, np.ndarray]]
    index: list[tuple[str, int]]
    _entry_dates: np.ndarray = field(default=None, repr=False)

    def entry_dates(self) -> np.ndarray:
        if self._entry_dates is None:
            self._entry_dates = np.array(
                [self.per_stock[c]["date"][i] for c, i in self.index]
            )
        return self._entry_dates

    def subset_by_date(self, start, end) -> list[tuple[str, int]]:
        dates = self.entry_dates()
        mask = (dates >= np.datetime64(start)) & (dates <= np.datetime64(end))
        return [self.index[k] for k in np.nonzero(mask)[0]]

    def meta_frame(self, entries: list[tuple[str, int]]) -> pd.DataFrame:
        rows = []
        for code, i in entries:
            rec = self.per_stock[code]
            rows.append({"date": rec["date"][i], "ts_code": code,
                        "fwd_ret": rec["fwd_ret"][i], "label": rec["label"][i]})
        return pd.DataFrame(rows)


def build_sequence_panel(df: pd.DataFrame, feature_cols: list[str],
                         window: int = 20) -> SequencePanel:
    """df 需含 date/ts_code/label/fwd_ret + feature_cols（即 build_train_frame 的输出）。"""
    per_stock: dict[str, dict[str, np.ndarray]] = {}
    index: list[tuple[str, int]] = []

    for code, g in df.sort_values("date").groupby("ts_code", sort=False):
        g = g.reset_index(drop=True)
        feat = g[feature_cols].to_numpy(dtype=np.float32)
        label = g["label"].to_numpy(dtype=np.float32)
        fwd = g["fwd_ret"].to_numpy(dtype=np.float32)
        date = pd.to_datetime(g["date"]).to_numpy()
        per_stock[code] = {"feat": feat, "label": label, "fwd_ret": fwd, "date": date}

        n = len(g)
        for i in range(window - 1, n):
            if np.isnan(feat[i - window + 1:i + 1]).any() or np.isnan(label[i]):
                continue
            index.append((code, i))

    return SequencePanel(feature_cols=feature_cols, window=window,
                         per_stock=per_stock, index=index)


class WindowDataset(Dataset):
    """给定一批 (code, position) 索引，惰性切片出 (window, n_features) 张量。"""

    def __init__(self, panel: SequencePanel, entries: list[tuple[str, int]]):
        self.panel = panel
        self.entries = entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        code, i = self.entries[idx]
        rec = self.panel.per_stock[code]
        w = self.panel.window
        x = rec["feat"][i - w + 1:i + 1]
        y = rec["label"][i]
        return x, np.float32(y)
