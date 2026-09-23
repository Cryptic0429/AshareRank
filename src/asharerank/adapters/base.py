"""模型数据适配器的统一接口。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AdaptedData:
    X: Any
    y: np.ndarray
    metadata: pd.DataFrame
    feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.y) != len(self.metadata):
            raise ValueError("y 与 metadata 的样本数不一致")


class BaseAdapter(ABC):
    """Adapter 必须先在训练集 fit，再用于验证集和测试集。"""

    @abstractmethod
    def fit(self, frame: pd.DataFrame) -> "BaseAdapter":
        raise NotImplementedError

    @abstractmethod
    def transform(self, frame: pd.DataFrame) -> AdaptedData:
        raise NotImplementedError

    def fit_transform(self, frame: pd.DataFrame) -> AdaptedData:
        return self.fit(frame).transform(frame)
