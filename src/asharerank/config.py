"""Panel 模型比较的公共配置。

这里只保存所有模型共同遵守的实验口径；模型专用超参数放在各自模型目录。
"""
from __future__ import annotations

from dataclasses import dataclass


STOCK_FEATURES = (
    "ret_1",
    "ret_5",
    "ret_10",
    "ret_20",
    "open_close_ratio",
    "high_close_ratio",
    "low_close_ratio",
    "vwap_close_ratio",
    "close_ma5_ratio",
    "close_ma10_ratio",
    "close_ma20_ratio",
    "volatility_5",
    "volatility_10",
    "volatility_20",
    "volume_ratio_5",
    "volume_ratio_20",
    "amount_ratio_5",
    "amount_ratio_20",
)

MARKET_FEATURES = (
    "market_ret_1",
    "market_ret_5",
    "market_ret_20",
    "market_volatility_5",
    "market_volatility_20",
    "market_amount_ratio_5",
    "market_amount_ratio_20",
    "market_drawdown_20",
)

SUPPORTED_MODELS = (
    "ridge",
    "random_forest",
    "lightgbm",
    "lite_rank",
    "master_lite",
)
RESERVED_MODELS = ("custom",)

# CSI 300 is the default research universe limit. Smaller samples should be
# selected explicitly for software development runs.
PANEL_STOCK_LIMIT = 300


@dataclass(frozen=True)
class ExperimentConfig:
    """一次公平模型比较使用的共享参数。"""

    lookback: int = 8
    horizon: int = 5
    target: str = "excess_return"
    valid_fraction: float = 0.15
    test_fraction: float = 0.15
    random_seed: int = 42
    min_cross_section: int = 5

    def __post_init__(self) -> None:
        if self.lookback < 1 or self.horizon < 1:
            raise ValueError("lookback 和 horizon 必须为正整数")
        if self.target not in {"return", "excess_return"}:
            raise ValueError("target 只能是 return 或 excess_return")
        if not 0 < self.valid_fraction < 1 or not 0 < self.test_fraction < 1:
            raise ValueError("验证集和测试集比例必须位于 (0, 1)")
        if self.valid_fraction + self.test_fraction >= 1:
            raise ValueError("验证集与测试集比例之和必须小于 1")
