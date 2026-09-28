"""StockMixer（AAAI-24）沪深300 训练/评估基线（独立于生产管线，不写入模型注册表）。

参考: https://github.com/SJTU-Quant/StockMixer
"""
from __future__ import annotations

from asharerank.forecasting.experimental.stockmixer.config import StockMixerConfig
from asharerank.forecasting.experimental.stockmixer.model import StockMixer
from asharerank.forecasting.experimental.stockmixer.train import run_stockmixer, run_stockmixer_multiseed

__all__ = ["StockMixerConfig", "StockMixer", "run_stockmixer", "run_stockmixer_multiseed"]
