"""LSTM 对比基线超参数。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LSTMConfig:
    window: int = 20          # 回看交易日数（决策：20日）
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.1
    epochs_per_fold: int = 8
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-5
    min_train_rows: int = 800
    min_valid_rows: int = 150
    device: str = "cpu"
    seed: int = 42


DEFAULT_LSTM_CONFIG = LSTMConfig()
