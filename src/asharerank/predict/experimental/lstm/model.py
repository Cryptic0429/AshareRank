"""单层/双层 LSTM + 线性头：吃过去 window 日截面因子特征，预测当日截面排名标签。"""
from __future__ import annotations

import torch
import torch.nn as nn


class LSTMRankPredictor(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 64,
                num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, window, n_features) → (B,) 预测标量。"""
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)
