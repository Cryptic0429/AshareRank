"""MASTER-Lite 专用超参数。"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class MasterLiteConfig:
    stock_features: int = 18
    market_features: int = 8
    hidden_dim: int = 64
    n_heads: int = 4
    intra_layers: int = 1
    inter_layers: int = 1
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 40
    patience: int = 6
    random_seed: int = 42
    device: str = "auto"

    def as_dict(self) -> dict:
        return asdict(self)
