"""精简 MASTER：市场门控、个股时序、股票间注意力与时间聚合。"""
from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np

from asharerank.models.base import BaseStockModel
from asharerank.models.master.config import MasterLiteConfig

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - 由未安装 torch 的环境覆盖
    torch = None
    nn = None


def torch_available() -> bool:
    return torch is not None


def _require_torch() -> None:
    if torch is None:
        raise ImportError(
            "训练 MASTER-Lite 需要 PyTorch。请先安装与当前 Python/硬件匹配的 torch。"
        )


if nn is not None:
    class MarketGuidedGating(nn.Module):
        def __init__(self, market_features: int, hidden_dim: int) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(market_features, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid(),
            )

        def forward(self, market):
            # 同一交易日的市场序列对所有股票相同，取横截面均值可容忍缺失股票。
            market_state = market.mean(dim=0)
            return self.network(market_state).unsqueeze(0)


    class TemporalAttention(nn.Module):
        def __init__(self, hidden_dim: int) -> None:
            super().__init__()
            self.score = nn.Linear(hidden_dim, 1)

        def forward(self, sequence):
            weights = torch.softmax(self.score(sequence), dim=1)
            return torch.sum(sequence * weights, dim=1)


    class MasterLiteNetwork(nn.Module):
        def __init__(self, config: MasterLiteConfig) -> None:
            super().__init__()
            self.stock_projection = nn.Linear(config.stock_features, config.hidden_dim)
            self.market_gate = MarketGuidedGating(config.market_features, config.hidden_dim)
            intra_layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.n_heads,
                dim_feedforward=config.hidden_dim * 2,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.intra_stock = nn.TransformerEncoder(
                intra_layer, config.intra_layers, enable_nested_tensor=False
            )
            inter_layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.n_heads,
                dim_feedforward=config.hidden_dim * 2,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.inter_stock = nn.TransformerEncoder(
                inter_layer, config.inter_layers, enable_nested_tensor=False
            )
            self.temporal_pool = TemporalAttention(config.hidden_dim)
            self.head = nn.Sequential(
                nn.LayerNorm(config.hidden_dim * 2),
                nn.Linear(config.hidden_dim * 2, config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, 1),
            )

        def forward(self, stock, market):
            stock_state = self.stock_projection(stock)
            gated = stock_state * self.market_gate(market)
            intra = self.intra_stock(gated)
            temporal = self.temporal_pool(intra)
            cross_stock = self.inter_stock(temporal.unsqueeze(0)).squeeze(0)
            return self.head(torch.cat([temporal, cross_stock], dim=-1)).squeeze(-1)
else:
    class MasterLiteNetwork:  # pragma: no cover - 只提供清晰的依赖错误
        def __init__(self, config: MasterLiteConfig) -> None:
            _require_torch()


class MasterLiteRegressor(BaseStockModel):
    name = "master_lite"

    def __init__(self, config: MasterLiteConfig | None = None) -> None:
        self.config = config or MasterLiteConfig()
        self.network = None
        self.history: list[dict[str, float]] = []
        self.device = "cpu"

    def _resolve_device(self) -> str:
        _require_torch()
        if self.config.device != "auto":
            return self.config.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _groups(dates: np.ndarray) -> list[np.ndarray]:
        # 按日期联合训练当日股票，保留 MASTER 的股票间注意力语义。
        _, inverse = np.unique(dates.astype("datetime64[ns]"), return_inverse=True)
        return [np.flatnonzero(inverse == index) for index in range(inverse.max() + 1)]

    def _loss(self, X, y, training: bool) -> float:
        groups = self._groups(np.asarray(X["dates"]))
        if training:
            random.shuffle(groups)
            self.network.train()
        else:
            self.network.eval()
        total = 0.0
        count = 0
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for indices in groups:
                stock = torch.as_tensor(X["stock"][indices], device=self.device)
                market = torch.as_tensor(X["market"][indices], device=self.device)
                target = torch.as_tensor(y[indices], device=self.device)
                prediction = self.network(stock, market)
                loss = torch.mean((prediction - target) ** 2)
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                    self.optimizer.step()
                total += float(loss.detach().cpu()) * len(indices)
                count += len(indices)
        return total / max(count, 1)

    def fit(self, X, y, X_valid=None, y_valid=None) -> "MasterLiteRegressor":
        _require_torch()
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.random_seed)
        self.device = self._resolve_device()
        self.network = MasterLiteNetwork(self.config).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        best_state = None
        best_valid = float("inf")
        stale_epochs = 0
        for epoch in range(1, self.config.max_epochs + 1):
            train_loss = self._loss(X, y, training=True)
            valid_loss = (
                self._loss(X_valid, y_valid, training=False)
                if X_valid is not None and y_valid is not None and len(y_valid)
                else train_loss
            )
            self.history.append({
                "epoch": float(epoch),
                "train_loss": train_loss,
                "valid_loss": valid_loss,
            })
            if valid_loss < best_valid - 1e-7:
                best_valid = valid_loss
                best_state = copy.deepcopy(self.network.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.config.patience:
                    break
        if best_state is not None:
            self.network.load_state_dict(best_state)
        return self

    def predict(self, X) -> np.ndarray:
        _require_torch()
        if self.network is None:
            raise RuntimeError("MASTER-Lite 尚未训练")
        result = np.empty(len(X["dates"]), dtype=float)
        self.network.eval()
        with torch.no_grad():
            for indices in self._groups(np.asarray(X["dates"])):
                stock = torch.as_tensor(X["stock"][indices], device=self.device)
                market = torch.as_tensor(X["market"][indices], device=self.device)
                result[indices] = self.network(stock, market).detach().cpu().numpy()
        return result

    def save(self, path: str | Path) -> None:
        _require_torch()
        if self.network is None:
            raise RuntimeError("MASTER-Lite 尚未训练")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "config": self.config.as_dict(),
            "state_dict": self.network.state_dict(),
            "history": self.history,
        }, destination)

    @classmethod
    def load(cls, path: str | Path) -> "MasterLiteRegressor":
        _require_torch()
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        model = cls(MasterLiteConfig(**payload["config"]))
        model.device = model._resolve_device()
        model.network = MasterLiteNetwork(model.config).to(model.device)
        model.network.load_state_dict(payload["state_dict"])
        model.history = payload.get("history", [])
        return model
