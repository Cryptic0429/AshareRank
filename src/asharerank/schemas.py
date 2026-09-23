"""Validated configuration schemas for model-comparison experiments."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from asharerank.config import PANEL_STOCK_LIMIT


class ModelCandidate(BaseModel):
    name: str
    available: bool
    role: str


class CompareModelsRequest(BaseModel):
    models: list[str] = Field(
        default_factory=lambda: [
            "ridge",
            "random_forest",
            "lightgbm",
            "lite_rank",
            "master_lite",
        ],
        min_length=1,
    )
    max_stocks: int = Field(
        default=PANEL_STOCK_LIMIT,
        ge=5,
        le=PANEL_STOCK_LIMIT,
        description="演示阶段最多使用沪深 300 前 50 只股票",
    )
    lookback: int = Field(default=8, ge=1)
    horizon: int = Field(default=5, ge=1)
    target: Literal["return", "excess_return"] = "excess_return"
    top_k: int = Field(default=10, ge=1)


class LivePredictRequest(BaseModel):
    models: list[str] = Field(
        default_factory=lambda: [
            "ridge",
            "random_forest",
            "lightgbm",
            "lite_rank",
            "master_lite",
        ],
        min_length=1,
    )
    top_k: int = Field(default=10, ge=1, le=PANEL_STOCK_LIMIT)
    refresh_data: bool = Field(
        default=True,
        description="先用 AkShare 增量更新最新日线；失败股票继续使用本地缓存",
    )
