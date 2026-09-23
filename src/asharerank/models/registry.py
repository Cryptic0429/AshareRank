"""候选模型工厂；不包含模型融合逻辑。"""
from __future__ import annotations

from asharerank.config import RESERVED_MODELS, SUPPORTED_MODELS


def list_candidate_models() -> list[dict[str, str | bool]]:
    return [
        {"name": "ridge", "available": True, "role": "linear_baseline"},
        {"name": "random_forest", "available": True, "role": "bagging_baseline"},
        {"name": "lightgbm", "available": True, "role": "strong_tabular_baseline"},
        {"name": "lite_rank", "available": _lightgbm_available(), "role": "lightweight_ranker"},
        {
            "name": "master_lite",
            "available": _master_available(),
            "role": "advanced_model",
        },
        {"name": "custom", "available": False, "role": "reserved"},
    ]


def _master_available() -> bool:
    from asharerank.models.master import torch_available

    return torch_available()


def _lightgbm_available() -> bool:
    try:
        import lightgbm  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


def create_model(name: str, **params):
    key = name.lower().strip()
    if key in RESERVED_MODELS:
        raise NotImplementedError("自研模型目录已预留，当前暂不实现")
    if key not in SUPPORTED_MODELS:
        raise ValueError(f"未知模型 {name!r}，可选: {', '.join(SUPPORTED_MODELS)}")
    if key == "ridge":
        from asharerank.models.ridge import RidgeModel

        return RidgeModel(**params)
    if key == "random_forest":
        from asharerank.models.random_forest import RandomForestModel

        return RandomForestModel(**params)
    if key == "lightgbm":
        from asharerank.models.lightgbm import LightGBMModel

        return LightGBMModel(**params)
    if key == "lite_rank":
        from asharerank.models.lite_rank import LiteRankModel

        return LiteRankModel(**params)
    from asharerank.models.master import MasterLiteRegressor

    return MasterLiteRegressor(**params)
