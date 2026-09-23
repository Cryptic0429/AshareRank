"""加载已训练模型并对最新统一特征执行推理。"""

from asharerank.inference.engine import (
    LiveModelOutput,
    list_saved_models,
    predict_saved_models,
)
from asharerank.inference.service import live_prediction_status, run_live_prediction

__all__ = [
    "LiveModelOutput",
    "list_saved_models",
    "live_prediction_status",
    "predict_saved_models",
    "run_live_prediction",
]
