"""MASTER-Lite 模型。PyTorch 仅在训练该模型时需要。"""

from asharerank.models.master.config import MasterLiteConfig
from asharerank.models.master.model import MasterLiteRegressor, torch_available

__all__ = ["MasterLiteConfig", "MasterLiteRegressor", "torch_available"]
