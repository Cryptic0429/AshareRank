"""把统一信息集转换为不同模型所需的输入形状。"""

from asharerank.adapters.base import AdaptedData, BaseAdapter
from asharerank.adapters.master import MasterAdapter
from asharerank.adapters.tabular import RankingTabularAdapter, TabularAdapter

__all__ = [
    "AdaptedData",
    "BaseAdapter",
    "MasterAdapter",
    "RankingTabularAdapter",
    "TabularAdapter",
]
