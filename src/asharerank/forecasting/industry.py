"""行业分类映射（tushare stock_basic，一次性拉全市场，供行业中性化标签使用）。"""
from __future__ import annotations

import json
import logging

import pandas as pd

from asharerank.settings import TUSHARE_TOKEN, UNIVERSE_DIR

logger = logging.getLogger(__name__)

INDUSTRY_META = UNIVERSE_DIR / "industry_meta.json"


def fetch_industry_map(force: bool = False) -> dict[str, str]:
    """ts_code → 行业名。本地缓存，避免重复调用。"""
    if INDUSTRY_META.exists() and not force:
        return json.loads(INDUSTRY_META.read_text())

    if not TUSHARE_TOKEN:
        raise RuntimeError("行业分类依赖 tushare stock_basic，请配置 TUSHARE_TOKEN")

    import tushare as ts

    ts.set_token(TUSHARE_TOKEN)
    pro = ts.pro_api()
    df = pro.stock_basic(exchange="", list_status="L",
                         fields="ts_code,name,industry")
    mapping = dict(zip(df["ts_code"], df["industry"].fillna("未分类")))
    INDUSTRY_META.write_text(json.dumps(mapping, ensure_ascii=False, indent=2))
    logger.info("行业分类已缓存: %d 只股票, %d 个行业",
                len(mapping), len(set(mapping.values())))
    return mapping


def load_industry_map() -> dict[str, str]:
    return fetch_industry_map(force=False)
