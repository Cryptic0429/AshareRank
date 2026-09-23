"""在不修改模型原始排名的前提下，生成分散度更高的候选股票。"""
from __future__ import annotations

import json

import pandas as pd

from asharerank.settings import CACHE_DIR


_UNKNOWN_GROUPS = {"", "其他", "未知", "nan", "none", "null"}


def _exchange_group(code: str) -> str:
    text = str(code)
    return text.rsplit(".", 1)[-1] if "." in text else "OTHER"


def _usable_group(frame: pd.DataFrame) -> tuple[pd.Series, str]:
    if "balance_group" in frame.columns:
        explicit = frame["balance_group"].fillna("其他").astype(str)
        if explicit.nunique() >= 2:
            method = (
                str(frame["balance_method"].dropna().iloc[0])
                if "balance_method" in frame.columns and frame["balance_method"].notna().any()
                else "custom"
            )
            return explicit, method
    if "industry" in frame.columns:
        industry = frame["industry"].fillna("其他").astype(str).str.strip()
        known = industry[~industry.str.lower().isin(_UNKNOWN_GROUPS)]
        coverage = len(known) / max(len(industry), 1)
        if known.nunique() >= 3 and coverage >= 0.6:
            return industry.where(~industry.str.lower().isin(_UNKNOWN_GROUPS), "其他"), "industry"
    exchange = frame["ts_code"].map(_exchange_group)
    return exchange, "exchange"


def balanced_top_k(frame: pd.DataFrame, k: int) -> tuple[pd.DataFrame, str]:
    """按行业（缺失时按交易所）轮转挑选，同时保留模型的原始 rank。"""
    if k < 1:
        raise ValueError("k 必须为正整数")
    if frame.empty:
        return frame.copy(), "none"
    required = {"ts_code", "rank", "raw_score"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"均衡选股缺少字段: {sorted(missing)}")

    ranked = frame.sort_values(["rank", "ts_code"]).reset_index(drop=True).copy()
    ranked["balance_group"], method = _usable_group(ranked)
    group_order = ranked.drop_duplicates("balance_group")["balance_group"].tolist()
    queues = {
        group: list(ranked.index[ranked["balance_group"] == group])
        for group in group_order
    }

    selected: list[int] = []
    while len(selected) < min(k, len(ranked)):
        progressed = False
        for group in group_order:
            if queues[group]:
                selected.append(queues[group].pop(0))
                progressed = True
                if len(selected) >= min(k, len(ranked)):
                    break
        if not progressed:
            break

    result = ranked.loc[selected].copy()
    result["balanced_pick_rank"] = range(1, len(result) + 1)
    result["balanced_selected"] = True
    return result.sort_values("balanced_pick_rank").reset_index(drop=True), method


def add_risk_balance_groups(frame: pd.DataFrame) -> pd.DataFrame:
    """行业元数据不足时，构造交易所 × 波动率的本地分组。"""
    result = frame.copy()
    industry = result.get(
        "industry", pd.Series("其他", index=result.index)
    ).fillna("其他").astype(str).str.strip()
    known = ~industry.str.lower().isin(_UNKNOWN_GROUPS)
    if known.mean() >= 0.6 and industry[known].nunique() >= 3:
        return result
    if "volatility_20" not in result.columns:
        return result
    risk_rank = result["volatility_20"].rank(pct=True, method="average")
    risk_bucket = pd.cut(
        risk_rank,
        bins=[0.0, 1 / 3, 2 / 3, 1.0],
        labels=["低波动", "中波动", "高波动"],
        include_lowest=True,
    ).astype("object").fillna("波动未知")
    exchange = result["ts_code"].astype(str).str.rsplit(".", n=1).str[-1]
    result["balance_group"] = exchange + " · " + risk_bucket.astype(str)
    result["balance_method"] = "risk_exchange"
    return result


def balanced_universe_codes(
    codes: list[str],
    info: pd.DataFrame,
    limit: int,
) -> tuple[list[str], dict[str, int]]:
    """从已有行情中分层抽取统一股票池，所有模型共享同一结果。"""
    unique_codes = list(dict.fromkeys(map(str, codes)))
    if limit < 1:
        raise ValueError("limit 必须为正整数")
    base = pd.DataFrame({"ts_code": unique_codes})
    columns = [column for column in ("ts_code", "industry", "market_cap") if column in info]
    if columns:
        base = base.merge(info[columns].drop_duplicates("ts_code"), on="ts_code", how="left")
    base["rank"] = range(1, len(base) + 1)
    market_cap = (
        base["market_cap"]
        if "market_cap" in base.columns
        else pd.Series(0.0, index=base.index)
    )
    base["raw_score"] = pd.to_numeric(market_cap, errors="coerce").fillna(0.0)
    # 行业内优先保留流动性/规模更稳定的股票；组间采用轮转。
    base = base.sort_values(["raw_score", "ts_code"], ascending=[False, True]).reset_index(drop=True)
    base["rank"] = range(1, len(base) + 1)
    selected, _ = balanced_top_k(base, min(limit, len(base)))
    distribution = selected["balance_group"].value_counts().sort_index().astype(int).to_dict()
    return selected["ts_code"].tolist(), distribution


def load_local_stock_info(codes: list[str]) -> pd.DataFrame:
    """只读取项目本地缓存，不为模型实验室触发逐股票外网请求。"""
    rows = {
        str(code): {
            "ts_code": str(code),
            "name": str(code),
            "industry": "其他",
            "market_cap": 0.0,
        }
        for code in codes
    }

    meta_path = CACHE_DIR / "universe" / "csi300_meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            for item in meta.get("stocks", []):
                code = str(item.get("ts_code", ""))
                if code in rows:
                    rows[code]["name"] = item.get("name") or code
        except (OSError, ValueError, TypeError):
            pass

    basic_path = CACHE_DIR / "stock_basic.csv"
    if basic_path.exists():
        try:
            basic = pd.read_csv(basic_path)
            for item in basic.to_dict(orient="records"):
                code = str(item.get("ts_code", ""))
                if code in rows:
                    rows[code]["name"] = item.get("name") or rows[code]["name"]
                    rows[code]["industry"] = item.get("industry") or "其他"
        except (OSError, ValueError, TypeError):
            pass

    for code in rows:
        info_path = CACHE_DIR / f"info_{code.replace('.', '_')}.json"
        if not info_path.exists():
            continue
        try:
            item = json.loads(info_path.read_text(encoding="utf-8"))
            rows[code]["name"] = item.get("name") or rows[code]["name"]
            rows[code]["industry"] = item.get("industry") or rows[code]["industry"]
            rows[code]["market_cap"] = float(item.get("market_cap") or 0.0)
        except (OSError, ValueError, TypeError):
            continue
    return pd.DataFrame(rows.values())
