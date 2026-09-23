"""持仓管理：账户模型（股数 + 可选成本价 + 现金），权重为派生量。

存储结构（JSON）：
{
  "cash": 100000.0,
  "holdings": [{"ts_code": "600519.SH", "shares": 100, "cost_price": 1450.0}, ...]
}
"""
from __future__ import annotations

import json

import pandas as pd

from asharerank.settings import PROCESSED_DATA_DIR

PORTFOLIO_FILE = PROCESSED_DATA_DIR / "portfolio.json"

DEFAULT_ACCOUNT = {
    "cash": 80000.0,
    "holdings": [
        {"ts_code": "600519.SH", "shares": 100,   "cost_price": 1400.0},
        {"ts_code": "300750.SZ", "shares": 400,   "cost_price": 260.0},
        {"ts_code": "601318.SH", "shares": 1800,  "cost_price": 52.0},
        {"ts_code": "600036.SH", "shares": 2500,  "cost_price": 38.0},
        {"ts_code": "000858.SZ", "shares": 700,   "cost_price": 135.0},
        {"ts_code": "601899.SH", "shares": 5000,  "cost_price": 18.0},
        {"ts_code": "002415.SZ", "shares": 2500,  "cost_price": 31.0},
        {"ts_code": "600900.SH", "shares": 3300,  "cost_price": 28.0},
        {"ts_code": "601012.SH", "shares": 4000,  "cost_price": 17.0},
        {"ts_code": "000333.SZ", "shares": 1200,  "cost_price": 72.0},
    ],
}


def load_account() -> dict:
    if PORTFOLIO_FILE.exists():
        data = json.loads(PORTFOLIO_FILE.read_text())
        # 旧格式（纯权重）迁移：直接替换为默认账户
        if isinstance(data, list) or "holdings" not in data or (
                data["holdings"] and "shares" not in data["holdings"][0]):
            save_account(DEFAULT_ACCOUNT)
            return DEFAULT_ACCOUNT
        return data
    save_account(DEFAULT_ACCOUNT)
    return DEFAULT_ACCOUNT


def save_account(account: dict) -> None:
    PORTFOLIO_FILE.parent.mkdir(exist_ok=True)
    PORTFOLIO_FILE.write_text(json.dumps(account, ensure_ascii=False, indent=2))


def compute_account_state(account: dict, latest_prices: pd.Series) -> dict:
    """由最新价计算账户状态：市值、权重（派生）、浮动盈亏、NAV。

    返回 {"nav", "cash", "stock_value", "positions": DataFrame}。
    positions 列：ts_code, shares, cost_price, price, market_value,
                  weight, unrealized_pnl, unrealized_pnl_pct
    """
    rows = []
    for h in account["holdings"]:
        code = h["ts_code"]
        price = float(latest_prices[code])
        # 未填写成本价时以当前价作为展示基准，使浮动盈亏为 0；
        # 风险计算始终只使用股数和当前市价，不受该展示字段影响。
        cost_price = float(h.get("cost_price") or price)
        mv = h["shares"] * price
        cost = h["shares"] * cost_price
        rows.append({
            "ts_code": code,
            "shares": h["shares"],
            "cost_price": cost_price,
            "price": price,
            "market_value": mv,
            "unrealized_pnl": mv - cost,
            "unrealized_pnl_pct": mv / cost - 1 if cost > 0 else 0.0,
        })
    pos = pd.DataFrame(rows).set_index("ts_code")
    stock_value = float(pos["market_value"].sum())
    cash = float(account.get("cash", 0.0))
    nav = stock_value + cash
    if stock_value <= 0:
        raise ValueError("持仓市值必须为正")
    # 风险计算口径：股票部分的权重（现金无市场风险，单独列示）
    pos["weight"] = pos["market_value"] / stock_value
    return {"nav": nav, "cash": cash, "stock_value": stock_value, "positions": pos}
