"""告警规则引擎：对风险报告应用阈值规则，输出告警列表。"""
from __future__ import annotations

import pandas as pd

DEFAULT_THRESHOLDS = {
    "var95_limit": 0.03,          # 95% 日 VaR 超过净值 3%
    "max_drawdown_limit": 0.20,   # 历史最大回撤超过 20%
    "industry_concentration": 0.35,  # 单行业权重超过 35%
    "single_stock_daily_loss": 0.05,  # 单只持仓单日跌幅超过 5%
    "single_stock_weight": 0.25,  # 单只持仓权重超过 25%
}


def evaluate_alerts(*, var95: float, max_dd: float,
                    industry_weights: pd.Series,
                    latest_returns: pd.Series,
                    weights: pd.Series,
                    names: dict[str, str],
                    thresholds: dict | None = None) -> list[dict]:
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    alerts: list[dict] = []

    if var95 > th["var95_limit"]:
        alerts.append({"level": "高", "rule": "VaR 超限",
                       "message": f"95% 日 VaR 为 {var95:.2%}，超过阈值 {th['var95_limit']:.0%}"})

    if max_dd > th["max_drawdown_limit"]:
        alerts.append({"level": "中", "rule": "最大回撤超限",
                       "message": f"历史最大回撤 {max_dd:.2%}，超过阈值 {th['max_drawdown_limit']:.0%}"})

    for industry, w in industry_weights.items():
        if w > th["industry_concentration"]:
            alerts.append({"level": "中", "rule": "行业集中度过高",
                           "message": f"行业「{industry}」权重 {w:.1%}，超过阈值 {th['industry_concentration']:.0%}"})

    for code, r in latest_returns.items():
        if r < -th["single_stock_daily_loss"]:
            alerts.append({"level": "高", "rule": "持仓单日大幅亏损",
                           "message": f"{names.get(code, code)}（{code}）最新单日下跌 {-r:.2%}，超过阈值 {th['single_stock_daily_loss']:.0%}"})

    for code, w in weights.items():
        if w > th["single_stock_weight"]:
            alerts.append({"level": "低", "rule": "个股权重过高",
                           "message": f"{names.get(code, code)}（{code}）权重 {w:.1%}，超过阈值 {th['single_stock_weight']:.0%}"})

    return alerts
