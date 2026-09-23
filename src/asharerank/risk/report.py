"""风险报告编排：串联数据层与各风险计算模块，生成完整报告（JSON 可序列化）。

口径说明：
- 面向用户的风险指标基于账户 NAV，现金作为零波动资产降低组合风险；
- VaR/CVaR 同时给出占账户 NAV 的比例和金额；
- 报告生成后将关键指标与告警落库（risk_metrics_daily / alerts 表）。
"""
from __future__ import annotations

import logging
import math

import pandas as pd

from asharerank import data_service, db
from asharerank.portfolio import compute_account_state, load_account
from asharerank.risk import alerts as alerts_mod
from asharerank.risk import backtest as backtest_mod
from asharerank.risk import shapley, stress
from asharerank.risk.covariance import condition_number, estimate_cov
from asharerank.risk.factor_model import factor_risk_decomposition
from asharerank.risk.market import align_market
from asharerank.risk.metrics import (
    assess_portfolio_risk,
    compute_var_cvar_all,
    summary_metrics,
    var_historical_bootstrap_ci,
)

logger = logging.getLogger(__name__)


def _sanitize(obj):
    """递归清洗 NaN/Inf，保证 JSON 可序列化。"""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def build_risk_report(account: dict | None = None, cov_method: str = "ledoit_wolf",
                      thresholds: dict | None = None, persist: bool = True) -> dict:
    account = account or load_account()
    codes = [h["ts_code"] for h in account["holdings"]]

    prices = data_service.get_price_history(codes)
    returns = prices.pct_change().dropna()
    info = data_service.get_stock_info(codes)
    market = data_service.get_index_returns()
    index_prices_full = data_service.get_index_history(full=True)["close"]
    stock_prices_full = data_service.get_long_price_history(codes)

    state = compute_account_state(account, prices.iloc[-1])
    pos = state["positions"].loc[returns.columns]
    stock_weights = pos["weight"].values
    nav, cash, stock_value = state["nav"], state["cash"], state["stock_value"]
    stock_exposure = stock_value / nav
    # 按账户 NAV 计算：权重之和等于股票仓位，剩余为零波动现金。
    risk_weights = stock_weights * stock_exposure

    cov = estimate_cov(returns, cov_method)
    summary = summary_metrics(returns, risk_weights, cov)
    var_cvar = compute_var_cvar_all(returns, risk_weights, cov)
    # 主要用户口径在等权历史法和近期加权历史法中取更保守值：
    # 既能快速反映近期波动聚集，也不会因短期平静而忽略长样本尾部损失。
    primary_estimate = {
        confidence: {
            metric: max(
                var_cvar[confidence]["historical"][metric],
                var_cvar[confidence]["weighted_historical"][metric],
            )
            for metric in ("var", "cvar")
        }
        for confidence in ("95", "99")
    }
    returns_m, market_m = align_market(returns, market)
    factor = factor_risk_decomposition(returns_m, info, risk_weights, market_m)
    shap_df = shapley.shapley_risk_attribution(cov, risk_weights)
    stress_df = stress.run_stress_tests(returns, risk_weights, market,
                                        index_prices_full, stock_prices_full,
                                        nav=nav)

    names = dict(zip(info["ts_code"], info["name"]))
    weights_s = pd.Series(risk_weights, index=returns.columns)
    industry_weights = factor["industry_weights"]
    alert_list = alerts_mod.evaluate_alerts(
        var95=primary_estimate["95"]["var"],
        max_dd=summary["max_drawdown"],
        industry_weights=industry_weights,
        latest_returns=returns.iloc[-1],
        weights=weights_s,
        names=names,
        thresholds=thresholds,
    )
    assessment = assess_portfolio_risk(
        cvar95=primary_estimate["95"]["cvar"],
        annualized_volatility=summary["annualized_volatility"],
        max_drawdown_value=summary["max_drawdown"],
        position_weights=risk_weights,
        industry_weights=industry_weights,
        stock_exposure=stock_exposure,
    )

    port_rets: pd.Series = summary["portfolio_returns"]
    drawdown: pd.Series = summary["drawdown_series"]
    nav_series = (1 + port_rets).cumprod()

    # ------------------------------------------------------------------
    # 数据质量：逐持仓行情数据源，按权重计算真实数据覆盖率
    # ------------------------------------------------------------------
    price_sources = data_service.get_price_source_map(returns.columns)
    real_data_weight = float(sum(
        weights_s[c] for c in returns.columns if price_sources.get(c) in ("tushare", "akshare")
    ))

    # ------------------------------------------------------------------
    # VaR 回测：用长历史做样本外滚动检验（权重固定为当前持仓）
    # ------------------------------------------------------------------
    long_returns = stock_prices_full[list(returns.columns)].pct_change().dropna()
    if len(long_returns) > 0:
        weights_bt = pos.loc[long_returns.columns, "weight"].values
        backtest_result = backtest_mod.backtest_var(long_returns, weights_bt)
    else:
        backtest_result = {"window": None, "n_total_obs": 0, "by_confidence": {},
                           "insufficient_data": True}

    # ------------------------------------------------------------------
    # 估计不确定性：历史法 VaR 的 bootstrap 置信区间 + 协方差矩阵诊断
    # ------------------------------------------------------------------
    var95_ci = var_historical_bootstrap_ci(port_rets, 0.95)
    var99_ci = var_historical_bootstrap_ci(port_rets, 0.99)
    uncertainty = {
        "var95_hist_ci": {"low": var95_ci[0], "high": var95_ci[1]},
        "var99_hist_ci": {"low": var99_ci[0], "high": var99_ci[1]},
        "cov_condition_number": condition_number(cov),
        "n_obs": len(returns),
        "n_assets": len(returns.columns),
        "obs_per_asset": len(returns) / len(returns.columns),
    }

    info_idx = info.set_index("ts_code")
    holdings_detail = []
    for code in returns.columns:
        p = pos.loc[code]
        holdings_detail.append({
            "ts_code": code,
            "name": info_idx.loc[code, "name"],
            "industry": info_idx.loc[code, "industry"],
            "shares": float(p["shares"]),
            "cost_price": float(p["cost_price"]),
            "price": float(p["price"]),
            "market_value": float(p["market_value"]),
            "weight": float(p["weight"]),
            "account_weight": float(p["market_value"] / nav),
            "unrealized_pnl": float(p["unrealized_pnl"]),
            "unrealized_pnl_pct": float(p["unrealized_pnl_pct"]),
            "latest_return": float(returns[code].iloc[-1]),
            "shapley_vol": float(shap_df.loc[code, "shapley_vol"]),
            "shapley_share": float(shap_df.loc[code, "share"]),
            "data_source": price_sources.get(code, "synthetic"),
        })

    as_of = str(returns.index[-1].date())
    report = _sanitize({
        "as_of": as_of,
        "data_source": data_service.data_source_status(),
        "data_quality": {
            "overall": data_service.data_source_status(),
            "by_holding": price_sources,
            "real_data_weight": real_data_weight,
        },
        "cov_method": cov_method,
        "n_obs": len(returns),
        "account": {
            "nav": nav,
            "cash": cash,
            "stock_value": stock_value,
            "total_unrealized_pnl": float(pos["unrealized_pnl"].sum()),
        },
        "summary": {
            "daily_volatility": summary["daily_volatility"],
            "annualized_volatility": summary["annualized_volatility"],
            "annualized_return": summary["annualized_return"],
            "sharpe": summary["sharpe"],
            "sortino": summary["sortino"],
            "downside_deviation": summary["downside_deviation"],
            "ulcer_index": summary["ulcer_index"],
            "max_drawdown": summary["max_drawdown"],
        },
        "risk_assessment": assessment,
        "primary_risk_estimate": primary_estimate,
        "var_cvar": var_cvar,
        "var_amounts": {
            "var95_amount": primary_estimate["95"]["var"] * nav,
            "cvar95_amount": primary_estimate["95"]["cvar"] * nav,
            "var99_amount": primary_estimate["99"]["var"] * nav,
            "cvar99_amount": primary_estimate["99"]["cvar"] * nav,
        },
        "factor": {
            "sigma_factor_model": factor["sigma_factor_model"],
            "factor_var_share": factor["factor_var_share"],
            "specific_var_share": factor["specific_var_share"],
            "style_exposures": factor["style_exposures"].to_dict(),
            "industry_exposures": factor["industry_exposures"].to_dict(),
            "industry_weights": factor["industry_weights"].to_dict(),
            "factor_contributions": factor["factor_contributions"].to_dict(),
        },
        "holdings": holdings_detail,
        "stress_tests": stress_df.to_dict(orient="records"),
        "alerts": alert_list,
        "backtest": backtest_result,
        "uncertainty": uncertainty,
        "series": {
            "dates": [str(d.date()) for d in port_rets.index],
            "portfolio_returns": [float(x) for x in port_rets],
            "nav": [float(x) for x in nav_series],
            "drawdown": [float(x) for x in drawdown],
        },
    })

    if persist:
        try:
            db.save_risk_metrics(as_of, {
                "var95_hist": var_cvar["95"]["historical"]["var"],
                "cvar95_hist": var_cvar["95"]["historical"]["cvar"],
                "var99_hist": var_cvar["99"]["historical"]["var"],
                "cvar99_hist": var_cvar["99"]["historical"]["cvar"],
                "annualized_volatility": report["summary"]["annualized_volatility"],
                "max_drawdown": report["summary"]["max_drawdown"],
                "nav": nav,
            })
            db.save_alerts(as_of, alert_list)
            logger.info("风险报告已落库: date=%s alerts=%d", as_of, len(alert_list))
        except Exception:
            logger.exception("风险指标落库失败")

    return report
