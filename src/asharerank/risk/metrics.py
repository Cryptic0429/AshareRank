"""组合风险指标：VaR/CVaR、下行风险、回撤与可解释风险评分。

约定：VaR/CVaR 以正数表示损失比例（占组合净值），日度口径。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from asharerank.settings import MC_N_SIMULATIONS, TRADING_DAYS_PER_YEAR


def portfolio_returns(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    return pd.Series(returns.values @ weights, index=returns.index, name="portfolio")


# ---------------------------------------------------------------------------
# VaR / CVaR
# ---------------------------------------------------------------------------

def var_cvar_parametric(mu: float, sigma: float, confidence: float) -> tuple[float, float]:
    """参数法（正态假设）。VaR_c = -(μ + σ Φ^{-1}(1-c))；
    CVaR_c = -(μ - σ φ(z)/(1-c))，z = Φ^{-1}(1-c)。"""
    z = stats.norm.ppf(1 - confidence)
    var = -(mu + sigma * z)
    cvar = -(mu - sigma * stats.norm.pdf(z) / (1 - confidence))
    return max(var, 0.0), max(cvar, 0.0)


def var_cvar_historical(port_rets: pd.Series, confidence: float) -> tuple[float, float]:
    """历史模拟法：经验分位数 VaR，分位以下平均损失为 CVaR。"""
    q = np.quantile(port_rets, 1 - confidence)
    tail = port_rets[port_rets <= q]
    cvar = -tail.mean() if len(tail) > 0 else -q
    return max(-q, 0.0), max(float(cvar), 0.0)


def var_cvar_weighted_historical(
    port_rets: pd.Series,
    confidence: float,
    decay: float = 0.97,
) -> tuple[float, float]:
    """时间加权历史法：越近的行情权重越高。

    相比等权历史法，该方法能更快反映波动率聚集和近期市场状态变化。
    VaR 取加权累计概率首次达到尾部概率的收益，CVaR 为该点及以下的加权均值。
    """
    values = np.asarray(port_rets, dtype=float)
    if values.size == 0:
        return 0.0, 0.0
    if not 0 < decay <= 1:
        raise ValueError("decay 必须介于 0 和 1 之间")

    chronological_weights = decay ** np.arange(values.size - 1, -1, -1)
    chronological_weights /= chronological_weights.sum()
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = chronological_weights[order]
    cutoff = 1 - confidence
    idx = min(int(np.searchsorted(np.cumsum(sorted_weights), cutoff, side="left")),
              values.size - 1)
    q = float(sorted_values[idx])
    tail_mask = sorted_values <= q
    tail_weights = sorted_weights[tail_mask]
    tail_mean = float(np.average(sorted_values[tail_mask], weights=tail_weights))
    return max(-q, 0.0), max(-tail_mean, 0.0)


def var_cvar_monte_carlo(returns: pd.DataFrame, weights: np.ndarray,
                         cov: pd.DataFrame, confidence: float,
                         n_sim: int = MC_N_SIMULATIONS,
                         seed: int = 42) -> tuple[float, float]:
    """蒙特卡洛法：多元正态模拟资产日收益，聚合为组合收益后取经验分位。"""
    rng = np.random.default_rng(seed)
    mu = returns.mean().values
    sims = rng.multivariate_normal(mu, cov.values, size=n_sim, method="cholesky")
    port = sims @ weights
    q = np.quantile(port, 1 - confidence)
    tail = port[port <= q]
    cvar = -tail.mean() if len(tail) > 0 else -q
    return max(-q, 0.0), max(float(cvar), 0.0)


def var_historical_bootstrap_ci(port_rets: pd.Series, confidence: float,
                                n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    """历史法 VaR 的自助法（bootstrap）抽样不确定性区间。

    对组合历史收益做有放回重抽样 n_boot 次，每次重算历史 VaR，
    取抽样分布的 5%~95% 分位作为置信区间，反映样本量有限带来的估计误差。
    区间越宽，说明当前历史窗口对该 VaR 的估计越不稳定。
    """
    rng = np.random.default_rng(seed)
    vals = port_rets.values
    n = len(vals)
    if n < 30:
        return 0.0, 0.0
    q = 1 - confidence
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_q = np.quantile(vals[idx], q, axis=1)
    boot_var = -boot_q
    lo, hi = np.percentile(boot_var, [5, 95])
    return max(float(lo), 0.0), max(float(hi), 0.0)


def compute_var_cvar_all(returns: pd.DataFrame, weights: np.ndarray,
                         cov: pd.DataFrame,
                         confidences: tuple[float, ...] = (0.95, 0.99)) -> dict:
    """三种方法 × 多置信度 的 VaR/CVaR 汇总。"""
    port = portfolio_returns(returns, weights)
    mu = float(port.mean())
    sigma = float(np.sqrt(weights @ cov.values @ weights))
    out = {}
    for c in confidences:
        key = f"{int(c * 100)}"
        v_p, cv_p = var_cvar_parametric(mu, sigma, c)
        v_h, cv_h = var_cvar_historical(port, c)
        v_w, cv_w = var_cvar_weighted_historical(port, c)
        v_m, cv_m = var_cvar_monte_carlo(returns, weights, cov, c)
        out[key] = {
            "parametric": {"var": v_p, "cvar": cv_p},
            "historical": {"var": v_h, "cvar": cv_h},
            "weighted_historical": {"var": v_w, "cvar": cv_w},
            "monte_carlo": {"var": v_m, "cvar": cv_m},
        }
    return out


# ---------------------------------------------------------------------------
# 其他指标
# ---------------------------------------------------------------------------

def annualized_volatility(sigma_daily: float) -> float:
    return sigma_daily * np.sqrt(TRADING_DAYS_PER_YEAR)


def max_drawdown(port_rets: pd.Series) -> tuple[float, pd.Series]:
    """返回 (最大回撤, 回撤序列)，回撤为正数比例。"""
    nav = (1 + port_rets).cumprod()
    # 把投资起点 1.0 纳入历史高点，否则样本首日下跌会被错认为 0 回撤。
    peak = nav.cummax().clip(lower=1.0)
    drawdown = 1 - nav / peak
    return float(drawdown.max()), drawdown


def summary_metrics(returns: pd.DataFrame, weights: np.ndarray,
                    cov: pd.DataFrame) -> dict:
    port = portfolio_returns(returns, weights)
    sigma_daily = float(np.sqrt(weights @ cov.values @ weights))
    mdd, drawdown = max_drawdown(port)
    ann_ret = float(port.mean() * TRADING_DAYS_PER_YEAR)
    ann_vol = annualized_volatility(sigma_daily)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    downside = port.clip(upper=0.0)
    downside_dev = float(np.sqrt((downside ** 2).mean()) * np.sqrt(TRADING_DAYS_PER_YEAR))
    sortino = ann_ret / downside_dev if downside_dev > 0 else 0.0
    ulcer_index = float(np.sqrt((drawdown ** 2).mean()))
    return {
        "daily_volatility": sigma_daily,
        "annualized_volatility": ann_vol,
        "annualized_return": ann_ret,
        "sharpe": sharpe,
        "sortino": sortino,
        "downside_deviation": downside_dev,
        "ulcer_index": ulcer_index,
        "max_drawdown": mdd,
        "drawdown_series": drawdown,
        "portfolio_returns": port,
    }


def _bounded_score(value: float, severe_at: float) -> float:
    """将风险量映射到 0-100，severe_at 对应 100 分。"""
    if severe_at <= 0:
        return 0.0
    return float(np.clip(value / severe_at * 100, 0, 100))


def assess_portfolio_risk(
    *,
    cvar95: float,
    annualized_volatility: float,
    max_drawdown_value: float,
    position_weights: np.ndarray,
    industry_weights: pd.Series,
    stock_exposure: float,
) -> dict:
    """生成 0-100 可解释组合风险评分（越高越危险）。

    评分不使用短期收益方向，避免把“最近在赚钱”错当成“低风险”。五个分量分别衡量：
    尾部损失 30%、历史回撤 25%、波动 20%、单一持仓集中 15%、行业集中 10%。
    position_weights / industry_weights 均应为占账户 NAV 的权重，因此现金能正确降低风险。
    """
    position_weights = np.asarray(position_weights, dtype=float)
    max_position = float(position_weights.max()) if position_weights.size else 0.0
    max_industry = float(industry_weights.max()) if len(industry_weights) else 0.0
    risky_weights = position_weights / stock_exposure if stock_exposure > 0 else position_weights
    hhi = float(np.square(risky_weights).sum())
    effective_holdings = float(1 / hhi) if hhi > 0 else 0.0

    components = {
        "tail_loss": _bounded_score(cvar95, 0.06),
        "drawdown": _bounded_score(max_drawdown_value, 0.35),
        "volatility": _bounded_score(annualized_volatility, 0.40),
        "position_concentration": _bounded_score(max_position, 0.40),
        "industry_concentration": _bounded_score(max_industry, 0.50),
    }
    component_weights = {
        "tail_loss": 0.30,
        "drawdown": 0.25,
        "volatility": 0.20,
        "position_concentration": 0.15,
        "industry_concentration": 0.10,
    }
    score = int(round(sum(components[k] * component_weights[k] for k in components)))
    if score <= 25:
        level = "low"
    elif score <= 50:
        level = "moderate"
    elif score <= 75:
        level = "high"
    else:
        level = "very_high"

    weighted_contributions = {
        key: components[key] * component_weights[key] for key in components
    }
    drivers = sorted(weighted_contributions, key=weighted_contributions.get, reverse=True)[:3]
    return {
        "score": score,
        "level": level,
        "components": components,
        "component_weights": component_weights,
        "weighted_contributions": weighted_contributions,
        "top_drivers": drivers,
        "stock_exposure": float(stock_exposure),
        "max_position_weight": max_position,
        "max_industry_weight": max_industry,
        "concentration_hhi": hhi,
        "effective_holdings": effective_holdings,
    }
