"""VaR 回测：滚动窗口历史模拟法 VaR 与实际收益比较，Kupiec POF 覆盖率检验。

动机：报告里的三种 VaR（参数法/历史法/蒙特卡洛法）都取自同一段历史窗口，
互相之间数字接近并不代表 VaR 模型本身准确——真正检验 VaR 有效性的行业惯例
（巴塞尔框架）是样本外滚动回测：用 t 时点之前的数据估计 VaR，检查 t 时点
实际收益是否击穿该 VaR，统计违约率是否落在统计上可接受的区间内。

方法：
- 用当前持仓权重回放到尽量长的历史（含个股上市前 NaN 已由调用方对齐），
  对每个时点用其前 window 天数据估计历史法 VaR（不使用未来信息），
  与该时点实际收益比较，得到「样本外」违约序列；
- Kupiec POF（Proportion of Failures）检验：H0 为「违约率等于名义显著性水平」，
  p 值 > 0.05 视为模型校准良好（不能拒绝 H0）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

MIN_TEST_OBS = 60  # 样本外观测数低于此值时，回测结果不具统计意义


def rolling_historical_var(port_rets: pd.Series, window: int, confidence: float) -> pd.Series:
    """滚动窗口历史模拟 VaR：t 时点的 VaR 仅用 [t-window, t) 的历史数据估计。"""
    values = port_rets.values
    n = len(values)
    q = 1 - confidence
    out = np.full(n, np.nan)
    for t in range(window, n):
        hist = values[t - window:t]
        out[t] = -np.quantile(hist, q)
    return pd.Series(out, index=port_rets.index)


def kupiec_pof_test(n: int, x: int, p: float) -> dict:
    """Kupiec 覆盖率检验（似然比检验，渐近服从自由度为 1 的卡方分布）。

    n: 样本外观测数；x: 违约（击穿 VaR）次数；p: 名义违约率（如 95% VaR 对应 0.05）。
    """
    if n == 0:
        return {"lr_stat": None, "p_value": None, "pass": None}
    x_hat = x / n
    if x == 0:
        lr = -2 * n * np.log(1 - p)
    elif x == n:
        lr = -2 * n * np.log(p)
    else:
        lr = -2 * (
            (n - x) * np.log(1 - p) + x * np.log(p)
            - (n - x) * np.log(1 - x_hat) - x * np.log(x_hat)
        )
    p_value = float(1 - stats.chi2.cdf(lr, df=1))
    return {
        "lr_stat": float(lr),
        "p_value": p_value,
        # 不能在 5% 显著性水平下拒绝「违约率符合预期」这一原假设，即视为校准良好
        "pass": bool(p_value > 0.05),
    }


def backtest_var(returns: pd.DataFrame, weights: np.ndarray,
                 window: int = 250,
                 confidences: tuple[float, ...] = (0.95, 0.99)) -> dict:
    """对当前持仓权重在尽量长的历史区间上做滚动 VaR 回测（walk-forward）。

    注：权重假设在整段历史中保持不变，是「若长期持有当前组合，VaR 模型
    历史上是否校准良好」的诊断，而非账户真实历史回测。
    """
    port = pd.Series(returns.values @ weights, index=returns.index)
    n_total = len(port)
    result: dict = {
        "window": window,
        "n_total_obs": n_total,
        "by_confidence": {},
        "insufficient_data": n_total < window + MIN_TEST_OBS,
    }
    if result["insufficient_data"]:
        return result

    for c in confidences:
        p = 1 - c
        var_series = rolling_historical_var(port, window, c)
        valid = var_series.dropna()
        actual = port.loc[valid.index]
        breach_mask = (actual < -valid).values
        n = len(valid)
        x = int(breach_mask.sum())
        test = kupiec_pof_test(n, x, p)
        breach_dates = valid.index[breach_mask]
        result["by_confidence"][f"{int(c * 100)}"] = {
            "n_obs": n,
            "n_violations": x,
            "violation_rate": x / n if n else None,
            "expected_rate": p,
            "expected_violations": n * p,
            "kupiec": test,
            "recent_violation_dates": [str(d.date()) for d in breach_dates[-10:]],
        }
    return result
