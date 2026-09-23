"""简化多因子风险模型（Barra 风格）。

因子体系：
- 行业因子：来自股票真实行业分类（0/1 哑变量暴露）
- 风格因子：SIZE（对数市值）、MOM（12-1 动量）、VOL（历史波动率）、
  BETA（对市场组合回归的 beta），均横截面标准化

模型：r_t = X f_t + ε_t（横截面回归逐日估计因子收益 f_t），
组合方差 σ_p² = w'X F X'w + w'D w，
因子 k 暴露 b_k = (X'w)_k，因子风险贡献 = b_k (F b)_k / σ_p²。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from asharerank.risk.market import estimate_betas

STYLE_FACTORS = ["SIZE", "MOM", "VOL", "BETA"]


def build_exposures(returns: pd.DataFrame, info: pd.DataFrame,
                    market: pd.Series) -> pd.DataFrame:
    """构建暴露矩阵 X（行=资产，列=因子）。风格因子横截面 z-score。
    market 为市场指数（沪深300）日收益序列。"""
    codes = list(returns.columns)
    info = info.set_index("ts_code").loc[codes]

    # 风格因子原始值
    size = np.log(info["market_cap"].astype(float).clip(lower=1))
    mom = returns.iloc[-252:-21].add(1).prod() - 1 if len(returns) >= 252 else \
        returns.iloc[:-21].add(1).prod() - 1
    vol = returns.rolling(60).std().iloc[-1]
    beta = estimate_betas(returns, market, shrink=True)

    style = pd.DataFrame({"SIZE": size, "MOM": mom, "VOL": vol, "BETA": beta})
    style = (style - style.mean()) / style.std(ddof=0).replace(0, 1)

    industry = pd.get_dummies(info["industry"]).astype(float)
    industry.columns = [f"IND_{c}" for c in industry.columns]

    X = pd.concat([style, industry], axis=1).fillna(0.0)
    return X


def estimate_factor_returns(returns: pd.DataFrame, X: pd.DataFrame,
                            ridge: float = 0.1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """逐日横截面岭回归估计因子收益，返回 (因子收益序列, 残差)。

    加岭正则的原因：演示组合资产数 N 可能小于因子数 K（欠定），
    普通最小二乘会完美拟合、残差恒为零，导致特有风险被低估为 0；
    岭回归同时缓解行业哑变量与风格因子的共线性。
    """
    Xv = X.values
    n, k = Xv.shape
    lam = ridge * np.trace(Xv.T @ Xv) / k
    solve_mat = np.linalg.solve(Xv.T @ Xv + lam * np.eye(k), Xv.T)  # K × N
    F_rets = returns.values @ solve_mat.T     # T × K
    resid = returns.values - F_rets @ Xv.T    # T × N
    f_df = pd.DataFrame(F_rets, index=returns.index, columns=X.columns)
    e_df = pd.DataFrame(resid, index=returns.index, columns=returns.columns)
    return f_df, e_df


def factor_risk_decomposition(returns: pd.DataFrame, info: pd.DataFrame,
                              weights: np.ndarray, market: pd.Series) -> dict:
    """完整因子风险分解。返回暴露、因子贡献、特有风险占比等。"""
    X = build_exposures(returns, info, market)
    f_rets, resid = estimate_factor_returns(returns, X)

    F = f_rets.cov().values                    # 因子协方差 K×K
    D = np.diag(resid.var().values)            # 特有风险对角阵 N×N
    Xv = X.values

    port_expo = Xv.T @ weights                 # b = X'w，K 维
    factor_var = float(port_expo @ F @ port_expo)
    specific_var = float(weights @ D @ weights)
    total_var = factor_var + specific_var
    sigma_p = np.sqrt(total_var)

    # 单因子边际贡献：b_k * (F b)_k / σ_p²
    marginal = port_expo * (F @ port_expo)
    contrib = pd.Series(marginal / total_var, index=X.columns)

    exposures = pd.Series(port_expo, index=X.columns)
    style_expo = exposures[STYLE_FACTORS]
    industry_expo = exposures[[c for c in X.columns if c.startswith("IND_")]]
    industry_expo.index = [c[4:] for c in industry_expo.index]

    # 行业权重占比（市值权重视角）
    ind_weights = pd.Series(weights, index=returns.columns).groupby(
        info.set_index("ts_code")["industry"]).sum()

    return {
        "sigma_factor_model": float(sigma_p),
        "factor_var_share": factor_var / total_var,
        "specific_var_share": specific_var / total_var,
        "style_exposures": style_expo,
        "industry_exposures": industry_expo,
        "industry_weights": ind_weights,
        "factor_contributions": contrib.sort_values(ascending=False),
        "exposure_matrix": X,
    }
