"""旧 LightGBM 路线使用的 IC / RankIC / ICIR 兼容函数。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from asharerank.settings import TRADING_DAYS_PER_YEAR


def daily_ic(pred: pd.Series, label: pd.Series, method: str = "pearson") -> float:
    if len(pred) < 5 or pred.nunique() < 2 or label.nunique() < 2:
        return np.nan
    if method == "spearman":
        return float(stats.spearmanr(pred, label).correlation)
    return float(stats.pearsonr(pred, label)[0])


def calc_ic_series(df: pd.DataFrame, pred_col: str = "pred",
                   label_col: str = "fwd_ret",
                   group_cols: tuple[str, ...] = ("date",)) -> pd.DataFrame:
    """按 group_cols 分组算截面 IC/RankIC。默认按日（全市场）；
    传 group_cols=("date","industry") 可算行业内 IC（行业中性化效果的评估口径）。
    """
    group_cols = list(group_cols)
    rows = []
    for key, g in df.groupby(group_cols):
        ic = daily_ic(g[pred_col], g[label_col], "pearson")
        ric = daily_ic(g[pred_col], g[label_col], "spearman")
        row = {"ic": ic, "rank_ic": ric}
        keys = key if isinstance(key, tuple) else (key,)
        row.update(dict(zip(group_cols, keys)))
        rows.append(row)
    return pd.DataFrame(rows).dropna(subset=["ic", "rank_ic"])


def calc_industry_neutral_ic(df: pd.DataFrame, industry_map: dict,
                             pred_col: str = "pred",
                             label_col: str = "fwd_ret") -> pd.DataFrame:
    """行业内 RankIC：同一天 + 同一行业内部排序的预测力（剔除行业 beta 影响后的纯 alpha 信号）。"""
    d = df.copy()
    d["industry"] = d["ts_code"].map(industry_map).fillna("未分类")
    return calc_ic_series(d, pred_col=pred_col, label_col=label_col,
                          group_cols=("date", "industry"))


def summarize_ic(ic_df: pd.DataFrame) -> dict:
    if ic_df.empty:
        return {"ic_mean": 0.0, "rank_ic_mean": 0.0, "icir": 0.0, "rank_icir": 0.0, "n_days": 0}
    ic_mean = float(ic_df["ic"].mean())
    ric_mean = float(ic_df["rank_ic"].mean())
    ic_std = float(ic_df["ic"].std(ddof=0)) or 1e-9
    ric_std = float(ic_df["rank_ic"].std(ddof=0)) or 1e-9
    return {
        "ic_mean": ic_mean,
        "rank_ic_mean": ric_mean,
        "icir": ic_mean / ic_std,
        "rank_icir": ric_mean / ric_std,
        "n_days": int(len(ic_df)),
    }



def precision_at_k(df: pd.DataFrame, top_k: int = 10,
                   pred_col: str = "pred", label_col: str = "fwd_ret") -> float:
    """Precision@K：逐日取预测 Top-K，未来 horizon 日真实收益(label_col)为正的占比，
    再对所有交易日取均值。

    定义与 `experimental/stockmixer/metrics.py::compute_metrics` 里的 Precision@K
    完全一致（"预测 Top-K 中收益为正的比例"，逐日命中率，不是"预测 Top-K 与真实
    Top-K 的重合度"），这样 StockMixer 和树模型/LSTM 基线才能在同一个定义下直接
    比较，而不是报告里各用各的口径。
    """
    precisions = []
    for _, g in df.groupby("date"):
        if len(g) < top_k:
            continue
        top = g.nlargest(top_k, pred_col)
        precisions.append(float((top[label_col] > 0).mean()))
    return float(np.mean(precisions)) if precisions else float("nan")


def _topk_backtest_one_offset(df: pd.DataFrame, dates: list, offset: int,
                              top_k: int, hold_days: int, cost_bps: float) -> list[dict]:
    """从 dates[offset] 开始、每 hold_days 天调仓一次的单条回测曲线。"""
    nav = 1.0
    curve = []
    prev_holdings: set = set()
    for d in dates[offset::hold_days]:
        g = df[df["date"] == d].nlargest(top_k, "pred")
        holdings = set(g["ts_code"])
        # 用 fwd_ret 作为持有期收益（与标签 horizon 一致时最干净）
        period_ret = float(g["fwd_ret"].mean()) if "fwd_ret" in g else 0.0
        turnover = 1.0 if not prev_holdings else (
            len(holdings - prev_holdings) + len(prev_holdings - holdings)
        ) / max(len(holdings | prev_holdings), 1)
        cost = turnover * cost_bps / 10000
        nav *= (1 + period_ret - cost)
        curve.append({"date": d, "nav": nav, "period_ret": period_ret - cost})
        prev_holdings = holdings
    return curve


def topk_backtest(df: pd.DataFrame, top_k: int = 30, hold_days: int = 5,
                  cost_bps: float = 10.0) -> dict:
    """等权 TopK，每 hold_days 调仓一次的多头组合回测；成本按换手近似扣减。

    ---- 2026-07-23 口径修正（重要）----
    旧版有三个系统性问题，会把结果算得"过于漂亮"且和
    `stockmixer/metrics.py::compute_metrics` 的口径互相矛盾（同一份预测两边
    Sharpe 能差出好几倍）：
      1. Sharpe 旧版用"复利年化收益 / 年化波动"，期均收益为正时系统性偏大
         （实测同一序列 2.60 vs 标准估计 1.94）。现改为标准估计
         `mean/std * sqrt(periods_per_year)`，与 compute_metrics 一致。
      2. annual_return 旧版用复利 `(1+mean)^P - 1`（期均收益 1.1% 时 76% vs
         算术 57%）。现改为算术年化 `mean * P`，与 compute_metrics 的
         AnnualizedReturn 同口径。
      3. 非重叠调仓序列只有 n_days/hold_days ≈ 20 出头个样本，结果对"从哪一天
         开始调仓"极其敏感（实测同一份预测，5 个起始偏移的 Sharpe 从 0.07 摆到
         2.30）。现对全部 hold_days 个起始偏移各回测一遍取平均作为主指标，并在
         `sharpe_per_offset` / `annual_return_per_offset` 里保留各偏移的原始值，
         方便看离散度——离散度大本身就是"样本太少、别过度解读"的警示。
    `nav_curve` 取 offset=0 那条（画图用）。
    """
    df = df.sort_values(["date", "pred"], ascending=[True, False])
    dates = sorted(df["date"].unique())
    periods_per_year = TRADING_DAYS_PER_YEAR / hold_days

    ann_list, sharpe_list, mdd_list = [], [], []
    nav_curve_offset0: list[dict] = []
    for offset in range(min(hold_days, max(len(dates), 1))):
        curve = _topk_backtest_one_offset(df, dates, offset, top_k, hold_days, cost_bps)
        if offset == 0:
            nav_curve_offset0 = curve
        if len(curve) < 2:
            continue
        s = pd.DataFrame(curve)
        rets = s["period_ret"]
        mean, std = float(rets.mean()), float(rets.std(ddof=0))
        ann_list.append(mean * periods_per_year)
        sharpe_list.append(mean / std * np.sqrt(periods_per_year) if std > 1e-12 else 0.0)
        peak = s["nav"].cummax()
        mdd_list.append(float((1 - s["nav"] / peak).max()))

    if not ann_list:
        return {"annual_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0, "nav_curve": [],
                "annual_return_per_offset": [], "sharpe_per_offset": []}
    return {
        "annual_return": float(np.mean(ann_list)),
        "sharpe": float(np.mean(sharpe_list)),
        "max_drawdown": float(np.mean(mdd_list)),
        "nav_curve": nav_curve_offset0,
        "annual_return_per_offset": [round(x, 6) for x in ann_list],
        "sharpe_per_offset": [round(x, 4) for x in sharpe_list],
    }
