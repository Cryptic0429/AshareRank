"""评估指标：IC / RankIC(RIC) / Precision@K / 夏普比率（Sharpe Ratio）。

均逐个交易日计算后再在时间维上汇总，这也是量化选股模型评估的标准做法
（截面相关性 -> 时间序列均值/标准差）。

---- 2026-07-23 修复（重要）----
`label_horizon > 1`（例如预测未来 5 日累计收益）时，"组合收益"序列里的每一个
样本实际上是一段 5 日持有期的收益，而不是 1 日收益。如果还按 1 日收益的方式
用 `sqrt(252)` / `*252` 做年化，会把 Sharpe 和年化收益都放大约 sqrt(horizon)
和 horizon 倍——这正是第一版报告里 Sharpe=2.54、年化收益=168% 这类"过于漂亮"
的数字的来源，并不是模型真的这么强。这里改成用
`periods_per_year = trading_days_per_year / horizon` 做年化，
并额外提供一份"非重叠抽样"（每 horizon 天取一次，避免相邻样本共享同一段
价格路径导致的自相关/方差低估）版本，作为更保守、更接近真实可执行策略的估计。
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def compute_metrics(preds: np.ndarray, targets: np.ndarray, masks: np.ndarray,
                    top_k: int = 10, trading_days_per_year: int = 252,
                    horizon: int = 1) -> dict:
    """
    Args:
        preds, targets, masks: 形状均为 [n_days, N]（masks 为 0/1）。
        top_k: Precision@K 与多头组合的持仓数。
        horizon: 标签的前瞻天数（次日收益率=1；预测未来 5 日收益则传 5），
            用于修正 Sharpe / 年化收益的年化系数——不影响 IC/RankIC/Precision@K
            （这三者是逐日横截面相关性/命中率，本身与持有期长度无关）。

    Returns:
        dict，包含 IC / RankIC / ICIR / Precision@K / Sharpe / 覆盖天数等。
        当 horizon > 1 时额外包含"非重叠抽样"版本的 Sharpe / 年化收益。
    """
    horizon = max(horizon, 1)
    ic_list: list[float] = []
    ric_list: list[float] = []
    prec_list: list[float] = []
    port_ret_list: list[float] = []  # 每日"做多预测 Top-K，等权持有 horizon 日"的组合收益
    day_idx_list: list[int] = []      # 对应的锚点日在 preds 数组里的行号，用于非重叠抽样

    n_days = preds.shape[0]
    for t in range(n_days):
        valid = masks[t] > 0.5
        if valid.sum() < max(top_k, 5):
            continue
        p, g = preds[t][valid], targets[t][valid]
        if np.std(p) < 1e-12 or np.std(g) < 1e-12:
            continue

        # ---- IC：预测值与真实收益的 Pearson 相关系数 ----
        ic = np.corrcoef(p, g)[0, 1]
        # ---- RankIC / RIC：预测值与真实收益的 Spearman 秩相关系数 ----
        ric, _ = spearmanr(p, g)
        if np.isnan(ic) or np.isnan(ric):
            continue
        ic_list.append(ic)
        ric_list.append(ric)

        # ---- Precision@K：预测 Top-K 中未来 horizon 日真实收益为正的占比 ----
        top_idx = np.argsort(-p)[:top_k]
        prec_list.append(float(np.mean(g[top_idx] > 0)))

        # ---- 组合收益：等权做多 Top-K，持有 horizon 日，用于计算 Sharpe ----
        port_ret_list.append(float(np.mean(g[top_idx])))
        day_idx_list.append(t)

    ic_arr = np.array(ic_list, dtype=np.float64)
    ric_arr = np.array(ric_list, dtype=np.float64)
    port_arr = np.array(port_ret_list, dtype=np.float64)
    day_arr = np.array(day_idx_list, dtype=np.int64)

    # 每年约有 trading_days_per_year / horizon 个"不重叠的 horizon 日持有周期"，
    # 这才是这段收益序列对应的正确年化频率（horizon=1 时退化为原来的 252）。
    periods_per_year = trading_days_per_year / horizon

    sharpe = 0.0
    if len(port_arr) > 1 and port_arr.std() > 1e-12:
        sharpe = float(port_arr.mean() / port_arr.std() * np.sqrt(periods_per_year))
    ann_return = float(port_arr.mean() * periods_per_year) if len(port_arr) else float("nan")

    # ---- 非重叠抽样：每 horizon 天只取一个样本，避免相邻锚点日共享同一段
    # 价格路径导致的自相关（会人为压低标准差、抬高 Sharpe），更接近真实可执行的
    # "到期后再开新仓"策略。样本量会降到约 1/horizon，仅在 horizon>1 时计算。----
    non_overlap_sharpe = float("nan")
    non_overlap_ann_return = float("nan")
    n_non_overlap = len(port_arr)
    if horizon > 1 and len(day_arr) > 1:
        keep_mask = np.zeros(len(day_arr), dtype=bool)
        last_kept_day = None
        for i in range(len(day_arr) - 1, -1, -1):  # 从最新的一天往前，保证互不重叠
            d = day_arr[i]
            if last_kept_day is None or last_kept_day - d >= horizon:
                keep_mask[i] = True
                last_kept_day = d
        sub_port = port_arr[keep_mask]
        n_non_overlap = int(keep_mask.sum())
        if len(sub_port) > 1 and sub_port.std() > 1e-12:
            non_overlap_sharpe = float(sub_port.mean() / sub_port.std() * np.sqrt(periods_per_year))
        if len(sub_port) >= 1:
            non_overlap_ann_return = float(sub_port.mean() * periods_per_year)

    icir = float(ic_arr.mean() / (ic_arr.std() + 1e-12)) if len(ic_arr) else float("nan")

    return {
        "IC": float(ic_arr.mean()) if len(ic_arr) else float("nan"),
        "RankIC": float(ric_arr.mean()) if len(ric_arr) else float("nan"),
        "ICIR": icir,
        f"Precision@{top_k}": float(np.mean(prec_list)) if prec_list else float("nan"),
        "Sharpe": sharpe,
        f"AnnualizedReturn@{top_k}": ann_return,
        "Sharpe_NonOverlap": non_overlap_sharpe,
        f"AnnualizedReturn_NonOverlap@{top_k}": non_overlap_ann_return,
        "horizon": horizon,
        "n_days": int(len(ic_arr)),
        "n_non_overlap_days": n_non_overlap,
    }


def format_metrics(name: str, m: dict) -> str:
    top_key = next((k for k in m if k.startswith("Precision@")), "Precision@K")
    ann_key = next((k for k in m if k.startswith("AnnualizedReturn@")), "AnnualizedReturn@K")
    horizon = m.get("horizon", 1)
    lines = [
        f"\n=== {name} 评估指标（覆盖 {m.get('n_days', 0)} 个交易日，标签horizon={horizon}日） ===",
        f"  IC (Information Coefficient)      : {m.get('IC', float('nan')):.4f}",
        f"  RIC / RankIC (秩相关系数)          : {m.get('RankIC', float('nan')):.4f}",
        f"  ICIR                                : {m.get('ICIR', float('nan')):.4f}",
        f"  {top_key:<34}: {m.get(top_key, float('nan')):.4f}",
        f"  Sharpe Ratio（年化，Top-K 多头组合）: {m.get('Sharpe', float('nan')):.4f}",
        f"  {ann_key:<34}: {m.get(ann_key, float('nan')):.2%}",
    ]
    if horizon > 1:
        lines.append(
            f"  —— 以上按 {horizon} 日持有期正确年化；下面是每 {horizon} 天取一次样本的"
            f"非重叠抽样版本（样本数 {m.get('n_non_overlap_days', 0)}，更保守/更接近真实可执行策略）——"
        )
        lines.append(f"  Sharpe（非重叠抽样）                : {m.get('Sharpe_NonOverlap', float('nan')):.4f}")
        ann_nonoverlap = m.get(f"AnnualizedReturn_NonOverlap@{top_key.split('@')[-1]}", float("nan"))
        lines.append(f"  年化收益（非重叠抽样）              : {ann_nonoverlap:.2%}")
    return "\n".join(lines)


# ============================================================
# 多 seed 复现：汇总各 seed 的测试集指标，计算均值/标准差，
# 用于判断单次结果（比如某一次 RankIC=0.097）是否只是"运气好"。
# ============================================================
_MULTISEED_KEYS = ("IC", "RankIC", "ICIR", "Sharpe")


def summarize_multiseed(per_seed_metrics: list[dict], top_k: int = 10) -> dict:
    """对多个 seed 的测试集指标做汇总统计（均值 / 标准差 / 最小值 / 最大值）。

    Args:
        per_seed_metrics: 每个 seed 一份 `compute_metrics` 返回的 dict（额外带 "seed" 字段）。
        top_k: 用于定位 `Precision@{top_k}` 字段名。

    Returns:
        dict，key 为指标名，value 为 {"mean", "std", "min", "max", "values"}。
        额外包含 "positive_rate"（RankIC > 0 的 seed 占比，用于判断信号方向是否稳定）。
    """
    prec_key = f"Precision@{top_k}"
    keys = list(_MULTISEED_KEYS) + [prec_key]
    summary: dict = {}
    for k in keys:
        vals = np.array([m[k] for m in per_seed_metrics if k in m and not np.isnan(m[k])], dtype=np.float64)
        summary[k] = {
            "mean": float(vals.mean()) if len(vals) else float("nan"),
            "std": float(vals.std()) if len(vals) else float("nan"),
            "min": float(vals.min()) if len(vals) else float("nan"),
            "max": float(vals.max()) if len(vals) else float("nan"),
            "values": vals.tolist(),
        }
    ric_vals = summary["RankIC"]["values"]
    summary["RankIC"]["positive_rate"] = (
        float(np.mean([v > 0 for v in ric_vals])) if ric_vals else float("nan")
    )
    return summary


def format_multiseed_summary(name: str, per_seed_metrics: list[dict], summary: dict, top_k: int = 10) -> str:
    prec_key = f"Precision@{top_k}"
    lines = [f"\n=== {name}：{len(per_seed_metrics)} 个随机种子汇总 ==="]
    lines.append(
        f"  {'seed':<8}{'best_epoch':<12}{'val_RankIC':<12}{'IC':<10}{'RankIC':<10}"
        f"{'ICIR':<10}{prec_key:<14}{'Sharpe':<10}"
    )
    for m in per_seed_metrics:
        lines.append(
            f"  {m.get('seed', ''):<8}{m.get('best_epoch', ''):<12}{m.get('best_val_rankic', float('nan')):<12.4f}"
            f"{m.get('IC', float('nan')):<10.4f}{m.get('RankIC', float('nan')):<10.4f}"
            f"{m.get('ICIR', float('nan')):<10.4f}{m.get(prec_key, float('nan')):<14.4f}"
            f"{m.get('Sharpe', float('nan')):<10.4f}"
        )
    lines.append("  ------------------------------------------------------------------------")
    for k in (*_MULTISEED_KEYS, prec_key):
        s = summary.get(k, {})
        extra = ""
        if k == "RankIC":
            extra = f"  | RankIC>0 占比={s.get('positive_rate', float('nan')):.0%}"
        lines.append(
            f"  {k:<10}: mean={s.get('mean', float('nan')):.4f}  std={s.get('std', float('nan')):.4f}  "
            f"min={s.get('min', float('nan')):.4f}  max={s.get('max', float('nan')):.4f}{extra}"
        )
    ric_mean = summary.get("RankIC", {}).get("mean", float("nan"))
    ric_std = summary.get("RankIC", {}).get("std", float("nan"))
    if not np.isnan(ric_mean) and not np.isnan(ric_std) and ric_std > 1e-9:
        lines.append(f"\n  RankIC 变异系数（std/|mean|）= {ric_std / max(abs(ric_mean), 1e-9):.2f}"
                    "（越小越说明信号稳定，>1 通常意味着方向/量级都不可靠）")
    return "\n".join(lines)
