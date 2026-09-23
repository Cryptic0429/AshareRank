"""统一 Benchmark 的评估合并层（架构路线 2→3→4 的第 3、4 步）。

**只做合并 + 评估**，不做任何模型训练：每个模型的训练动作分别在各自独立的
子进程里完成（`benchmark_stockmixer_step.py` / `benchmark_lgbm_step.py` /
`benchmark_xgboost_step.py` import lightgbm/xgboost 相关代码，
`benchmark_stockmixer_step.py` / `benchmark_lstm_step.py` import torch）——
这是刻意的进程隔离，原因见 `benchmark_stockmixer_step.py` 顶部注释（PyTorch
与 LightGBM/XGBoost 各自绑定的 OpenMP 运行时在同一进程里会在 macOS 上直接
segfault）。本模块因此可以放心地被 `scripts/research/run_benchmark.py` 这个
编排入口 import，不会触发那个冲突。

评估口径：
  - `full_universe`：各模型自己覆盖到的最大样本（用于看模型自身能力上限）。
  - `common_universe`：**参与对比的全部模型**预测都覆盖到的 (date, ts_code)
    交集（唯一允许用来"排名"的口径，避免"样本少但分数高"的模型被误判为更强）。
  - 所有模型的 `fwd_ret` 都会被 `canonical.build_canonical_label` 的结果覆盖，
    确保不是在比较不同定义的"收益率"。

支持任意数量的模型（不再硬编码 StockMixer/LightGBM 两个）：调用方把
`{model_key: PredictionFrame}` 传进来即可，`model_key` 建议用简短英文名
（如 "stockmixer"/"lgbm"/"xgboost"/"lstm"），展示名通过 `display_names` 映射。
"""
from __future__ import annotations

import logging

import pandas as pd

from asharerank.predict import canonical
from asharerank.predict.evaluate import calc_ic_series, precision_at_k, summarize_ic, topk_backtest
from asharerank.predict.experimental.report import save_report
from asharerank.predict.universe import load_panel

logger = logging.getLogger(__name__)

REPORT_NAME = "benchmark_all_models"


def _evaluate(name: str, frame: pd.DataFrame, top_k: int, hold_days: int, cost_bps: float) -> dict:
    ic_df = calc_ic_series(frame, pred_col="pred", label_col="fwd_ret")
    ic_summary = summarize_ic(ic_df)
    bt = topk_backtest(frame, top_k=top_k, hold_days=hold_days, cost_bps=cost_bps)
    # Precision@K：和 Sharpe 用的是同一个"预测 Top-K"，但统计口径不同——Precision@K
    # 逐日看"这一天选出的 K 只里有多少只未来收益为正"（命中率，样本数=交易日数≈112），
    # Sharpe 看的是"每 hold_days 天调仓一次的组合收益序列"（样本数只有 22~23 个调仓点）。
    # 两者对照能区分"选股本身经常选不准"还是"选得还行，只是撞上了小样本里的几次极端亏损"。
    precision = precision_at_k(frame, top_k=top_k, pred_col="pred", label_col="fwd_ret")
    return {
        "model": name,
        "n_days": ic_summary["n_days"],
        "n_pairs": int(frame[["date", "ts_code"]].drop_duplicates().shape[0]),
        "ic_mean": ic_summary["ic_mean"],
        "rank_ic_mean": ic_summary["rank_ic_mean"],
        "icir": ic_summary["icir"],
        "rank_icir": ic_summary["rank_icir"],
        f"precision_at_{top_k}": precision,
        "annual_return": bt["annual_return"],
        "sharpe": bt["sharpe"],
        "max_drawdown": bt["max_drawdown"],
    }


def _format_report(report: dict) -> str:
    lines = [
        f"\n=== 统一 Benchmark：{len(report['full_universe'])} 个模型同窗口/同股票池对照 ===",
        f"测试窗口: {report['test_window'][0]} ~ {report['test_window'][1]}",
        f"股票池: {report['n_codes']} 只 | 共同样本股票池: {report['n_codes_common']} 只",
    ]
    sm_meta = report.get("model_meta", {}).get("stockmixer", {})
    if sm_meta.get("seeds"):
        lines.append(f"StockMixer 口径: {len(sm_meta['seeds'])}-seed 等权集成 {sm_meta['seeds']}")
    elif sm_meta.get("seed") is not None:
        lines.append(f"StockMixer 口径: 单 seed={sm_meta['seed']}（未做多 seed 集成，结果可能受随机性影响较大）")
    top_k = report.get("top_k", 10)
    prec_key = f"precision_at_{top_k}"
    for scope, cn in (("full_universe", "各模型最大覆盖"), ("common_universe", "全部模型交集（唯一允许排名的口径）")):
        lines.append(f"\n--- {scope}（{cn}） ---")
        lines.append(
            f"  {'模型':<28}{'RankIC':<10}{'ICIR':<10}{f'Precision@{top_k}':<14}"
            f"{'Sharpe':<10}{'年化收益':<10}{'覆盖交易日':<10}{'样本数':<10}"
        )
        for m in report[scope].values():
            lines.append(
                f"  {m['model']:<28}{m['rank_ic_mean']:<10.4f}{m['icir']:<10.4f}"
                f"{m.get(prec_key, float('nan')):<14.4f}"
                f"{m['sharpe']:<10.4f}{m['annual_return']:<10.2%}{m['n_days']:<10}{m['n_pairs']:<10}"
            )
    return "\n".join(lines)


def combine_and_evaluate(frames: dict[str, pd.DataFrame], *,
                         display_names: dict[str, str], metas: dict[str, dict] | None = None,
                         codes: list[str], test_start: str, test_end: str,
                         horizon: int, top_k: int = 10, cost_bps: float = 10.0,
                         experiment_config: dict | None = None) -> dict:
    """合并各模型 benchmark 子进程产出的 PredictionFrame，用 canonical 层统一
    fwd_ret，计算全部模型的共同样本交集，最后复用 `evaluate.py` 统一评估——
    全流程不 import torch / lightgbm / xgboost。

    Args:
        frames: `{model_key: (date, ts_code, pred, fwd_ret) 长表}`，至少 1 个。
        display_names: `{model_key: 报告里展示的模型名}`，缺失的 key 直接用 model_key 本身。
        metas: `{model_key: 附加元信息}`（如 StockMixer 的 seeds/best_epoch），可选。
        codes: 统一股票池（用于加载 canonical 标签的 panel，通常传参与对比的全部
            模型代码的并集；哪个模型实际覆盖了多少只，由各自 frame 决定）。
        test_start/test_end: 报告展示用的测试窗口（通常是各模型对齐的公共测试窗口）。
    """
    if not frames:
        raise ValueError("至少需要一个模型的预测结果")
    metas = metas or {}
    test_start_ts = pd.Timestamp(test_start)
    test_end_ts = pd.Timestamp(test_end)

    logger.info("=== [3/4] 用 canonical 层的 fwd_ret 统一覆盖，并计算 %d 个模型的共同样本交集 ===", len(frames))
    panel = load_panel(codes=codes)
    canonical_label = canonical.build_canonical_label(panel, horizon=horizon)
    frames = {k: canonical.overwrite_with_canonical_label(f, canonical_label) for k, f in frames.items()}

    # N 路 (date, ts_code) 交集：所有模型都覆盖到，才算"共同样本"
    common_keys = None
    for f in frames.values():
        keys = f[["date", "ts_code"]].drop_duplicates()
        common_keys = keys if common_keys is None else common_keys.merge(keys, on=["date", "ts_code"], how="inner")
    common_frames = {k: f.merge(common_keys, on=["date", "ts_code"], how="inner") for k, f in frames.items()}
    logger.info("各模型样本数=%s, 共同样本数=%d",
                {k: len(f) for k, f in frames.items()}, len(common_keys))

    logger.info("=== [4/4] 统一评估（evaluate.py 同一套代码，所有模型完全一致） ===")
    report = {
        "test_window": [str(test_start_ts.date()), str(test_end_ts.date())],
        "n_codes": len(codes),
        "n_codes_common": int(common_keys["ts_code"].nunique()) if len(common_keys) else 0,
        "full_universe": {
            k: _evaluate(display_names.get(k, k), f, top_k, horizon, cost_bps)
            for k, f in frames.items()
        },
        "common_universe": {
            k: _evaluate(display_names.get(k, k), f, top_k, horizon, cost_bps)
            for k, f in common_frames.items()
        },
        "model_meta": {k: {kk: vv for kk, vv in m.items() if kk != "codes"} for k, m in metas.items()},
        "cost_bps": cost_bps,
        "top_k": top_k,
    }
    print(_format_report(report))
    report_path = save_report(
        REPORT_NAME,
        report,
        config={
            **(experiment_config or {}),
            "models": list(frames),
            "codes": codes,
            "horizon": horizon,
            "top_k": top_k,
            "cost_bps": cost_bps,
            "test_start": str(test_start_ts.date()),
            "test_end": str(test_end_ts.date()),
            "model_meta": metas,
        },
    )
    run_dir = report_path.parent
    for model, frame in frames.items():
        frame.to_parquet(run_dir / f"predictions_{model}.parquet", index=False)
    for model, frame in common_frames.items():
        frame.to_parquet(run_dir / f"common_predictions_{model}.parquet", index=False)
    return report
