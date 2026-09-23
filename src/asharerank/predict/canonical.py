"""统一 Benchmark 的"唯一真值"层：原始前瞻收益 + 股票有效性判定 + 预测帧适配。

背景（架构设计"2→3→4"路线的第 2 步）：树模型（LightGBM/XGBoost）用长表 2D
`(date, ts_code)` 结构、标签是横截面排名分位数；StockMixer 用 `[D,N,F]` 张量、
标签是训练集统计量标准化后的收益率。两者的标签定义、有效样本判定逻辑是各自
独立实现的，直接拿两边的指标数字互相比较并不严谨（可能不是同一批样本、不是
同一段时间、甚至不是同一个"收益率"定义）。

这个模块不改变任何模型现有的训练代码，只提供三个可复用的原语：
  1. `build_canonical_label`   —— 唯一的"原始前瞻收益"计算入口，不做任何模型
     专属变换（不做 rank_norm，不做 z-score，不 clip，不 ffill）。评估阶段用它
     统一覆盖各模型自己算出来的 fwd_ret，确保比较的是同一个数字。
  2. `build_eligibility_mask` / `intersect_eligibility` —— 把"上市年限 + 停牌
     覆盖率"这类过滤规则通用化、可跨模型复用，避免每个模型各自定义一套"什么样
     本算数"的规则。
  3. `to_prediction_frame` —— 唯一的"张量 -> 长表"适配器，把 StockMixer 这类
     `[n_days, N]` 数组形状的输出拍平成 `(date, ts_code, pred, fwd_ret)` 长表，
     使其可以直接复用 `asharerank/predict/evaluate.py` 里树模型那一套
     IC/RankIC/TopK 回测评估代码，不用再维护第二套评估实现。

用法见 `asharerank/predict/benchmark.py`。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_canonical_label(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """原始前瞻收益：fwd_ret_raw = close[t+horizon] / close[t] - 1。

    刻意不做任何下游变换（树模型的 cs_rank_norm、StockMixer 的 z-score/clip 都
    是各自的下游算子，不下沉到这一层），公式与
    `asharerank/predict/dataset.py::make_labels` 里的 `fwd_ret` 完全一致，
    是所有模型评估时"唯一允许使用"的收益率真值来源。
    """
    d = panel.sort_values(["ts_code", "date"]).copy()
    d["date"] = pd.to_datetime(d["date"])
    d["fwd_ret_raw"] = d.groupby("ts_code")["close"].transform(
        lambda x: x.shift(-horizon) / x - 1)
    return d[["date", "ts_code", "fwd_ret_raw"]]


def build_eligibility_mask(panel: pd.DataFrame, *,
                           min_listed_days: int = 250,
                           max_missing_ratio: float = 0.20,
                           listed_cutoff: "pd.Timestamp | str | None" = None) -> pd.DataFrame:
    """通用化自 `stockmixer/data.py::_filter_stocks` 的"上市年限 + 停牌覆盖率"
    过滤逻辑，返回逐 (date, ts_code) 的布尔标签长表。

    与 StockMixer 内部实现的关键区别：这里只打标签、不删行、不抛异常——是否
    使用、以及如何跟其他模型的判定结果取交集，交给调用方（通常是
    `asharerank/predict/benchmark.py`）决定，避免"过滤规则"和"训练逻辑"
    耦合在一起、每个模型各自为政。

    Args:
        listed_cutoff: 判定"上市年限"的截止日期（通常传评估窗口的起点）；
            为 None 时用面板里最新日期兜底。
    """
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    cutoff = pd.Timestamp(listed_cutoff) if listed_cutoff is not None else panel["date"].max()

    hist_counts = panel.loc[panel["date"] < cutoff, "ts_code"].value_counts()
    listed_ok = set(hist_counts[hist_counts >= min_listed_days].index)

    dates_all = sorted(panel["date"].unique())
    n_dates = max(len(dates_all), 1)
    coverage = panel.groupby("ts_code").size() / n_dates
    coverage_ok = set(coverage[coverage >= (1 - max_missing_ratio)].index)

    eligible_codes = listed_ok & coverage_ok
    out = panel[["date", "ts_code"]].drop_duplicates().reset_index(drop=True)
    out["eligible"] = out["ts_code"].isin(eligible_codes)
    return out


def intersect_eligibility(*masks: pd.DataFrame) -> pd.DataFrame:
    """多个模型各自的 eligibility 长表按 (date, ts_code) 求交集
    （所有模型都判定为可用，才算"共同样本"）。"""
    if not masks:
        raise ValueError("至少需要一个 mask")
    merged = masks[0][["date", "ts_code", "eligible"]].rename(columns={"eligible": "eligible_0"})
    for i, m in enumerate(masks[1:], start=1):
        merged = merged.merge(
            m[["date", "ts_code", "eligible"]].rename(columns={"eligible": f"eligible_{i}"}),
            on=["date", "ts_code"], how="inner",
        )
    elig_cols = [c for c in merged.columns if c.startswith("eligible_")]
    merged["eligible"] = merged[elig_cols].all(axis=1)
    return merged[["date", "ts_code", "eligible"]]


def to_prediction_frame(dates: list, codes: list[str],
                        preds: np.ndarray, targets: np.ndarray, masks: np.ndarray) -> pd.DataFrame:
    """把 `[n_days, N]` 形状的预测 / 原始收益 / 有效性 mask 拍平成
    `(date, ts_code, pred, fwd_ret)` 长表 —— 唯一的"张量 -> 长表"转换入口。

    转换之后即可直接喂给 `evaluate.calc_ic_series` / `evaluate.topk_backtest`，
    不需要再为 StockMixer 这类序列/截面模型单独维护一套评估实现。

    Args:
        dates: 长度 n_days，每个 anchor 对应的实际交易日期（不是数组下标）。
        codes: 长度 N，与 preds 第二维顺序一致的股票代码列表。
        preds, targets, masks: 形状均为 `[n_days, N]`；masks 中 1 表示该
            (日, 股票) 样本有效（无停牌/无缺失），0 表示应剔除。
    """
    preds = np.asarray(preds)
    targets = np.asarray(targets)
    masks = np.asarray(masks)
    n_days, n_codes = preds.shape
    if len(dates) != n_days or len(codes) != n_codes:
        raise ValueError(
            f"形状不匹配: len(dates)={len(dates)} vs preds.shape[0]={n_days}, "
            f"len(codes)={len(codes)} vs preds.shape[1]={n_codes}"
        )
    mask_bool = masks > 0.5
    day_idx, code_idx = np.where(mask_bool)
    return pd.DataFrame({
        "date": [pd.Timestamp(dates[i]) for i in day_idx],
        "ts_code": [codes[j] for j in code_idx],
        "pred": preds[day_idx, code_idx],
        "fwd_ret": targets[day_idx, code_idx],
    })


def overwrite_with_canonical_label(frame: pd.DataFrame, canonical_label: pd.DataFrame) -> pd.DataFrame:
    """用 canonical fwd_ret_raw 覆盖 frame 里模型自己算的 fwd_ret，确保多模型评估时
    用的是同一个收益率定义（消除各模型内部 clip/ffill 等细节差异带来的误差）。
    inner join：canonical 里没有的 (date, ts_code) 组合会被自然丢弃。
    """
    out = frame.drop(columns=[c for c in ("fwd_ret",) if c in frame.columns])
    out = out.merge(
        canonical_label.rename(columns={"fwd_ret_raw": "fwd_ret"}),
        on=["date", "ts_code"], how="inner",
    )
    return out
