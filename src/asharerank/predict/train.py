"""兼容旧接口的 LightGBM 滚动训练流水线（仅预测指标，不含回测）。"""
from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import pandas as pd

from asharerank.settings import (
    LGBM_STEP_MONTHS,
    LGBM_TRAIN_YEARS,
    LGBM_VALID_MONTHS,
    PREDICT_HORIZON,
)
from asharerank.predict.booster import get_backend, train_booster
from asharerank.data.splits import purge_tail_dates
from asharerank.predict.dataset import build_train_frame, rolling_splits
from asharerank.predict.evaluate import calc_ic_series, summarize_ic
from asharerank.predict.registry import save_model
from asharerank.predict.universe import load_panel

logger = logging.getLogger(__name__)

# In-process training state for command-line orchestration and logs.
_JOB_STATE: dict = {"status": "idle", "progress": 0.0, "message": "", "result": None}


def get_train_status() -> dict:
    return dict(_JOB_STATE)


def _set_state(**kwargs):
    _JOB_STATE.update(kwargs)


def run_training(max_stocks: int | None = None,
                 model_id: str | None = None,
                 top_k: int = 30) -> dict:
    """完整训练：读数据 → 特征标签 → 滚动 CV → 全样本终训 → 注册。"""
    _set_state(status="running", progress=0.05, message="加载行情面板…", result=None)
    try:
        panel = load_panel()
        if max_stocks:
            codes = sorted(panel["ts_code"].unique())[:max_stocks]
            panel = panel[panel["ts_code"].isin(codes)]
        n_stocks = panel["ts_code"].nunique()
        logger.info("面板: %d 只股票, %d 行", n_stocks, len(panel))

        _set_state(progress=0.15, message=f"构建特征与标签（{n_stocks} 只）…")
        df = build_train_frame(panel, horizon=PREDICT_HORIZON)
        feature_cols = df.attrs["feature_cols"]
        dates = sorted(df["date"].unique())
        splits = rolling_splits(
            dates, LGBM_TRAIN_YEARS, LGBM_VALID_MONTHS, LGBM_STEP_MONTHS,
            purge_days=PREDICT_HORIZON,
        )
        if not splits:
            raise ValueError("无法生成滚动窗口，历史数据过短")

        backend = get_backend()
        fold_preds = []
        fold_metrics = []
        fold_splits = []
        for i, split in enumerate(splits):
            tr_s, tr_e, va_s, va_e = split
            _set_state(progress=0.2 + 0.5 * i / len(splits),
                       message=f"滚动训练 {i+1}/{len(splits)} ({backend})…")
            train = df[(df["date"] >= tr_s) & (df["date"] <= tr_e)]
            valid = df[(df["date"] >= va_s) & (df["date"] <= va_e)]
            if len(train) < 1000 or len(valid) < 200:
                continue
            X_tr = train[feature_cols].values
            y_tr = train["label"].values
            X_va = valid[feature_cols].values
            y_va = valid["label"].values
            booster = train_booster(X_tr, y_tr, X_va, y_va, feature_cols)
            pred = booster.predict(X_va)
            part = valid[["date", "ts_code", "fwd_ret", "label"]].copy()
            part["pred"] = pred
            fold_preds.append(part)
            fold_splits.append(split.as_dict())
            ic_sum = summarize_ic(calc_ic_series(part, label_col="fwd_ret"))
            fold_metrics.append(ic_sum)
            logger.info("fold %d RankIC=%.4f ICIR=%.3f", i + 1,
                        ic_sum["rank_ic_mean"], ic_sum["rank_icir"])

        if not fold_preds:
            raise ValueError("所有滚动窗口样本不足，请增加股票数或拉长历史")

        oos = pd.concat(fold_preds, ignore_index=True)
        oos_ic = summarize_ic(calc_ic_series(oos, label_col="fwd_ret"))
        _set_state(progress=0.8, message="终训全样本模型…")
        # 用最后一段 valid 之前的全部数据做终训
        last_valid_start = splits[-1].valid_start
        train_final = df[df["date"] < last_valid_start]
        valid_final = df[(df["date"] >= last_valid_start)]
        if len(valid_final) < 200:
            # 兜底：时间切 90/10
            cut = dates[int(len(dates) * 0.9)]
            train_final = df[df["date"] < cut]
            valid_final = df[df["date"] >= cut]
        train_final, purged_final_train_dates = purge_tail_dates(
            train_final, PREDICT_HORIZON
        )
        if train_final.empty or valid_final.empty:
            raise ValueError("终训切分在五日标签清除后为空，请增加历史数据")
        booster = train_booster(
            train_final[feature_cols].values, train_final["label"].values,
            valid_final[feature_cols].values, valid_final["label"].values,
            feature_cols,
        )

        model_id = model_id or f"lgbm_{datetime.now().strftime('%Y%m%d_%H%M')}"
        metrics = dict(oos_ic)
        meta = save_model(
            model_id, booster, feature_cols, metrics,
            extra={
                "n_stocks": n_stocks,
                "n_samples": len(df),
                "n_folds": len(fold_metrics),
                "horizon": PREDICT_HORIZON,
                "fold_metrics": fold_metrics,
                "fold_splits": fold_splits,
                "final_fit_split": {
                    "train_start": str(pd.Timestamp(train_final["date"].min()).date()),
                    "train_end": str(pd.Timestamp(train_final["date"].max()).date()),
                    "valid_start": str(pd.Timestamp(valid_final["date"].min()).date()),
                    "valid_end": str(pd.Timestamp(valid_final["date"].max()).date()),
                    "purge_days": PREDICT_HORIZON,
                    "purged_train_dates": [str(value.date()) for value in purged_final_train_dates],
                },
            },
        )
        result = {
            "model_id": model_id,
            "backend": backend,
            "metrics": metrics,
            "meta": meta,
        }
        _set_state(status="done", progress=1.0, message="训练完成", result=result)
        return result
    except Exception as exc:
        logger.exception("训练失败")
        _set_state(status="error", progress=0.0, message=str(exc), result=None)
        raise
