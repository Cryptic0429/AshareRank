"""LightGBM 对比基线（复用生产 booster.py，与 xgboost_baseline 同口径，供集成/消融实验用）。

独立于生产训练管线（asharerank.predict.train）：不写入 runs/models/registry.json，
产物落在 runs/experiments/，专供 XGBoost / LSTM / 特征消融做同口径对比。
"""
from __future__ import annotations

import logging

import joblib
import pandas as pd

from asharerank.settings import (
    LGBM_STEP_MONTHS,
    LGBM_TRAIN_YEARS,
    LGBM_VALID_MONTHS,
    PREDICT_HORIZON,
)
from asharerank.predict.booster import train_booster
from asharerank.predict.dataset import build_train_frame, rolling_splits
from asharerank.predict.evaluate import calc_ic_series, summarize_ic
from asharerank.predict.experimental.report import EXPERIMENTAL_DIR, print_report, save_report
from asharerank.predict.universe import load_panel

logger = logging.getLogger(__name__)


def run_lgbm_baseline(max_stocks: int | None = None,
                      params: dict | None = None,
                      codes: list[str] | None = None,
                      industry_map: dict | None = None,
                      extra_features: pd.DataFrame | None = None,
                      tag: str = "lgbm") -> dict:
    """Walk-forward 训练 + OOS RankIC 汇总；与 xgboost_baseline 参数/产出结构对齐。"""
    panel = load_panel()
    if codes:
        panel = panel[panel["ts_code"].isin(codes)]
    elif max_stocks:
        picked = sorted(panel["ts_code"].unique())[:max_stocks]
        panel = panel[panel["ts_code"].isin(picked)]
    n_stocks = panel["ts_code"].nunique()
    logger.info("LightGBM baseline[%s]: %d 只股票", tag, n_stocks)

    df = build_train_frame(panel, horizon=PREDICT_HORIZON,
                           industry_map=industry_map, extra_features=extra_features)
    feature_cols = df.attrs["feature_cols"]
    dates = sorted(df["date"].unique())
    splits = rolling_splits(
        dates, LGBM_TRAIN_YEARS, LGBM_VALID_MONTHS, LGBM_STEP_MONTHS,
        purge_days=PREDICT_HORIZON,
    )
    if not splits:
        raise ValueError("无法生成滚动窗口，历史数据过短")

    fold_preds, fold_metrics, fold_splits = [], [], []
    last_model = None
    backend_used = None
    for i, split in enumerate(splits):
        tr_s, tr_e, va_s, va_e = split
        train = df[(df["date"] >= tr_s) & (df["date"] <= tr_e)]
        valid = df[(df["date"] >= va_s) & (df["date"] <= va_e)]
        if len(train) < 1000 or len(valid) < 200:
            continue
        X_tr, y_tr = train[feature_cols].values, train["label"].values
        X_va, y_va = valid[feature_cols].values, valid["label"].values

        booster = train_booster(X_tr, y_tr, X_va, y_va, feature_cols, params=params)
        backend_used = booster.backend
        last_model = booster
        pred = booster.predict(X_va)
        part = valid[["date", "ts_code", "fwd_ret", "label"]].copy()
        part["pred"] = pred
        fold_preds.append(part)
        fold_splits.append(split.as_dict())

        ic_sum = summarize_ic(calc_ic_series(part, label_col="fwd_ret"))
        fold_metrics.append(ic_sum)
        logger.info("fold %d RankIC=%.4f RankICIR=%.3f",
                    i + 1, ic_sum["rank_ic_mean"], ic_sum["rank_icir"])

    if not fold_preds:
        raise ValueError("所有滚动窗口样本不足，请增加股票数或拉长历史")

    oos = pd.concat(fold_preds, ignore_index=True)
    oos_ic = summarize_ic(calc_ic_series(oos, label_col="fwd_ret"))

    if last_model is not None:
        joblib.dump(
            {"booster": last_model, "feature_names": feature_cols},
            EXPERIMENTAL_DIR / f"{tag}_last_fold.joblib",
        )
    oos_path = EXPERIMENTAL_DIR / f"{tag}_oos_predictions.parquet"
    oos.to_parquet(oos_path, index=False)

    report = {
        "model_type": "LightGBM",
        "tag": tag,
        "backend": backend_used,
        "n_stocks": n_stocks,
        "n_samples": len(df),
        "n_folds": len(fold_metrics),
        "horizon": PREDICT_HORIZON,
        "oos_metrics": oos_ic,
        "fold_metrics": fold_metrics,
        "fold_splits": fold_splits,
        "oos_predictions_path": str(oos_path),
    }
    save_report(tag, report)
    print_report("LightGBM", oos_ic, fold_metrics,
                 extra={"n_stocks": n_stocks, "n_samples": len(df), "backend": backend_used})
    return report
