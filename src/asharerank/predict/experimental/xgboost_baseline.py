"""XGBoost 截面收益预测（对比基线，与 LightGBM 同源特征/标签/切分）。

独立于生产训练管线：不写入 runs/models/registry.json，
只用于和 LightGBM baseline 做 RankIC / RankICIR 对比。
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
from asharerank.predict.dataset import build_train_frame, rolling_splits
from asharerank.predict.evaluate import calc_ic_series, summarize_ic
from asharerank.predict.experimental.report import EXPERIMENTAL_DIR, print_report, save_report
from asharerank.predict.universe import load_panel

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 1.0,
    "reg_lambda": 5.0,
    "min_child_weight": 5,
}


def _fit_one_fold(X_tr, y_tr, X_va, y_va, params: dict):
    from asharerank.predict.booster import _ensure_libomp
    _ensure_libomp()  # macOS: 复用 LightGBM 的 libomp 探测（XGBoost 同样依赖 OpenMP）
    from xgboost import XGBRegressor

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        min_child_weight=params["min_child_weight"],
        n_jobs=-1,
        early_stopping_rounds=40,
        eval_metric="rmse",
        random_state=42,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return model


def run_xgboost_baseline(max_stocks: int | None = None,
                         params: dict | None = None,
                         codes: list[str] | None = None,
                         industry_map: dict | None = None,
                         extra_features: pd.DataFrame | None = None,
                         tag: str = "xgboost") -> dict:
    """Walk-forward 训练 + OOS RankIC 汇总；产出与 lgbm baseline 同口径的报告。

    Args:
        codes: 指定股票代码列表（优先于 max_stocks，用于和 extra_features 覆盖范围对齐）。
        industry_map: 传入则做行业中性化标签（见 dataset.make_labels）。
        extra_features: 额外特征表（如基本面因子），见 dataset.build_train_frame。
        tag: 报告文件名前缀，便于同时保留多组对比实验（如 xgboost_fundamentals）。
    """
    params = {**DEFAULT_PARAMS, **(params or {})}

    panel = load_panel()
    if codes:
        panel = panel[panel["ts_code"].isin(codes)]
    elif max_stocks:
        picked = sorted(panel["ts_code"].unique())[:max_stocks]
        panel = panel[panel["ts_code"].isin(picked)]
    n_stocks = panel["ts_code"].nunique()
    logger.info("XGBoost baseline[%s]: %d 只股票", tag, n_stocks)

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
    for i, split in enumerate(splits):
        tr_s, tr_e, va_s, va_e = split
        train = df[(df["date"] >= tr_s) & (df["date"] <= tr_e)]
        valid = df[(df["date"] >= va_s) & (df["date"] <= va_e)]
        if len(train) < 1000 or len(valid) < 200:
            continue
        X_tr, y_tr = train[feature_cols].values, train["label"].values
        X_va, y_va = valid[feature_cols].values, valid["label"].values

        model = _fit_one_fold(X_tr, y_tr, X_va, y_va, params)
        last_model = model
        pred = model.predict(X_va)
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
            {"model": last_model, "feature_names": feature_cols, "backend": "xgboost"},
            EXPERIMENTAL_DIR / f"{tag}_last_fold.joblib",
        )
    oos_path = EXPERIMENTAL_DIR / f"{tag}_oos_predictions.parquet"
    oos.to_parquet(oos_path, index=False)

    report = {
        "model_type": "XGBoost",
        "tag": tag,
        "n_stocks": n_stocks,
        "n_samples": len(df),
        "n_folds": len(fold_metrics),
        "horizon": PREDICT_HORIZON,
        "params": params,
        "oos_metrics": oos_ic,
        "fold_metrics": fold_metrics,
        "fold_splits": fold_splits,
        "oos_predictions_path": str(oos_path),
    }
    save_report(tag, report)
    print_report("XGBoost", oos_ic, fold_metrics, extra={"n_stocks": n_stocks, "n_samples": len(df)})
    return report
