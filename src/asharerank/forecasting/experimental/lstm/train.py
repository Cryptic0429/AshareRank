"""LSTM 对比基线：与 LightGBM/XGBoost 同源特征/标签/切分，walk-forward + RankIC。"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from asharerank.settings import (
    LGBM_STEP_MONTHS,
    LGBM_TRAIN_YEARS,
    LGBM_VALID_MONTHS,
    PREDICT_HORIZON,
)
from asharerank.forecasting.dataset import build_train_frame, rolling_splits
from asharerank.forecasting.evaluate import calc_ic_series, summarize_ic
from asharerank.forecasting.experimental.lstm.config import DEFAULT_LSTM_CONFIG, LSTMConfig
from asharerank.forecasting.experimental.lstm.dataset import (
    SequencePanel,
    WindowDataset,
    build_sequence_panel,
)
from asharerank.forecasting.experimental.lstm.model import LSTMRankPredictor
from asharerank.forecasting.experimental.report import EXPERIMENTAL_DIR, print_report, save_report
from asharerank.forecasting.universe import load_panel

logger = logging.getLogger(__name__)


def _train_one_fold(model: LSTMRankPredictor, train_entries, panel: SequencePanel,
                    cfg: LSTMConfig, device: torch.device) -> None:
    loader = DataLoader(WindowDataset(panel, train_entries), batch_size=cfg.batch_size,
                        shuffle=True, drop_last=False)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()
    model.train()
    for epoch in range(cfg.epochs_per_fold):
        running, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(loss.item()) * x.shape[0]
            n += x.shape[0]
        logger.debug("  epoch %d/%d train_mse=%.6f", epoch + 1, cfg.epochs_per_fold, running / max(n, 1))


@torch.no_grad()
def _predict(model: LSTMRankPredictor, entries, panel: SequencePanel,
            cfg: LSTMConfig, device: torch.device) -> np.ndarray:
    loader = DataLoader(WindowDataset(panel, entries), batch_size=cfg.batch_size, shuffle=False)
    model.eval()
    preds = []
    for x, _ in loader:
        x = x.to(device)
        preds.append(model(x).cpu().numpy())
    return np.concatenate(preds) if preds else np.array([])


def run_lstm_baseline(max_stocks: int | None = None,
                      cfg: LSTMConfig | None = None) -> dict:
    cfg = cfg or DEFAULT_LSTM_CONFIG
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    panel_df = load_panel()
    if max_stocks:
        codes = sorted(panel_df["ts_code"].unique())[:max_stocks]
        panel_df = panel_df[panel_df["ts_code"].isin(codes)]
    n_stocks = panel_df["ts_code"].nunique()
    logger.info("LSTM baseline: %d 只股票, window=%d", n_stocks, cfg.window)

    df = build_train_frame(panel_df, horizon=PREDICT_HORIZON)
    feature_cols = df.attrs["feature_cols"]
    dates = sorted(df["date"].unique())
    splits = rolling_splits(
        dates, LGBM_TRAIN_YEARS, LGBM_VALID_MONTHS, LGBM_STEP_MONTHS,
        purge_days=PREDICT_HORIZON,
    )
    if not splits:
        raise ValueError("无法生成滚动窗口，历史数据过短")

    seq_panel = build_sequence_panel(df, feature_cols, window=cfg.window)
    logger.info("序列样本总数: %d（原始截面行数: %d）", len(seq_panel.index), len(df))

    fold_preds, fold_metrics, fold_splits = [], [], []
    last_model = None
    for i, split in enumerate(splits):
        tr_s, tr_e, va_s, va_e = split
        train_entries = seq_panel.subset_by_date(tr_s, tr_e)
        valid_entries = seq_panel.subset_by_date(va_s, va_e)
        if len(train_entries) < cfg.min_train_rows or len(valid_entries) < cfg.min_valid_rows:
            continue

        model = LSTMRankPredictor(len(feature_cols), cfg.hidden_size,
                                  cfg.num_layers, cfg.dropout).to(device)
        _train_one_fold(model, train_entries, seq_panel, cfg, device)
        last_model = model

        pred = _predict(model, valid_entries, seq_panel, cfg, device)
        part = seq_panel.meta_frame(valid_entries)
        part["pred"] = pred
        fold_preds.append(part)
        fold_splits.append(split.as_dict())

        ic_sum = summarize_ic(calc_ic_series(part, label_col="fwd_ret"))
        fold_metrics.append(ic_sum)
        logger.info("fold %d RankIC=%.4f RankICIR=%.3f (train=%d valid=%d)",
                    i + 1, ic_sum["rank_ic_mean"], ic_sum["rank_icir"],
                    len(train_entries), len(valid_entries))

    if not fold_preds:
        raise ValueError("所有滚动窗口序列样本不足，请增加股票数、拉长历史或调小 window")

    oos = pd.concat(fold_preds, ignore_index=True)
    oos_ic = summarize_ic(calc_ic_series(oos, label_col="fwd_ret"))

    if last_model is not None:
        torch.save(last_model.state_dict(), EXPERIMENTAL_DIR / "lstm_last_fold.pt")

    report = {
        "model_type": "LSTM",
        "n_stocks": n_stocks,
        "n_sequences": len(seq_panel.index),
        "n_folds": len(fold_metrics),
        "horizon": PREDICT_HORIZON,
        "window": cfg.window,
        "hidden_size": cfg.hidden_size,
        "oos_metrics": oos_ic,
        "fold_metrics": fold_metrics,
        "fold_splits": fold_splits,
    }
    save_report("lstm", report)
    print_report("LSTM", oos_ic, fold_metrics,
                 extra={"n_stocks": n_stocks, "n_sequences": len(seq_panel.index), "window": cfg.window})
    return report
