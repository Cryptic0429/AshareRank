"""Benchmark 子步骤：LSTM 对照模型，训练/验证/测试边界与 StockMixer 完全对齐。

**必须在独立进程里运行**（本模块 import torch），原因见
`benchmark_stockmixer_step.py` 顶部注释（PyTorch + LightGBM/XGBoost 的 OpenMP
运行时在同一进程里会在 macOS 上直接 segfault）。可以和 StockMixer 共用同一个
torch 子进程，但 `run_benchmark.py` 仍让每个模型各起一个独立子进程，保持
"一个模型一个子进程"的一致模式。

与 `experimental/lstm/train.py` 的 walk-forward baseline 的关键区别：
  1. 数据切分：这里是固定单次切分，train/val/test 边界直接复用 StockMixer 的
     `date_bounds`，而不是多年滚动窗口——对齐的是"同一批股票、同一段训练历史、
     同一个早停验证窗口、同一个最终测试窗口"。
  2. 训练目标：与 StockMixer 一致的组合损失（MSE + λ·Pairwise Rank Loss + 逐日
     横截面标准化，复用 `stockmixer/losses.py::combined_loss`，不重新实现一遍）。
  3. 稳定性：与 StockMixer 一致的"权重 EMA + 早停判据用滑动平均 val_RankIC"
     （详见 `stockmixer/train.py` 里 2026-07-23 的实测记录与注释）。
  4. 序列窗口长度默认与 StockMixer 的 `--lookback` 对齐（而不是 baseline 自己的
     `window=20`），确保"回看多少天历史"这个超参数也是同一个数字。

---- 2026-07-23 补充（关键改动）----
第一版实现里 DataLoader 用 `shuffle=True` + 打平的 `(股票, 日期)` 样本，逐样本
算 MSE——这样"同一个 batch 里混进了不同交易日的股票"，天然没法算 Pairwise Rank
Loss（排序损失的前提是"同一天内两两比较"）。四模型 benchmark 实测发现 LSTM 的
Precision@10（0.446）几乎等于当期市场基准命中率（0.449），是四个模型里唯一
"零选股能力"的——怀疑正是因为它只学了"预测各股票各自的绝对标签值"，完全没有
显式的跨股票排序信号。这里改成按交易日分组构造 batch（`_build_day_batches`），
让 LSTM 也能用上和 StockMixer 一样的排序损失。
"""
from __future__ import annotations

import copy
import logging
from collections import deque

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from asharerank.settings import PREDICT_HORIZON
from asharerank.forecasting.dataset import build_train_frame
from asharerank.forecasting.evaluate import calc_ic_series, summarize_ic
from asharerank.forecasting.experimental.lstm.dataset import SequencePanel, WindowDataset, build_sequence_panel
from asharerank.forecasting.experimental.lstm.model import LSTMRankPredictor
from asharerank.forecasting.experimental.stockmixer.losses import combined_loss
from asharerank.forecasting.experimental.stockmixer.train import _ema_update_, get_device, set_seed
from asharerank.forecasting.universe import load_panel

logger = logging.getLogger(__name__)

DEFAULT_HIDDEN_SIZE = 64
DEFAULT_NUM_LAYERS = 1
DEFAULT_DROPOUT = 0.1
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-5
DEFAULT_BATCH_SIZE = 256
# 组合损失超参数，直接沿用 StockMixer 实测调优过的默认值（见 stockmixer/config.py 注释）
DEFAULT_RANK_LOSS_WEIGHT = 0.5
DEFAULT_CROSS_SECTIONAL_NORM = True
DEFAULT_EMA_DECAY = 0.9
DEFAULT_RANKIC_SMOOTH_WINDOW = 3


def _build_day_batches(panel: SequencePanel, entries: list[tuple[str, int]]
                       ) -> list[tuple[np.ndarray, np.ndarray]]:
    """把 (code, position) 索引按"日期"分组，拼成 `[N_day, window, F]` 的截面
    batch——Pairwise Rank Loss / 横截面标准化都要求"同一天的股票放一起比较"，
    原来打平按 (股票,日期) 随机 shuffle 的 DataLoader 做不到这一点。
    """
    by_date: dict[np.datetime64, list[tuple[str, int]]] = {}
    for code, i in entries:
        d = panel.per_stock[code]["date"][i]
        by_date.setdefault(d, []).append((code, i))

    w = panel.window
    batches = []
    for d in sorted(by_date):
        ents = by_date[d]
        x = np.stack([panel.per_stock[c]["feat"][i - w + 1:i + 1] for c, i in ents]).astype(np.float32)
        y = np.array([panel.per_stock[c]["label"][i] for c, i in ents], dtype=np.float32)
        batches.append((x, y))
    return batches


def _purge_tail_entries(
    panel: SequencePanel,
    entries: list[tuple[str, int]],
    horizon: int,
) -> tuple[list[tuple[str, int]], tuple[pd.Timestamp, ...]]:
    """Drop anchors whose forward labels can cross the end of a time partition."""
    if horizon < 0:
        raise ValueError("horizon 不能为负数")
    dates = sorted({
        pd.Timestamp(panel.per_stock[code]["date"][position])
        for code, position in entries
    })
    removed = tuple(dates[-horizon:]) if horizon else ()
    removed_set = set(removed)
    kept = [
        entry for entry in entries
        if pd.Timestamp(panel.per_stock[entry[0]]["date"][entry[1]]) not in removed_set
    ]
    return kept, removed


def run_lstm_step(codes: list[str], *, train_start: str, train_end: str,
                  val_start: str, val_end: str, test_start: str, test_end: str,
                  horizon: int = PREDICT_HORIZON, window: int = 16,
                  epochs: int = 60, patience: int = 10, seed: int = 42,
                  rank_loss_weight: float = DEFAULT_RANK_LOSS_WEIGHT,
                  cross_sectional_norm: bool = DEFAULT_CROSS_SECTIONAL_NORM,
                  ema_decay: float = DEFAULT_EMA_DECAY,
                  rankic_smooth_window: int = DEFAULT_RANKIC_SMOOTH_WINDOW,
                  device: str | None = None) -> pd.DataFrame:
    """返回测试窗口内 `(date, ts_code, pred, fwd_ret)` 长表。

    训练目标/稳定性策略与 StockMixer 对齐（见模块顶部 2026-07-23 注释）：
    按交易日分组构造 batch → 组合损失（MSE + λ·Pairwise Rank Loss + 横截面标准化）
    → 权重 EMA → 早停判据用最近 `rankic_smooth_window` 轮 val_RankIC 的滑动平均。
    """
    set_seed(seed)
    dev = get_device(device)

    train_start_ts, train_end_ts = pd.Timestamp(train_start), pd.Timestamp(train_end)
    val_start_ts, val_end_ts = pd.Timestamp(val_start), pd.Timestamp(val_end)
    test_start_ts, test_end_ts = pd.Timestamp(test_start), pd.Timestamp(test_end)

    panel = load_panel(codes=codes)
    df = build_train_frame(panel, horizon=horizon)
    feature_cols = df.attrs["feature_cols"]

    seq_panel = build_sequence_panel(df, feature_cols, window=window)
    train_entries = seq_panel.subset_by_date(train_start_ts, train_end_ts)
    val_entries = seq_panel.subset_by_date(val_start_ts, val_end_ts)
    test_entries = seq_panel.subset_by_date(test_start_ts, test_end_ts)
    train_entries, purged_train_dates = _purge_tail_entries(seq_panel, train_entries, horizon)
    val_entries, purged_val_dates = _purge_tail_entries(seq_panel, val_entries, horizon)
    if not train_entries or not val_entries or not test_entries:
        raise ValueError(
            f"benchmark LSTM 序列样本不足（train={len(train_entries)}, val={len(val_entries)}, "
            f"test={len(test_entries)}），请检查窗口长度/训练历史是否够长"
        )
    logger.info("[benchmark/LSTM] 序列样本：train=%d val=%d test=%d，window=%d",
                len(train_entries), len(val_entries), len(test_entries), window)
    logger.info("[benchmark/LSTM] 标签边界 purge：train=%d 天，val=%d 天",
                len(purged_train_dates), len(purged_val_dates))

    train_batches = _build_day_batches(seq_panel, train_entries)
    logger.info("[benchmark/LSTM] 按交易日分组后训练 batch 数=%d（每个 batch=1 个交易日的截面）",
                len(train_batches))

    model = LSTMRankPredictor(len(feature_cols), DEFAULT_HIDDEN_SIZE,
                              DEFAULT_NUM_LAYERS, DEFAULT_DROPOUT).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=DEFAULT_LR, weight_decay=DEFAULT_WEIGHT_DECAY)
    val_loader = DataLoader(WindowDataset(seq_panel, val_entries), batch_size=DEFAULT_BATCH_SIZE, shuffle=False)
    val_meta = seq_panel.meta_frame(val_entries)

    def _predict_loader(loader) -> np.ndarray:
        model.eval()
        preds = []
        with torch.no_grad():
            for x, _ in loader:
                preds.append(model(x.to(dev)).cpu().numpy())
        return np.concatenate(preds) if preds else np.array([])

    # ---- 权重 EMA：和 StockMixer 一样，用训练过程内部的自集成对抗单 epoch 噪声 ----
    ema_state = copy.deepcopy(model.state_dict())

    best_val_rankic = float("-inf")   # 记录"滑动平均后"的最佳值
    best_state = None                 # 保存最佳 epoch 对应的 EMA 权重
    best_epoch = -1
    patience_left = patience
    rankic_window: deque[float] = deque(maxlen=max(rankic_smooth_window, 1))
    rng = np.random.default_rng(seed)

    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train_batches))
        for bi in order:
            x_np, y_np = train_batches[bi]
            x = torch.from_numpy(x_np).to(dev)              # [N_day, window, F]
            y = torch.from_numpy(y_np).to(dev).unsqueeze(0)  # [1, N_day]
            mask = torch.ones_like(y)
            pred = model(x).unsqueeze(0)                     # [1, N_day]
            loss, _, _ = combined_loss(pred, y, mask, rank_weight=rank_loss_weight,
                                       cross_sectional_norm=cross_sectional_norm)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        _ema_update_(ema_state, model.state_dict(), ema_decay)

        val_pred = _predict_loader(val_loader)
        val_frame = val_meta.copy()
        val_frame["pred"] = val_pred
        val_rankic_raw = summarize_ic(calc_ic_series(val_frame, label_col="fwd_ret"))["rank_ic_mean"]
        if not np.isnan(val_rankic_raw):
            rankic_window.append(val_rankic_raw)
        val_rankic_smooth = float(np.mean(rankic_window)) if rankic_window else float("nan")
        logger.info("[benchmark/LSTM] epoch %d/%d val_RankIC=%.4f (滑动平均=%.4f)",
                    epoch, epochs, val_rankic_raw, val_rankic_smooth)

        if not np.isnan(val_rankic_smooth) and val_rankic_smooth > best_val_rankic + 1e-6:
            best_val_rankic = val_rankic_smooth
            best_state = copy.deepcopy(ema_state)
            best_epoch = epoch
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info("[benchmark/LSTM] 验证集 RankIC(滑动平均) 连续 %d 轮未改善，触发早停"
                            "（最佳 epoch=%d, val_RankIC 滑动平均=%.4f）",
                            patience, best_epoch, best_val_rankic)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        logger.warning("[benchmark/LSTM] 所有 epoch 的 val_RankIC 均为 NaN，使用最后一轮 EMA 权重作为兜底")
        model.load_state_dict(ema_state)
        best_epoch = epochs

    test_loader = DataLoader(WindowDataset(seq_panel, test_entries), batch_size=DEFAULT_BATCH_SIZE, shuffle=False)
    test_pred = _predict_loader(test_loader)
    test_frame = seq_panel.meta_frame(test_entries)
    test_frame["pred"] = test_pred
    logger.info("[benchmark/LSTM] 测试集评估（使用最佳验证 RankIC(滑动平均) 对应的 EMA 权重, "
                "epoch=%d, val_RankIC=%.4f）", best_epoch, best_val_rankic)
    result = test_frame[["date", "ts_code", "pred", "fwd_ret"]].reset_index(drop=True)
    result.attrs["split_metadata"] = {
        "horizon": horizon,
        "alignment": "stockmixer",
        "purged_train_dates": [str(value.date()) for value in purged_train_dates],
        "purged_val_dates": [str(value.date()) for value in purged_val_dates],
        "n_train": len(train_entries),
        "n_validation": len(val_entries),
        "best_epoch": best_epoch,
        "best_validation_rankic": best_val_rankic,
    }
    return result
