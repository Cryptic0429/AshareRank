"""Benchmark 子步骤 1：StockMixer 原生训练 -> 导出测试集 PredictionFrame。

**必须在独立进程里运行**，不能和 `benchmark_lgbm_step.py` import 到同一个
Python 进程：本模块会 import torch，LightGBM 又会在同一进程里 dlopen 自己
绑定的 OpenMP 运行时，两者的 libomp 在 macOS 上会冲突导致直接 segfault
（`exit code 139`，不是训练逻辑的 bug）。`scripts/research/run_benchmark.py`
用 `subprocess` 分别调用本模块和 `benchmark_lgbm_step.py` 来规避这个问题。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from asharerank.forecasting import canonical
from asharerank.forecasting.experimental.stockmixer.config import StockMixerConfig
from asharerank.forecasting.experimental.stockmixer.data import prepare_dataset
from asharerank.forecasting.experimental.stockmixer.train import get_device, prepare_loaders, _train_one_run

logger = logging.getLogger(__name__)


def get_universe_and_bounds(cfg: StockMixerConfig) -> dict:
    """只做 StockMixer 的数据准备（上市年限/停牌覆盖率过滤 + train/val/test 切分），
    不训练模型——给"训练数据对齐"场景（比如重训一版和 StockMixer 用同一批股票池、
    同一段时间窗口的 LightGBM 候选模型）提供元信息，不用为了拿这几个字段去跑一次
    完整训练。仍然要在独立进程里跑（本模块 import torch），原因见模块顶部注释。
    """
    data = prepare_dataset(cfg)
    train_start, train_end, val_start, val_end, test_start, test_end = data.date_bounds
    return {
        "codes": data.codes,
        "train_start": str(train_start.date()), "train_end": str(train_end.date()),
        "val_start": str(val_start.date()), "val_end": str(val_end.date()),
        "test_start": str(test_start.date()), "test_end": str(test_end.date()),
        "purge_days": cfg.label_horizon,
        "train_effective_end": str(data.dates_all[data.train_idx[-1]].date()),
        "val_effective_end": str(data.dates_all[data.val_idx[-1]].date()),
        "purged_train_dates": [str(d.date()) for d in data.purged_train_dates],
        "purged_val_dates": [str(d.date()) for d in data.purged_val_dates],
    }


def run_stockmixer_step(cfg: StockMixerConfig, seed: int | None = None) -> tuple[pd.DataFrame, dict]:
    """按 StockMixer 现有的 `_train_one_run` 训练（不改一行模型/训练代码），
    把测试集预测转成 `(date, ts_code, pred, fwd_ret)` 长表 + 元信息。
    """
    seed = cfg.seed if seed is None else seed
    device = get_device(cfg.device)
    data, train_loader, val_loader, test_loader = prepare_loaders(cfg)
    result = _train_one_run(cfg, data, train_loader, val_loader, test_loader, device, seed,
                            log_prefix="[benchmark/StockMixer] ")

    test_dates = [data.dates_all[i] for i in data.test_idx]
    frame = canonical.to_prediction_frame(
        dates=test_dates, codes=data.codes,
        preds=result["test_preds"], targets=result["test_targets"], masks=result["test_masks"],
    )
    train_start, train_end, val_start, val_end, test_start, test_end = data.date_bounds
    meta = {
        "codes": data.codes,
        "train_start": str(train_start.date()), "train_end": str(train_end.date()),
        "val_start": str(val_start.date()), "val_end": str(val_end.date()),
        "test_start": str(test_start.date()), "test_end": str(test_end.date()),
        "purge_days": cfg.label_horizon,
        "train_effective_end": str(data.dates_all[data.train_idx[-1]].date()),
        "val_effective_end": str(data.dates_all[data.val_idx[-1]].date()),
        "purged_train_dates": [str(d.date()) for d in data.purged_train_dates],
        "purged_val_dates": [str(d.date()) for d in data.purged_val_dates],
        "best_epoch": result["best_epoch"], "seed": seed,
        "native_test_metrics": result["test_metrics"],
    }
    return frame, meta


def run_stockmixer_step_multiseed(cfg: StockMixerConfig, seeds: tuple[int, ...]) -> tuple[pd.DataFrame, dict]:
    """多 seed 版本：同一份数据只准备一次，依次用不同 seed 训练（`_train_one_run`
    不改一行），把各 seed 的测试集预测**等权平均**后再转成 PredictionFrame。

    背景：`run_benchmark.py` 早期版本只用单个 seed（如 42）训练 StockMixer，
    但已经在 `train_stockmixer_multiseed.py` 的实验里验证过——单 seed 的 RankIC
    波动区间能从 0.005 摆到 0.099（std=0.029），拿单 seed 的结果去跟 LightGBM
    做"架构对比"很容易把"这次运气好/差"误判成"架构更强/更弱"。等权集成能把
    seed 间独立的噪声互相抵消，是目前验证过的、成本最低的稳定化手段（见
    `stockmixer_multiseed_report.json` 里 ensemble_metrics 的实测结果）。
    """
    device = get_device(cfg.device)
    data, train_loader, val_loader, test_loader = prepare_loaders(cfg)

    per_seed_preds: dict[int, np.ndarray] = {}
    per_seed_metrics: list[dict] = []
    test_targets = test_masks = None
    for i, seed in enumerate(seeds, start=1):
        prefix = f"[benchmark/StockMixer seed {i}/{len(seeds)}={seed}] "
        logger.info("%s开始训练 ...", prefix)
        result = _train_one_run(cfg, data, train_loader, val_loader, test_loader, device, seed,
                                log_prefix=prefix)
        per_seed_preds[seed] = result["test_preds"]
        per_seed_metrics.append({
            "seed": seed,
            "best_epoch": result["best_epoch"],
            "best_val_rankic": result["best_val_rankic"],
            **result["test_metrics"],
        })
        if test_targets is None:
            test_targets, test_masks = result["test_targets"], result["test_masks"]
        logger.info("%s完成：best_epoch=%d RankIC=%.4f Sharpe=%.4f",
                    prefix, result["best_epoch"], result["test_metrics"]["RankIC"], result["test_metrics"]["Sharpe"])

    ensemble_pred = np.mean(np.stack(list(per_seed_preds.values()), axis=0), axis=0)

    test_dates = [data.dates_all[i] for i in data.test_idx]
    frame = canonical.to_prediction_frame(
        dates=test_dates, codes=data.codes,
        preds=ensemble_pred, targets=test_targets, masks=test_masks,
    )
    train_start, train_end, val_start, val_end, test_start, test_end = data.date_bounds
    meta = {
        "codes": data.codes,
        "train_start": str(train_start.date()), "train_end": str(train_end.date()),
        "val_start": str(val_start.date()), "val_end": str(val_end.date()),
        "test_start": str(test_start.date()), "test_end": str(test_end.date()),
        "purge_days": cfg.label_horizon,
        "train_effective_end": str(data.dates_all[data.train_idx[-1]].date()),
        "val_effective_end": str(data.dates_all[data.val_idx[-1]].date()),
        "purged_train_dates": [str(d.date()) for d in data.purged_train_dates],
        "purged_val_dates": [str(d.date()) for d in data.purged_val_dates],
        "seeds": list(seeds),
        "per_seed_metrics": per_seed_metrics,
    }
    return frame, meta
