"""StockMixer 训练 / 验证 / 测试主流程。

用法（见 `scripts/research/train_stockmixer.py` / `scripts/research/train_stockmixer_multiseed.py`）：
    python scripts/research/train_stockmixer.py
    python scripts/research/train_stockmixer.py --max-stocks 60 --epochs 30
    python scripts/research/train_stockmixer_multiseed.py --seeds 42,43,44,45,46
"""
from __future__ import annotations

# 必须在 import torch 之前设置：某些算子在 Apple Silicon 的 MPS 后端上还没有
# 原生实现，设置该环境变量后 PyTorch 会自动把这些算子 fallback 到 CPU 执行，
# 而不是直接抛异常中断训练。
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import copy
import logging
import random
import time
from collections import deque

import numpy as np
import torch
from torch.utils.data import DataLoader

from asharerank.forecasting.experimental.report import EXPERIMENTAL_DIR, save_report
from asharerank.forecasting.experimental.stockmixer.config import DEFAULT_CONFIG, StockMixerConfig
from asharerank.forecasting.experimental.stockmixer.data import PreparedData, StockMixerDataset, prepare_dataset
from asharerank.forecasting.experimental.stockmixer.losses import combined_loss
from asharerank.forecasting.experimental.stockmixer.metrics import (
    compute_metrics,
    format_metrics,
    format_multiseed_summary,
    summarize_multiseed,
)
from asharerank.forecasting.experimental.stockmixer.model import StockMixer

logger = logging.getLogger(__name__)


# ============================================================
# 硬件适配：Apple M2 -> MPS，不支持的算子自动 fallback CPU
# ============================================================
def get_device(preferred: str | None = None) -> torch.device:
    """自动检测并优先启用 Apple Silicon 的 `mps` 后端；不可用时依次降级到
    `cuda` / `cpu`。MPS 上不支持的算子由上方的 `PYTORCH_ENABLE_MPS_FALLBACK`
    环境变量自动 fallback 到 CPU 执行，无需手动处理。"""
    if preferred:
        return torch.device(preferred)
    if torch.backends.mps.is_available():
        logger.info("检测到 Apple Silicon MPS 加速，使用 device=mps"
                    "（不支持的算子将自动 fallback 到 CPU）")
        return torch.device("mps")
    if torch.cuda.is_available():
        logger.info("检测到 CUDA，使用 device=cuda")
        return torch.device("cuda")
    logger.info("未检测到 GPU 加速，使用 device=cpu")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _train_years_tag(train_years: float) -> str:
    """报告/权重文件名标签，例如 3 -> ty3，4.5 -> ty4p5。"""
    if abs(train_years - round(train_years)) < 1e-9:
        return f"ty{int(round(train_years))}"
    return "ty" + str(train_years).replace(".", "p")


def _run_tag(cfg: StockMixerConfig) -> str:
    """组合实验标签，避免 5y+mld120 覆盖 5y+mld250 等结果。"""
    parts = [_train_years_tag(cfg.train_years)]
    if cfg.min_listed_days != 250:
        parts.append(f"mld{cfg.min_listed_days}")
    if cfg.freeze_codes:
        parts.append(f"fz{len(cfg.freeze_codes)}")
    if cfg.use_fundamentals:
        fields = cfg.fundamental_fields
        if fields is None:
            parts.append("fund")
        else:
            parts.append("fund_" + "-".join(fields))
        if cfg.fund_missing_mask:
            parts.append("fmiss")
        if abs(cfg.dropout - 0.1) > 1e-12:
            parts.append(f"do{str(cfg.dropout).replace('.', 'p')}")
        if abs(cfg.weight_decay - 1e-5) > 1e-15:
            parts.append(f"wd{cfg.weight_decay:.0e}".replace("+", "").replace("-", "m"))
    return "_".join(parts)


# ============================================================
# 训练 / 验证 / 测试一个 epoch
# ============================================================
def _run_epoch(model: StockMixer, loader: DataLoader, device: torch.device,
               cfg: StockMixerConfig, optimizer: torch.optim.Optimizer | None) -> dict:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = total_mse = total_rank = 0.0
    n_batches = 0
    all_preds, all_targets, all_masks = [], [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for x, y, m in loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
            pred = model(x)
            loss, mse, rank = combined_loss(pred, y, m, rank_weight=cfg.rank_loss_weight,
                                            cross_sectional_norm=cfg.cross_sectional_norm)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()

            total_loss += float(loss.item())
            total_mse += float(mse.item())
            total_rank += float(rank.item())
            n_batches += 1

            all_preds.append(pred.detach().cpu().numpy())
            all_targets.append(y.detach().cpu().numpy())
            all_masks.append(m.detach().cpu().numpy())

    preds = np.concatenate(all_preds, axis=0) if all_preds else np.zeros((0, 0))
    targets = np.concatenate(all_targets, axis=0) if all_targets else np.zeros((0, 0))
    masks = np.concatenate(all_masks, axis=0) if all_masks else np.zeros((0, 0))

    n_batches = max(n_batches, 1)
    # ---- 诊断指标：模型原始预测在"每一天横截面"上的标准差 ----
    # 若这个值持续接近 0，说明模型对同一天里的所有股票给出几乎相同的预测
    # （"塌缩"到只会预测截面均值），是排序损失/横截面标准化仍未生效的信号。
    pred_std = float("nan")
    if preds.size:
        mask_bool = masks > 0.5
        daily_std = []
        for t in range(preds.shape[0]):
            valid = mask_bool[t]
            if valid.sum() >= 2:
                daily_std.append(preds[t][valid].std())
        if daily_std:
            pred_std = float(np.mean(daily_std))

    return {
        "loss": total_loss / n_batches,
        "mse": total_mse / n_batches,
        "rank_loss": total_rank / n_batches,
        "pred_std": pred_std,
        "preds": preds, "targets": targets, "masks": masks,
    }


# ============================================================
# 数据准备（多 seed 共用同一份数据，只在模型初始化 + 训练时改变随机性）
# ============================================================
def prepare_loaders(cfg: StockMixerConfig) -> tuple[PreparedData, DataLoader, DataLoader, DataLoader]:
    logger.info("=== [1/4] 数据获取与特征工程 ===")
    data = prepare_dataset(cfg)
    logger.info("股票数 N=%d, 特征维度 F=%d, lookback T=%d, 特征列=%s",
                data.n_stocks, data.n_features, data.lookback, data.feature_names)

    train_ds = StockMixerDataset(data, data.train_idx)
    val_ds = StockMixerDataset(data, data.val_idx)
    test_ds = StockMixerDataset(data, data.test_idx)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers)
    return data, train_loader, val_loader, test_loader


# ============================================================
# 单次训练（固定 seed）：构建模型 -> 训练 + 早停 -> 测试集评估
# ============================================================
def _ema_update_(ema_state: dict, model_state: dict, decay: float) -> None:
    """就地把 `ema_state` 往 `model_state` 的方向做一次指数滑动平均更新
    （只对浮点张量做平均；整型/布尔 buffer 直接覆盖，避免类型报错，且这类
    buffer 本身也不参与梯度更新，覆盖等价于保持不变）。"""
    for k, v in model_state.items():
        if torch.is_floating_point(v):
            ema_state[k].mul_(decay).add_(v, alpha=1 - decay)
        else:
            ema_state[k].copy_(v)


def _ema_warmup_decay(max_decay: float, epoch: int) -> float:
    """EMA 的 warmup 衰减系数：min(max_decay, (1+t)/(10+t))。

    第一版实现固定用 decay=0.9 且从"随机初始权重"开始累积，导致 best_epoch=4~7
    就早停的 seed 保存的 EMA 权重里残留 0.9^t ≈ 48%~66% 的未训练随机初始化（2026-07-23
    5 seed 实测 RankIC 均值 0.050→0.013 的主因）。warmup 让训练早期 decay 很小
    （t=1 时 0.18、t=5 时 0.4、t=30 时 0.775），EMA 快速"忘掉"初始化、贴近当前权重，
    训练后期才逐渐逼近 max_decay 的长窗口平滑。"""
    return min(max_decay, (1 + epoch) / (10 + epoch))


def _train_one_run(cfg: StockMixerConfig, data: PreparedData,
                   train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader,
                   device: torch.device, seed: int, log_prefix: str = "") -> dict:
    set_seed(seed)

    model = StockMixer(
        n_stocks=data.n_stocks, time_steps=data.lookback, n_features=data.n_features,
        market_hidden=cfg.market_hidden, scales=cfg.scales,
        activation=cfg.activation, dropout=cfg.dropout,
        market_layerscale_init=cfg.market_layerscale_init,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("%s模型参数量: %d (seed=%d)", log_prefix, n_params, seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    # ---- 权重 EMA（可选，cfg.use_ema）：训练过程内部的"自集成" ----
    # 修复版两个要点（第一版的缺陷复盘见 config.py 注释）：
    #   1) decay 用 warmup（_ema_warmup_decay），训练早期 EMA 快速贴近当前权重，
    #      不被随机初始化污染；
    #   2) 开启 EMA 时，验证集直接评 EMA 权重（评什么就存什么），而不是"评原始权重、
    #      存 EMA 权重"的错位。
    ema_state = copy.deepcopy(model.state_dict()) if cfg.use_ema else None

    best_val_loss = float("inf")
    best_val_rankic = float("-inf")       # 记录判据（滑动平均后）的最佳值，供日志/报告展示
    best_state = None
    best_epoch = -1
    patience_left = cfg.patience
    # ---- 早停 / checkpoint 判据：最近 rankic_smooth_window 轮 val_RankIC 的滑动平均 ----
    # window=1 时退化为裸单轮值（基线行为）。动机：验证集只有 100 出头个锚点，单轮
    # RankIC 噪声很大，容易把一次偶然的高点误判成"最佳模型"。
    rankic_window: deque[float] = deque(maxlen=max(cfg.rankic_smooth_window, 1))

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_stats = _run_epoch(model, train_loader, device, cfg, optimizer)

        if cfg.use_ema:
            _ema_update_(ema_state, model.state_dict(), _ema_warmup_decay(cfg.ema_decay, epoch))
            # 用 EMA 权重做验证（评什么存什么）：临时换入 EMA，评完再换回原始权重继续训练
            raw_state = copy.deepcopy(model.state_dict())
            model.load_state_dict(ema_state)
            val_stats = _run_epoch(model, val_loader, device, cfg, optimizer=None)
            model.load_state_dict(raw_state)
        else:
            val_stats = _run_epoch(model, val_loader, device, cfg, optimizer=None)
        # 学习率调度仍按 val_loss（更平滑、对 ReduceLROnPlateau 更合适），
        # 与下面 checkpoint 选择标准（滑动平均 val_RankIC）分开，互不影响。
        scheduler.step(val_stats["loss"])

        val_metrics = compute_metrics(val_stats["preds"], val_stats["targets"], val_stats["masks"],
                                      top_k=cfg.top_k, trading_days_per_year=cfg.trading_days_per_year,
                                      horizon=cfg.label_horizon)
        val_rankic_raw = val_metrics["RankIC"]
        if not np.isnan(val_rankic_raw):
            rankic_window.append(val_rankic_raw)
        val_rankic_smooth = float(np.mean(rankic_window)) if rankic_window else float("nan")
        elapsed = time.time() - t0
        logger.info(
            "%sepoch %3d/%d | train_loss=%.6f (mse=%.6f, rank=%.6f) | "
            "val_loss=%.6f | val_IC=%.4f val_RankIC=%.4f (滑动平均=%.4f) | pred_std=%.4f | %.1fs",
            log_prefix, epoch, cfg.epochs, train_stats["loss"], train_stats["mse"], train_stats["rank_loss"],
            val_stats["loss"], val_metrics["IC"], val_rankic_raw, val_rankic_smooth, val_stats["pred_std"], elapsed,
        )

        # ---- checkpoint / 早停选择标准：val_RankIC（可选滑动平均），评什么存什么 ----
        # 开启 EMA 时上面评的是 EMA 权重，这里存的也是 EMA 权重；未开启时评/存的都是
        # 原始权重——保证"验证集上选出来的"和"最终拿去测试的"是同一份权重。
        if not np.isnan(val_rankic_smooth) and val_rankic_smooth > best_val_rankic + 1e-6:
            best_val_rankic = val_rankic_smooth
            best_val_loss = val_stats["loss"]
            best_state = copy.deepcopy(ema_state if cfg.use_ema else model.state_dict())
            best_epoch = epoch
            patience_left = cfg.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info("%s验证集 RankIC%s 连续 %d 轮未改善，触发早停（最佳 epoch=%d, 判据值=%.4f）",
                            log_prefix, "(滑动平均)" if cfg.rankic_smooth_window > 1 else "",
                            cfg.patience, best_epoch, best_val_rankic)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        # 极端兜底：所有 epoch 的 val_RankIC 都是 NaN（例如验证集样本太少），
        # 用最后一轮的权重，避免直接崩溃。
        logger.warning("%s所有 epoch 的 val_RankIC 均为 NaN，使用最后一轮权重作为兜底", log_prefix)
        if cfg.use_ema:
            model.load_state_dict(ema_state)
        best_epoch = cfg.epochs

    logger.info("%s测试集评估（使用最佳验证 RankIC 对应的%s权重, epoch=%d, 判据值=%.4f）",
                log_prefix, "EMA " if cfg.use_ema else "", best_epoch, best_val_rankic)
    test_stats = _run_epoch(model, test_loader, device, cfg, optimizer=None)
    test_metrics = compute_metrics(test_stats["preds"], test_stats["targets"], test_stats["masks"],
                                   top_k=cfg.top_k, trading_days_per_year=cfg.trading_days_per_year,
                                   horizon=cfg.label_horizon)

    return {
        "seed": seed, "model": model, "n_params": n_params,
        "best_epoch": best_epoch, "best_val_loss": best_val_loss, "best_val_rankic": best_val_rankic,
        "test_metrics": test_metrics,
        "test_preds": test_stats["preds"], "test_targets": test_stats["targets"], "test_masks": test_stats["masks"],
    }


# ============================================================
# 主流程：单次训练（cfg.seed）
# ============================================================
def run_stockmixer(cfg: StockMixerConfig | None = None) -> dict:
    cfg = cfg or DEFAULT_CONFIG
    device = get_device(cfg.device)

    data, train_loader, val_loader, test_loader = prepare_loaders(cfg)

    logger.info("=== [2/4] 构建 StockMixer 模型 ===")
    logger.info("=== [3/4] 训练（epochs=%d, batch_size=%d, device=%s, seed=%d） ===",
                cfg.epochs, cfg.batch_size, device, cfg.seed)
    result = _train_one_run(cfg, data, train_loader, val_loader, test_loader, device, cfg.seed)

    logger.info("=== [4/4] 测试集评估结果 ===")
    print(format_metrics("StockMixer 测试集（沪深300）", result["test_metrics"]))

    ty_tag = _run_tag(cfg)
    bounds = data.date_bounds
    date_bounds = {
        "train": [str(bounds[0].date()), str(bounds[1].date())],
        "val": [str(bounds[2].date()), str(bounds[3].date())],
        "test": [str(bounds[4].date()), str(bounds[5].date())],
    }
    ckpt_path = EXPERIMENTAL_DIR / f"stockmixer_{ty_tag}.pt"
    payload = {"model_state": result["model"].state_dict(), "codes": data.codes,
               "feature_names": data.feature_names, "cfg": cfg.__dict__,
               # 上线到生产 /alpha 推理时，最新窗口的 z-score 标准化必须复用训练集统计量
               # （见 prediction/inference/stockmixer.py），否则推理时特征分布和训练时不一致。
               "mean": data.mean, "std": data.std}
    torch.save(payload, ckpt_path)
    # 默认 3 年仍写兼容路径，避免 deploy --skip-training 找不到旧文件名
    if abs(cfg.train_years - 3.0) < 1e-9 and cfg.min_listed_days == 250 and not cfg.freeze_codes:
        compat = EXPERIMENTAL_DIR / "stockmixer_best.pt"
        torch.save(payload, compat)
        logger.info("模型权重已保存: %s（兼容副本 %s）", ckpt_path, compat)
    else:
        logger.info("模型权重已保存: %s", ckpt_path)

    report = {
        "model_type": "StockMixer",
        "pool": cfg.pool,
        "train_years": cfg.train_years,
        "min_listed_days": cfg.min_listed_days,
        "freeze_n_codes": len(cfg.freeze_codes) if cfg.freeze_codes else 0,
        "date_bounds": date_bounds,
        "n_stocks": data.n_stocks,
        "n_features": data.n_features,
        "lookback": data.lookback,
        "label_horizon": cfg.label_horizon,
        "scales": list(cfg.scales),
        "market_hidden": cfg.market_hidden,
        "activation": cfg.activation,
        "rank_loss_weight": cfg.rank_loss_weight,
        "cross_sectional_norm": cfg.cross_sectional_norm,
        "n_train": len(data.train_idx),
        "n_val": len(data.val_idx),
        "n_test": len(data.test_idx),
        "seed": cfg.seed,
        "best_epoch": result["best_epoch"],
        "best_val_loss": result["best_val_loss"],
        "best_val_rankic": result["best_val_rankic"],
        "test_metrics": result["test_metrics"],
        "device": str(device),
        "n_params": result["n_params"],
        "ckpt_path": str(ckpt_path),
    }
    save_report(f"stockmixer_{ty_tag}", report)
    return report


# ============================================================
# 主流程：多 seed 复现（同一份数据只准备一次，重复训练+评估 N 次）
# ============================================================
def run_stockmixer_multiseed(cfg: StockMixerConfig | None = None,
                             seeds: tuple[int, ...] = (42, 43, 44, 45, 46)) -> dict:
    """用同一份数据、同一套超参数，跑多个随机种子，检验单次结果（如 RankIC=0.097）
    是不是"运气好"——如果不同 seed 之间 RankIC 方差很大甚至符号不稳定，说明单次
    结果不可信；如果多个 seed 都稳定为正且量级接近，才说明这是真实可复现的信号。
    """
    cfg = cfg or DEFAULT_CONFIG
    device = get_device(cfg.device)

    logger.info("=== 数据获取与特征工程（多 seed 共用同一份数据） ===")
    data, train_loader, val_loader, test_loader = prepare_loaders(cfg)

    per_seed_metrics = []
    per_seed_preds = {}
    per_seed_states = []
    test_targets = test_masks = None
    for i, seed in enumerate(seeds, start=1):
        prefix = f"[seed {i}/{len(seeds)}={seed}] "
        logger.info("%s开始训练 ...", prefix)
        result = _train_one_run(cfg, data, train_loader, val_loader, test_loader, device, seed, log_prefix=prefix)
        m = dict(result["test_metrics"])
        m["seed"] = seed
        m["best_epoch"] = result["best_epoch"]
        m["best_val_rankic"] = result["best_val_rankic"]
        per_seed_metrics.append(m)
        per_seed_preds[seed] = result["test_preds"]
        # 权重也要留一份：等权集成不是"训一个模型"，是"N 份权重的预测取平均"，
        # 要把集成结果真正部署到线上推理，必须把每个 seed 的权重都存下来
        # （见下方 ckpt_path 的保存），不能只留指标。
        per_seed_states.append(copy.deepcopy(result["model"].state_dict()))
        # 测试集的 targets/masks 与 seed 无关（同一份数据切分），只需存一份供集成评估复用
        if test_targets is None:
            test_targets, test_masks = result["test_targets"], result["test_masks"]
        logger.info(
            "%s完成：best_epoch=%d IC=%.4f RankIC=%.4f ICIR=%.4f Precision@%d=%.4f Sharpe=%.4f",
            prefix, result["best_epoch"], m["IC"], m["RankIC"], m["ICIR"], cfg.top_k,
            m.get(f"Precision@{cfg.top_k}", float("nan")), m["Sharpe"],
        )

    summary = summarize_multiseed(per_seed_metrics, top_k=cfg.top_k)
    print(format_multiseed_summary("StockMixer 多 seed 复现（沪深300）", per_seed_metrics, summary, top_k=cfg.top_k))

    # ---- 等权集成：把 N 个 seed 的测试集预测取算术平均，再统一算一遍指标 ----
    # 直觉：每个 seed 的预测里都混有"真实信号 + 该次训练特有的噪声"，如果噪声在 seed 间
    # 基本独立，平均后噪声会被互相抵消、信号部分保留，集成后的 RankIC 有机会比"单个 seed
    # 的均值"更高、比"最好的单个 seed"更稳（不需要事后才知道哪个 seed 运气好）。
    ensemble_pred = np.mean(np.stack(list(per_seed_preds.values()), axis=0), axis=0)
    ensemble_metrics = compute_metrics(ensemble_pred, test_targets, test_masks,
                                       top_k=cfg.top_k, trading_days_per_year=cfg.trading_days_per_year,
                                       horizon=cfg.label_horizon)
    print(format_metrics(f"StockMixer {len(seeds)}-seed 等权集成（沪深300）", ensemble_metrics))
    logger.info(
        "集成（%d seed 等权平均）：IC=%.4f RankIC=%.4f ICIR=%.4f Precision@%d=%.4f Sharpe=%.4f，"
        "对比单 seed 均值 RankIC=%.4f、最佳单 seed RankIC=%.4f",
        len(seeds), ensemble_metrics["IC"], ensemble_metrics["RankIC"], ensemble_metrics["ICIR"], cfg.top_k,
        ensemble_metrics.get(f"Precision@{cfg.top_k}", float("nan")), ensemble_metrics["Sharpe"],
        summary["RankIC"]["mean"], summary["RankIC"]["max"],
    )

    ty_tag = _run_tag(cfg)
    bounds = data.date_bounds
    date_bounds = {
        "train": [str(bounds[0].date()), str(bounds[1].date())],
        "val": [str(bounds[2].date()), str(bounds[3].date())],
        "test": [str(bounds[4].date()), str(bounds[5].date())],
    }
    ckpt_path = EXPERIMENTAL_DIR / f"stockmixer_multiseed_{ty_tag}.pt"
    payload = {
        "model_states": per_seed_states, "seeds": list(seeds),
        "codes": data.codes, "feature_names": data.feature_names, "cfg": cfg.__dict__,
        "mean": data.mean, "std": data.std,
    }
    torch.save(payload, ckpt_path)
    if abs(cfg.train_years - 3.0) < 1e-9 and cfg.min_listed_days == 250 and not cfg.freeze_codes:
        compat = EXPERIMENTAL_DIR / "stockmixer_multiseed_best.pt"
        torch.save(payload, compat)
        logger.info("%d-seed 集成权重已保存: %s（兼容副本 %s）", len(seeds), ckpt_path, compat)
    else:
        logger.info("%d-seed 集成权重已保存: %s", len(seeds), ckpt_path)

    report = {
        "model_type": "StockMixer-MultiSeed",
        "pool": cfg.pool,
        "train_years": cfg.train_years,
        "min_listed_days": cfg.min_listed_days,
        "freeze_n_codes": len(cfg.freeze_codes) if cfg.freeze_codes else 0,
        "date_bounds": date_bounds,
        "n_stocks": data.n_stocks,
        "n_features": data.n_features,
        "lookback": data.lookback,
        "label_horizon": cfg.label_horizon,
        "rank_loss_weight": cfg.rank_loss_weight,
        "cross_sectional_norm": cfg.cross_sectional_norm,
        "n_train": len(data.train_idx),
        "n_val": len(data.val_idx),
        "n_test": len(data.test_idx),
        "seeds": list(seeds),
        "per_seed_metrics": per_seed_metrics,
        "summary": summary,
        "ensemble_metrics": ensemble_metrics,
        "device": str(device),
        "ckpt_path": str(ckpt_path),
    }
    save_report(f"stockmixer_multiseed_{ty_tag}", report)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_stockmixer()
