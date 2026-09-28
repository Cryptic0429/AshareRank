"""StockMixer 生产在线推理：加载 .pt 权重 → 构造最新窗口张量 → 前向打分。

与树模型（booster.py/service.py 的表格截面管线）完全独立的一条推理路径：
StockMixer 的输入是固定股票池、固定顺序的 `[N, T, F]` 张量（N=训练时冻结的股票
数量与顺序，T=lookback 滑动窗口，F=特征维度），不能像 LightGBM 那样直接吃
`_latest_feature_frame()` 产出的表格截面。

特征工程严格复用 `experimental/stockmixer/data.py` 里训练时用到的同一批函数
（`_pivot` / `_build_fundamental_frames`），只是只算「最新一个窗口」而不是
整段训练历史，且标准化必须使用训练 checkpoint 里保存的 mean/std（而不是现算的
均值方差），否则推理时的特征分布会和训练时不一致。
"""
from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from asharerank.settings import MODEL_DIR
from asharerank.forecasting.experimental.stockmixer.config import StockMixerConfig
from asharerank.forecasting.experimental.stockmixer.data import _build_fundamental_frames, _pivot
from asharerank.forecasting.experimental.stockmixer.model import StockMixer
from asharerank.forecasting.registry import (
    get_active_model_id,
    get_model_meta,
    save_stockmixer_model,
)
from asharerank.forecasting.universe import load_panel

logger = logging.getLogger(__name__)

_MODEL_CACHE: dict[str, tuple] = {}

# metrics dict（compute_metrics 的原始输出，键名如 "RankIC"）与 registry meta
# （已映射成小写键，如 "rank_ic"）之间的别名表，方便同一个 key 两边都能查到。
_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "ic": ("IC", "ic", "ic_mean"),
    "rank_ic": ("RankIC", "rank_ic", "rank_ic_mean"),
    "icir": ("ICIR", "icir"),
    "sharpe": ("Sharpe", "sharpe"),
}


def _metric_value(d: dict | None, key: str) -> float | None:
    if not d:
        return None
    for alias in _METRIC_ALIASES.get(key, (key,)):
        if d.get(alias) is not None:
            return d[alias]
    if key == "annual_return":
        for k, v in d.items():
            if k.startswith("AnnualizedReturn@") and v is not None:
                return v
        return d.get("annual_return")
    return None


def _load_checkpoint(path: str | Path) -> dict:
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def load_stockmixer_model(meta: dict):
    """按 registry 元信息加载 StockMixer 模型（进程内缓存，避免每次请求都反序列化）。

    支持两种 checkpoint 格式：
      - 单 seed（`model_state`，一份权重）；
      - 多 seed 等权集成（`model_states`，N 份权重，见 `run_stockmixer_multiseed`）——
        集成不是"一个模型"，是"N 份权重的预测取平均"，因此统一返回 `models: list`
        （单 seed 时列表长度为 1），推理侧对列表取平均即可，不需要区分两种格式。

    Returns:
        (models: list[StockMixer], cfg, codes, feature_names, mean, std)
    """
    model_id = meta["model_id"]
    path = meta["path"]
    cached = _MODEL_CACHE.get(model_id)
    if cached is not None:
        return cached

    ckpt = _load_checkpoint(path)
    for key in ("codes", "feature_names", "cfg", "mean", "std"):
        if key not in ckpt:
            raise ValueError(
                f"StockMixer 权重文件缺少必要字段 '{key}'（{path}）："
                f"请用最新版 experimental/stockmixer/train.py 重新训练并部署。"
            )
    if "model_states" in ckpt:
        state_dicts = list(ckpt["model_states"])
    elif "model_state" in ckpt:
        state_dicts = [ckpt["model_state"]]
    else:
        raise ValueError(f"checkpoint 既没有 'model_state' 也没有 'model_states'（{path}）")

    codes: list[str] = list(ckpt["codes"])
    feature_names: list[str] = list(ckpt["feature_names"])
    # 只吸收 StockMixerConfig 认识的字段，避免旧/新 checkpoint 字段漂移直接炸
    cfg_kwargs = {
        k: v for k, v in dict(ckpt["cfg"]).items()
        if k in StockMixerConfig.__dataclass_fields__
    }
    cfg = StockMixerConfig(**cfg_kwargs)
    # 与权重训练时的特征列对齐：旧 ckpt 无 fund_missing 时不能因默认 True 多拼一维
    cfg.fund_missing_mask = "fund_missing" in feature_names
    mean = np.asarray(ckpt["mean"], dtype=np.float32)
    std = np.asarray(ckpt["std"], dtype=np.float32)

    models = []
    for state_dict in state_dicts:
        model = StockMixer(
            n_stocks=len(codes), time_steps=cfg.lookback, n_features=len(feature_names),
            market_hidden=cfg.market_hidden, scales=cfg.scales,
            activation=cfg.activation, dropout=cfg.dropout,
            market_layerscale_init=cfg.market_layerscale_init,
        )
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)

    result = (models, cfg, codes, feature_names, mean, std)
    _MODEL_CACHE[model_id] = result
    return result


def _latest_window(codes: list[str], cfg: StockMixerConfig, feature_names: list[str],
                   mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, pd.Timestamp, np.ndarray]:
    """构造「最新一天」的 `[N, T, F]` 特征窗口（用训练 checkpoint 的 mean/std 标准化）。

    Returns:
        x: [N, T, F] float32，已标准化、缺失位置填 0
        last_date: 该窗口对应的最新交易日
        valid: [N] bool，该股票在这个窗口内是否全程有效（无缺失）
    """
    panel = load_panel(codes=codes)
    panel["date"] = pd.to_datetime(panel["date"])
    dates_all = sorted(panel["date"].unique())
    warmup = cfg.lookback + max(cfg.ma_windows) + 5
    if len(dates_all) < warmup:
        raise RuntimeError(f"本地行情历史不足（{len(dates_all)} 天 < 需要 {warmup} 天），"
                           f"请先运行 scripts/research/fetch_universe.py 补齐")

    open_w = _pivot(panel, dates_all, codes, "open").ffill(limit=cfg.ffill_limit)
    high_w = _pivot(panel, dates_all, codes, "high").ffill(limit=cfg.ffill_limit)
    low_w = _pivot(panel, dates_all, codes, "low").ffill(limit=cfg.ffill_limit)
    close_w = _pivot(panel, dates_all, codes, "close").ffill(limit=cfg.ffill_limit)
    vol_w = _pivot(panel, dates_all, codes, "volume").ffill(limit=cfg.ffill_limit)

    close_prev = close_w.shift(1)
    clip = cfg.clip_return

    o_rel = (open_w / close_prev - 1).clip(-clip, clip)
    h_rel = (high_w / close_prev - 1).clip(-clip, clip)
    l_rel = (low_w / close_prev - 1).clip(-clip, clip)
    c_rel = (close_w / close_prev - 1).clip(-clip, clip)
    vol_log = np.log1p(vol_w.clip(lower=0))
    vol_chg = (vol_log - vol_log.shift(1)).clip(-3, 3)

    ma_rel_frames = []
    for w in cfg.ma_windows:
        ma = close_w.rolling(window=w, min_periods=w).mean()
        ma_rel_frames.append((ma / close_prev - 1).clip(-clip, clip))

    price_frames = [o_rel, h_rel, l_rel, c_rel, vol_chg, *ma_rel_frames]
    price_arr = np.stack([f.to_numpy(dtype=np.float32) for f in price_frames], axis=-1)  # [D,N,Fp]
    feat_valid = ~np.isnan(price_arr).any(axis=-1)  # [D,N]

    if cfg.use_fundamentals:
        fund_frames, _ = _build_fundamental_frames(panel, dates_all, codes, close_w, cfg)
        if fund_frames:
            fund_arr = np.stack([f.to_numpy(dtype=np.float32) for f in fund_frames], axis=-1)
            raw_feat_arr = np.concatenate([price_arr, fund_arr], axis=-1)
        else:
            raw_feat_arr = price_arr
    else:
        raw_feat_arr = price_arr

    if raw_feat_arr.shape[-1] != len(feature_names):
        raise RuntimeError(
            f"最新窗口特征维度 {raw_feat_arr.shape[-1]} 与模型训练时的特征维度 "
            f"{len(feature_names)} 不一致，可能是 use_fundamentals/ma_windows 配置漂移"
        )

    feat_norm = (raw_feat_arr - mean) / std
    feat_arr = np.nan_to_num(feat_norm, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    lookback = cfg.lookback
    window = feat_arr[-lookback:]                     # [T, N, F]
    window_valid = feat_valid[-lookback:].all(axis=0)  # [N]
    x = np.transpose(window, (1, 0, 2))                 # [N, T, F]
    return x, dates_all[-1], window_valid


def predict_latest_stockmixer(meta: dict, top_k: int = 20) -> pd.DataFrame:
    """对最新交易日的固定股票池打分，返回含 date/ts_code/alpha_score 的长表。

    多 seed 集成时，等权平均每个 seed 模型的预测（与
    `run_stockmixer_multiseed` 里离线评估集成指标时用的聚合方式一致）。
    """
    models, cfg, codes, feature_names, mean, std = load_stockmixer_model(meta)
    x, last_date, valid = _latest_window(codes, cfg, feature_names, mean, std)

    with torch.no_grad():
        xt = torch.from_numpy(x).unsqueeze(0)   # [1, N, T, F]
        preds = np.stack([m(xt).squeeze(0).numpy() for m in models], axis=0)  # [n_seed, N]
        pred = preds.mean(axis=0)                # [N]

    day = pd.DataFrame({
        "date": last_date, "ts_code": codes,
        "alpha_score": pred.astype(float),
    })
    day = day[valid].reset_index(drop=True)
    if day.empty:
        raise RuntimeError("StockMixer 最新窗口全部股票特征缺失，无法出具推荐")
    return day


def deploy_stockmixer_checkpoint(ckpt_path: str | Path, model_id: str | None = None,
                                 metrics: dict | None = None, status: str = "已部署",
                                 extra: dict | None = None, compare_metric: str = "rank_ic",
                                 force: bool = False) -> dict:
    """把 `experimental/stockmixer/train.py::run_stockmixer` 训练产出的 checkpoint
    正式接入生产 `/alpha` 推理：复制权重到 `MODEL_DIR`（避免后续重新训练覆盖掉线上
    正在使用的权重）并登记到 registry。

    供 `scripts/research/deploy_stockmixer.py` 调用；不在 import 期间做任何训练，
    只负责「已训练好的 checkpoint → 注册为可上线模型」这一步。

    安全默认行为（"上线的模型应该是指标最好的模型"）：`status="已部署"` 时，不会
    无脑顶替当前线上激活模型——先按 `compare_metric`（默认 RankIC）跟当前激活模型
    比一次，新模型更差就只注册为"候选"（不影响线上），并打印清晰提示，而不是
    "谁最后训练谁说了算"。训练本身有随机性，同一份代码不同 seed 跑出来的指标可能
    忽好忽坏，这层比较避免"运气差的一次训练"意外顶替"运气好的上一次训练"。
    传 `force=True` 跳过比较，强制按 `status` 执行（例如你已经用
    `promote_best_model.py` 之类的工具人工确认过要切换）。
    """
    ckpt_path = Path(ckpt_path)
    ckpt = _load_checkpoint(ckpt_path)
    for key in ("codes", "feature_names", "cfg", "mean", "std"):
        if key not in ckpt:
            raise ValueError(f"checkpoint 缺少必要字段 '{key}'，无法部署（{ckpt_path}）")
    if "model_state" not in ckpt and "model_states" not in ckpt:
        raise ValueError(f"checkpoint 既没有 'model_state' 也没有 'model_states'，无法部署（{ckpt_path}）")
    n_seeds = len(ckpt["model_states"]) if "model_states" in ckpt else 1

    model_id = model_id or f"stockmixer_{datetime.now():%Y%m%d_%H%M}"

    effective_status = status
    if status == "已部署" and not force:
        current_id = get_active_model_id()
        current_meta = get_model_meta(current_id) if current_id else None
        cur_val = _metric_value(current_meta, compare_metric)
        new_val = _metric_value(metrics, compare_metric)
        if current_meta is not None and cur_val is not None:
            if new_val is None:
                effective_status = "候选"
                logger.warning(
                    "新模型 %s 没有 %s 指标，无法与当前线上激活模型 %s（%s=%.4f）比较，"
                    "先注册为候选，不自动上线（可用 promote_best_model.py 人工确认后上线，"
                    "或传 force=True 强制上线）",
                    model_id, compare_metric, current_id, compare_metric, cur_val,
                )
            elif new_val < cur_val:
                effective_status = "候选"
                logger.warning(
                    "新模型 %s 的 %s=%.4f 低于当前线上激活模型 %s（%s=%.4f），"
                    "不自动顶替：先注册为候选，不影响线上推荐。确认要切换可用 "
                    "promote_best_model.py，或本次调用传 force=True。",
                    model_id, compare_metric, new_val, current_id, compare_metric, cur_val,
                )

    dest = MODEL_DIR / f"{model_id}.pt"
    if dest.resolve() != ckpt_path.resolve():
        shutil.copy2(ckpt_path, dest)
    merged_extra = {"n_seeds": n_seeds, **(extra or {})}
    meta = save_stockmixer_model(
        model_id, dest, feature_names=ckpt["feature_names"],
        metrics=metrics or {}, extra=merged_extra, status=effective_status,
    )
    logger.info("StockMixer 已注册: %s（%d-seed，status=%s）-> %s",
               model_id, n_seeds, effective_status, dest)
    return meta


def explain_top_features_stockmixer(meta: dict, ts_code: str, top_n: int = 8) -> dict:
    """StockMixer 没有树模型的特征重要性，改用输入梯度（saliency）近似局部贡献：
    对目标股票的预测分数反传，`|梯度| × 特征值` 越大表示该特征对这次预测的
    影响越大（与 booster 路径「特征值 × 全局重要性」的近似解释思路一致）。
    多 seed 集成时，对每个 seed 模型分别算梯度贡献再取平均。
    """
    models, cfg, codes, feature_names, mean, std = load_stockmixer_model(meta)
    if ts_code not in codes:
        raise ValueError(f"当日无该股票特征: {ts_code}（不在该模型的固定股票池内）")
    idx = codes.index(ts_code)

    x, last_date, valid = _latest_window(codes, cfg, feature_names, mean, std)
    if not valid[idx]:
        raise ValueError(f"当日无该股票特征: {ts_code}")

    contrib_per_seed = []
    for model in models:
        xt = torch.from_numpy(x).unsqueeze(0).requires_grad_(True)  # [1, N, T, F]
        pred = model(xt).squeeze(0)          # [N]
        model.zero_grad(set_to_none=True)
        pred[idx].backward()
        grad = xt.grad[0, idx].detach().numpy()   # [T, F]
        value = x[idx]                             # [T, F]
        # 沿时间维取均值，把 [T, F] 收口成每个特征一个标量贡献
        contrib_per_seed.append((np.abs(grad) * np.abs(value)).mean(axis=0))  # [F]

    contrib = np.mean(contrib_per_seed, axis=0)  # [F]
    total = float(contrib.sum()) or 1.0
    importance = contrib / total
    last_value = x[idx][-1]  # 展示用：窗口最后一天（最新交易日）的标准化特征值

    order = np.argsort(-contrib)[:top_n]
    return {
        "ts_code": ts_code,
        "model_id": meta["model_id"],
        "features": [
            {"name": feature_names[i], "value": float(last_value[i]),
             "importance": float(importance[i]), "contribution": float(contrib[i])}
            for i in order
        ],
    }
