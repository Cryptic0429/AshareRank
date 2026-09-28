"""Load the latest unified benchmark report for programmatic analysis."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from asharerank.settings import CACHE_DIR
from asharerank.forecasting.evaluate import calc_ic_series
from asharerank.forecasting.experimental.report import EXPERIMENTAL_DIR
from asharerank.forecasting.benchmark import REPORT_NAME

logger = logging.getLogger(__name__)

MODEL_ORDER = ("lgbm", "xgboost", "lstm", "stockmixer")

MODEL_LABELS = {
    "lgbm": "LightGBM",
    "xgboost": "XGBoost",
    "lstm": "LSTM",
    "stockmixer": "StockMixer",
}

FRAME_FILES = {
    "stockmixer": "sm_frame.parquet",
    "lgbm": "lgbm_frame.parquet",
    "xgboost": "xgboost_frame.parquet",
    "lstm": "lstm_frame.parquet",
}

_TMP_DIR = EXPERIMENTAL_DIR / "_benchmark_tmp"


def _latest_report_path() -> Path:
    return EXPERIMENTAL_DIR / f"{REPORT_NAME}_report.json"


def _relpath(path: Path) -> str:
    try:
        return str(path.relative_to(CACHE_DIR))
    except ValueError:
        return str(path)


def _normalize_metrics(raw: dict, top_k: int) -> dict:
    """把报告里的字段统一成前端友好的稳定键名。"""
    prec_key = f"precision_at_{top_k}"
    precision = raw.get("precision_at_k", raw.get(prec_key))
    return {
        "ic": raw.get("ic_mean"),
        "rank_ic": raw.get("rank_ic_mean"),
        "icir": raw.get("icir"),
        "rank_icir": raw.get("rank_icir"),
        "precision_at_k": precision,
        "annual_return": raw.get("annual_return"),
        "sharpe": raw.get("sharpe"),
        "max_drawdown": raw.get("max_drawdown"),
        "n_days": raw.get("n_days"),
        "n_pairs": raw.get("n_pairs"),
    }


def _registry_status_map() -> dict[str, dict]:
    """把生产注册表条目映射到四模型 key。"""
    try:
        from asharerank.forecasting.registry import get_active_model_id, list_models
    except Exception:
        return {}

    active = get_active_model_id()
    out: dict[str, dict] = {}
    for model in list_models():
        backend = str(model.get("backend") or "").lower()
        mtype = str(model.get("type") or "").lower()
        mid = str(model.get("model_id") or "")
        key = None
        if "stockmixer" in backend or "stockmixer" in mtype or "stockmixer" in mid:
            key = "stockmixer"
        elif "xgb" in backend or "xgboost" in mtype or "xgboost" in mid:
            key = "xgboost"
        elif "lstm" in backend or "lstm" in mtype or "lstm" in mid:
            key = "lstm"
        elif (
            "lgbm" in backend
            or "lightgbm" in backend
            or "hgb" in backend
            or "histgbm" in mtype
            or "lightgbm" in mtype
            or mid.startswith("lgbm")
        ):
            key = "lgbm"
        if key is None:
            continue
        servable = model.get("servable", True)
        if mid == active and servable is not False:
            status = "deployed"
        elif servable is False:
            status = "contrast"
        else:
            status = "registered"
        prev = out.get(key)
        # 优先保留已部署；否则保留更新的一条
        if prev and prev.get("status") == "deployed" and status != "deployed":
            continue
        out[key] = {
            "status": status,
            "model_id": mid,
            "trained_at": model.get("trained_at"),
            "servable": servable is not False,
        }
    return out


def _load_ic_series() -> dict[str, list[dict]]:
    """若 benchmark 临时预测帧仍在，计算逐日 IC/RankIC 供稳定性图。"""
    series: dict[str, list[dict]] = {}
    if not _TMP_DIR.exists():
        return series
    for key, filename in FRAME_FILES.items():
        path = _TMP_DIR / filename
        if not path.exists():
            continue
        try:
            frame = pd.read_parquet(path)
            ic_df = calc_ic_series(frame)
            if ic_df.empty:
                continue
            rows = []
            for _, row in ic_df.sort_values("date").iterrows():
                rows.append({
                    "date": str(pd.Timestamp(row["date"]).date()),
                    "ic": float(row["ic"]),
                    "rank_ic": float(row["rank_ic"]),
                })
            series[key] = rows
        except Exception as exc:
            logger.warning("读取 benchmark 预测帧失败 %s: %s", path, exc)
    return series


def load_benchmark_latest(*, include_series: bool = True) -> dict:
    """返回前端模型对比页所需的结构化 payload。"""
    path = _latest_report_path()
    if not path.exists():
        return {
            "available": False,
            "message": "尚未找到统一 Benchmark 报告，请先运行 scripts/research/run_benchmark.py",
            "report_path": _relpath(path),
        }

    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return {"available": False, "message": f"Benchmark 报告无法解析: {exc}"}

    top_k = int(report.get("top_k") or 10)
    common = report.get("common_universe") or {}
    full = report.get("full_universe") or {}
    meta = report.get("model_meta") or {}
    sm_meta = meta.get("stockmixer") or {}
    registry = _registry_status_map()

    fairness = {
        "train_start": sm_meta.get("train_start"),
        "train_end": sm_meta.get("train_end"),
        "val_start": sm_meta.get("val_start"),
        "val_end": sm_meta.get("val_end"),
        "test_start": sm_meta.get("test_start") or (report.get("test_window") or [None, None])[0],
        "test_end": sm_meta.get("test_end") or (report.get("test_window") or [None, None])[1],
        "n_codes": report.get("n_codes"),
        "n_codes_common": report.get("n_codes_common"),
        "top_k": top_k,
        "cost_bps": report.get("cost_bps"),
        "horizon": None,
        "seeds": sm_meta.get("seeds"),
    }
    per_seed = sm_meta.get("per_seed_metrics") or []
    if per_seed and isinstance(per_seed[0], dict):
        fairness["horizon"] = per_seed[0].get("horizon", 5)
    if fairness["horizon"] is None:
        fairness["horizon"] = 5

    models = []
    for key in MODEL_ORDER:
        row = common.get(key) or full.get(key)
        if not row:
            continue
        item = {
            "key": key,
            "label": MODEL_LABELS.get(key, row.get("model") or key),
            "metrics": _normalize_metrics(row, top_k),
            "full_metrics": _normalize_metrics(full[key], top_k) if key in full else None,
            "registry": registry.get(key) or {"status": "unregistered", "model_id": None},
        }
        models.append(item)

    payload = {
        "available": bool(models),
        "scope": "common_universe",
        "saved_at": report.get("saved_at"),
        "git_commit": report.get("git_commit"),
        "fairness": fairness,
        "models": models,
        "seed_stability": {
            "stockmixer": [
                {
                    "seed": s.get("seed"),
                    "rank_ic": s.get("RankIC"),
                    "ic": s.get("IC"),
                    "icir": s.get("ICIR"),
                    "precision_at_k": s.get(f"Precision@{top_k}") or s.get("Precision@10"),
                }
                for s in per_seed
            ]
        } if per_seed else {},
        "report_path": _relpath(path),
    }
    if include_series:
        payload["ic_series"] = _load_ic_series()
    else:
        payload["ic_series"] = {}
    if not models:
        payload["message"] = "Benchmark 报告中没有可展示的模型指标"
    return payload
