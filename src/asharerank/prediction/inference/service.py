"""最新行情更新与已训练模型推理服务。"""
from __future__ import annotations

import json
import logging
import time
from itertools import combinations
from threading import Lock

import numpy as np
import pandas as pd

from asharerank.settings import MODEL_DIR
from asharerank.config import PANEL_STOCK_LIMIT, SUPPORTED_MODELS
from asharerank.prediction.inference.engine import list_saved_models, predict_saved_models
from asharerank.forecasting.universe import (
    list_available_codes,
    load_panel,
    refresh_cached_universe,
)
from asharerank.prediction import (
    add_risk_balance_groups,
    balanced_top_k,
    load_local_stock_info,
)

logger = logging.getLogger(__name__)

_LIVE_DIR = MODEL_DIR / "panel_comparison" / "live"
_LIVE_RESULT_FILE = _LIVE_DIR / "latest_result.json"
_LIVE_PREDICTIONS_FILE = _LIVE_DIR / "latest_predictions.parquet"
_STATE_LOCK = Lock()
_RUN_LOCK = Lock()
_STATE: dict = {
    "status": "idle",
    "progress": 0.0,
    "message": "",
    "result": None,
}


def _set_state(**values) -> None:
    with _STATE_LOCK:
        _STATE.update(values)


def _json_records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(
        frame.to_json(orient="records", force_ascii=False, date_format="iso")
    )


def _load_persisted_result() -> dict | None:
    if not _LIVE_RESULT_FILE.exists():
        return None
    try:
        return json.loads(_LIVE_RESULT_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        logger.exception("读取最近实时推理结果失败")
        return None


def live_prediction_status() -> dict:
    with _STATE_LOCK:
        if _STATE["status"] == "idle" and _STATE["result"] is None:
            persisted = _load_persisted_result()
            if persisted:
                _STATE.update(
                    status="done",
                    progress=1.0,
                    message="已加载最近一次在线推理结果",
                    result=persisted,
                )
        return {
            **_STATE,
            "saved_models": list_saved_models(),
        }


def _rank_correlation(combined: pd.DataFrame) -> list[dict]:
    pivot = combined.pivot(index="ts_code", columns="model_name", values="rank")
    correlation = pivot.corr(method="spearman")
    rows = []
    for left in correlation.columns:
        for right in correlation.columns:
            value = correlation.loc[left, right]
            rows.append({
                "model_a": left,
                "model_b": right,
                "correlation": float(value) if pd.notna(value) else None,
            })
    return rows


def _top_k_overlap(combined: pd.DataFrame, top_k: int) -> list[dict]:
    top_sets = {
        name: set(group.nsmallest(min(top_k, len(group)), "rank")["ts_code"])
        for name, group in combined.groupby("model_name")
    }
    rows = []
    for left, right in combinations(top_sets, 2):
        intersection = top_sets[left] & top_sets[right]
        union = top_sets[left] | top_sets[right]
        rows.append({
            "model_a": left,
            "model_b": right,
            "intersection": len(intersection),
            "jaccard": len(intersection) / len(union) if union else 1.0,
        })
    return rows


def _rank_matrix(combined: pd.DataFrame, info: pd.DataFrame) -> pd.DataFrame:
    ranks = combined.pivot(index="ts_code", columns="model_name", values="rank")
    ranks.columns = [f"rank_{column}" for column in ranks.columns]
    percentiles = combined.pivot(index="ts_code", columns="model_name", values="rank_pct")
    percentiles.columns = [f"pct_{column}" for column in percentiles.columns]
    matrix = ranks.join(percentiles).reset_index()
    return info.merge(matrix, on="ts_code", how="right").sort_values("ts_code")


def run_live_prediction(
    models: list[str] | None = None,
    top_k: int = 10,
    refresh_data: bool = True,
) -> dict:
    """更新日线并使用已有权重预测；此路径不会训练任何模型。"""
    if not 1 <= top_k <= PANEL_STOCK_LIMIT:
        raise ValueError(f"top_k 必须位于 1 到 {PANEL_STOCK_LIMIT}")
    requested = models or list(SUPPORTED_MODELS)
    if not _RUN_LOCK.acquire(blocking=False):
        raise RuntimeError("已有最新行情预测任务正在运行")
    started = time.perf_counter()
    _set_state(status="running", progress=0.05, message="准备更新最新日线", result=None)
    try:
        codes = list_available_codes()[:PANEL_STOCK_LIMIT]
        if refresh_data:
            refresh = refresh_cached_universe(codes=codes)
        else:
            panel_dates = load_panel(codes=codes).groupby("ts_code")["date"].max()
            refresh = {
                "requested": len(codes),
                "updated": 0,
                "failed": 0,
                "failed_codes": [],
                "latest_market_date": pd.to_datetime(panel_dates.max()).strftime("%Y-%m-%d"),
                "source": "local_cache",
                "fallback_to_cache": True,
            }
        _set_state(progress=0.38, message="加载统一特征和已训练模型")
        panel = load_panel(codes=codes)
        outputs, errors, coverage = predict_saved_models(panel, requested)
        if not outputs:
            raise RuntimeError(f"没有模型完成推理: {errors}")

        combined = pd.concat([output.predictions for output in outputs], ignore_index=True)
        info = load_local_stock_info(sorted(combined["ts_code"].unique()))
        enriched = combined.merge(info, on="ts_code", how="left")
        enriched["name"] = enriched["name"].fillna(enriched["ts_code"])
        enriched["industry"] = enriched["industry"].fillna("其他")
        enriched = add_risk_balance_groups(enriched)

        rankings: dict[str, list[dict]] = {}
        balanced_rankings: dict[str, list[dict]] = {}
        summaries: list[dict] = []
        for output in outputs:
            model_rows = enriched[enriched["model_name"] == output.model_name].copy()
            model_rows = model_rows.sort_values("rank")
            rankings[output.model_name] = _json_records(model_rows.head(top_k))
            balanced_rows, method = balanced_top_k(model_rows, top_k)
            balanced_rankings[output.model_name] = _json_records(balanced_rows)
            leader = model_rows.iloc[0]
            summaries.append({
                "model_name": output.model_name,
                "top_stock": leader["ts_code"],
                "top_stock_name": leader["name"],
                "top_score": float(leader["raw_score"]),
                "inference_seconds": output.inference_seconds,
                "trained_at": output.trained_at,
                "artifact_dir": output.artifact_dir,
                "balance_method": method,
            })

        matrix = _rank_matrix(enriched, info)
        elapsed = time.perf_counter() - started
        payload = {
            "as_of": coverage["prediction_date"],
            "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
            "refresh": refresh,
            "coverage": coverage,
            "models_completed": [output.model_name for output in outputs],
            "errors": errors,
            "top_k": top_k,
            "duration_seconds": elapsed,
            "model_summaries": summaries,
            "rankings": rankings,
            "balanced_rankings": balanced_rankings,
            "rank_matrix": _json_records(matrix),
            "rank_correlation": _rank_correlation(enriched),
            "top_k_overlap": _top_k_overlap(enriched, top_k),
            "prediction_file": str(_LIVE_PREDICTIONS_FILE),
            "training_performed": False,
        }
        _LIVE_DIR.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(_LIVE_PREDICTIONS_FILE, index=False)
        _LIVE_RESULT_FILE.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _set_state(
            status="done",
            progress=1.0,
            message="最新行情预测完成（未重新训练）",
            result=payload,
        )
        return payload
    except Exception as exc:
        logger.exception("最新行情预测失败")
        _set_state(status="error", progress=0.0, message=str(exc), result=None)
        raise
    finally:
        _RUN_LOCK.release()
