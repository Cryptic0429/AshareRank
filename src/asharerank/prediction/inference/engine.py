"""离线训练、在线推理的核心引擎。"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from asharerank.settings import MODEL_DIR
from asharerank.config import PANEL_STOCK_LIMIT, SUPPORTED_MODELS
from asharerank.data.features import build_common_features
from asharerank.prediction import build_prediction_frame, rank_predictions


@dataclass(frozen=True)
class LiveModelOutput:
    model_name: str
    predictions: pd.DataFrame
    artifact_dir: str
    trained_at: str | None
    inference_seconds: float


def latest_saved_artifact(model_name: str) -> Path | None:
    root = MODEL_DIR / "panel_comparison" / model_name
    if not root.exists():
        return None
    model_filename = "model.pt" if model_name == "master_lite" else "model.joblib"
    candidates = [
        path.parent
        for path in root.glob(f"*/{model_filename}")
        if (path.parent / "adapter.joblib").exists()
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def list_saved_models() -> list[dict]:
    rows = []
    for name in SUPPORTED_MODELS:
        artifact = latest_saved_artifact(name)
        manifest = {}
        if artifact is not None and (artifact / "manifest.json").exists():
            manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
        rows.append({
            "name": name,
            "trained": artifact is not None,
            "artifact_dir": str(artifact) if artifact else None,
            "trained_at": manifest.get("created_at"),
            "test_end": (manifest.get("date_split") or {}).get("test", {}).get("end"),
        })
    return rows


def _load_saved_model(model_name: str, artifact: Path):
    if model_name == "master_lite":
        from asharerank.models.master import MasterLiteRegressor

        return MasterLiteRegressor.load(artifact / "model.pt")
    model = joblib.load(artifact / "model.joblib")
    # 在线横截面只有几十只股票，单线程更快且避免 Windows 服务进程创建管道失败。
    estimator = getattr(model, "estimator", None)
    if estimator is not None and hasattr(estimator, "n_jobs"):
        estimator.n_jobs = 1
    return model


def prepare_latest_feature_frame(
    panel: pd.DataFrame,
    minimum_coverage: float = 0.8,
) -> tuple[pd.DataFrame, pd.Timestamp, dict]:
    """构造无未来标签的最新横截面，并避免使用只有少量股票更新的日期。"""
    if panel.empty:
        raise ValueError("最新行情为空")
    features = build_common_features(panel)
    total_stocks = int(panel["ts_code"].nunique())
    required = max(5, math.ceil(total_stocks * minimum_coverage))
    counts = features.groupby("date")["ts_code"].nunique().sort_index()
    eligible = counts[counts >= required]
    if eligible.empty:
        raise ValueError(f"没有覆盖至少 {required} 只股票的共同预测日期")
    prediction_date = pd.Timestamp(eligible.index[-1])
    features["label"] = np.nan
    features.loc[features["date"] == prediction_date, "label"] = 0.0
    coverage = {
        "prediction_date": prediction_date.strftime("%Y-%m-%d"),
        "stocks_on_date": int(eligible.iloc[-1]),
        "total_stocks": total_stocks,
        "coverage_ratio": float(eligible.iloc[-1] / max(total_stocks, 1)),
    }
    return features, prediction_date, coverage


def predict_saved_models(
    panel: pd.DataFrame,
    model_names: list[str] | tuple[str, ...] | None = None,
) -> tuple[list[LiveModelOutput], dict[str, str], dict]:
    """加载最近一次保存的权重；本函数不会调用 fit。"""
    features, prediction_date, coverage = prepare_latest_feature_frame(panel)
    outputs: list[LiveModelOutput] = []
    errors: dict[str, str] = {}
    for raw_name in model_names or SUPPORTED_MODELS:
        name = raw_name.lower().strip()
        if name not in SUPPORTED_MODELS:
            errors[name] = "未知模型"
            continue
        artifact = latest_saved_artifact(name)
        if artifact is None:
            errors[name] = "没有已训练模型产物"
            continue
        try:
            adapter = joblib.load(artifact / "adapter.joblib")
            adapted = adapter.transform(features)
            model = _load_saved_model(name, artifact)
            started = time.perf_counter()
            scores = model.predict(adapted.X)
            elapsed = time.perf_counter() - started
            predictions = rank_predictions(
                build_prediction_frame(adapted.metadata, scores, name)
            )
            predictions = predictions[predictions["date"] == prediction_date].copy()
            manifest = {}
            manifest_path = artifact / "manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            outputs.append(LiveModelOutput(
                model_name=name,
                predictions=predictions,
                artifact_dir=str(artifact),
                trained_at=manifest.get("created_at"),
                inference_seconds=elapsed,
            ))
        except Exception as exc:
            errors[name] = str(exc)
    coverage["models_requested"] = len(model_names or SUPPORTED_MODELS)
    coverage["models_completed"] = len(outputs)
    coverage["stock_limit"] = PANEL_STOCK_LIMIT
    return outputs, errors, coverage
