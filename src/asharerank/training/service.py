"""Reusable model-comparison workflow and in-process progress state."""
from __future__ import annotations

import json
import logging
from threading import Lock

import pandas as pd

from asharerank.settings import MODEL_DIR
from asharerank.config import ExperimentConfig, PANEL_STOCK_LIMIT, SUPPORTED_MODELS
from asharerank.data import build_common_dataset
from asharerank.evaluation.comparison import compare_prediction_frames
from asharerank.evaluation.report import save_comparison_report
from asharerank.forecasting.universe import load_panel
from asharerank.prediction import (
    add_risk_balance_groups,
    balanced_top_k,
    balanced_universe_codes,
    load_local_stock_info,
    rank_predictions,
)
from asharerank.training.trainer import run_experiments

logger = logging.getLogger(__name__)

_REPORT_DIR = MODEL_DIR / "panel_comparison" / "reports"
_LATEST_RESULT_FILE = _REPORT_DIR / "latest_result.json"
_COMPARISON_FILE = _REPORT_DIR / "model_comparison.json"
_RANK_CONTEXT_FILE = _REPORT_DIR / "rank_context.parquet"

_STATE_LOCK = Lock()
_STATE: dict = {
    "status": "idle",
    "progress": 0.0,
    "message": "",
    "result": None,
}


def _load_persisted_result() -> dict | None:
    """优先恢复完整任务结果，兼容已经存在的旧版指标报告。"""
    try:
        if _LATEST_RESULT_FILE.exists():
            return json.loads(_LATEST_RESULT_FILE.read_text(encoding="utf-8"))
        if _COMPARISON_FILE.exists():
            comparison = json.loads(_COMPARISON_FILE.read_text(encoding="utf-8"))
            return {
                "models_completed": [row.get("model_name") for row in comparison],
                "errors": {},
                "comparison": comparison,
                "reports": {
                    "csv": str(_REPORT_DIR / "model_comparison.csv"),
                    "json": str(_COMPARISON_FILE),
                },
                "split": {},
            }
    except (OSError, ValueError, TypeError):
        logger.exception("读取上一次模型比较结果失败")
    return None


def comparison_status() -> dict:
    with _STATE_LOCK:
        if _STATE["status"] == "idle" and _STATE["result"] is None:
            persisted = _load_persisted_result()
            if persisted:
                _STATE.update(
                    status="done",
                    progress=1.0,
                    message="已加载上一次模型比较结果",
                    result=persisted,
                )
        return dict(_STATE)


def _set_state(**values) -> None:
    with _STATE_LOCK:
        _STATE.update(values)


def _safe_stock_info(codes: list[str]) -> pd.DataFrame:
    return load_local_stock_info(codes)


def save_rank_context(dataset) -> str:
    """缓存排名展示所需的波动率，不把它作为额外模型输入。"""
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    context = dataset.frame[["date", "ts_code", "volatility_20"]].copy()
    context.to_parquet(_RANK_CONTEXT_FILE, index=False)
    return str(_RANK_CONTEXT_FILE)


def rebuild_rank_context() -> str:
    panel = load_panel()
    dataset = build_common_dataset(panel, config=ExperimentConfig())
    return save_rank_context(dataset)


def _add_risk_balance_groups(daily: pd.DataFrame, selected_date: pd.Timestamp) -> pd.DataFrame:
    result = daily.copy()
    if "volatility_20" not in result.columns:
        if not _RANK_CONTEXT_FILE.exists():
            return result
        context = pd.read_parquet(_RANK_CONTEXT_FILE)
        context["date"] = pd.to_datetime(context["date"])
        context = context[context["date"].dt.normalize() == selected_date][
            ["ts_code", "volatility_20"]
        ]
        result = result.merge(context, on="ts_code", how="left")
    return add_risk_balance_groups(result)


def _latest_prediction_file(model_name: str):
    model_root = MODEL_DIR / "panel_comparison" / model_name
    if not model_root.exists():
        return None
    files = list(model_root.glob("*/test_predictions.parquet"))
    return max(files, key=lambda path: path.stat().st_mtime) if files else None


def available_ranking_models() -> list[str]:
    """返回至少有一份测试集排名产物的模型。"""
    return [name for name in SUPPORTED_MODELS if _latest_prediction_file(name)]


def load_latest_prediction_frames(
    model_names: list[str] | tuple[str, ...] | None = None,
) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for name in model_names or SUPPORTED_MODELS:
        path = _latest_prediction_file(name)
        if path is not None:
            frames.append(pd.read_parquet(path))
    return frames


def _attach_training_metadata(
    comparison: pd.DataFrame,
    metrics_by_model: dict[str, dict],
) -> pd.DataFrame:
    result = comparison.copy()
    fields = (
        "train_seconds",
        "inference_seconds",
        "n_train",
        "n_valid",
        "n_test",
        "validation_rank_ic",
        "orientation",
        "best_iteration",
    )
    for field in fields:
        result[field] = result["model_name"].map(
            lambda name: metrics_by_model.get(name, {}).get(field)
        )
    return result


def rebuild_latest_comparison(top_k: int = 10) -> dict:
    """用每个模型最近一次同口径产物重建五模型报告。"""
    frames: list[pd.DataFrame] = []
    metrics_by_model: dict[str, dict] = {}
    split: dict = {}
    for name in SUPPORTED_MODELS:
        path = _latest_prediction_file(name)
        if path is None:
            continue
        frames.append(pd.read_parquet(path))
        manifest_path = path.parent / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            metrics_by_model[name] = manifest.get("metrics") or {}
            split = split or manifest.get("date_split") or {}
    if not frames:
        raise FileNotFoundError("没有可用于重建报告的模型排名产物")

    comparison = compare_prediction_frames(
        frames,
        target_column="future_excess_return",
        k=top_k,
    )
    comparison = _attach_training_metadata(comparison, metrics_by_model)
    report_paths = save_comparison_report(comparison, _REPORT_DIR)
    payload = {
        "models_completed": comparison["model_name"].tolist(),
        "errors": {},
        "comparison": json.loads(comparison.to_json(orient="records")),
        "reports": report_paths,
        "split": split,
        "ranking_models": available_ranking_models(),
    }
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    _LATEST_RESULT_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _set_state(
        status="done",
        progress=1.0,
        message="已加载五模型比较与独立股票排名",
        result=payload,
    )
    return payload


def get_model_rankings(
    model_name: str,
    date: str | None = None,
    top_k: int = 20,
    balanced: bool = True,
) -> dict:
    """读取某一模型的历史测试排名；均衡选择不会覆盖模型原始 rank。"""
    name = model_name.lower().strip()
    if name not in SUPPORTED_MODELS:
        raise ValueError(f"未知模型 {model_name!r}")
    if not 1 <= top_k <= PANEL_STOCK_LIMIT:
        raise ValueError(f"top_k 必须位于 1 到 {PANEL_STOCK_LIMIT}")
    path = _latest_prediction_file(name)
    if path is None:
        raise FileNotFoundError(f"{name} 暂无排名产物，请先训练该模型")

    predictions = pd.read_parquet(path)
    predictions["date"] = pd.to_datetime(predictions["date"])
    predictions = rank_predictions(predictions)
    dates = sorted(predictions["date"].dropna().unique())
    if not dates:
        raise ValueError(f"{name} 的排名产物没有有效日期")
    selected_date = pd.Timestamp(date).normalize() if date else pd.Timestamp(dates[-1]).normalize()
    daily = predictions[predictions["date"].dt.normalize() == selected_date].copy()
    if daily.empty:
        raise ValueError(f"{name} 在 {selected_date.date()} 没有排名数据")

    codes = daily["ts_code"].astype(str).tolist()
    info = _safe_stock_info(codes)
    daily = daily.merge(
        info[["ts_code", "name", "industry", "market_cap"]].drop_duplicates("ts_code"),
        on="ts_code",
        how="left",
    )
    daily["name"] = daily["name"].fillna(daily["ts_code"])
    daily["industry"] = daily["industry"].fillna("其他")
    daily = daily.sort_values(["rank", "ts_code"]).reset_index(drop=True)
    daily = _add_risk_balance_groups(daily, selected_date)

    if balanced:
        selected, method = balanced_top_k(daily, top_k)
    else:
        selected = daily.head(top_k).copy()
        selected["balance_group"] = selected["industry"]
        selected["balanced_pick_rank"] = range(1, len(selected) + 1)
        selected["balanced_selected"] = False
        method = "model_rank"

    selected["date"] = selected["date"].dt.strftime("%Y-%m-%d")
    row_columns = [column for column in (
        "date",
        "model_name",
        "rank",
        "balanced_pick_rank",
        "balanced_selected",
        "ts_code",
        "name",
        "industry",
        "balance_group",
        "market_cap",
        "raw_score",
        "rank_pct",
        "rank_score",
        "future_return",
        "future_excess_return",
    ) if column in selected.columns]
    rows = json.loads(selected[row_columns].to_json(orient="records", force_ascii=False))
    return {
        "model_name": name,
        "date": selected_date.strftime("%Y-%m-%d"),
        "available_dates": [pd.Timestamp(value).strftime("%Y-%m-%d") for value in dates],
        "top_k": len(selected),
        "balanced": balanced,
        "selection_method": method,
        "group_count": int(selected["balance_group"].nunique()),
        "total_candidates": int(len(daily)),
        "artifact": str(path),
        "rows": rows,
    }


def run_comparison_job(
    models: list[str],
    max_stocks: int = PANEL_STOCK_LIMIT,
    lookback: int = 8,
    horizon: int = 5,
    target: str = "excess_return",
    top_k: int = 10,
) -> dict:
    """依次训练模型并返回指标表；不运行回测，也不融合模型分数。"""
    _set_state(status="running", progress=0.05, message="加载统一股票数据", result=None)
    try:
        panel = load_panel()
        stock_limit = min(max_stocks, PANEL_STOCK_LIMIT)
        all_codes = sorted(panel["ts_code"].astype(str).unique())
        stock_info = _safe_stock_info(all_codes)
        selected, universe_distribution = balanced_universe_codes(
            all_codes, stock_info, stock_limit
        )
        panel = panel[panel["ts_code"].isin(selected)]
        config = ExperimentConfig(
            lookback=lookback,
            horizon=horizon,
            target=target,
        )
        _set_state(progress=0.15, message="构建统一特征、标签和时间切分")
        dataset = build_common_dataset(panel, config=config)
        save_rank_context(dataset)
        results = []
        errors: dict[str, str] = {}
        for index, name in enumerate(models):
            _set_state(
                progress=0.2 + 0.65 * index / max(len(models), 1),
                message=f"独立训练 {name}（{index + 1}/{len(models)}）",
            )
            completed, failed = run_experiments(
                dataset,
                [name],
                experiment_config=config,
                top_k=top_k,
                save_artifacts=True,
                continue_on_error=True,
            )
            results.extend(completed)
            errors.update(failed)
        target_column = "future_excess_return" if target == "excess_return" else "future_return"
        comparison = compare_prediction_frames(
            [result.predictions for result in results],
            target_column=target_column,
            k=top_k,
        )
        comparison = _attach_training_metadata(
            comparison,
            {result.model_name: result.metrics for result in results},
        )
        report_paths = save_comparison_report(comparison, _REPORT_DIR)
        payload = {
            "models_completed": [result.model_name for result in results],
            "errors": errors,
            "comparison": comparison.to_dict(orient="records"),
            "reports": report_paths,
            "split": dataset.split.as_dict(),
            "universe": {
                "n_stocks": len(selected),
                "codes": selected,
                "group_distribution": universe_distribution,
            },
            "ranking_models": [result.model_name for result in results],
        }
        _REPORT_DIR.mkdir(parents=True, exist_ok=True)
        _LATEST_RESULT_FILE.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _set_state(status="done", progress=1.0, message="模型比较完成", result=payload)
        return payload
    except Exception as exc:
        logger.exception("模型比较任务失败")
        _set_state(status="error", progress=0.0, message=str(exc), result=None)
        raise
