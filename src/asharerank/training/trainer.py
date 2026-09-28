"""在相同数据、标签与时间切分下独立训练和评价候选模型。"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from asharerank.settings import MODEL_DIR
from asharerank.adapters import (
    AdaptedData,
    MasterAdapter,
    RankingTabularAdapter,
    TabularAdapter,
)
from asharerank.config import ExperimentConfig
from asharerank.data.dataset import CommonDataset
from asharerank.evaluation.comparison import evaluate_predictions
from asharerank.evaluation.efficiency_metrics import measure_time
from asharerank.models.registry import create_model
from asharerank.prediction import build_prediction_frame, rank_predictions
from asharerank.training.reproducibility import set_random_seed


@dataclass(frozen=True)
class ExperimentResult:
    model_name: str
    metrics: dict[str, Any]
    predictions: pd.DataFrame
    train_seconds: float
    inference_seconds: float
    artifact_dir: str | None = None


def _adapter_for(model_name: str, lookback: int):
    if model_name == "master_lite":
        return MasterAdapter(lookback=lookback)
    if model_name == "lite_rank":
        return RankingTabularAdapter(lookback=lookback)
    return TabularAdapter(lookback=lookback, scale=(model_name == "ridge"))


def _slice_X(X, mask: np.ndarray):
    if isinstance(X, dict):
        return {key: np.asarray(value)[mask] for key, value in X.items()}
    return np.asarray(X)[mask]


def _slice_adapted(adapted: AdaptedData, dates) -> AdaptedData:
    selected_dates = set(pd.to_datetime(dates))
    mask = pd.to_datetime(adapted.metadata["date"]).isin(selected_dates).to_numpy()
    if not mask.any():
        raise ValueError("时间切分后没有可用的时序样本")
    return AdaptedData(
        X=_slice_X(adapted.X, mask),
        y=adapted.y[mask],
        metadata=adapted.metadata.loc[mask].reset_index(drop=True),
        feature_names=adapted.feature_names,
    )


def _save_artifacts(
    result: ExperimentResult,
    model,
    adapter,
    dataset: CommonDataset,
    experiment_config: ExperimentConfig,
    output_root: str | Path | None,
) -> str:
    root = Path(output_root) if output_root else MODEL_DIR / "panel_comparison"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destination = root / result.model_name / run_id
    destination.mkdir(parents=True, exist_ok=False)
    model_suffix = ".pt" if result.model_name == "master_lite" else ".joblib"
    model.save(destination / f"model{model_suffix}")
    joblib.dump(adapter, destination / "adapter.joblib")
    result.predictions.to_parquet(destination / "test_predictions.parquet", index=False)
    from asharerank.research_record import environment_record, input_manifest
    manifest = {
        **environment_record(),
        "run_id": run_id,
        "data_manifest": input_manifest(),
        "model_name": result.model_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "experiment_config": asdict(experiment_config),
        "date_split": dataset.split.as_dict(),
        "feature_names": list(result.predictions.attrs.get("feature_names", [])),
        "metrics": result.metrics,
        "train_seconds": result.train_seconds,
        "inference_seconds": result.inference_seconds,
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return str(destination)


def run_experiment(
    dataset: CommonDataset,
    model_name: str,
    experiment_config: ExperimentConfig | None = None,
    model_params: dict[str, Any] | None = None,
    top_k: int = 10,
    save_artifacts: bool = True,
    output_root: str | Path | None = None,
) -> ExperimentResult:
    """训练一个模型并只在冻结测试集上输出独立预测与指标。"""
    config = experiment_config or ExperimentConfig()
    name = model_name.lower().strip()
    set_random_seed(config.random_seed)

    train_frame = dataset.partition("train")
    adapter = _adapter_for(name, config.lookback)
    adapter.fit(train_frame)
    # 在完整时间轴上构造窗口，使验证/测试首日可以使用此前已知的历史数据。
    adapted = adapter.transform(dataset.frame)
    train = _slice_adapted(adapted, dataset.split.train_dates)
    valid = _slice_adapted(adapted, dataset.split.valid_dates)
    test = _slice_adapted(adapted, dataset.split.test_dates)

    model = create_model(name, **(model_params or {}))
    with measure_time() as train_timer:
        model.fit(train.X, train.y, valid.X, valid.y)
    with measure_time() as inference_timer:
        scores = model.predict(test.X)

    predictions = rank_predictions(build_prediction_frame(test.metadata, scores, name))
    target_column = (
        "future_excess_return" if config.target == "excess_return" else "future_return"
    )
    metrics = evaluate_predictions(predictions, target_column=target_column, k=top_k)
    metrics.update({
        "train_seconds": train_timer.seconds,
        "inference_seconds": inference_timer.seconds,
        "n_train": len(train.y),
        "n_valid": len(valid.y),
        "n_test": len(test.y),
    })
    for attribute in ("validation_rank_ic", "orientation", "best_iteration"):
        if hasattr(model, attribute):
            metrics[attribute] = getattr(model, attribute)
    predictions.attrs["feature_names"] = list(adapted.feature_names)
    initial = ExperimentResult(
        model_name=name,
        metrics=metrics,
        predictions=predictions,
        train_seconds=train_timer.seconds,
        inference_seconds=inference_timer.seconds,
    )
    if not save_artifacts:
        return initial
    artifact_dir = _save_artifacts(
        initial, model, adapter, dataset, config, output_root=output_root
    )
    return ExperimentResult(
        model_name=initial.model_name,
        metrics=initial.metrics,
        predictions=initial.predictions,
        train_seconds=initial.train_seconds,
        inference_seconds=initial.inference_seconds,
        artifact_dir=artifact_dir,
    )


def run_experiments(
    dataset: CommonDataset,
    model_names: list[str] | tuple[str, ...],
    experiment_config: ExperimentConfig | None = None,
    model_params: dict[str, dict[str, Any]] | None = None,
    top_k: int = 10,
    save_artifacts: bool = True,
    output_root: str | Path | None = None,
    continue_on_error: bool = True,
) -> tuple[list[ExperimentResult], dict[str, str]]:
    """依次运行独立实验；此函数不会融合任何模型预测。"""
    results: list[ExperimentResult] = []
    errors: dict[str, str] = {}
    for name in model_names:
        try:
            results.append(run_experiment(
                dataset=dataset,
                model_name=name,
                experiment_config=experiment_config,
                model_params=(model_params or {}).get(name),
                top_k=top_k,
                save_artifacts=save_artifacts,
                output_root=output_root,
            ))
        except Exception as exc:
            errors[name] = str(exc)
            if not continue_on_error:
                raise
    return results, errors
