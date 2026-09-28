"""梯度提升后端：优先 LightGBM，缺 libomp 时回退到 sklearn HistGradientBoosting。"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_BACKEND = None


def _ensure_libomp() -> None:
    """macOS 上为 LightGBM 补齐 libomp（仅 Homebrew；勿注入 torch 自带的 libomp）。

    其它项目 / 本机 venv 里的 ``torch/lib/libomp.dylib`` 与 LightGBM 在同进程
    混用常直接 SIGSEGV（exit -11），因此明确禁止扫描 torch 路径。
    未安装时请: ``brew install libomp``，否则 ``get_backend()`` 回退 HistGBM。
    """
    if "libomp" in os.environ.get("DYLD_LIBRARY_PATH", ""):
        return
    candidates = [
        Path("/opt/homebrew/opt/libomp/lib"),
        Path("/usr/local/opt/libomp/lib"),
    ]
    # Homebrew 可能装在非默认前缀
    for brew_bin in ("/opt/homebrew/bin/brew", "/usr/local/bin/brew"):
        if Path(brew_bin).exists():
            try:
                import subprocess
                prefix = subprocess.check_output(
                    [brew_bin, "--prefix", "libomp"],
                    text=True, stderr=subprocess.DEVNULL, timeout=3,
                ).strip()
                if prefix:
                    candidates.append(Path(prefix) / "lib")
            except Exception:
                pass
            break
    for d in candidates:
        dylib = d / "libomp.dylib"
        if not dylib.exists():
            continue
        cur = os.environ.get("DYLD_LIBRARY_PATH", "")
        os.environ["DYLD_LIBRARY_PATH"] = f"{d}{os.pathsep}{cur}" if cur else str(d)
        try:
            import ctypes
            ctypes.CDLL(str(dylib))
        except Exception as exc:
            logger.warning("加载 %s 失败: %s", dylib, exc)
            continue
        logger.info("已注入 libomp 路径: %s", d)
        return
    logger.info("未找到 Homebrew libomp，LightGBM 可能不可用（brew install libomp）")


def get_backend() -> str:
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    _ensure_libomp()
    try:
        import lightgbm  # noqa: F401
        _BACKEND = "lightgbm"
    except Exception as exc:
        logger.warning("LightGBM 不可用（%s），回退到 HistGradientBoosting。"
                       "若在 macOS，可执行: brew install libomp", exc)
        _BACKEND = "sklearn_hgb"
    return _BACKEND


@dataclass
class BoosterModel:
    backend: str
    model: Any
    feature_names: list[str]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(X), dtype=float)


def train_booster(X_train, y_train, X_valid, y_valid,
                  feature_names: list[str],
                  params: dict | None = None) -> BoosterModel:
    backend = get_backend()
    params = params or {}

    if backend == "lightgbm":
        import lightgbm as lgb

        lgb_params = {
            "objective": "regression",
            "metric": "l2",
            "learning_rate": 0.05,
            "num_leaves": 64,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 1.0,
            "reg_lambda": 5.0,
            "min_child_samples": 50,
            "verbosity": -1,
            "n_jobs": -1,
            **params,
        }
        dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
        dvalid = lgb.Dataset(X_valid, label=y_valid, feature_name=feature_names,
                             reference=dtrain)
        model = lgb.train(
            lgb_params, dtrain,
            num_boost_round=params.get("num_boost_round", 400),
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(40, verbose=False),
                       lgb.log_evaluation(period=0)],
        )
        return BoosterModel(backend="lightgbm", model=model,
                            feature_names=feature_names)

    from sklearn.ensemble import HistGradientBoostingRegressor

    # 仅在训练集上 fit；早停用训练集内部分割，严禁把 valid 拼进训练（否则 OOS 指标虚高）
    model = HistGradientBoostingRegressor(
        max_depth=params.get("max_depth", 6),
        learning_rate=params.get("learning_rate", 0.05),
        max_iter=params.get("num_boost_round", 300),
        l2_regularization=params.get("reg_lambda", 5.0),
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=40,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return BoosterModel(backend="sklearn_hgb", model=model,
                        feature_names=feature_names)
