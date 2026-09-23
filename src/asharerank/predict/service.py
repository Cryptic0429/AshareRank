"""预测服务：批量推理、写入 predictions 表、Daily Picks / SHAP。"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from asharerank import data_service, db
from asharerank.settings import FEATURE_WARMUP
from asharerank.predict.features import build_feature_panel
from asharerank.predict.registry import (
    get_active_model_id,
    get_model_meta,
    list_models,
    load_booster,
)
from asharerank.predict.universe import load_constituents, load_panel

logger = logging.getLogger(__name__)


def _should_isolate_tree_infer() -> bool:
    """是否必须把树模型推理放到子进程。

    macOS 上 PyTorch 与 LightGBM 的 OpenMP 运行时同进程会 SIGSEGV（整站「无法连接」）。
    只要当前进程已加载 torch，就必须隔离；在 Darwin 上默认也隔离，避免先 LGBM
    后 StockMixer 时反向踩雷。
    """
    if "torch" in sys.modules:
        return True
    return sys.platform == "darwin"


def _run_tree_infer_worker(args: list[str]) -> dict:
    """在干净子进程里跑 tree_infer_worker，返回 result 字典。"""
    with tempfile.TemporaryDirectory(prefix="asharerank_tree_") as tmp:
        out_path = Path(tmp) / "result.json"
        cmd = [
            sys.executable, "-m", "asharerank.predict.tree_infer_worker",
            *args, "--out", str(out_path),
        ]
        logger.info("树模型隔离推理: %s", " ".join(cmd))
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if proc.returncode != 0 and not out_path.exists():
            detail = (proc.stderr or proc.stdout or "").strip()[-2000:]
            raise RuntimeError(
                f"树模型子进程失败（exit={proc.returncode}）: {detail or '无输出'}"
            )
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        if not payload.get("ok"):
            raise RuntimeError(payload.get("error") or "树模型子进程返回失败")
        return payload["result"]


def _pick_coverage_date(feats: pd.DataFrame, min_ratio: float = 0.5,
                        min_count: int = 30) -> pd.Timestamp:
    """选「覆盖股票数足够多」的最新交易日。

    个股行情增量刷新常常参差不齐（部分股票先拉到最新日），若直接取全局
    max(date) 会导致截面只剩个别抢先更新的股票。这里改为在最近若干个交易日中，
    找覆盖数达到近期满仓水平一定比例的最新日期。
    """
    counts = feats.groupby("date").size().sort_index()
    if counts.empty:
        return feats["date"].max()
    full = counts.tail(10).max()
    threshold = max(min_count, full * min_ratio)
    covered = counts[counts >= threshold]
    return covered.index.max() if not covered.empty else counts.index.max()


def _latest_feature_frame(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    if panel is None:
        # 产品推理只打 CSI300 成分股，避免 data 里多余 parquet 污染截面
        try:
            codes = load_constituents()["ts_code"].tolist()
            panel = load_panel(codes=codes)
        except Exception:
            panel = load_panel()
    dates = sorted(panel["date"].unique())
    keep_from = dates[-(FEATURE_WARMUP + 30):][0] if len(dates) > FEATURE_WARMUP + 30 else dates[0]
    panel = panel[panel["date"] >= keep_from]
    feats = build_feature_panel(panel)
    feature_cols = [c for c in feats.columns if c not in ("date", "ts_code")]
    last_date = _pick_coverage_date(feats)
    day = feats[feats["date"] == last_date].copy()
    for col in feature_cols:
        mu, sigma = day[col].mean(), day[col].std(ddof=0)
        day[col] = 0.0 if not sigma else (day[col] - mu) / sigma
    return day.dropna(subset=feature_cols)


def _resolve_model_meta(model_id: str | None) -> dict:
    mid = model_id or get_active_model_id()
    if not mid:
        raise FileNotFoundError("尚无已训练模型，请先运行 scripts/research/train_lgbm.py "
                                "或 scripts/research/deploy_stockmixer.py")
    meta = get_model_meta(mid)
    if meta is None:
        raise FileNotFoundError(f"未知模型: {mid}")
    return meta


def _augment_scores(day: pd.DataFrame) -> pd.DataFrame:
    """在打分结果上补充 rank_pct / pred_return_5d / confidence，两条推理路径共用。"""
    day = day.copy()
    # average：并列时分位相同（科学上更合理）；展示排序用稳定次序
    day["rank_pct"] = day["alpha_score"].rank(pct=True, method="average")
    # 展示用「伪 5 日收益」：按分位线性缩放到约 ±3%（非模型直接输出）
    day["pred_return_5d"] = day["rank_pct"].map(lambda r: (r - 0.5) * 0.06)
    # 置信度：相对截面分数离散度，避免头部全是 95%
    mu = float(day["alpha_score"].mean())
    sigma = float(day["alpha_score"].std(ddof=0)) or 1e-9
    day["confidence"] = ((day["alpha_score"] - mu).abs() / sigma).clip(0, 3) / 3 * 0.45 + 0.50
    day["confidence"] = day["confidence"].clip(0.50, 0.95)
    return day


def predict_latest_tree_inplace(model_id: str | None = None, top_k: int = 20) -> dict:
    """同进程树模型推理（仅应由干净子进程 / 未加载 torch 的进程调用）。"""
    booster, feature_names, meta = load_booster(model_id)
    day = _latest_feature_frame()
    missing = [c for c in feature_names if c not in day.columns]
    if missing:
        raise RuntimeError(f"特征缺失: {missing[:5]}…")
    X = day[feature_names].values
    day = day.copy()
    day["alpha_score"] = booster.predict(X)

    day = _augment_scores(day)
    as_of = str(pd.Timestamp(day["date"].iloc[0]).date())
    mid = meta["model_id"]
    rows = [
        {"date": as_of, "ts_code": r.ts_code, "model_id": mid,
         "alpha_score": float(r.alpha_score), "rank_pct": float(r.rank_pct)}
        for r in day.itertuples()
    ]
    db.init_db()
    _save_predictions(rows)

    return _format_picks_response(
        day, as_of=as_of, model_id=mid, backend=meta.get("backend"), top_k=top_k,
    )


def predict_latest(model_id: str | None = None, top_k: int = 20) -> dict:
    """对最新交易日全市场打分，写库并返回 TopK。按模型 backend 分派到对应推理路径。"""
    meta = _resolve_model_meta(model_id)
    if meta.get("backend") == "stockmixer":
        from asharerank.predict.stockmixer_infer import predict_latest_stockmixer

        day = predict_latest_stockmixer(meta, top_k=top_k)
        day = _augment_scores(day)
        as_of = str(pd.Timestamp(day["date"].iloc[0]).date())
        mid = meta["model_id"]
        rows = [
            {"date": as_of, "ts_code": r.ts_code, "model_id": mid,
             "alpha_score": float(r.alpha_score), "rank_pct": float(r.rank_pct)}
            for r in day.itertuples()
        ]
        db.init_db()
        _save_predictions(rows)
        return _format_picks_response(
            day, as_of=as_of, model_id=mid, backend=meta.get("backend"), top_k=top_k,
        )

    if _should_isolate_tree_infer():
        args = ["predict", "--top-k", str(int(top_k))]
        if model_id:
            args += ["--model-id", model_id]
        return _run_tree_infer_worker(args)
    return predict_latest_tree_inplace(model_id=model_id, top_k=top_k)


def get_daily_picks(model_id: str | None = None, top_k: int = 20,
                    force_refresh: bool = False) -> dict:
    """优先读当日已落库预测；没有则现场推理。"""
    if not force_refresh:
        cached = _picks_from_db(model_id=model_id, top_k=top_k)
        if cached is not None:
            return cached
    return predict_latest(model_id=model_id, top_k=top_k)


def _stock_names(codes: list[str]) -> dict[str, str]:
    names: dict[str, str] = {}
    try:
        info = data_service.get_stock_info(codes)
        names.update(dict(zip(info["ts_code"], info["name"])))
    except Exception:
        pass
    try:
        cons = load_constituents().set_index("ts_code")["name"].to_dict()
        for k, v in cons.items():
            if v and v != k:
                names[k] = v
    except Exception:
        pass
    return names


def _format_picks_response(day: pd.DataFrame, *, as_of: str, model_id: str,
                           backend: str | None, top_k: int) -> dict:
    names = _stock_names(day["ts_code"].tolist())
    day = day.copy()
    if "pred_return_5d" not in day.columns:
        day["pred_return_5d"] = day["rank_pct"].map(lambda r: (r - 0.5) * 0.06)
    if "confidence" not in day.columns:
        mu = float(day["alpha_score"].mean())
        sigma = float(day["alpha_score"].std(ddof=0)) or 1e-9
        day["confidence"] = ((day["alpha_score"] - mu).abs() / sigma).clip(0, 3) / 3 * 0.45 + 0.50
        day["confidence"] = day["confidence"].clip(0.50, 0.95)
    # 同分时按代码稳定排序，避免每次顺序乱跳
    top = day.sort_values(["alpha_score", "ts_code"], ascending=[False, True]).head(top_k)
    picks = []
    for r in top.itertuples():
        picks.append({
            "ts_code": r.ts_code,
            "name": names.get(r.ts_code, r.ts_code),
            "pred_return_5d": float(r.pred_return_5d),
            "alpha_score": float(r.alpha_score),
            "rank_pct": float(r.rank_pct),
            "confidence": float(r.confidence),
        })
    return {
        "stub": False,
        "date": as_of,
        "model_id": model_id,
        "backend": backend,
        "n_universe": int(len(day)),
        "picks": picks,
        "source": "live",
    }


def _picks_from_db(model_id: str | None, top_k: int) -> dict | None:
    """若 predictions 表已有「面板最新交易日」且覆盖足够股票，直接组装 TopK。"""
    from asharerank.predict.registry import get_model_meta
    from asharerank.predict.universe import panel_asof_date

    mid = model_id or get_active_model_id()
    if not mid:
        return None
    as_of = panel_asof_date()
    if not as_of:
        return None
    db.init_db()
    with db.get_conn() as conn:
        rows = conn.execute(
            """SELECT ts_code, alpha_score, rank_pct FROM predictions
               WHERE date = ? AND model_id = ?
               ORDER BY alpha_score DESC""",
            (as_of, mid),
        ).fetchall()
    if len(rows) < 30:
        return None
    day = pd.DataFrame([dict(r) for r in rows])
    meta = get_model_meta(mid) or {}
    out = _format_picks_response(
        day, as_of=as_of, model_id=mid, backend=meta.get("backend"), top_k=top_k,
    )
    out["source"] = "db"
    return out


def _save_predictions(rows: list[dict]) -> None:
    if not rows:
        return
    as_of = rows[0]["date"]
    mid = rows[0]["model_id"]
    with db.get_conn() as conn:
        conn.execute(
            "DELETE FROM predictions WHERE date = ? AND model_id = ?",
            (as_of, mid),
        )
        conn.executemany(
            """INSERT OR REPLACE INTO predictions
               (date, ts_code, model_id, alpha_score, rank_pct)
               VALUES (:date, :ts_code, :model_id, :alpha_score, :rank_pct)""",
            rows,
        )


def explain_top_features_tree_inplace(ts_code: str, model_id: str | None = None,
                                      top_n: int = 8) -> dict:
    """同进程树模型解释（仅应由干净子进程 / 未加载 torch 的进程调用）。"""
    booster, feature_names, meta = load_booster(model_id)
    day = _latest_feature_frame()
    row = day[day["ts_code"] == ts_code]
    if row.empty:
        raise ValueError(f"当日无该股票特征: {ts_code}")
    x = row[feature_names].iloc[0]
    # 简化：用特征值 × 全局重要性近似（避免 SHAP 在无 libomp 环境下额外依赖问题）
    importance = _feature_importance(booster, feature_names)
    contrib = (x.abs() * pd.Series(importance)).sort_values(ascending=False).head(top_n)
    return {
        "ts_code": ts_code,
        "model_id": meta["model_id"],
        "features": [
            {"name": k, "value": float(x[k]), "importance": float(importance.get(k, 0)),
             "contribution": float(v)}
            for k, v in contrib.items()
        ],
    }


def explain_top_features(ts_code: str, model_id: str | None = None,
                         top_n: int = 8) -> dict:
    """用特征对预测的局部贡献做简化解释（增益近似 / 可选 SHAP）。"""
    meta = _resolve_model_meta(model_id)
    if meta.get("backend") == "stockmixer":
        from asharerank.predict.stockmixer_infer import explain_top_features_stockmixer

        return explain_top_features_stockmixer(meta, ts_code, top_n=top_n)

    if _should_isolate_tree_infer():
        args = ["explain", "--ts-code", ts_code, "--top-n", str(int(top_n))]
        if model_id:
            args += ["--model-id", model_id]
        return _run_tree_infer_worker(args)
    return explain_top_features_tree_inplace(ts_code, model_id=model_id, top_n=top_n)


def _feature_importance(booster, feature_names: list[str]) -> dict[str, float]:
    if booster.backend == "lightgbm":
        imp = booster.model.feature_importance(importance_type="gain")
        total = float(imp.sum()) or 1.0
        return {n: float(v) / total for n, v in zip(feature_names, imp)}
    # sklearn HistGBM
    if hasattr(booster.model, "feature_importances_"):
        imp = booster.model.feature_importances_
        total = float(imp.sum()) or 1.0
        return {n: float(v) / total for n, v in zip(feature_names, imp)}
    return {n: 1.0 / len(feature_names) for n in feature_names}


def models_for_api() -> dict:
    models = list_models()
    return {"stub": False, "models": models, "active": get_active_model_id()}
