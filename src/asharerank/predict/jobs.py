"""Alpha 日终任务：增量更新行情 → 推理 → 落库。"""
from __future__ import annotations

import logging
import time
from typing import Any

from asharerank.predict.service import predict_latest
from asharerank.predict.universe import (
    DEFAULT_POOL,
    download_universe,
    panel_asof_date,
)

logger = logging.getLogger(__name__)


def run_daily_alpha_refresh(
    max_stocks: int | None = None,
    top_k: int = 30,
    sleep_s: float = 0.25,
    lookback_days: int = 5,
    skip_download: bool = False,
    pool: str = DEFAULT_POOL,
) -> dict[str, Any]:
    """
    交易日收盘后流水线：
      1. 增量更新 universe OHLCV（默认中证800）
      2. 用已部署模型全市场打分并写入 predictions
    """
    t0 = time.time()
    asof_before = panel_asof_date()
    logger.info(
        "Alpha 日终刷新开始 asof_before=%s pool=%s max_stocks=%s",
        asof_before, pool, max_stocks,
    )

    download_stats: dict[str, Any] = {"skipped": True}
    if not skip_download:
        download_stats = download_universe(
            max_stocks=max_stocks,
            sleep_s=sleep_s,
            update_existing=True,
            lookback_days=lookback_days,
            pool=pool,
        )
        logger.info("行情增量完成: %s", download_stats)

    asof_after = panel_asof_date()
    picks = predict_latest(model_id=None, top_k=top_k)
    elapsed = time.time() - t0
    result = {
        "ok": True,
        "asof_before": asof_before,
        "asof_after": asof_after,
        "download": download_stats,
        "predict_date": picks.get("date"),
        "model_id": picks.get("model_id"),
        "n_universe": picks.get("n_universe"),
        "n_picks": len(picks.get("picks") or []),
        "elapsed_sec": round(elapsed, 1),
    }
    logger.info(
        "Alpha 日终刷新完成 date=%s model=%s universe=%s elapsed=%.1fs",
        result["predict_date"], result["model_id"],
        result["n_universe"], elapsed,
    )
    return result


def daily_alpha_job() -> None:
    """APScheduler 入口：全量增量刷新（失败只打日志，不抛到调度器外）。"""
    try:
        run_daily_alpha_refresh(max_stocks=None, top_k=30)
    except Exception:
        logger.exception("Alpha 日终刷新失败")
