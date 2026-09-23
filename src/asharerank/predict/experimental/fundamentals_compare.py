"""基本面因子消融对比：同一批股票池 + 相同切分，有/无基本面特征的 RankIC 对比。

只在已下载到基本面数据的股票子集上做对比（否则两组样本覆盖范围不一致，对比不公平）。
"""
from __future__ import annotations

import logging

from asharerank.predict.experimental.xgboost_baseline import run_xgboost_baseline
from asharerank.predict.fundamentals import build_fundamental_feature_panel, list_available_codes as list_fund_codes
from asharerank.predict.universe import load_panel

logger = logging.getLogger(__name__)


def run_fundamentals_compare() -> dict:
    codes = list_fund_codes()
    if len(codes) < 10:
        raise ValueError(f"基本面数据覆盖股票太少({len(codes)})，请先跑 scripts/research/fetch_fundamentals.py")
    logger.info("基本面因子消融对比：%d 只股票（有基本面数据覆盖的子集）", len(codes))

    panel = load_panel()
    panel = panel[panel["ts_code"].isin(codes)]
    fund_panel = build_fundamental_feature_panel(panel, codes=codes)
    coverage = fund_panel.dropna(how="all", subset=[c for c in fund_panel.columns if c not in ("date", "ts_code")])
    logger.info("基本面特征覆盖行数: %d / %d", len(coverage), len(fund_panel))

    logger.info("=== baseline: 无基本面特征（纯技术因子） ===")
    report_base = run_xgboost_baseline(codes=codes, tag="xgboost_no_fundamentals")

    logger.info("=== baseline: 技术因子 + 基本面因子 ===")
    report_fund = run_xgboost_baseline(codes=codes, extra_features=fund_panel, tag="xgboost_with_fundamentals")

    base_ic = report_base["oos_metrics"]
    fund_ic = report_fund["oos_metrics"]
    print("\n=== 基本面因子消融对比（同批 %d 只股票）===" % len(codes))
    print(f"  无基本面: RankIC={base_ic['rank_ic_mean']:.4f}  RankICIR={base_ic['rank_icir']:.3f}  "
          f"样本数={report_base['n_samples']}")
    print(f"  加基本面: RankIC={fund_ic['rank_ic_mean']:.4f}  RankICIR={fund_ic['rank_icir']:.3f}  "
          f"样本数={report_fund['n_samples']}")

    return {"no_fundamentals": report_base, "with_fundamentals": report_fund}
