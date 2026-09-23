"""行业中性化标签 vs 全市场标签：对比两种评估口径下的 RankIC。

- 全市场 RankIC：预测分数 vs 全市场当日排名（推荐产品实际使用的口径——用户从全市场选 TopK）。
- 行业内 RankIC：预测分数 vs 同行业内排名（剔除行业 beta，衡量纯 alpha 信号强弱）。

行业中性化标签训练出来的模型，通常全市场 RankIC 会略降（因为它主动放弃了"押行业"这部分
经常有效的信号），但行业内 RankIC 应该更高（选股能力更纯粹）。两个数字都要看，
不能只看一个就下结论。
"""
from __future__ import annotations

import logging

from asharerank.predict.evaluate import calc_ic_series, calc_industry_neutral_ic, summarize_ic
from asharerank.predict.experimental.report import EXPERIMENTAL_DIR, save_report
from asharerank.predict.experimental.xgboost_baseline import run_xgboost_baseline
from asharerank.predict.industry import load_industry_map

logger = logging.getLogger(__name__)


def run_industry_compare(max_stocks: int | None = None) -> dict:
    industry_map = load_industry_map()

    logger.info("=== baseline: 全市场标签 ===")
    report_full = run_xgboost_baseline(max_stocks=max_stocks, tag="xgboost_market_label")

    logger.info("=== baseline: 行业中性化标签 ===")
    report_neutral = run_xgboost_baseline(max_stocks=max_stocks, industry_map=industry_map,
                                          tag="xgboost_industry_label")

    import pandas as pd
    oos_full = pd.read_parquet(report_full["oos_predictions_path"])
    oos_neutral = pd.read_parquet(report_neutral["oos_predictions_path"])

    result = {}
    for name, oos in (("market_label", oos_full), ("industry_label", oos_neutral)):
        market_ic = summarize_ic(calc_ic_series(oos, label_col="fwd_ret"))
        ind_ic_df = calc_industry_neutral_ic(oos, industry_map, label_col="fwd_ret")
        ind_ic = summarize_ic(ind_ic_df)
        result[name] = {"market_rankic": market_ic, "industry_neutral_rankic": ind_ic}

    print("\n=== 行业中性化对比（同一批股票池 / 相同 walk-forward 切分）===")
    for name, r in result.items():
        m, ind = r["market_rankic"], r["industry_neutral_rankic"]
        print(f"  [{name}]")
        print(f"    全市场 RankIC={m['rank_ic_mean']:.4f}  RankICIR={m['rank_icir']:.3f}")
        print(f"    行业内 RankIC={ind['rank_ic_mean']:.4f}  RankICIR={ind['rank_icir']:.3f}  "
              f"(n_groups={ind['n_days']})")

    save_report("industry_neutral_compare", result)
    return result
