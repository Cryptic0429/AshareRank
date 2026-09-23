"""在同一数据集上独立训练模型并生成预测指标对比，不执行回测。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from asharerank.settings import MODEL_DIR
from asharerank.config import ExperimentConfig, PANEL_STOCK_LIMIT
from asharerank.data import build_common_dataset
from asharerank.evaluation.comparison import compare_prediction_frames
from asharerank.evaluation.report import save_comparison_report
from asharerank.predict.universe import load_panel
from asharerank.prediction import balanced_universe_codes, load_local_stock_info
from asharerank.training import run_experiments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["ridge", "random_forest", "lightgbm", "lite_rank", "master_lite"],
    )
    parser.add_argument("--max-stocks", type=int, default=PANEL_STOCK_LIMIT)
    parser.add_argument("--lookback", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--target", choices=["return", "excess_return"], default="excess_return")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    panel = load_panel()
    stock_limit = min(args.max_stocks, PANEL_STOCK_LIMIT)
    all_codes = sorted(panel["ts_code"].astype(str).unique())
    selected, universe_distribution = balanced_universe_codes(
        all_codes,
        load_local_stock_info(all_codes),
        stock_limit,
    )
    panel = panel[panel["ts_code"].isin(selected)]
    config = ExperimentConfig(
        lookback=args.lookback,
        horizon=args.horizon,
        target=args.target,
    )
    dataset = build_common_dataset(panel, config=config)
    results, errors = run_experiments(
        dataset,
        model_names=args.models,
        experiment_config=config,
        top_k=args.top_k,
        save_artifacts=not args.no_save,
    )
    comparison = compare_prediction_frames(
        [result.predictions for result in results],
        target_column=("future_excess_return" if args.target == "excess_return" else "future_return"),
        k=args.top_k,
    )
    reports = save_comparison_report(comparison, MODEL_DIR / "panel_comparison" / "reports")
    print(json.dumps({
        "models_completed": [result.model_name for result in results],
        "errors": errors,
        "reports": reports,
        "comparison": comparison.to_dict(orient="records"),
        "universe": {
            "codes": selected,
            "group_distribution": universe_distribution,
        },
    }, ensure_ascii=False, indent=2))
    if not results:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
