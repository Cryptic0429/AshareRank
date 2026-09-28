"""构建并缓存 Panel 五模型共用的数据集。"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from asharerank.settings import PROCESSED_DATA_DIR
from asharerank.config import ExperimentConfig, PANEL_STOCK_LIMIT
from asharerank.data import build_common_dataset
from asharerank.forecasting.universe import load_panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-stocks", type=int, default=PANEL_STOCK_LIMIT)
    parser.add_argument("--lookback", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--target", choices=["return", "excess_return"], default="excess_return")
    parser.add_argument("--output", type=Path, default=PROCESSED_DATA_DIR / "common_dataset.parquet")
    args = parser.parse_args()
    from asharerank.research_record import record_arguments
    record_arguments(args)

    panel = load_panel()
    stock_limit = min(args.max_stocks, PANEL_STOCK_LIMIT)
    selected = sorted(panel["ts_code"].unique())[:stock_limit]
    panel = panel[panel["ts_code"].isin(selected)]
    config = ExperimentConfig(
        lookback=args.lookback,
        horizon=args.horizon,
        target=args.target,
    )
    dataset = build_common_dataset(panel, config=config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.frame.to_parquet(args.output, index=False)
    metadata_path = args.output.with_suffix(".json")
    from asharerank.research_record import environment_record, input_manifest
    metadata_path.write_text(json.dumps({
        **environment_record(),
        "data_manifest": input_manifest(),
        "config": asdict(config),
        "split": dataset.split.as_dict(),
        "rows": len(dataset.frame),
        "stocks": int(dataset.frame["ts_code"].nunique()),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "dataset": str(args.output),
        "metadata": str(metadata_path),
        "rows": len(dataset.frame),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    from asharerank.research_record import run_recorded
    run_recorded(main)
