"""列出 `runs/experiments/runs/index.json` 里记录的全部历史实验。

覆盖式的 `{name}_report.json` 只保留最新一次结果；每次调用
`asharerank/forecasting/experimental/report.py::save_report` 时，都会额外在
`runs/{run_id}/report.json` 存一份版本化快照，并把摘要 append 进
`runs/index.json`——这个脚本就是读那份索引，按时间列出所有实验，方便横向
对比"这周跑的第 5 版 XGBoost 对齐实验" vs "上周那版"。

用法:
  python scripts/research/list_experiments.py                # 全部实验
  python scripts/research/list_experiments.py --name xgboost # 只看某个模型
  python scripts/research/list_experiments.py --last 5       # 只看最近 5 条
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from asharerank.forecasting.experimental.report import load_run_index


def _rank_ic(m: dict) -> float | None:
    """不同报告的 RankIC 字段命名不统一：树模型/benchmark 评估用小写
    `rank_ic_mean`（见 evaluate.py::summarize_ic），StockMixer 原生指标用
    大写 `RankIC`（见 stockmixer/metrics.py::compute_metrics）。两种都兼容。
    """
    for key in ("rank_ic_mean", "RankIC"):
        if key in m and m[key] is not None:
            return m[key]
    return None


def _fmt_summary(entry: dict, max_width: int = 58) -> str:
    summary = entry.get("summary") or {}
    value = summary.get("value")
    if not value:
        return "-"
    # common_universe / full_universe 是 {model_key: metrics_dict}；其余大多是单个 metrics_dict。
    text = "-"
    if isinstance(value, dict) and value and isinstance(next(iter(value.values())), dict):
        parts = []
        for model_key, m in value.items():
            ric = _rank_ic(m) if isinstance(m, dict) else None
            if ric is not None:
                parts.append(f"{model_key}:RankIC={ric:.4f}")
        text = " | ".join(parts) if parts else "-"
    elif isinstance(value, dict):
        ric = _rank_ic(value)
        if ric is not None:
            icir = value.get("icir", value.get("ICIR", 0)) or 0
            text = f"RankIC={ric:.4f} ICIR={icir:.3f}"
    if len(text) > max_width:
        text = text[: max_width - 1] + "…"
    return text


def main():
    p = argparse.ArgumentParser(description="查看历史实验索引")
    p.add_argument("--name", default=None, help="只看某个模型名（子串匹配），如 xgboost")
    p.add_argument("--last", type=int, default=None, help="只看最近 N 条")
    args = p.parse_args()

    index = load_run_index()
    if args.name:
        index = [e for e in index if args.name.lower() in e["name"].lower()]
    if args.last:
        index = index[-args.last:]

    if not index:
        print("暂无实验记录（运行任意 train_*.py / run_benchmark.py 后会自动写入）。")
        return

    print(f"{'时间':<20}{'模型':<28}{'commit':<16}{'关键指标':<60}{'run_id'}")
    for e in index:
        print(f"{e['saved_at']:<20}{e['name']:<28}{(e.get('git_commit') or '-'):<16}"
             f"{_fmt_summary(e):<60}{e['run_id']}")
    print(f"\n共 {len(index)} 条记录。报告位于 runs/experiments/runs/<run_id>/report.json")


if __name__ == "__main__":
    main()
