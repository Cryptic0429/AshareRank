"""内部脚本：训练"训练数据与 StockMixer 对齐"的 LightGBM 候选模型并注册。

由 `scripts/research/train_lgbm_aligned.py` 通过 `subprocess` 调用，跑在独立进程里
（不 import torch），原因见 `asharerank/predict/benchmark_stockmixer_step.py`
顶部注释。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from asharerank.predict.benchmark_lgbm_step import train_aligned_candidate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--universe-file", required=True, help="_bm_get_universe.py 的输出 json")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--model-id", default="lgbm_v4")
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--status", default="候选", choices=["候选", "已部署"],
                   help="注册状态：候选=不影响当前生产模型；已部署=直接替换生产模型")
    p.add_argument("--out", required=True, help="结果摘要输出 json 路径")
    args = p.parse_args()

    info = json.loads(Path(args.universe_file).read_text())
    result = train_aligned_candidate(
        info["codes"],
        train_start=info["train_start"], train_end=info["train_end"],
        val_start=info["val_start"], val_end=info["val_end"],
        test_start=info["test_start"], test_end=info["test_end"],
        horizon=args.horizon, model_id=args.model_id, top_k=args.top_k, status=args.status,
    )
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    print(f"LGBM_CANDIDATE_DONE model_id={result['model_id']} out={args.out}")


if __name__ == "__main__":
    main()
