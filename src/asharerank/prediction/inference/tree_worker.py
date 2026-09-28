"""Tree-model (LightGBM / HistGBM) inference subprocess entry point.

PyTorch and LightGBM can load incompatible OpenMP runtimes in some environments.
Isolating tree inference in a clean subprocess keeps benchmark and inference runs
reproducible across those environments.

Usage (normally called by alpha.py):
  python -m asharerank.prediction.inference.tree_worker predict --model-id lgbm_v4 --top-k 800 --out /tmp/out.json
  python -m asharerank.prediction.inference.tree_worker explain --model-id lgbm_v4 --ts-code 600519.SH --out /tmp/out.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _predict(model_id: str | None, top_k: int) -> dict:
    from asharerank.prediction.inference.alpha import predict_latest_tree_inplace

    return predict_latest_tree_inplace(model_id=model_id, top_k=top_k)


def _explain(model_id: str | None, ts_code: str, top_n: int) -> dict:
    from asharerank.prediction.inference.alpha import explain_top_features_tree_inplace

    return explain_top_features_tree_inplace(ts_code, model_id=model_id, top_n=top_n)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="树模型隔离推理（内部 worker）")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_pred = sub.add_parser("predict")
    p_pred.add_argument("--model-id", default=None)
    p_pred.add_argument("--top-k", type=int, default=20)
    p_pred.add_argument("--out", required=True)

    p_exp = sub.add_parser("explain")
    p_exp.add_argument("--model-id", default=None)
    p_exp.add_argument("--ts-code", required=True)
    p_exp.add_argument("--top-n", type=int, default=8)
    p_exp.add_argument("--out", required=True)

    args = p.parse_args(argv)
    try:
        if args.cmd == "predict":
            result = _predict(args.model_id, args.top_k)
        else:
            result = _explain(args.model_id, args.ts_code, args.top_n)
    except Exception as exc:
        logger.exception("tree_infer_worker 失败")
        Path(args.out).write_text(
            json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False),
            encoding="utf-8",
        )
        return 1

    Path(args.out).write_text(
        json.dumps({"ok": True, "result": result}, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"TREE_INFER_DONE cmd={args.cmd} out={args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
