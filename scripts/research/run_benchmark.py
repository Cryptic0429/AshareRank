"""统一 Benchmark：StockMixer vs LightGBM vs XGBoost vs LSTM（同 train/val/test
边界 + 同股票池，公平对照）。

不改变任一模型现有的训练代码：StockMixer 按现有流程原生训练一次（多 seed 等权
集成，默认 5 个 seed），拿到它的股票池 + train/val/test 时间边界后，LightGBM /
XGBoost / LSTM 各自训练一个"边界与 StockMixer 完全对齐"的对照版本（不注册、不
覆盖任何生产模型）。四者的预测统一转成长表、用同一套 evaluate.py 评估，同时汇报：
  - full_universe   : 各模型自己覆盖到的最大样本
  - common_universe : 全部参与对比的模型预测都覆盖到的 (date, ts_code) 交集
                       （唯一允许排名的口径）

对齐策略（"对齐评估窗口"的默认答案）：LightGBM/XGBoost/LSTM 默认都使用
StockMixer 实际切出来的 train/val/test 边界（同一批股票、同一段训练历史、
同一个早停验证窗口、同一个最终测试窗口）——比早期版本"只对齐测试窗口起点"
更严格，避免"训练历史长度不同"这个额外的混杂变量。传 `--loose-align` 可以
退回旧的宽松模式（只对齐测试窗口，训练集用尽可能长的历史），用于想看"给
树模型更多历史数据是否能补偿架构差距"这类问题。

实现细节：PyTorch（StockMixer/LSTM）和 LightGBM/XGBoost 分别跑在独立子进程里
（`_bm_stockmixer_step.py` / `_bm_lstm_step.py` / `_bm_lgbm_step.py` /
`_bm_xgboost_step.py`），因为前者绑定的 OpenMP 运行时库和后者在同一进程里会在
macOS 上直接 segfault（exit code 139）。本脚本本身不 import torch / lightgbm /
xgboost，只做子进程编排 + 结果合并评估。

用法:
  python scripts/research/run_benchmark.py
  python scripts/research/run_benchmark.py --max-stocks 60 --epochs 20      # 调试用小样本
  python scripts/research/run_benchmark.py --models stockmixer,lgbm         # 只跑部分模型
  python scripts/research/run_benchmark.py --label-horizon 5 --top-k 10
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from asharerank.settings import RUNS_DIR
from asharerank.forecasting.benchmark import combine_and_evaluate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent

ALL_MODELS = ("stockmixer", "lgbm", "xgboost", "lstm")
DISPLAY_NAMES = {
    "stockmixer": "StockMixer",
    "lgbm": "LightGBM(对照/非生产)",
    "xgboost": "XGBoost(对照/非生产)",
    "lstm": "LSTM(对照/非生产)",
}


def main():
    p = argparse.ArgumentParser(description="统一 Benchmark：StockMixer / LightGBM / XGBoost / LSTM 对照")
    p.add_argument("--models", default=",".join(ALL_MODELS),
                  help=f"逗号分隔的参与对比模型列表，默认全部：{','.join(ALL_MODELS)}"
                       "（stockmixer 必须包含在内，因为它的 train/val/test 边界是其余模型对齐的基准）")
    p.add_argument("--pool", default="csi300", choices=["csi300", "csi500", "csi800"])
    p.add_argument("--max-stocks", type=int, default=None, help="调试用：限制股票数量")
    p.add_argument("--lookback", type=int, default=16, help="StockMixer 滑动窗口长度 T，也是 LSTM 默认序列窗口长度")
    p.add_argument("--label-horizon", type=int, default=5, help="预测目标前瞻天数，全部模型统一使用")
    p.add_argument("--epochs", type=int, default=60, help="StockMixer / LSTM 共用的最大训练轮数（早停会提前结束）")
    p.add_argument("--batch-size", type=int, default=8, help="仅 StockMixer 使用")
    p.add_argument("--lr", type=float, default=2e-3, help="仅 StockMixer 使用")
    p.add_argument("--dropout", type=float, default=0.1, help="仅 StockMixer 使用")
    p.add_argument("--weight-decay", type=float, default=1e-5, help="仅 StockMixer 使用")
    p.add_argument("--patience", type=int, default=10, help="StockMixer / LSTM 共用的早停 patience")
    p.add_argument("--rank-loss-weight", type=float, default=0.5, help="仅 StockMixer 使用")
    p.add_argument("--use-fundamentals", action="store_true", help="仅 StockMixer 使用")
    p.add_argument("--use-ema", action="store_true", help="仅 StockMixer：启用权重 EMA")
    p.add_argument("--ema-decay", type=float, default=0.9, help="仅 StockMixer：EMA 衰减上限")
    p.add_argument("--rankic-smooth-window", type=int, default=1,
                  help="仅 StockMixer：早停判据用最近 N 轮 val_RankIC 滑动平均；1=裸单轮值")
    p.add_argument("--market-layerscale-init", type=float, default=1.0,
                  help="仅 StockMixer：MarketAwareStockMixing 残差门控初始值")
    p.add_argument("--seed", type=int, default=42, help="仅单 seed 模式生效（未传 --seeds/--n-seeds>1 时）；也用作 LSTM 的种子")
    p.add_argument("--seeds", type=str, default=None,
                  help="逗号分隔的随机种子列表，例如 '42,43,44,45,46'；传了就走多 seed 等权集成，"
                       "覆盖 --seed/--n-seeds/--seed-start（仅 StockMixer）")
    p.add_argument("--n-seeds", type=int, default=5,
                  help="未传 --seeds 时，StockMixer 生成的随机种子个数；默认 5（多 seed 等权集成，避免单 seed 运气"
                       "误判架构优劣，见 stockmixer_multiseed 实测）；传 1 则退化为旧的单 seed 行为")
    p.add_argument("--seed-start", type=int, default=42, help="未传 --seeds 时，随机种子的起始值（依次+1）")
    p.add_argument("--loose-align", action="store_true",
                  help="LightGBM/XGBoost 退回宽松对齐模式（只对齐测试窗口起点，训练集用尽可能长的历史），"
                       "默认严格对齐 StockMixer 的完整 train/val/test 边界")
    p.add_argument("--valid-months", type=int, default=3, help="仅 --loose-align 模式下 LightGBM/XGBoost 早停验证集月数")
    p.add_argument("--top-k", type=int, default=10, help="Precision@K / 组合回测的 K")
    p.add_argument("--cost-bps", type=float, default=10.0, help="双边换手成本假设（基点）")
    p.add_argument("--device", default=None, choices=[None, "mps", "cuda", "cpu"])
    args = p.parse_args()
    from asharerank.research_record import record_arguments
    record_arguments(args)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = set(models) - set(ALL_MODELS)
    if unknown:
        raise SystemExit(f"未知模型: {unknown}，可选: {ALL_MODELS}")
    if "stockmixer" not in models:
        raise SystemExit("stockmixer 必须包含在 --models 里：其余模型的 train/val/test 边界都对齐它的切分结果")

    tmp_dir = RUNS_DIR / "_benchmark_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    codes_path = tmp_dir / "codes.json"

    if args.seeds:
        seeds_desc = args.seeds
    else:
        seeds_desc = ",".join(str(args.seed_start + i) for i in range(args.n_seeds))
    logger.info("=== [1/%d] 子进程训练 StockMixer（PyTorch 独立进程，seeds=%s） ===", len(models), seeds_desc)
    sm_frame_path = tmp_dir / "sm_frame.parquet"
    sm_meta_path = tmp_dir / "sm_meta.json"
    sm_cmd = [
        sys.executable, str(SCRIPT_DIR / "_bm_stockmixer_step.py"),
        "--pool", args.pool,
        "--lookback", str(args.lookback),
        "--label-horizon", str(args.label_horizon),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--dropout", str(args.dropout),
        "--weight-decay", str(args.weight_decay),
        "--patience", str(args.patience),
        "--rank-loss-weight", str(args.rank_loss_weight),
        "--ema-decay", str(args.ema_decay),
        "--rankic-smooth-window", str(args.rankic_smooth_window),
        "--market-layerscale-init", str(args.market_layerscale_init),
        "--seed", str(args.seed),
        "--n-seeds", str(args.n_seeds),
        "--seed-start", str(args.seed_start),
        "--out-frame", str(sm_frame_path),
        "--out-meta", str(sm_meta_path),
    ]
    if args.seeds:
        sm_cmd += ["--seeds", args.seeds]
    if args.max_stocks:
        sm_cmd += ["--max-stocks", str(args.max_stocks)]
    if args.use_fundamentals:
        sm_cmd += ["--use-fundamentals"]
    if args.use_ema:
        sm_cmd += ["--use-ema"]
    if args.device:
        sm_cmd += ["--device", args.device]
    subprocess.run(sm_cmd, check=True)

    sm_meta = json.loads(sm_meta_path.read_text())
    codes_path.write_text(json.dumps(sm_meta["codes"]))
    logger.info(
        "StockMixer 边界：train (%s, %s] | val (%s, %s] | test (%s, %s] | 股票池=%d 只",
        sm_meta["train_start"], sm_meta["train_end"], sm_meta["val_start"], sm_meta["val_end"],
        sm_meta["test_start"], sm_meta["test_end"], len(sm_meta["codes"]),
    )

    frames = {"stockmixer": None}
    metas = {"stockmixer": sm_meta}
    import pandas as pd
    frames["stockmixer"] = pd.read_parquet(sm_frame_path)

    align_kwargs = {}
    if not args.loose_align:
        align_kwargs = {
            "train_start": sm_meta["train_start"], "train_end": sm_meta["train_end"],
            "val_start": sm_meta["val_start"], "val_end": sm_meta["val_end"],
        }

    step_no = 2
    if "lgbm" in models:
        logger.info("=== [%d/%d] 子进程训练 LightGBM 对照模型（不 import torch，独立进程） ===", step_no, len(models))
        lgbm_frame_path = tmp_dir / "lgbm_frame.parquet"
        lgbm_meta_path = tmp_dir / "lgbm_meta.json"
        lgbm_cmd = [
            sys.executable, str(SCRIPT_DIR / "_bm_lgbm_step.py"),
            "--codes-file", str(codes_path),
            "--test-start", sm_meta["test_start"], "--test-end", sm_meta["test_end"],
            "--horizon", str(args.label_horizon), "--valid-months", str(args.valid_months),
            "--out-frame", str(lgbm_frame_path),
            "--out-meta", str(lgbm_meta_path),
        ]
        if not args.loose_align:
            lgbm_cmd += [
                "--train-start", sm_meta["train_start"], "--train-end", sm_meta["train_end"],
                "--val-start", sm_meta["val_start"], "--val-end", sm_meta["val_end"],
            ]
        subprocess.run(lgbm_cmd, check=True)
        frames["lgbm"] = pd.read_parquet(lgbm_frame_path)
        metas["lgbm"] = json.loads(lgbm_meta_path.read_text())
        step_no += 1

    if "xgboost" in models:
        logger.info("=== [%d/%d] 子进程训练 XGBoost 对照模型（不 import torch，独立进程） ===", step_no, len(models))
        xgb_frame_path = tmp_dir / "xgboost_frame.parquet"
        xgb_meta_path = tmp_dir / "xgboost_meta.json"
        xgb_cmd = [
            sys.executable, str(SCRIPT_DIR / "_bm_xgboost_step.py"),
            "--codes-file", str(codes_path),
            "--test-start", sm_meta["test_start"], "--test-end", sm_meta["test_end"],
            "--horizon", str(args.label_horizon), "--valid-months", str(args.valid_months),
            "--out-frame", str(xgb_frame_path),
            "--out-meta", str(xgb_meta_path),
        ]
        if not args.loose_align:
            xgb_cmd += [
                "--train-start", sm_meta["train_start"], "--train-end", sm_meta["train_end"],
                "--val-start", sm_meta["val_start"], "--val-end", sm_meta["val_end"],
            ]
        subprocess.run(xgb_cmd, check=True)
        frames["xgboost"] = pd.read_parquet(xgb_frame_path)
        metas["xgboost"] = json.loads(xgb_meta_path.read_text())
        step_no += 1

    if "lstm" in models:
        logger.info("=== [%d/%d] 子进程训练 LSTM 对照模型（PyTorch 独立进程） ===", step_no, len(models))
        lstm_frame_path = tmp_dir / "lstm_frame.parquet"
        lstm_meta_path = tmp_dir / "lstm_meta.json"
        lstm_cmd = [
            sys.executable, str(SCRIPT_DIR / "_bm_lstm_step.py"),
            "--codes-file", str(codes_path),
            "--train-start", sm_meta["train_start"], "--train-end", sm_meta["train_end"],
            "--val-start", sm_meta["val_start"], "--val-end", sm_meta["val_end"],
            "--test-start", sm_meta["test_start"], "--test-end", sm_meta["test_end"],
            "--horizon", str(args.label_horizon), "--window", str(args.lookback),
            "--epochs", str(args.epochs), "--patience", str(args.patience), "--seed", str(args.seed),
            "--out-frame", str(lstm_frame_path),
            "--out-meta", str(lstm_meta_path),
        ]
        if args.device:
            lstm_cmd += ["--device", args.device]
        subprocess.run(lstm_cmd, check=True)
        frames["lstm"] = pd.read_parquet(lstm_frame_path)
        metas["lstm"] = json.loads(lstm_meta_path.read_text())
        step_no += 1

    logger.info("=== [%d/%d] 合并结果并统一评估 ===", step_no, len(models))
    combine_and_evaluate(
        frames, display_names=DISPLAY_NAMES, metas=metas,
        codes=sm_meta["codes"], test_start=sm_meta["test_start"], test_end=sm_meta["test_end"],
        horizon=args.label_horizon, top_k=args.top_k, cost_bps=args.cost_bps,
        experiment_config={
            "arguments": vars(args),
            "selected_models": models,
            "effective_boundaries": {
                key: sm_meta[key]
                for key in ("train_start", "train_end", "val_start", "val_end", "test_start", "test_end")
                if key in sm_meta
            },
        },
    )


if __name__ == "__main__":
    from asharerank.research_record import run_recorded
    run_recorded(main)
