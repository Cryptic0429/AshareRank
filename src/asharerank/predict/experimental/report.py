"""实验模型报告落盘/打印（与生产模型注册表完全隔离）。

落盘分两层，解决"覆盖式 JSON 丢历史"的问题：
  1. `{name}_report.json`      —— 覆盖式"最新视图"，保持向后兼容（脚本/文档里
     习惯直接读这个固定路径看最新一次结果）。
  2. `runs/{run_id}/report.json` —— 版本化快照，每次调用 `save_report` 都单独
     存一份，历史实验永久保留、不会被下一次运行覆盖；同时把摘要 append 进
     `runs/index.json`，作为可横向检索的实验索引（时间、模型、git commit、
     关键指标、报告路径），配合 `scripts/research/list_experiments.py` 查看。
"""
from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from importlib import metadata

from asharerank.settings import RUNS_DIR as PROJECT_RUNS_DIR

logger = logging.getLogger(__name__)

EXPERIMENTAL_DIR = PROJECT_RUNS_DIR / "experiments"
EXPERIMENTAL_DIR.mkdir(exist_ok=True)

RUNS_DIR = EXPERIMENTAL_DIR / "runs"
RUNS_DIR.mkdir(exist_ok=True, parents=True)
RUN_INDEX_PATH = RUNS_DIR / "index.json"

# 从 payload 里按优先级尝试提取的"关键指标摘要"字段名——不同模型的报告结构不完全
# 一致（walk-forward 基线是 oos_metrics，benchmark 编排是 common_universe，
# StockMixer 原生报告是 test_metrics），尽量都能在实验索引里看到一行摘要。
_SUMMARY_KEYS = (
    "common_universe", "oos_metrics", "test_metrics",
    "native_test_metrics", "ensemble_metrics", "metrics",
)


def _git_commit() -> str:
    """当前仓库的 commit hash（含未提交改动时打上 -dirty 后缀），用于实验可追溯性。
    不是 git 仓库或命令失败时返回 "unknown"，不影响报告落盘。
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[4], capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        if not sha:
            return "unknown"
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).resolve().parents[4], capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except Exception:
        return "unknown"


def _extract_summary(payload: dict) -> dict:
    for key in _SUMMARY_KEYS:
        if key in payload and payload[key]:
            return {"from": key, "value": payload[key]}
    return {}


def _append_run_index(run_id: str, name: str, saved_at: str, run_path: Path, payload: dict) -> None:
    index: list[dict] = []
    if RUN_INDEX_PATH.exists():
        try:
            index = json.loads(RUN_INDEX_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            logger.warning("实验索引 %s 解析失败，将重建（旧内容不会被物理删除，仍在 runs/ 目录下）",
                          RUN_INDEX_PATH)
            index = []
    index.append({
        "run_id": run_id,
        "name": name,
        "saved_at": saved_at,
        "git_commit": payload.get("git_commit"),
        "report_path": str(run_path.relative_to(EXPERIMENTAL_DIR)),
        "summary": _extract_summary(payload),
    })
    RUN_INDEX_PATH.write_text(json.dumps(index, ensure_ascii=False, indent=2, default=str))


def save_report(name: str, payload: dict, *, config: dict | None = None) -> Path:
    """写入 `{name}_report.json`（覆盖式最新视图）+ `runs/{run_id}/report.json`
    （版本化快照）+ `runs/index.json`（实验索引，append-only）。

    Args:
        config: 可选，本次实验的超参数/配置快照（如未随 payload 内联给出）；
            会原样塞进落盘的 payload["config"]，方便日后复现实验。
    """
    now = datetime.now()
    saved_at = now.strftime("%Y-%m-%d %H:%M:%S.%f")
    run_id = f"{now.strftime('%Y%m%d_%H%M%S_%f')}_{name}"
    git_commit = _git_commit()
    payload = {
        **payload,
        "run_id": run_id,
        "saved_at": saved_at,
        "git_commit": git_commit,
    }
    if config is not None:
        payload = {**payload, "config": config}
    body = json.dumps(payload, ensure_ascii=False, indent=2, default=str)

    # 1) 覆盖式"最新视图"（向后兼容旧的固定路径读取方式）
    latest_path = EXPERIMENTAL_DIR / f"{name}_report.json"
    latest_path.write_text(body)

    # 2) 版本化快照：每次运行单独一份目录，历史不会被覆盖
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    run_path = run_dir / "report.json"
    run_path.write_text(body)

    distributions = ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "xgboost", "torch")
    versions = {}
    for distribution in distributions:
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            continue
    manifest = {
        "run_id": run_id,
        "name": name,
        "created_at": saved_at,
        "git_revision": git_commit,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
        "command": list(sys.argv),
        "config": config if config is not None else payload.get("config"),
        "data_manifest": payload.get("data_manifest"),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str)
    )
    if manifest["config"] is not None:
        (run_dir / "config.json").write_text(
            json.dumps(manifest["config"], ensure_ascii=False, indent=2, default=str)
        )

    # 3) 索引 append（供横向对比历史所有实验用，见 list_experiments.py）
    _append_run_index(run_id, name, saved_at, run_path, payload)

    logger.info("实验报告已保存: %s（最新视图） + %s（版本化存档，run_id=%s）",
                latest_path, run_path, run_id)
    return run_path


def load_run_index() -> list[dict]:
    """读取 `runs/index.json` 里的全部历史实验摘要（按落盘顺序，最早的在前）。"""
    if not RUN_INDEX_PATH.exists():
        return []
    try:
        return json.loads(RUN_INDEX_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def print_report(name: str, oos_metrics: dict, fold_metrics: list[dict],
                 extra: dict | None = None) -> None:
    print(f"\n=== {name} — walk-forward OOS 指标 ===")
    print(f"  RankIC mean : {oos_metrics.get('rank_ic_mean', 0):.4f}")
    print(f"  RankICIR    : {oos_metrics.get('rank_icir', 0):.3f}")
    print(f"  IC mean     : {oos_metrics.get('ic_mean', 0):.4f}")
    print(f"  ICIR        : {oos_metrics.get('icir', 0):.3f}")
    print(f"  OOS 天数    : {oos_metrics.get('n_days', 0)}")
    if extra:
        for k, v in extra.items():
            print(f"  {k}: {v}")
    print(f"\n  逐 fold RankIC:")
    for i, m in enumerate(fold_metrics):
        print(f"    fold {i+1}: RankIC={m.get('rank_ic_mean', 0):.4f}  "
             f"RankICIR={m.get('rank_icir', 0):.3f}  n_days={m.get('n_days', 0)}")
