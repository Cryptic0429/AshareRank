"""把模型比较结果导出为 CSV 和 JSON。"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def save_comparison_report(comparison: pd.DataFrame, output_dir: str | Path) -> dict[str, str]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "model_comparison.csv"
    json_path = destination / "model_comparison.json"
    comparison.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(comparison.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"csv": str(csv_path), "json": str(json_path)}
