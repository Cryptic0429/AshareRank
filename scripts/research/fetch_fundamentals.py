"""下载基本面因子（ROE/营收增速等，akshare 逐股票，较慢）到 data/processed/universe/fundamentals/。

用法:
  python scripts/research/fetch_fundamentals.py --max 60      # 先验证一小批
  python scripts/research/fetch_fundamentals.py               # 全量（较慢，建议后台跑）
  python scripts/research/fetch_fundamentals.py --csi300-missing  # 只补沪深300缺的（优先）
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from asharerank.settings import UNIVERSE_DIR
from asharerank.predict.fundamentals import download_fundamentals, list_available_codes as list_fund_codes
from asharerank.predict.universe import list_available_codes

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")


def _csi300_codes() -> list[str]:
    meta_path = UNIVERSE_DIR / "csi300_meta.json"
    meta = json.loads(meta_path.read_text())
    return [s["ts_code"] for s in meta["stocks"]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max", type=int, default=None, help="最多下载股票数")
    p.add_argument("--sleep", type=float, default=0.5)
    p.add_argument("--start-year", type=str, default="2017")
    p.add_argument("--csi300-missing", action="store_true",
                   help="只下载沪深300中尚未缓存基本面的股票（优先把覆盖率拉到 90%+）")
    args = p.parse_args()

    if args.csi300_missing:
        have = set(list_fund_codes())
        codes = [c for c in _csi300_codes() if c not in have]
        print(f"CSI300 缺基本面: {len(codes)} 只（已有 {len(have & set(_csi300_codes()))}/300）")
    else:
        codes = list_available_codes()  # 复用已有 OHLCV 股票池
        if args.max:
            codes = codes[:args.max]
        print(f"目标股票数: {len(codes)}")

    if args.max and args.csi300_missing:
        codes = codes[:args.max]
        print(f"截断到 --max={args.max} → {len(codes)} 只")

    if not codes:
        print("DOWNLOAD_DONE", {"total": 0, "downloaded": 0, "skipped": 0, "failed": 0})
        return

    stats = download_fundamentals(codes, start_year=args.start_year, sleep_s=args.sleep)
    print("DOWNLOAD_DONE", stats)


if __name__ == "__main__":
    main()
