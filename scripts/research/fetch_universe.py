"""下载指数成分股前复权日线到 data/processed/universe/ohlcv/（断点续传）。

用法:
  python scripts/research/fetch_universe.py                    # 沪深300 全量约 300 只
  python scripts/research/fetch_universe.py --pool csi500      # 中证500
  python scripts/research/fetch_universe.py --pool csi800      # 中证800 = 300∪500
  python scripts/research/fetch_universe.py --pool csi800 --max 80
  python scripts/research/fetch_universe.py --update           # 日终增量刷新
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from asharerank.forecasting.universe import INDEX_POOLS, download_universe, fetch_index_constituents

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", choices=list(INDEX_POOLS), default="csi300",
                   help="指数池：csi300 / csi500 / csi800")
    p.add_argument("--max", type=int, default=None, help="最多下载股票数")
    p.add_argument("--sleep", type=float, default=0.35, help="请求间隔秒")
    p.add_argument("--start", type=str, default="20180101")
    p.add_argument("--update", action="store_true",
                   help="增量更新已有文件（补最新交易日），而非跳过")
    args = p.parse_args()
    from asharerank.research_record import record_arguments
    record_arguments(args)

    cons = fetch_index_constituents(args.pool)
    print(f"{args.pool} 成分股 {len(cons)} 只")
    stats = download_universe(
        max_stocks=args.max, sleep_s=args.sleep, start=args.start,
        update_existing=args.update, pool=args.pool,
    )
    print("DOWNLOAD_DONE", stats)
    if stats["downloaded"] + stats["skipped"] + stats.get("updated", 0) < 30:
        sys.exit(1)


if __name__ == "__main__":
    from asharerank.research_record import run_recorded
    run_recorded(main)
