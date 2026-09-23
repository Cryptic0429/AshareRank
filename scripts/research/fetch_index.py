"""独立脚本：分段拉取沪深300 全历史（2005 至今）并写入 data。

index_daily 单次调用有行数与频率限制，因此按 2 年分段、段间等待，
支持断点续传（已有缓存则只补缺失年份）。
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import datetime as dt

import pandas as pd
import tushare as ts

from asharerank.settings import CACHE_DIR, TUSHARE_TOKEN
from asharerank.data_service import INDEX_CODE, INDEX_HISTORY_START

SEGMENT_YEARS = 2
WAIT_BETWEEN_CALLS_S = 65
MAX_RETRY_PER_SEGMENT = 3

ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()
path = CACHE_DIR / f"index_{INDEX_CODE.replace('.', '_')}.csv"

existing = None
if path.exists():
    existing = pd.read_csv(path, index_col=0, parse_dates=True)
    print(f"已有缓存 {len(existing)} 行: {existing.index[0].date()} ~ {existing.index[-1].date()}")

start_year = int(INDEX_HISTORY_START[:4])
this_year = dt.date.today().year
segments = []
for y in range(start_year, this_year + 1, SEGMENT_YEARS):
    seg_start = f"{y}0101" if y > start_year else INDEX_HISTORY_START
    seg_end = f"{min(y + SEGMENT_YEARS - 1, this_year)}1231"
    segments.append((seg_start, seg_end))

frames = [existing] if existing is not None else []
for i, (s, e) in enumerate(segments):
    # 断点续传：该段已完整覆盖则跳过（段末在缓存范围内）
    if existing is not None and not existing.empty:
        seg_end_dt = pd.Timestamp(e)
        seg_start_dt = pd.Timestamp(s)
        covered = existing.loc[seg_start_dt:seg_end_dt]
        expected_days = (min(seg_end_dt, existing.index[-1]) - seg_start_dt).days
        if len(covered) > expected_days * 0.6 and seg_end_dt <= existing.index[-1]:
            print(f"段 {s}~{e} 已覆盖，跳过")
            continue
    for attempt in range(1, MAX_RETRY_PER_SEGMENT + 1):
        try:
            raw = pro.index_daily(ts_code=INDEX_CODE, start_date=s, end_date=e)
            if raw is not None and not raw.empty:
                raw = raw[["trade_date", "close"]].sort_values("trade_date")
                raw["trade_date"] = pd.to_datetime(raw["trade_date"])
                frames.append(raw.set_index("trade_date"))
                print(f"段 {s}~{e}: {len(raw)} 行")
            else:
                print(f"段 {s}~{e}: 空")
            break
        except Exception as exc:
            print(f"段 {s}~{e} 第 {attempt} 次失败: {exc}")
            if attempt < MAX_RETRY_PER_SEGMENT:
                time.sleep(WAIT_BETWEEN_CALLS_S)
    # 每段成功后立即合并落盘，中断也不丢进度
    if frames:
        merged = pd.concat(frames)
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        merged.to_csv(path)
    time.sleep(WAIT_BETWEEN_CALLS_S)

final = pd.read_csv(path, index_col=0, parse_dates=True) if path.exists() else None
if final is not None and len(final) > 4000:
    print(f"INDEX_FETCH_OK rows={len(final)} range={final.index[0].date()}~{final.index[-1].date()}")
    sys.exit(0)
print(f"INDEX_FETCH_PARTIAL rows={0 if final is None else len(final)}")
sys.exit(1)
