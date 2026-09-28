"""基本面因子：ROE、营收/净利润增速、每股净资产等（akshare 东财指标，逐股票拉取）。

数据源说明：
- tushare 的 daily_basic / fina_indicator 在当前 token 等级下不可用
  （分别被限到 1 次/小时 与完全无权限），故改用 akshare 免费的
  stock_financial_analysis_indicator（逐股票、按季度）。
- 财报"日期"列是报告期（如 2024-03-31），不是公告日；为避免未来函数，
  统一加保守滞后 REPORT_LAG_DAYS 天作为"信息可得日"，之后才允许被当日特征使用。
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

from asharerank.settings import UNIVERSE_DIR

logger = logging.getLogger(__name__)

FUNDAMENTALS_DIR = UNIVERSE_DIR / "fundamentals"
FUNDAMENTALS_DIR.mkdir(parents=True, exist_ok=True)

# akshare 中文列名 → 英文特征名
COLUMN_MAP: dict[str, str] = {
    "摊薄每股收益(元)": "eps",
    "每股净资产_调整后(元)": "bvps",
    "净资产收益率(%)": "roe",
    "主营业务收入增长率(%)": "rev_growth",
    "净利润增长率(%)": "profit_growth",
    "资产负债率(%)": "debt_ratio",
    "销售净利率(%)": "net_margin",
}
FEATURE_COLS = list(COLUMN_MAP.values())
REPORT_LAG_DAYS = 45  # 财报公告相对报告期的保守滞后（避免用未公开信息，即防未来函数）


def _path(ts_code: str) -> Path:
    return FUNDAMENTALS_DIR / f"{ts_code.replace('.', '_')}.parquet"


def fetch_one(ts_code: str, start_year: str = "2017") -> pd.DataFrame | None:
    """拉取单只股票财务指标，返回带 avail_date（信息可得日）的整理表。"""
    import akshare as ak

    symbol = ts_code.split(".")[0]
    try:
        df = ak.stock_financial_analysis_indicator(symbol=symbol, start_year=start_year)
    except Exception as exc:
        logger.warning("基本面拉取 %s 失败: %s", ts_code, exc)
        return None
    if df is None or df.empty or "日期" not in df.columns:
        return None

    df = df.rename(columns=COLUMN_MAP)
    keep = ["日期"] + [c for c in FEATURE_COLS if c in df.columns]
    df = df[keep].rename(columns={"日期": "report_date"})
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
    df = df.dropna(subset=["report_date"])
    df["avail_date"] = df["report_date"] + pd.Timedelta(days=REPORT_LAG_DAYS)
    df["ts_code"] = ts_code
    for c in FEATURE_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("avail_date").reset_index(drop=True)


def download_fundamentals(codes: list[str], start_year: str = "2017",
                          sleep_s: float = 0.5) -> dict:
    """逐股票下载财务指标（断点续传：已有文件直接跳过）。"""
    ok = skip = fail = 0
    for i, code in enumerate(codes):
        path = _path(code)
        if path.exists():
            skip += 1
            continue
        df = fetch_one(code, start_year=start_year)
        if df is None or df.empty:
            fail += 1
            logger.warning("[%d/%d] 跳过 %s（无数据）", i + 1, len(codes), code)
            time.sleep(sleep_s)
            continue
        df.to_parquet(path, index=False)
        ok += 1
        logger.info("[%d/%d] 基本面 %s %d 条", i + 1, len(codes), code, len(df))
        time.sleep(sleep_s)
    return {"total": len(codes), "downloaded": ok, "skipped": skip, "failed": fail}


def list_available_codes() -> list[str]:
    out = []
    for p in FUNDAMENTALS_DIR.glob("*.parquet"):
        stem = p.stem
        code = stem[:-3] + "." + stem[-2:] if stem.endswith(("_SH", "_SZ")) else stem.replace("_", ".")
        out.append(code)
    return sorted(out)


def load_fundamentals(codes: list[str] | None = None) -> pd.DataFrame:
    paths = list(FUNDAMENTALS_DIR.glob("*.parquet"))
    if codes:
        wanted = {c.replace(".", "_") for c in codes}
        paths = [p for p in paths if p.stem in wanted]
    if not paths:
        return pd.DataFrame(columns=["avail_date", "ts_code", *FEATURE_COLS])
    frames = [pd.read_parquet(p) for p in paths]
    return pd.concat(frames, ignore_index=True)


def build_fundamental_feature_panel(calendar: pd.DataFrame,
                                    codes: list[str] | None = None) -> pd.DataFrame:
    """calendar: 含 date/ts_code 的日历面板（如 OHLCV panel）；
    按 as-of（截至当日已公开的最新一份财报）合并出逐日基本面特征。
    """
    fund = load_fundamentals(codes)
    if fund.empty:
        return pd.DataFrame(columns=["date", "ts_code", *FEATURE_COLS])
    feature_cols = [c for c in FEATURE_COLS if c in fund.columns]

    out_parts = []
    cal = calendar[["date", "ts_code"]].drop_duplicates()
    for code, g_cal in cal.groupby("ts_code", sort=False):
        g_fund = fund[fund["ts_code"] == code].sort_values("avail_date")
        if g_fund.empty:
            continue
        g_cal = g_cal.sort_values("date")
        merged = pd.merge_asof(
            g_cal, g_fund[["avail_date", *feature_cols]],
            left_on="date", right_on="avail_date", direction="backward",
        )
        merged["ts_code"] = code
        out_parts.append(merged[["date", "ts_code", *feature_cols]])
    if not out_parts:
        return pd.DataFrame(columns=["date", "ts_code", *feature_cols])
    return pd.concat(out_parts, ignore_index=True)
