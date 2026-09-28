"""宽基股票池：沪深300/中证500/中证800 成分股 OHLCV 拉取（tushare → akshare）与 Parquet 存储。"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd

from asharerank.settings import PREDICT_START_DATE, UNIVERSE_DIR

logger = logging.getLogger(__name__)

# 指数池：pool 名 → (中证指数代码, 展示名)
INDEX_POOLS: dict[str, tuple[str, str]] = {
    "csi300": ("000300", "沪深300"),
    "csi500": ("000905", "中证500"),
    "csi800": ("000906", "中证800"),  # 官方 CSI800 = CSI300 ∪ CSI500，直接用该指数更权威
}
DEFAULT_POOL = "csi800"

OHLCV_DIR = UNIVERSE_DIR / "ohlcv"
OHLCV_DIR.mkdir(parents=True, exist_ok=True)


def _meta_path(pool: str) -> Path:
    return UNIVERSE_DIR / f"{pool}_meta.json"


# 向后兼容旧调用点
UNIVERSE_META = _meta_path(DEFAULT_POOL)

# 单次 download/refresh 会话内：tushare 触发频控后本轮改走 akshare
_TUSHARE_DISABLED_UNTIL: float = 0.0


def _tushare_allowed() -> bool:
    return time.time() >= _TUSHARE_DISABLED_UNTIL


def _disable_tushare(seconds: float = 65.0) -> None:
    global _TUSHARE_DISABLED_UNTIL
    _TUSHARE_DISABLED_UNTIL = time.time() + seconds
    logger.warning("tushare 频控，本轮 %.0fs 内改走 akshare", seconds)


def to_ts_code(symbol: str, exchange: str | None = None) -> str:
    """6 位代码 → ts_code（如 600519.SH）。"""
    s = str(symbol).zfill(6)
    if exchange:
        ex = "SH" if "上海" in exchange or exchange.upper() in {"SH", "SSE"} else "SZ"
        return f"{s}.{ex}"
    if s.startswith(("5", "6", "9")):
        return f"{s}.SH"
    return f"{s}.SZ"


def fetch_index_constituents(pool: str = DEFAULT_POOL) -> pd.DataFrame:
    """拉取指定指数池（csi300 / csi500 / csi800）最新成分股列表并落盘元数据。"""
    import akshare as ak

    if pool not in INDEX_POOLS:
        raise ValueError(f"未知股票池: {pool}，可选: {list(INDEX_POOLS)}")
    index_code, index_name = INDEX_POOLS[pool]

    df = ak.index_stock_cons_csindex(symbol=index_code)
    # 中证接口固定列：成分券代码 / 成分券名称 / 交易所（避免「代码」匹配到指数代码）
    rename = {}
    for c in df.columns:
        if c in ("成分券代码", "品种代码") or c.lower() == "con_code":
            rename[c] = "symbol"
        elif c in ("成分券名称", "品种名称") or c.lower() == "con_name":
            rename[c] = "name"
        elif c == "交易所" or c.lower() == "exchange":
            rename[c] = "exchange"
    df = df.rename(columns=rename)
    if "symbol" not in df.columns:
        raise RuntimeError(f"无法解析成分股列: {list(df.columns)}")
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    df["ts_code"] = [
        to_ts_code(sym, ex if "exchange" in df.columns else None)
        for sym, ex in zip(df["symbol"], df["exchange"] if "exchange" in df.columns
                           else [None] * len(df))
    ]
    if "name" not in df.columns:
        df["name"] = df["ts_code"]
    out = df[["ts_code", "name"]].drop_duplicates("ts_code").reset_index(drop=True)
    _meta_path(pool).write_text(json.dumps(
        {"index": index_code, "index_name": index_name, "pool": pool, "n": len(out),
         "stocks": out.to_dict(orient="records")},
        ensure_ascii=False, indent=2))
    return out


def fetch_csi300_constituents() -> pd.DataFrame:
    """向后兼容别名：拉取沪深300 成分股。"""
    return fetch_index_constituents("csi300")


def load_constituents(pool: str = DEFAULT_POOL) -> pd.DataFrame:
    path = _meta_path(pool)
    if path.exists():
        meta = json.loads(path.read_text())
        return pd.DataFrame(meta["stocks"])
    return fetch_index_constituents(pool)


def _ohlcv_path(ts_code: str) -> Path:
    return OHLCV_DIR / f"{ts_code.replace('.', '_')}.parquet"


def _ak_symbol(ts_code: str) -> str:
    """600519.SH → sh600519。"""
    num, exch = ts_code.split(".")
    return f"{exch.lower()}{num}"


def _fetch_ohlcv_tushare(ts_code: str, start: str, end: str) -> pd.DataFrame | None:
    """tushare 前复权日线（pro_bar）。"""
    from asharerank.settings import TUSHARE_TOKEN

    if not TUSHARE_TOKEN or not _tushare_allowed():
        return None
    try:
        import tushare as ts

        ts.set_token(TUSHARE_TOKEN)
        df = ts.pro_bar(ts_code=ts_code, adj="qfq", start_date=start, end_date=end)
        if df is None or df.empty:
            return None
        out = df.rename(columns={"trade_date": "date", "vol": "volume"})
        keep = [c for c in ["date", "open", "high", "low", "close", "volume", "amount"]
                if c in out.columns]
        out = out[keep].copy()
        out["date"] = pd.to_datetime(out["date"])
        out["ts_code"] = ts_code
        return out.sort_values("date").reset_index(drop=True)
    except Exception as exc:
        # pro_bar 频控时异常信息常仅为 "ERROR."，一并冷却改走 akshare
        _disable_tushare(90.0)
        logger.warning("tushare 拉取 %s 失败: %s", ts_code, exc)
        return None


def _fetch_ohlcv_akshare(ts_code: str, start: str, end: str,
                         retries: int = 3) -> pd.DataFrame | None:
    """akshare 前复权日线。优先新浪，失败再试东财。"""
    import akshare as ak

    sina_sym = _ak_symbol(ts_code)
    em_sym = ts_code.split(".")[0]
    df = None
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            df = ak.stock_zh_a_daily(symbol=sina_sym, start_date=start,
                                    end_date=end, adjust="qfq")
            if df is not None and not df.empty:
                break
        except Exception as exc:
            last_err = exc
        try:
            df = ak.stock_zh_a_hist(symbol=em_sym, period="daily",
                                    start_date=start, end_date=end, adjust="qfq")
            if df is not None and not df.empty:
                break
        except Exception as exc:
            last_err = exc
            logger.warning("akshare 拉取 %s 第 %d 次失败: %s", ts_code, attempt, last_err)
            time.sleep(1.2 * attempt)
            df = None
    if df is None or df.empty:
        if last_err:
            logger.warning("akshare 拉取 %s 最终失败: %s", ts_code, last_err)
        return None
    rename = {
        "日期": "date", "开盘": "open", "收盘": "close",
        "最高": "high", "最低": "low", "成交量": "volume",
        "成交额": "amount", "换手率": "turnover",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    keep = [c for c in ["date", "open", "high", "low", "close", "volume", "amount", "turnover"]
            if c in df.columns]
    df = df[keep].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["ts_code"] = ts_code
    return df.sort_values("date").reset_index(drop=True)


def fetch_stock_ohlcv(ts_code: str, start: str = PREDICT_START_DATE,
                      end: str | None = None, retries: int = 3) -> pd.DataFrame | None:
    """个股前复权日线：tushare → akshare。"""
    import datetime as dt

    end = end or dt.date.today().strftime("%Y%m%d")
    df = _fetch_ohlcv_tushare(ts_code, start, end)
    if df is not None and not df.empty:
        return df
    return _fetch_ohlcv_akshare(ts_code, start, end, retries=retries)


def _local_last_date(path: Path) -> pd.Timestamp | None:
    if not path.exists():
        return None
    try:
        existing = pd.read_parquet(path, columns=["date"])
        if existing.empty:
            return None
        return pd.to_datetime(existing["date"]).max()
    except Exception:
        return None


def _merge_ohlcv(path: Path, new_df: pd.DataFrame) -> pd.DataFrame:
    """与本地 parquet 合并去重，按 date 升序。"""
    if path.exists():
        old = pd.read_parquet(path)
        old["date"] = pd.to_datetime(old["date"])
        merged = pd.concat([old, new_df], ignore_index=True)
    else:
        merged = new_df
    merged["date"] = pd.to_datetime(merged["date"])
    merged = (
        merged.sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )
    return merged


def download_universe(max_stocks: int | None = None, sleep_s: float = 0.35,
                      start: str = PREDICT_START_DATE,
                      update_existing: bool = False,
                      lookback_days: int = 5,
                      pool: str = DEFAULT_POOL) -> dict:
    """下载 / 增量更新成分股 OHLCV。

    Args:
        update_existing: False 时已有文件直接跳过（首次拉取）；
            True 时从本地最后日期往前 lookback_days 起补拉并合并（日终刷新）。
        lookback_days: 增量时回看天数，兼顾前复权回溯调整。
        pool: 指数池，csi300 / csi500 / csi800（csi800 = csi300 ∪ csi500，官方指数）。
            OHLCV 统一存到同一个 ohlcv/ 目录，按 ts_code 去重，扩池只补拉新增股票。
    """
    cons = load_constituents(pool)
    if max_stocks:
        cons = cons.head(max_stocks)
    ok = skip = updated = fail = 0
    for _, row in cons.iterrows():
        code = row["ts_code"]
        path = _ohlcv_path(code)
        last = _local_last_date(path)

        if path.exists() and not update_existing:
            skip += 1
            continue

        if last is not None and update_existing:
            fetch_start = (last - pd.Timedelta(days=lookback_days)).strftime("%Y%m%d")
        else:
            fetch_start = start

        df = fetch_stock_ohlcv(code, start=fetch_start)
        if df is None or df.empty:
            fail += 1
            logger.warning("跳过 %s（拉取失败）", code)
            time.sleep(sleep_s)
            continue

        if path.exists():
            before = last
            merged = _merge_ohlcv(path, df)
            merged.to_parquet(path, index=False)
            after = pd.to_datetime(merged["date"]).max()
            if before is not None and after > before:
                updated += 1
                logger.info("[%d/%d] 更新 %s %s → %s (%d 行)",
                            ok + skip + updated + fail, len(cons), code,
                            before.date(), after.date(), len(merged))
            else:
                skip += 1
        else:
            if len(df) <= 100:
                fail += 1
                logger.warning("跳过 %s（数据不足 %d 行）", code, len(df))
            else:
                df.to_parquet(path, index=False)
                ok += 1
                logger.info("[%d/%d] 新建 %s %s 行",
                            ok + skip + updated + fail, len(cons), code, len(df))
        time.sleep(sleep_s)

    return {
        "total": len(cons),
        "downloaded": ok,
        "updated": updated,
        "skipped": skip,
        "failed": fail,
    }


def panel_asof_date() -> str | None:
    """本地 universe 最新交易日（YYYY-MM-DD）。"""
    paths = list(OHLCV_DIR.glob("*.parquet"))
    if not paths:
        return None
    latest = None
    for p in paths:
        d = _local_last_date(p)
        if d is None:
            continue
        latest = d if latest is None else max(latest, d)
    return None if latest is None else str(latest.date())


def refresh_cached_universe(
    codes: list[str] | None = None,
    *,
    lookback_days: int = 5,
    sleep_s: float = 0.25,
) -> dict:
    """增量刷新指定股票的本地 OHLCV；失败则保留缓存并标记 fallback。

    供模型实验室「最新行情预测」调用；返回结构与 inference.service 约定一致。
    """
    targets = list(codes) if codes else list_available_codes()
    updated = 0
    failed_codes: list[str] = []
    source = "tushare_or_akshare"

    for code in targets:
        path = _ohlcv_path(code)
        last = _local_last_date(path)
        if last is not None:
            fetch_start = (last - pd.Timedelta(days=lookback_days)).strftime("%Y%m%d")
        else:
            fetch_start = PREDICT_START_DATE
        try:
            df = fetch_stock_ohlcv(code, start=fetch_start)
            if df is None or df.empty:
                failed_codes.append(code)
                time.sleep(sleep_s)
                continue
            if path.exists():
                before = last
                merged = _merge_ohlcv(path, df)
                merged.to_parquet(path, index=False)
                after = pd.to_datetime(merged["date"]).max()
                if before is None or after > before:
                    updated += 1
            else:
                df.to_parquet(path, index=False)
                updated += 1
        except Exception as exc:
            logger.warning("刷新 %s 失败，保留本地缓存: %s", code, exc)
            failed_codes.append(code)
        time.sleep(sleep_s)

    latest = panel_asof_date()
    fallback = bool(failed_codes) or updated == 0
    if updated == 0:
        source = "local_cache"
    return {
        "requested": len(targets),
        "updated": updated,
        "failed": len(failed_codes),
        "failed_codes": failed_codes,
        "latest_market_date": latest,
        "source": source,
        "fallback_to_cache": fallback,
    }


def _path_to_code(p: Path) -> str:
    # 600519_SH.parquet → 600519.SH
    stem = p.stem
    if stem.endswith("_SH"):
        return stem[:-3] + ".SH"
    if stem.endswith("_SZ"):
        return stem[:-3] + ".SZ"
    return stem.replace("_", ".")


def list_available_codes() -> list[str]:
    return sorted(_path_to_code(p) for p in OHLCV_DIR.glob("*.parquet"))


def load_panel(codes: list[str] | None = None) -> pd.DataFrame:
    """加载本地 OHLCV 长表。"""
    paths = list(OHLCV_DIR.glob("*.parquet"))
    if codes:
        wanted = {c.replace(".", "_") for c in codes}
        paths = [p for p in paths if p.stem in wanted]
    if not paths:
        raise FileNotFoundError(
            f"未找到行情 Parquet，请先运行: python scripts/research/fetch_universe.py")
    frames = [pd.read_parquet(p) for p in paths]
    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    return panel.sort_values(["date", "ts_code"]).reset_index(drop=True)
