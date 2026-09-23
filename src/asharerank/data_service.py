"""数据层：真实行情多级回退（tushare → akshare → 模拟数据），本地缓存。

对外暴露：
- get_price_history(ts_codes) -> DataFrame(index=交易日, columns=ts_code, values=收盘价)
- get_stock_info(ts_codes)   -> DataFrame(ts_code, name, industry, market_cap)
- data_source_status()       -> 当前实际生效的数据源
"""
from __future__ import annotations

import datetime as dt
import json
import logging
from typing import Iterable

import numpy as np
import pandas as pd

from asharerank.settings import CACHE_DIR, LOOKBACK_DAYS, TUSHARE_TOKEN

logger = logging.getLogger(__name__)

_pro = None
_source = "uninitialized"  # "tushare" | "akshare" | "synthetic"
_code_price_sources: dict[str, str] = {}  # 逐标的行情数据源，供可信度展示用


def _mark_source(src: str) -> None:
    """记录实际生效的数据源。synthetic 最劣，一旦出现则整体标记为 synthetic。"""
    global _source
    priority = {"tushare": 0, "akshare": 1, "synthetic": 2, "uninitialized": -1}
    if priority.get(src, 2) > priority.get(_source, -1):
        _source = src


def get_price_source_map(ts_codes: Iterable[str]) -> dict[str, str]:
    """返回逐标的最近一次 get_price_history 实际生效的数据源，
    未拉取过的标的视为 synthetic（保守判断，避免误报为真实数据）。"""
    return {c: _code_price_sources.get(c, "synthetic") for c in ts_codes}


def _get_pro():
    global _pro
    if _pro is not None:
        return _pro
    if not TUSHARE_TOKEN:
        return None
    try:
        import tushare as ts

        ts.set_token(TUSHARE_TOKEN)
        _pro = ts.pro_api()
        # 不做连通性探测（探测接口本身有限频），由各数据拉取调用自行失败回退
    except Exception as exc:  # SDK 缺失等
        logger.warning("tushare 初始化失败: %s", exc)
        _pro = None
    return _pro


def data_source_status() -> str:
    return "synthetic" if _source == "uninitialized" else _source


# ---------------------------------------------------------------------------
# 行情
# ---------------------------------------------------------------------------

def _cache_path(ts_code: str) -> "Path":
    return CACHE_DIR / f"daily_{ts_code.replace('.', '_')}.csv"


def _fetch_daily_tushare(ts_code: str, start: str, end: str) -> pd.DataFrame | None:
    pro = _get_pro()
    if pro is None:
        return None
    try:
        df = pro.daily(ts_code=ts_code, start_date=start, end_date=end)
        if df is None or df.empty:
            return None
        df = df[["trade_date", "close"]].sort_values("trade_date")
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        return df.set_index("trade_date")
    except Exception as exc:
        logger.warning("tushare 拉取 %s 失败: %s", ts_code, exc)
        return None


def _fetch_daily_akshare(ts_code: str, start: str, end: str) -> pd.DataFrame | None:
    """akshare 东财日线（前复权收盘价）。symbol 不带交易所后缀。"""
    try:
        import akshare as ak

        df = ak.stock_zh_a_hist(symbol=ts_code.split(".")[0], period="daily",
                                start_date=start, end_date=end, adjust="qfq")
        if df is None or df.empty:
            return None
        df = df[["日期", "收盘"]].rename(columns={"日期": "trade_date", "收盘": "close"})
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        return df.sort_values("trade_date").set_index("trade_date")
    except Exception as exc:
        logger.warning("akshare 拉取 %s 失败: %s", ts_code, exc)
        return None


def _fetch_daily(ts_code: str, start: str, end: str) -> tuple[pd.DataFrame | None, str]:
    """按 tushare → akshare 顺序拉取，返回 (数据, 来源)。"""
    df = _fetch_daily_tushare(ts_code, start, end)
    if df is not None:
        return df, "tushare"
    df = _fetch_daily_akshare(ts_code, start, end)
    if df is not None:
        return df, "akshare"
    return None, "synthetic"


def _synthetic_daily(ts_code: str, n_days: int) -> pd.DataFrame:
    """可复现的几何布朗运动模拟价格，种子由代码决定，保证多次调用一致。"""
    # Python's built-in hash is process-randomized. A stable digest makes the
    # same synthetic ticker reproducible across runs and machines.
    import hashlib

    seed = int.from_bytes(hashlib.sha256(ts_code.encode("utf-8")).digest()[:4], "big")
    rng = np.random.default_rng(seed)
    mu, sigma = rng.uniform(0.0, 0.10), rng.uniform(0.18, 0.45)  # 年化
    dt_ = 1 / 252
    rets = rng.normal((mu - 0.5 * sigma**2) * dt_, sigma * np.sqrt(dt_), n_days)
    # 注入少量共同市场因子，让资产间有真实的相关结构
    market_rng = np.random.default_rng(20240101)
    market = market_rng.normal(0, 0.012, n_days)
    beta = rng.uniform(0.5, 1.5)
    rets = rets + beta * market
    prices = 50 * rng.uniform(0.5, 4.0) * np.exp(np.cumsum(rets))
    dates = pd.bdate_range(end=dt.date.today(), periods=n_days)
    return pd.DataFrame({"close": prices}, index=dates)


def _cache_src_path(path) -> "Path":
    return path.with_suffix(".src")


def _load_cache(path, max_stale_days: int = 3) -> tuple[pd.DataFrame | None, str]:
    if not path.exists():
        return None, "synthetic"
    cached = pd.read_csv(path, index_col=0, parse_dates=True)
    if cached.empty or (dt.date.today() - cached.index[-1].date()).days > max_stale_days:
        return None, "synthetic"
    src_file = _cache_src_path(path)
    src = src_file.read_text().strip() if src_file.exists() else "akshare"
    return cached, src


def _save_cache(path, df: pd.DataFrame, src: str) -> None:
    df.to_csv(path)
    _cache_src_path(path).write_text(src)


def get_price_history(ts_codes: Iterable[str], lookback: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """返回收盘价宽表，列为 ts_code。优先本地缓存，其次 tushare/akshare，最后模拟。"""
    end = dt.date.today()
    start = end - dt.timedelta(days=int(lookback * 1.6) + 30)
    start_s, end_s = start.strftime("%Y%m%d"), end.strftime("%Y%m%d")

    series = {}
    for code in ts_codes:
        path = _cache_path(code)
        df, src = _load_cache(path)
        if df is None:
            df, src = _fetch_daily(code, start_s, end_s)
            if df is not None:
                _save_cache(path, df, src)
        if df is None:
            df = _synthetic_daily(code, lookback + 30)
            src = "synthetic"
        _mark_source(src)
        _code_price_sources[code] = src
        series[code] = df["close"]

    prices = pd.DataFrame(series).sort_index().ffill().dropna()
    return prices.tail(lookback)


def get_returns(ts_codes: Iterable[str], lookback: int = LOOKBACK_DAYS) -> pd.DataFrame:
    prices = get_price_history(ts_codes, lookback)
    return prices.pct_change().dropna()


# ---------------------------------------------------------------------------
# 指数行情（市场组合：沪深300）
# ---------------------------------------------------------------------------

INDEX_CODE = "000300.SH"
INDEX_HISTORY_START = "20050408"  # 沪深300 发布日


def _synthetic_index(n_days: int) -> pd.DataFrame:
    """模拟模式下的指数：复用模拟个股共享的市场因子（同一种子），保证 beta 可估。"""
    rng = np.random.default_rng(20240101)
    market = rng.normal(0.0002, 0.012, n_days)
    prices = 3500 * np.exp(np.cumsum(market))
    dates = pd.bdate_range(end=dt.date.today(), periods=n_days)
    return pd.DataFrame({"close": prices}, index=dates)


def _fetch_index_akshare(index_code: str) -> pd.DataFrame | None:
    """akshare 新浪指数日线（全历史）。'000300.SH' → 'sh000300'。"""
    try:
        import akshare as ak

        num, exch = index_code.split(".")
        df = ak.stock_zh_index_daily(symbol=f"{exch.lower()}{num}")
        if df is None or df.empty:
            return None
        df = df[["date", "close"]].rename(columns={"date": "trade_date"})
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        return df.sort_values("trade_date").set_index("trade_date")
    except Exception as exc:
        logger.warning("akshare 拉取指数 %s 失败: %s", index_code, exc)
        return None


# 指数数据进程内缓存与失败冷却（index_daily 限频 1 次/分钟，不能反复重试）
_index_mem: dict[str, pd.DataFrame] = {}
_index_fail_until: dict[str, dt.datetime] = {}
_INDEX_FAIL_COOLDOWN = dt.timedelta(minutes=15)


def get_index_history(index_code: str = INDEX_CODE, full: bool = False) -> pd.DataFrame:
    """沪深300 收盘价序列。full=True 返回 2005 年至今的全部历史（供历史情景压测）。

    一次拉全并缓存到磁盘（3 天内新鲜直接用）；进程内记忆化；
    拉取失败进入 15 分钟冷却期，期间用过期磁盘缓存或模拟数据兜底。
    """
    if index_code in _index_mem:
        df = _index_mem[index_code]
        return df if full else df.tail(LOOKBACK_DAYS)

    path = CACHE_DIR / f"index_{index_code.replace('.', '_')}.csv"
    end = dt.date.today()
    df = None
    stale = None
    if path.exists():
        cached = pd.read_csv(path, index_col=0, parse_dates=True)
        if not cached.empty:
            if (end - cached.index[-1].date()).days <= 3:
                df = cached
            else:
                stale = cached

    # 指数全历史优先 akshare（免费无限频）；tushare index_daily 积分不足时仅 1 次/小时
    now = dt.datetime.now()
    in_cooldown = now < _index_fail_until.get(index_code, dt.datetime.min)
    if df is None and not in_cooldown:
        df = _fetch_index_akshare(index_code)
        if df is not None:
            df.to_csv(path)
            _mark_source("akshare")
            logger.info("指数 %s 已由 akshare 拉取，共 %d 行", index_code, len(df))
        else:
            pro = _get_pro()
            if pro is not None:
                try:
                    raw = pro.index_daily(ts_code=index_code,
                                          start_date=INDEX_HISTORY_START,
                                          end_date=end.strftime("%Y%m%d"))
                    if raw is not None and not raw.empty:
                        raw = raw[["trade_date", "close"]].sort_values("trade_date")
                        raw["trade_date"] = pd.to_datetime(raw["trade_date"])
                        df = raw.set_index("trade_date")
                        df.to_csv(path)
                        _mark_source("tushare")
                except Exception as exc:
                    logger.warning("tushare 拉取指数 %s 失败: %s", index_code, exc)
            if df is None:
                _index_fail_until[index_code] = now + _INDEX_FAIL_COOLDOWN
                logger.warning("指数 %s 拉取失败（冷却 15 分钟）", index_code)

    if df is None:
        df = stale  # 过期缓存好于模拟数据
    if df is None:
        df = _synthetic_index(LOOKBACK_DAYS + 30)
    else:
        _index_mem[index_code] = df
    return df if full else df.tail(LOOKBACK_DAYS)


def get_index_returns(full: bool = False) -> pd.Series:
    prices = get_index_history(full=full)
    return prices["close"].pct_change().dropna()


def get_long_price_history(ts_codes: Iterable[str]) -> pd.DataFrame:
    """持仓个股 2005 年至今的长历史收盘价（供历史情景真实回放），逐只缓存。
    个股未上市期间为 NaN。"""
    end = dt.date.today()
    series = {}
    for code in ts_codes:
        path = CACHE_DIR / f"daily_long_{code.replace('.', '_')}.csv"
        df = None
        if path.exists():
            cached = pd.read_csv(path, index_col=0, parse_dates=True)
            if not cached.empty and (end - cached.index[-1].date()).days <= 7:
                df = cached
        if df is None:
            df, _src = _fetch_daily(code, INDEX_HISTORY_START, end.strftime("%Y%m%d"))
            if df is not None:
                df.to_csv(path)
        if df is None:
            df = _synthetic_daily(code, LOOKBACK_DAYS + 30)
        series[code] = df["close"]
    return pd.DataFrame(series).sort_index()


# ---------------------------------------------------------------------------
# 股票基本信息（名称/行业/市值）
# ---------------------------------------------------------------------------

# 模拟数据模式下的兜底信息（示例组合成分）
_FALLBACK_INFO = {
    "600519.SH": ("贵州茅台", "白酒", 21000),
    "300750.SZ": ("宁德时代", "电池", 11000),
    "601318.SH": ("中国平安", "保险", 9500),
    "600036.SH": ("招商银行", "银行", 9000),
    "000858.SZ": ("五粮液", "白酒", 5500),
    "601899.SH": ("紫金矿业", "有色金属", 4800),
    "002415.SZ": ("海康威视", "电子设备", 3000),
    "600900.SH": ("长江电力", "电力", 6800),
    "601012.SH": ("隆基绿能", "光伏", 1500),
    "000333.SZ": ("美的集团", "家电", 5200),
}


_INFO_FAIL_COOLDOWN = dt.timedelta(minutes=30)
_info_fail_until: dict[str, dt.datetime] = {}
_basic_cache: pd.DataFrame | None = None
_basic_fail_until: dt.datetime = dt.datetime.min
_mv_cache: pd.Series | None = None
_mv_fail_until: dt.datetime = dt.datetime.min
_stock_info_mem: dict[tuple[str, ...], tuple[dt.datetime, pd.DataFrame]] = {}
_STOCK_INFO_TTL = dt.timedelta(minutes=10)


def _info_cache_path(ts_code: str) -> "Path":
    return CACHE_DIR / f"info_{ts_code.replace('.', '_')}.json"


def _seed_fallback_info(ts_code: str) -> dict | None:
    """把内置兜底写入磁盘，避免后续请求反复打外部接口。"""
    if ts_code not in _FALLBACK_INFO:
        return None
    name, industry, mktcap = _FALLBACK_INFO[ts_code]
    result = {"name": name, "industry": industry, "market_cap": float(mktcap),
              "source": "fallback"}
    _info_cache_path(ts_code).write_text(json.dumps(result, ensure_ascii=False))
    return result


def _load_cached_info(ts_code: str) -> dict | None:
    path = _info_cache_path(ts_code)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return None


def _fetch_info_akshare(ts_code: str) -> dict | None:
    """akshare 东财个股信息。成功则永久缓存；失败进入冷却并尽量用兜底落盘。"""
    cached = _load_cached_info(ts_code)
    if cached is not None:
        return cached

    now = dt.datetime.now()
    if now < _info_fail_until.get(ts_code, dt.datetime.min):
        return _seed_fallback_info(ts_code)

    try:
        import akshare as ak

        info = ak.stock_individual_info_em(symbol=ts_code.split(".")[0])
        d = dict(zip(info["item"], info["value"]))
        result = {
            "name": str(d.get("股票简称", ts_code)),
            "industry": str(d.get("行业", "其他")),
            "market_cap": float(d.get("总市值", 0)) / 1e8,
            "source": "akshare",
        }
        _info_cache_path(ts_code).write_text(json.dumps(result, ensure_ascii=False))
        return result
    except Exception as exc:
        _info_fail_until[ts_code] = now + _INFO_FAIL_COOLDOWN
        # 只打一次警告级别日志，后续冷却期内静默走兜底
        logger.warning("akshare 个股信息失败 %s（冷却 30 分钟）: %s", ts_code, exc)
        return _seed_fallback_info(ts_code)


def _get_stock_basic() -> pd.DataFrame | None:
    """stock_basic 进程内缓存；限频失败冷却 30 分钟，不再每次 /portfolio 都打。"""
    global _basic_cache, _basic_fail_until
    if _basic_cache is not None:
        return _basic_cache
    if dt.datetime.now() < _basic_fail_until:
        return None
    path = CACHE_DIR / "stock_basic.csv"
    if path.exists():
        try:
            cached = pd.read_csv(path).set_index("ts_code")
            # 一周内缓存视为可用
            age = dt.datetime.now() - dt.datetime.fromtimestamp(path.stat().st_mtime)
            if age < dt.timedelta(days=7):
                _basic_cache = cached
                return cached
        except Exception:
            pass
    pro = _get_pro()
    if pro is None:
        return None
    try:
        basic = pro.stock_basic(exchange="", list_status="L",
                                fields="ts_code,name,industry")
        if basic is not None and not basic.empty:
            basic.to_csv(path, index=False)
            _basic_cache = basic.set_index("ts_code")
            return _basic_cache
    except Exception as exc:
        _basic_fail_until = dt.datetime.now() + _INFO_FAIL_COOLDOWN
        logger.warning("拉取 stock_basic 失败（冷却 30 分钟）: %s", exc)
    return None


def _get_market_caps(codes: list[str]) -> pd.Series | None:
    """daily_basic 市值：成功则缓存；限频后冷却，避免刷屏。"""
    global _mv_cache, _mv_fail_until
    if _mv_cache is not None:
        return _mv_cache.reindex(codes)
    if dt.datetime.now() < _mv_fail_until:
        return None
    path = CACHE_DIR / "daily_basic_mv.csv"
    if path.exists():
        age = dt.datetime.now() - dt.datetime.fromtimestamp(path.stat().st_mtime)
        if age < dt.timedelta(days=1):
            try:
                _mv_cache = pd.read_csv(path, index_col=0).squeeze("columns")
                return _mv_cache.reindex(codes)
            except Exception:
                pass
    pro = _get_pro()
    if pro is None:
        return None
    try:
        trade_date = dt.date.today()
        for _ in range(7):
            db = pro.daily_basic(trade_date=trade_date.strftime("%Y%m%d"),
                                 fields="ts_code,total_mv")
            if db is not None and not db.empty:
                series = db.set_index("ts_code")["total_mv"] / 1e4  # 万元->亿元
                series.to_csv(path)
                _mv_cache = series
                return series.reindex(codes)
            trade_date -= dt.timedelta(days=1)
    except Exception as exc:
        _mv_fail_until = dt.datetime.now() + _INFO_FAIL_COOLDOWN
        logger.warning("拉取 daily_basic 失败（冷却 30 分钟）: %s", exc)
    return None


def get_stock_info(ts_codes: Iterable[str]) -> pd.DataFrame:
    """返回 ts_code, name, industry, market_cap(亿元)。

    优先级：磁盘/内存缓存 → 内置兜底（示例成分股直接用，不再打外网）
    → tushare stock_basic → akshare → 兜底。
    外网失败进入 30 分钟冷却，避免 /portfolio 每次刷日志。
    """
    codes = list(ts_codes)
    key = tuple(sorted(codes))
    now = dt.datetime.now()
    if key in _stock_info_mem:
        cached_at, cached_df = _stock_info_mem[key]
        if now - cached_at < _STOCK_INFO_TTL:
            return cached_df.copy()

    # 示例组合成分：优先磁盘缓存 / 内置表，不主动打 akshare
    basic = _get_stock_basic()
    rows = []
    for code in codes:
        name = industry = None
        mktcap = 1000.0
        cached = _load_cached_info(code)
        if cached is not None:
            name, industry = cached["name"], cached["industry"]
            mktcap = float(cached.get("market_cap") or mktcap)
        elif code in _FALLBACK_INFO:
            seeded = _seed_fallback_info(code)
            name, industry = seeded["name"], seeded["industry"]
            mktcap = seeded["market_cap"]
        elif basic is not None and code in basic.index:
            name = basic.loc[code, "name"]
            industry = basic.loc[code, "industry"] or "其他"
        else:
            ak_info = _fetch_info_akshare(code)
            if ak_info is not None:
                name, industry = ak_info["name"], ak_info["industry"]
                if ak_info.get("market_cap", 0) > 0:
                    mktcap = float(ak_info["market_cap"])
        if name is None:
            name, industry = code, "其他"
        if basic is not None and code in basic.index:
            # 有 stock_basic 时用官方名称/行业覆盖兜底
            name = basic.loc[code, "name"]
            industry = basic.loc[code, "industry"] or industry
        rows.append({"ts_code": code, "name": name, "industry": industry,
                     "market_cap": mktcap})

    df = pd.DataFrame(rows)
    mv = _get_market_caps(codes)
    if mv is not None:
        df["market_cap"] = df["ts_code"].map(mv).fillna(df["market_cap"])

    _stock_info_mem[key] = (now, df.copy())
    return df
