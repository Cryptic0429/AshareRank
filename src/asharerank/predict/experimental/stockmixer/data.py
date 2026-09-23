"""A 股（沪深300）数据获取与特征工程，构造 StockMixer 所需的
`[Batch, N, T, F]` 张量数据集。

数据源：akshare（通过项目已有的 `asharerank.predict.universe` 模块，
新浪/东财接口，支持本地缓存 + 断点续传，避免重复拉取）。

处理流程：
  1. 下载 / 复用本地缓存的沪深300 成分股前复权日线；
  2. 剔除上市不满 `min_listed_days` 个交易日的新股；
  3. 对齐到统一交易日历，短暂停牌（<= ffill_limit 天）前向填充，
     长期停牌/数据覆盖率过低的股票整体剔除；
  4. 特征工程：开高低收相对上一日收盘的涨跌幅 + 成交量对数变化 +
     5/10 日均线相对收盘价的偏离度，共 F = 5 + len(ma_windows) 维量价特征；
     `cfg.use_fundamentals=True` 时额外拼接基本面因子（默认 7 维，可用
     `fundamental_fields` 少维消融），并按 `industry_neutralize_fundamentals`
     做行业内截面标准化；`fund_missing_mask=True` 时再拼一维缺测掩码；
  5. 标签为未来 `cfg.label_horizon` 日累计收益率（默认 1 日，可调大以对齐
     其他基线模型的 5 日 horizon、降低单日噪声）；用训练集统计量做 z-score
     标准化（避免验证/测试集信息泄露）；
  6. 按"近 3 年训练 / 近 6 月验证 / 近 6 月测试"切分锚点日期；
  7. 打包成 `StockMixerDataset`，`__getitem__` 返回单日 `(X[N,T,F], y[N], mask[N])`，
     经 DataLoader 按 batch_size 打包多日即得到 `[B,N,T,F]`。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from asharerank.predict.experimental.stockmixer.config import StockMixerConfig
from asharerank.predict.fundamentals import build_fundamental_feature_panel
from asharerank.predict.industry import load_industry_map
from asharerank.predict.universe import download_universe, load_constituents, load_panel

logger = logging.getLogger(__name__)

# 基础量价特征（顺序固定，供 feat_arr 的最后一维索引使用）
_BASE_COLS = ["open", "high", "low", "close", "volume"]


# ============================================================
# 1) 拉取 / 复用本地缓存的沪深300 行情
# ============================================================
def fetch_raw_panel(cfg: StockMixerConfig) -> pd.DataFrame:
    """下载（或复用缓存）沪深300 成分股前复权日线，返回长表 DataFrame。"""
    stats = download_universe(max_stocks=cfg.max_stocks, pool=cfg.pool, start=cfg.fetch_start)
    logger.info("沪深300 行情下载/校验完成: %s", stats)

    cons = load_constituents(cfg.pool)
    codes = cons["ts_code"].tolist()
    if cfg.max_stocks:
        codes = codes[: cfg.max_stocks]

    panel = load_panel(codes=codes)
    panel["date"] = pd.to_datetime(panel["date"])
    missing_cols = [c for c in _BASE_COLS if c not in panel.columns]
    if missing_cols:
        raise RuntimeError(f"行情数据缺少必要列: {missing_cols}")
    logger.info("原始行情: %d 只股票, %d 行, 区间 %s ~ %s",
                panel["ts_code"].nunique(), len(panel),
                panel["date"].min().date(), panel["date"].max().date())
    return panel


# ============================================================
# 2) 训练 / 验证 / 测试日期边界
# ============================================================
def compute_date_bounds(dates_all: list[pd.Timestamp], cfg: StockMixerConfig):
    """按"近3年训练 / 近6月验证 / 近6月测试"从最新日期往前切分。"""
    test_end = dates_all[-1]
    test_start = test_end - pd.DateOffset(months=cfg.test_months)
    val_end = test_start
    val_start = val_end - pd.DateOffset(months=cfg.valid_months)
    train_end = val_start
    train_start = train_end - pd.DateOffset(years=int(cfg.train_years)) - \
        pd.DateOffset(months=round((cfg.train_years % 1) * 12))
    return train_start, train_end, val_start, val_end, test_start, test_end


# ============================================================
# 3) 上市年限 / 停牌覆盖率过滤（高容错率要求）
# ============================================================
def _filter_stocks(panel: pd.DataFrame, cfg: StockMixerConfig,
                   train_start: pd.Timestamp, test_end: pd.Timestamp) -> pd.DataFrame:
    n_before = panel["ts_code"].nunique()
    if cfg.freeze_codes:
        # 控变量消融：跳过上市年限过滤，强制使用指定股票池（仍做覆盖率过滤）
        eligible = set(cfg.freeze_codes)
        present = set(panel["ts_code"].unique())
        missing = sorted(eligible - present)
        eligible = eligible & present
        logger.info("冻结股票池：指定 %d 只，行情中命中 %d 只（缺失 %d 只）",
                    len(cfg.freeze_codes), len(eligible), len(missing))
        if missing[:5]:
            logger.warning("冻结池中无行情的示例: %s%s",
                           missing[:5], " ..." if len(missing) > 5 else "")
    else:
        # ---- 3.1 剔除上市不满 min_listed_days 交易日的新股 ----
        hist_counts = panel.loc[panel["date"] < train_start, "ts_code"].value_counts()
        eligible = set(hist_counts[hist_counts >= cfg.min_listed_days].index)
        logger.info("上市年限过滤：剔除 %d / %d 只（训练起点前历史 < %d 个交易日）",
                    n_before - len(eligible), n_before, cfg.min_listed_days)

    warmup_days = cfg.lookback + max(cfg.ma_windows) + 5
    buffer_start = train_start - pd.Timedelta(days=int(warmup_days * 2.2))
    panel = panel[
        panel["ts_code"].isin(eligible)
        & (panel["date"] >= buffer_start)
        & (panel["date"] <= test_end)
    ].copy()

    # ---- 3.2 停牌 / 数据覆盖率过滤：整段区间缺失过多的股票整体剔除 ----
    dates_all = sorted(panel["date"].unique())
    n_dates = max(len(dates_all), 1)
    coverage = panel.groupby("ts_code").size() / n_dates
    keep_codes = sorted(coverage[coverage >= (1 - cfg.max_missing_ratio)].index)
    logger.info("停牌覆盖率过滤：保留 %d / %d 只（覆盖率 >= %.0f%%）",
                len(keep_codes), panel["ts_code"].nunique(), (1 - cfg.max_missing_ratio) * 100)

    panel = panel[panel["ts_code"].isin(keep_codes)]
    if len(keep_codes) < 10:
        raise ValueError(
            f"过滤后仅剩 {len(keep_codes)} 只股票，样本过少，请调低 min_listed_days / "
            f"max_missing_ratio，或先运行 fetch_universe.py 补齐历史数据"
        )
    return panel


def _pivot(panel: pd.DataFrame, dates_all: list, codes: list, col: str) -> pd.DataFrame:
    wide = panel.pivot_table(index="date", columns="ts_code", values=col, aggfunc="last")
    return wide.reindex(index=dates_all, columns=codes)


# ============================================================
# 3.5) 基本面因子 + 行业中性化（2026-07-23 新增）
# ============================================================
def _industry_neutralize_wide(wide: pd.DataFrame, codes: list[str],
                              industry_map: dict[str, str]) -> pd.DataFrame:
    """对宽表 [D, N] 按"每日截面 + 行业分组"做 z-score：同行业股票互相比较，
    剥离行业整体水平差异（例如银行 ROE 天然比科技股高，不做处理会让模型学到
    "行业哑变量"而不是"行业内相对强弱"）。行业信息缺失或组内样本 < 3 的
    股票，退化为跨全市场（不分行业）的截面 z-score，保证仍有输出而不是 NaN。
    """
    industries = pd.Series([industry_map.get(c, "未分类") for c in codes], index=codes)
    out = pd.DataFrame(index=wide.index, columns=wide.columns, dtype=np.float64)
    market_mu = wide.mean(axis=1)
    market_sigma = wide.std(axis=1).replace(0, np.nan)
    out_market = wide.sub(market_mu, axis=0).div(market_sigma, axis=0)

    for ind, members in industries.groupby(industries):
        cols = members.index.tolist()
        if len(cols) < 3:
            out[cols] = out_market[cols]
            continue
        sub = wide[cols]
        mu = sub.mean(axis=1)
        sigma = sub.std(axis=1).replace(0, np.nan)
        neutralized = sub.sub(mu, axis=0).div(sigma, axis=0)
        # 单日行业内有效样本 < 3 时该日的行业 z-score 不可靠，退化为全市场版本
        n_valid = sub.notna().sum(axis=1)
        thin_days = n_valid < 3
        neutralized.loc[thin_days, :] = out_market.loc[thin_days, cols]
        out[cols] = neutralized
    return out.astype(np.float32)


def _build_fundamental_frames(panel: pd.DataFrame, dates_all: list[pd.Timestamp],
                              codes: list[str], close_w: pd.DataFrame,
                              cfg: StockMixerConfig) -> tuple[list[pd.DataFrame], list[str]]:
    """构造基本面特征宽表：5 个质量类指标（ROE/净利率/资产负债率/营收增速/利润增速，
    天然无量纲） + 2 个衍生估值比率（ep_ratio=EPS/收盘价、bp_ratio=BVPS/收盘价，
    比原始 EPS/BVPS 绝对值更适合跨股票比较）。数据源为季度财报，已用
    REPORT_LAG_DAYS 滞后处理防未来函数；覆盖不到的股票在标准化后填 0（视为
    "行业/市场平均水平"的中性假设），不会因为基本面缺失被整体剔除出股票池。
    """
    calendar = panel[["date", "ts_code"]].drop_duplicates()
    fund_long = build_fundamental_feature_panel(calendar, codes=codes)
    if fund_long.empty:
        logger.warning("未取得任何基本面数据（先运行 scripts/research/fetch_fundamentals.py），"
                       "本次训练跳过基本面特征，仅用量价特征")
        return [], []

    if cfg.fundamental_fields is not None:
        wanted = tuple(cfg.fundamental_fields)
    else:
        wanted = tuple(cfg.fundamental_quality_fields) + ("ep_ratio", "bp_ratio")
    wanted_set = set(wanted)

    raw_frames: dict[str, pd.DataFrame] = {}
    for field in wanted:
        if field in ("ep_ratio", "bp_ratio"):
            continue
        if field not in fund_long.columns:
            continue
        wide = fund_long.pivot_table(index="date", columns="ts_code", values=field, aggfunc="last")
        raw_frames[field] = wide.reindex(index=dates_all, columns=codes)

    if "ep_ratio" in wanted_set and "eps" in fund_long.columns:
        eps_wide = fund_long.pivot_table(index="date", columns="ts_code", values="eps", aggfunc="last")
        eps_wide = eps_wide.reindex(index=dates_all, columns=codes)
        raw_frames["ep_ratio"] = (eps_wide / close_w).replace([np.inf, -np.inf], np.nan).clip(-1, 1)
    if "bp_ratio" in wanted_set and "bvps" in fund_long.columns:
        bvps_wide = fund_long.pivot_table(index="date", columns="ts_code", values="bvps", aggfunc="last")
        bvps_wide = bvps_wide.reindex(index=dates_all, columns=codes)
        raw_frames["bp_ratio"] = (bvps_wide / close_w).replace([np.inf, -np.inf], np.nan).clip(-5, 5)

    n_covered = fund_long["ts_code"].nunique()
    logger.info("基本面特征：%d/%d 只股票有覆盖（%.0f%%），字段=%s",
                n_covered, len(codes), 100 * n_covered / max(len(codes), 1), list(raw_frames.keys()))
    if n_covered < len(codes):
        logger.warning("基本面覆盖不全，缺失的 %d 只股票该组特征在标准化后会填 0（中性假设）%s，"
                       "建议运行 scripts/research/fetch_fundamentals.py --csi300-missing 补齐后重新训练",
                       len(codes) - n_covered,
                       "；已附加 fund_missing 掩码" if cfg.fund_missing_mask else "")

    if cfg.industry_neutralize_fundamentals:
        industry_map = load_industry_map()
        frames = [_industry_neutralize_wide(f, codes, industry_map) for f in raw_frames.values()]
    else:
        frames = list(raw_frames.values())
    names = [f"fund_{k}" for k in raw_frames.keys()]

    # 缺测掩码：用行业中性化之前的原始 NaN 判断（中性化不会"创造"覆盖），
    # 全部基本面字段均为 NaN → 1，否则 → 0。不参与行业中性化。
    if cfg.fund_missing_mask and raw_frames:
        stacked = np.stack([f.to_numpy(dtype=np.float32) for f in raw_frames.values()], axis=-1)
        missing = np.isnan(stacked).all(axis=-1).astype(np.float32)  # [D, N]
        frames.append(pd.DataFrame(missing, index=dates_all, columns=codes))
        names.append("fund_missing")
        logger.info("已附加 fund_missing 掩码：截面平均缺测率=%.1f%%",
                    100.0 * float(missing.mean()))

    return frames, names


# ============================================================
# 4) 特征工程 + 5) 标准化 —— 汇总为特征/标签/有效性数组
# ============================================================
class PreparedData:
    """准备好的全量数据：dense [D, N, F] 张量 + 训练/验证/测试锚点索引。"""

    def __init__(self, feat_arr: np.ndarray, label_arr: np.ndarray,
                feat_valid: np.ndarray, label_valid: np.ndarray,
                dates_all: list[pd.Timestamp], codes: list[str],
                feature_names: list[str], lookback: int,
                train_idx: list[int], val_idx: list[int], test_idx: list[int],
                date_bounds: tuple, mean: np.ndarray, std: np.ndarray,
                purged_train_dates: tuple[pd.Timestamp, ...] = (),
                purged_val_dates: tuple[pd.Timestamp, ...] = ()):
        self.feat_arr = feat_arr          # [D, N, F] float32，已标准化，缺失位置填 0
        self.label_arr = label_arr        # [D, N] float32，label_arr[d] 对应未来 horizon 日标签
        self.feat_valid = feat_valid      # [D, N] bool
        self.label_valid = label_valid    # [D, N] bool
        self.dates_all = dates_all
        self.codes = codes
        self.feature_names = feature_names
        self.lookback = lookback
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx
        self.date_bounds = date_bounds
        self.purged_train_dates = purged_train_dates
        self.purged_val_dates = purged_val_dates
        # 训练集统计量（[F]），推理时对最新窗口做同口径 z-score 标准化必须复用这份，
        # 不能用推理当天现算的均值/方差（否则训练/推理的特征分布对不齐）。
        self.mean = mean
        self.std = std

    @property
    def n_stocks(self) -> int:
        return len(self.codes)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)


def prepare_dataset(cfg: StockMixerConfig) -> PreparedData:
    panel = fetch_raw_panel(cfg)
    all_dates = sorted(panel["date"].unique())
    train_start, train_end, val_start, val_end, test_start, test_end = compute_date_bounds(all_dates, cfg)

    panel = _filter_stocks(panel, cfg, train_start, test_end)
    dates_all = sorted(panel["date"].unique())
    codes = sorted(panel["ts_code"].unique())
    logger.info("最终建模股票池: %d 只，交易日: %d 天（%s ~ %s）",
                len(codes), len(dates_all), dates_all[0].date(), dates_all[-1].date())

    # ---- 4.1 宽表 pivot + 短暂停牌前向填充 ----
    open_w = _pivot(panel, dates_all, codes, "open").ffill(limit=cfg.ffill_limit)
    high_w = _pivot(panel, dates_all, codes, "high").ffill(limit=cfg.ffill_limit)
    low_w = _pivot(panel, dates_all, codes, "low").ffill(limit=cfg.ffill_limit)
    close_w = _pivot(panel, dates_all, codes, "close").ffill(limit=cfg.ffill_limit)
    vol_w = _pivot(panel, dates_all, codes, "volume").ffill(limit=cfg.ffill_limit)

    close_prev = close_w.shift(1)
    clip = cfg.clip_return

    # ---- 4.2 相对特征（对不同股价水平的股票做归一化，避免尺度差异）----
    o_rel = (open_w / close_prev - 1).clip(-clip, clip)
    h_rel = (high_w / close_prev - 1).clip(-clip, clip)
    l_rel = (low_w / close_prev - 1).clip(-clip, clip)
    c_rel = (close_w / close_prev - 1).clip(-clip, clip)
    vol_log = np.log1p(vol_w.clip(lower=0))
    vol_chg = (vol_log - vol_log.shift(1)).clip(-3, 3)

    ma_rel_frames = []
    for w in cfg.ma_windows:
        ma = close_w.rolling(window=w, min_periods=w).mean()
        ma_rel_frames.append((ma / close_prev - 1).clip(-clip, clip))

    price_feature_frames = [o_rel, h_rel, l_rel, c_rel, vol_chg, *ma_rel_frames]
    price_feature_names = ["open_rel", "high_rel", "low_rel", "close_rel", "volume_log_chg"] + \
        [f"ma{w}_rel" for w in cfg.ma_windows]

    # ---- 4.2.5 基本面 / 行业中性化因子（可选）----
    # feat_valid（窗口是否可用于训练）只由量价特征决定，不受基本面覆盖率影响：
    # 否则 207/300 只暂无基本面数据的股票会被整体判定为"窗口无效"而被排除出每一天的
    # 训练/评估截面，等于变相把股票池砍掉大半。基本面缺失的位置在标准化后填 0（中性假设）。
    price_arr = np.stack([f.to_numpy(dtype=np.float32) for f in price_feature_frames], axis=-1)  # [D,N,Fp]
    feat_valid = ~np.isnan(price_arr).any(axis=-1)  # [D,N]，仅由量价特征决定

    if cfg.use_fundamentals:
        fund_frames, fund_names = _build_fundamental_frames(panel, dates_all, codes, close_w, cfg)
    else:
        fund_frames, fund_names = [], []

    if fund_frames:
        fund_arr = np.stack([f.to_numpy(dtype=np.float32) for f in fund_frames], axis=-1)  # [D,N,Ff]
        raw_feat_arr = np.concatenate([price_arr, fund_arr], axis=-1)  # [D,N,Fp+Ff]
        feature_names = price_feature_names + fund_names
    else:
        raw_feat_arr = price_arr
        feature_names = price_feature_names

    # ---- 4.3 标签：未来 label_horizon 日累计收益率 r_i^{t->t+H} = (close_{t+H}-close_t)/close_t ----
    # H=1 即用户原始需求里的"次日收益率"；H>1 时是多日累计收益（与 LightGBM/XGBoost
    # 基线的 PREDICT_HORIZON=5 对齐，用于降低单日噪声、便于横向比较）。
    horizon = max(cfg.label_horizon, 1)
    label_raw = (close_w.shift(-horizon) / close_w - 1).clip(-clip, clip).to_numpy(dtype=np.float32)  # [D,N]
    label_valid = ~np.isnan(label_raw)

    # ---- 4.4 用训练集统计量做 z-score 标准化（防止验证/测试信息泄露）----
    # 用 feat_valid（仅量价）筛选训练期的行，但用 nanmean/nanstd 逐列统计——基本面列
    # 在这些行里可能仍有 NaN（覆盖不到的股票），nan-aware 统计量不会被这些 NaN 污染，
    # 也不会像 np.mean 那样让整列直接变成 NaN。
    train_day_mask = np.array([train_start < d <= train_end for d in dates_all])
    train_vals = raw_feat_arr[train_day_mask][feat_valid[train_day_mask]]
    if len(train_vals) == 0:
        raise ValueError("训练区间没有任何有效样本，请检查日期切分/数据覆盖率设置")
    mean = np.nan_to_num(np.nanmean(train_vals, axis=0), nan=0.0)
    std = np.nan_to_num(np.nanstd(train_vals, axis=0), nan=1.0)
    std[std < 1e-8] = 1.0
    # fund_missing 是 {0,1} 指示变量，保持原值不做 z-score（mean=0/std=1 等价恒等变换）
    for i, name in enumerate(feature_names):
        if name == "fund_missing":
            mean[i] = 0.0
            std[i] = 1.0

    feat_norm = (raw_feat_arr - mean) / std          # 缺失位置仍为 NaN
    feat_arr = np.nan_to_num(feat_norm, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    label_arr = np.nan_to_num(label_raw, nan=0.0).astype(np.float32)

    # ---- 4.5 切分训练 / 验证 / 测试锚点索引 ----
    n_dates = len(dates_all)
    lookback = cfg.lookback
    anchors = list(range(lookback - 1, n_dates - horizon))  # 需要 T 天历史 + H 天可用标签
    train_candidates = [d for d in anchors if train_start < dates_all[d] <= train_end]
    val_candidates = [d for d in anchors if val_start < dates_all[d] <= val_end]
    # Keep a label wholly within its assigned partition. Labels ending in the
    # following fold are excluded from fitting and early-stopping metrics.
    train_idx = [d for d in train_candidates if dates_all[d + horizon] <= train_end]
    val_idx = [d for d in val_candidates if dates_all[d + horizon] <= val_end]
    train_kept = set(train_idx)
    val_kept = set(val_idx)
    purged_train_dates = tuple(dates_all[d] for d in train_candidates if d not in train_kept)
    purged_val_dates = tuple(dates_all[d] for d in val_candidates if d not in val_kept)
    test_idx = [d for d in anchors if test_start < dates_all[d] <= test_end]
    logger.info("锚点样本数：train=%d, val=%d, test=%d", len(train_idx), len(val_idx), len(test_idx))
    if min(len(train_idx), len(val_idx), len(test_idx)) == 0:
        raise ValueError("训练/验证/测试集中存在空集合，历史数据长度不足，请拉取更长历史或调小窗口/年限")

    return PreparedData(
        feat_arr=feat_arr, label_arr=label_arr,
        feat_valid=feat_valid, label_valid=label_valid,
        dates_all=dates_all, codes=codes, feature_names=feature_names,
        lookback=lookback, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
        date_bounds=(train_start, train_end, val_start, val_end, test_start, test_end),
        mean=mean, std=std,
        purged_train_dates=purged_train_dates,
        purged_val_dates=purged_val_dates,
    )


# ============================================================
# 6) PyTorch Dataset：单样本 = 单个交易日的全市场截面
# ============================================================
class StockMixerDataset(Dataset):
    """`__getitem__` 返回 (X[N,T,F], y[N], mask[N])；DataLoader 按 batch_size 打包多日
    即得到 `[B,N,T,F]` / `[B,N]` / `[B,N]`。"""

    def __init__(self, data: PreparedData, anchor_indices: list[int]):
        self.data = data
        self.anchors = anchor_indices
        self.lookback = data.lookback

    def __len__(self) -> int:
        return len(self.anchors)

    def __getitem__(self, idx: int):
        d = self.anchors[idx]
        t0 = d - self.lookback + 1
        x = self.data.feat_arr[t0:d + 1]                      # [T, N, F]
        x = np.transpose(x, (1, 0, 2))                          # [N, T, F]
        y = self.data.label_arr[d]                              # [N]  次日收益率
        window_ok = self.data.feat_valid[t0:d + 1].all(axis=0)  # [N] 整段窗口均非缺失
        mask = (window_ok & self.data.label_valid[d]).astype(np.float32)
        return (
            torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32)),
            torch.from_numpy(mask),
        )

    def anchor_dates(self) -> list[pd.Timestamp]:
        return [self.data.dates_all[d] for d in self.anchors]
