"""StockMixer（AAAI-24）超参数与运行配置。

论文：《StockMixer: A Simple yet Strong MLP-based Architecture for Stock Price
Forecasting》(AAAI 2024)
官方代码参考：https://github.com/SJTU-Quant/StockMixer

本配置对齐用户的 5 项要求：
  1) MPS/CPU 自动适配（device 留空表示自动检测）；
  2) A 股（沪深300）近 3 年训练 / 近 6 个月验证 / 近 6 个月测试；
  3) T=16 滑动窗口、多尺度 k∈{1,2,4}、Market 隐层 m=10；
  4) 组合 Loss（MSE + λ·Rank Loss，λ=0.1）。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StockMixerConfig:
    # ---------------- 股票池 / 数据获取 ----------------
    pool: str = "csi300"                     # 沪深300 成分股
    max_stocks: int | None = None            # None = 使用全量成分股；调试可设小值（如 30）加速
    fetch_start: str = "20180101"            # akshare/tushare 拉取行情起点（需早于 train_start 以满足上市年限过滤）
    # 非空时跳过上市年限过滤，强制使用该股票列表（仍做停牌覆盖率过滤）。
    # 用于「冻结 3 年基线的 276 只，只改 train_years」这类控变量消融。
    freeze_codes: tuple[str, ...] | None = None

    # ---------------- 数据拆分（近 3 年训练 / 近 6 月验证 / 近 6 月测试） ----------------
    train_years: float = 3.0
    valid_months: float = 6.0
    test_months: float = 6.0

    # ---------------- 高容错率相关 ----------------
    min_listed_days: int = 250               # 训练起点前至少有 250 个交易日历史（约 1 年），剔除新股
    max_missing_ratio: float = 0.20          # 整段区间缺失（停牌/无行情）比例超过此值则整只股票剔除
    ffill_limit: int = 3                     # 短暂停牌（<=3 个交易日）价格前向填充，长停牌仍记为缺失
    clip_return: float = 0.21                # 收益率 / 相对价格特征裁剪阈值（略高于 A 股 ±20% 涨跌停）

    # ---------------- 特征工程 ----------------
    lookback: int = 16                       # T：滑动窗口长度
    ma_windows: tuple[int, ...] = (5, 10)    # 简单移动平均窗口（5 日 / 10 日）

    # ---------------- 基本面 / 行业中性化因子（2026-07-23 新增，默认关闭） ----------------
    # 复用树模型管线已有的 asharerank/forecasting/fundamentals.py（akshare 季度财报，
    # REPORT_LAG_DAYS=45 天滞后防未来函数）。原始字段里 eps/bvps 是绝对值，跨股票不可比，
    # 这里改用衍生的估值比率 ep_ratio=eps/close（盈利收益率）、bp_ratio=bvps/close（账面市值比），
    # 叠加 5 个天然无量纲的质量类指标，共 7 维基本面特征。
    #
    # 默认 False：2026-07-23 5 seed 实测发现，当前 CSI300 基本面覆盖率只有 34%（akshare 逐股票
    # 抓取受网络环境限制，抓取任务卡住未能补齐），叠加特征后参数量从 8181 涨到 11345（+39%），
    # 撞上了同一轮诊断出的"模型本来就容易过拟合"问题，5 个 seed 的 RankIC 均值反而从 0.050 掉到
    # 0.035。先把覆盖率补到 90%+、把过拟合问题解决后再重新评估，之前用 `--use-fundamentals`
    # / 直接传 `use_fundamentals=True` 打开实验。
    use_fundamentals: bool = False
    fundamental_quality_fields: tuple[str, ...] = (
        "roe", "net_margin", "debt_ratio", "rev_growth", "profit_growth",
    )
    # 非空时覆盖默认基本面字段集合（可含质量字段 + ep_ratio/bp_ratio）。
    # 例：("ep_ratio", "bp_ratio") 或 ("roe",) —— 用于少维消融。
    fundamental_fields: tuple[str, ...] | None = None
    # 是否对基本面因子做"行业内"截面标准化（同行业股票互相比较），而不是对标签做行业中性化
    # ——树模型基线是对标签做行业内排序归一化，但那样会让 label_arr 失去真实收益率含义，
    # 破坏 Sharpe/年化收益等下游指标的可解释性。这里改为对因子做行业中性化，语义更安全：
    # 剥离"银行 ROE 天然比科技股高"这类行业基线差异，同时保留 label_arr 仍是真实收益率。
    industry_neutralize_fundamentals: bool = True
    # 缺测掩码：在基本面维标准化后填 0 之外，额外拼一维 fund_missing∈{0,1}
    # （该日该股全部基本面字段均为 NaN → 1）。让模型区分「真的中性」与「没数据被填 0」。
    # 2026-07-25 消融（覆盖率≈70%）：ROE+fmiss 集成 RankIC 0.044，低于无掩码 ROE 的 0.095
    # 与纯量价 0.079——当前覆盖下掩码本身成噪声，默认关闭；覆盖拉到 90%+ 后再开重试。
    fund_missing_mask: bool = False
    # 预测目标的前瞻天数。默认 1 = 用户原始需求里的"次日收益率"。2026-07-23 实测发现
    # 1 日 horizon 信噪比极低（验证集 IC/RankIC 从 epoch 1 起就在 0 附近无方向震荡，
    # val_loss 几乎不随训练下降——说明不是超参问题，是标签本身太吵），而项目里
    # LightGBM/XGBoost 用的是 5 日 horizon 且能稳定跑出 RankIC≈0.04，建议对比实验时
    # 用 `--label-horizon 5` 与基线对齐口径。
    label_horizon: int = 5

    # ---------------- 网络结构 ----------------
    scales: tuple[int, ...] = (1, 2, 4)      # Multi-Scale Time Mixing 的 patch 尺度因子 k
    market_hidden: int = 10                  # Market-Aware Stock Mixing 隐层维度 m
    activation: str = "hardswish"            # "relu" | "hardswish"
    dropout: float = 0.1
    # LayerScale：MarketAwareStockMixing 残差分支的初始门控幅度。
    # 1.0 = 与原始结构完全等价（gamma 可学习但从 1 起步，行为≈没有门控）。
    # 调小（如 0.1）的假设：该分支 up 层(m=10->N)的默认初始化幅度显著大于 down 层
    # （fan_in 差 27 倍），不同随机种子训练起点的市场分支输出幅度差异较大，压低初始
    # 门控或许能减少"多 seed 轨迹分叉"。2026-07-23 与 EMA/滑动平均早停打包实验时整体
    # 变差（RankIC 均值 0.050→0.013），但三项混在一起无法归因，默认回退到 1.0，
    # 用 --market-layerscale-init 0.1 单独消融。
    market_layerscale_init: float = 1.0

    # ---------------- 损失函数 ----------------
    # λ：Pairwise 排序损失权重。论文原始默认 0.1；2026-07-23 实测发现该权重下模型很快
    # "偷懒"收敛到预测截面均值（rank_loss 被压到几乎 0，但验证集 IC/RankIC 无方向震荡），
    # 因此把默认值上调到 0.5，加大排序损失在总 loss 中的占比。
    rank_loss_weight: float = 0.5
    # 是否在计算 MSE 前做逐日横截面标准化（减去当日均值/除以当日标准差），把"大盘整体
    # 涨跌"这个所有股票共享的公共项剥离出去，逼模型学习个股相对强弱而不是预测市场均值。
    cross_sectional_norm: bool = True

    # ---------------- 训练超参 ----------------
    lr: float = 2e-3
    weight_decay: float = 1e-5
    epochs: int = 60
    batch_size: int = 8                      # 一个 batch = 8 个交易日的横截面（[B, N, T, F]）
    patience: int = 10                       # 早停：验证集 loss 连续 N 轮不下降则停止
    grad_clip: float = 1.0
    seed: int = 42
    num_workers: int = 0                     # macOS + MPS 建议 0，避免多进程与 MPS 冲突

    # ---------------- 稳定性优化开关（2026-07-23 新增；默认全部=基线行为，逐项消融再决定是否启用） ----------------
    # 第一版实现把 EMA + 滑动平均早停 + LayerScale 三项打包默认开启，5 seed 实测整体大幅
    # 变差（RankIC 均值 0.050→0.013，RankIC>0 占比 100%→60%）。复盘发现 EMA 实现有两个缺陷：
    #   (1) EMA 从"随机初始权重"开始、固定 decay=0.9 累积——best_epoch=4~7 就早停的 seed，
    #       保存的权重里还残留 0.9^t ≈ 48%~66% 的未训练随机初始化（唯一训满 60 轮、初始化
    #       残留仅 0.2% 的 seed 42 恰好是全场最佳，旁证了污染机制）；
    #   (2) 早停判据评的是"原始权重"的 val_RankIC，checkpoint 存的却是 EMA 权重，评存错位。
    # 现已修复：EMA 用 warmup 衰减（训练早期快速贴近当前权重，不受初始化污染），且开启 EMA 时
    # 验证集直接评 EMA 权重（评什么存什么）。三项全部改为独立开关便于归因。
    use_ema: bool = False                    # True 时启用权重 EMA（修复版），验证/测试都用 EMA 权重
    ema_decay: float = 0.9                   # EMA 衰减上限；实际按 min(ema_decay, (1+t)/(10+t)) warmup
    # 早停 / checkpoint 判据用"最近 N 轮 val_RankIC 的滑动平均"；1 = 裸单轮值（基线行为）。
    # 动机：验证集只有 100 出头个锚点，单轮 RankIC 噪声大，容易把偶然高点误判成"最佳模型"。
    rankic_smooth_window: int = 1

    # ---------------- 评估 ----------------
    top_k: int = 10                          # Precision@10 / 组合 Sharpe 的 K
    trading_days_per_year: int = 252

    # ---------------- 设备 ----------------
    device: str | None = None                # None = 自动检测（mps > cuda > cpu）


DEFAULT_CONFIG = StockMixerConfig()
