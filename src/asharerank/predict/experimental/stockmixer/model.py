"""StockMixer（AAAI-24）网络结构复现。

对照原论文与官方开源代码（https://github.com/SJTU-Quant/StockMixer 的
`src/model.py`）重新实现，划分为用户要求的三个核心模块：

  1) IndicatorMixing          —— 指标混合：对指标/特征维度 F 做全连接混合；
  2) MultiScaleTimeMixing      —— 多尺度时间混合：因果掩码 + 多尺度 patch（k∈{1,2,4}）；
  3) MarketAwareStockMixing    —— 市场感知的股票混合：超图式降维(N→m)/升维(m→N)。

最终 `StockMixer` 把三者拼装成与官方实现一致的「时间分支 + 市场分支相加」结构，
并把输入维度从官方的 `[N, T, F]`（单日全市场截面）泛化为 `[B, N, T, F]`
（B 个交易日的截面打包成一个 batch），所有子模块均通过在正确的维度上做
矩阵乘法/卷积来天然支持这层批量维度，不需要显式 for 循环。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 工具：激活函数选择（论文在金融时序上验证较好的 ReLU / HardSwish）
# ============================================================
def get_activation(name: str) -> nn.Module:
    name = (name or "hardswish").lower()
    if name == "relu":
        return nn.ReLU()
    if name in ("hardswish", "hswish"):
        return nn.Hardswish()
    raise ValueError(f"不支持的激活函数: {name}，可选 'relu' / 'hardswish'")


# ============================================================
# 模块 1：因果时间混合层（Causal Time Mixing）
# ============================================================
class CausalTimeMixing(nn.Module):
    """时间维上的因果全连接混合层。

    标准做法 y = Linear(x)（在时间维 T 上）会让输出的第 t 个位置看到
    输入序列中 *所有* 时刻（包括 t 之后）的信息，这对"用历史预测未来"的
    时序任务是一种数据泄露。

    这里显式构造一个下三角（tril）0/1 掩码 `causal_mask`，并在前向时用它
    去遮蔽全连接层的权重矩阵：`causal_mask[t, s] = 1` 当且仅当 `s <= t`。
    这样输出位置 t 的取值只能是输入位置 `0..t` 的线性组合，天然保证了
    "不泄露未来信息"（Causal Mask）。
    """

    def __init__(self, time_steps: int):
        super().__init__()
        self.time_steps = time_steps
        self.weight = nn.Parameter(torch.empty(time_steps, time_steps))
        self.bias = nn.Parameter(torch.zeros(time_steps))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        # 下三角掩码（含对角线）：causal_mask[t, s] = 1 表示允许 t 看到 s（s<=t）
        causal_mask = torch.tril(torch.ones(time_steps, time_steps))
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., time_steps]，在最后一维（时间维）做因果混合
        masked_weight = self.weight * self.causal_mask
        return F.linear(x, masked_weight, self.bias)


# ============================================================
# 模块 1：指标混合（Indicator Mixing）
# ============================================================
class IndicatorMixing(nn.Module):
    """对指标/特征维度 F 做全连接混合的两层 MLP（带残差 + LayerNorm）。

    对应论文里的 channel-mixing：让开盘价、收盘价、成交量、均线等不同量价
    指标之间做非线性组合（例如捕捉"量价背离""放量突破均线"等组合特征）。
    """

    def __init__(self, n_features: int, hidden_dim: int | None = None,
                activation: str = "hardswish", dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or max(n_features * 2, 8)
        self.norm = nn.LayerNorm(n_features)
        self.fc1 = nn.Linear(n_features, hidden_dim)
        self.act = get_activation(activation)
        self.fc2 = nn.Linear(hidden_dim, n_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., F]
        h = self.norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.fc2(h)
        return x + self.dropout(h)  # 残差连接


# ============================================================
# 模块 2：单尺度 Mixer 块（patch + 因果时间混合 + 指标混合）
# ============================================================
class ScaleMixerBlock(nn.Module):
    """尺度因子 k 对应的 Mixer 块。

    1) 若 k>1：用 `Conv1d(kernel_size=k, stride=k)` 沿时间维做不重叠 patch
       提取，把 T 天压缩成 T/k 个"patch"（例如 k=4 相当于把每 4 个交易日
       打包看作一个"周期"，用于捕捉更低频、更长周期的走势）；k=1 时为原始
       日频分辨率，不做压缩。
    2) 对（patch 后的）序列做 LayerNorm → CausalTimeMixing（因果时间混合）
       → 激活 → 残差；
    3) 再做一次 IndicatorMixing（指标混合，内部已含残差）。
    """

    def __init__(self, time_steps: int, n_features: int, scale: int,
                activation: str = "hardswish", dropout: float = 0.0):
        super().__init__()
        if time_steps % scale != 0:
            raise ValueError(f"lookback={time_steps} 必须能被尺度因子 {scale} 整除")
        self.scale = scale
        self.patch = (
            nn.Conv1d(n_features, n_features, kernel_size=scale, stride=scale)
            if scale > 1 else nn.Identity()
        )
        self.patched_len = time_steps // scale
        self.norm = nn.LayerNorm([self.patched_len, n_features])
        self.time_mix = CausalTimeMixing(self.patched_len)
        self.act = get_activation(activation)
        self.indicator_mix = IndicatorMixing(n_features, activation=activation, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [BN, T, F]
        if self.scale > 1:
            xp = x.permute(0, 2, 1)          # [BN, F, T]
            xp = self.patch(xp)              # [BN, F, T/k]
            x = xp.permute(0, 2, 1)          # [BN, T/k, F]

        h = self.norm(x)
        h = h.permute(0, 2, 1)               # [BN, F, T/k]，在时间维上做因果混合
        h = self.time_mix(h)
        h = self.act(h)
        h = h.permute(0, 2, 1)               # [BN, T/k, F]
        x = x + h                            # 残差
        x = self.indicator_mix(x)            # 指标混合（内部已含残差）
        return x                             # [BN, T/k, F]


# ============================================================
# 模块 2：多尺度时间混合（Multi-Scale Time Mixing）
# ============================================================
class MultiScaleTimeMixing(nn.Module):
    """把 k∈{1,2,4} 三个尺度的 ScaleMixerBlock 输出沿时间维拼接。

    k=1 保留日频细节，k=2/4 分别捕捉"两日/四日"级别的中低频趋势，三者拼接
    后同时保留了短期波动与中长期趋势信息，且每一路内部都通过因果掩码
    (CausalTimeMixing) 保证不会用到未来的数据。
    """

    def __init__(self, time_steps: int, n_features: int,
                scales: tuple[int, ...] = (1, 2, 4),
                activation: str = "hardswish", dropout: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            ScaleMixerBlock(time_steps, n_features, k, activation, dropout)
            for k in scales
        ])
        self.out_len = sum(time_steps // k for k in scales)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [BN, T, F] -> [BN, out_len, F]
        outs = [block(x) for block in self.blocks]
        return torch.cat(outs, dim=1)


# ============================================================
# 模块 3：市场感知的股票混合（Market-Aware Stock Mixing）
# ============================================================
class MarketAwareStockMixing(nn.Module):
    """借鉴超图（hypergraph）思想的股票间混合层。

    把 N 只股票的表示看成超图上的 N 个节点，通过一个可学习的线性映射
    H ∈ R^{N×m}（"降维"，相当于把节点聚合到 m 条超边/市场因子上），再用另一
    个映射 H' ∈ R^{m×N}（"升维"，把市场因子广播回每只股票）完成信息交换：

        z = x + Up( Act( Down( LayerNorm(x) ) ) )

    这样每只股票的最终表示都融合了全市场的公共信息（板块联动、大盘风格等），
    同时避免显式构造/存储一张 N×N 的（超）图关系矩阵，参数量只有
    O(N·m)，对沪深300（N≈300）这种规模的股票池非常轻量（m=10）。
    """

    def __init__(self, n_stocks: int, market_hidden: int = 10, activation: str = "hardswish",
                layerscale_init: float = 1.0):
        super().__init__()
        self.norm = nn.LayerNorm(n_stocks)
        self.down = nn.Linear(n_stocks, market_hidden)   # 降维：N -> m（超边聚合）
        self.act = get_activation(activation)
        self.up = nn.Linear(market_hidden, n_stocks)      # 升维：m -> N（广播回各股票）
        # ---- LayerScale：可学习的标量门控（默认 1.0 = 与原始结构行为等价） ----
        # 调小初始值（如 0.1）的假设：PyTorch 默认初始化下 up 层(fan_in=10)的权重幅度
        # ∝ 1/√10，明显大于 down 层(fan_in≈276)的 1/√276，市场分支残差在训练起点的
        # 输出幅度较大且随 seed 波动，压低初始门控或许能减少"多 seed 轨迹分叉"。
        # 该假设需通过单独消融验证（见 config.py market_layerscale_init 注释），
        # 默认不改变原始行为。
        self.gamma = nn.Parameter(torch.full((1,), float(layerscale_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, N]，在最后一维（股票维）上做降维-升维混合
        h = self.norm(x)
        h = self.down(h)
        h = self.act(h)
        h = self.up(h)
        return x + self.gamma * h  # 残差：叠加市场公共因子（幅度由 gamma 门控）


# ============================================================
# 整体网络：StockMixer
# ============================================================
class StockMixer(nn.Module):
    """StockMixer 主干网络。

    输入:  x [B, N, T, F]
           B = batch 内的交易日数；N = 股票数（沪深300）；
           T = lookback（默认16）；F = 每日指标特征维度。
    输出:  pred [B, N]  —— 每只股票的次日收益率预测。

    结构（与官方实现一致的"双分支相加"设计）：
        1. MultiScaleTimeMixing + IndicatorMixing  在 (B*N) 维度上对每只
           股票各自的时序做混合，输出 [B*N, L, F]；
        2. channel_fc: Linear(F->1) 把指标维收口成标量 -> [B, N, L]；
        3. 时间分支：time_head: Linear(L->1) 直接得到时间维的预测；
        4. 市场分支：先做 MarketAwareStockMixing（在 N 维上降维/升维），
           再用 market_head: Linear(L->1) 得到市场感知后的预测；
        5. 两分支相加作为最终输出（对应官方代码 `return y + z`）。
    """

    def __init__(self, n_stocks: int, time_steps: int, n_features: int,
                market_hidden: int = 10, scales: tuple[int, ...] = (1, 2, 4),
                activation: str = "hardswish", dropout: float = 0.1,
                market_layerscale_init: float = 0.1):
        super().__init__()
        self.n_stocks = n_stocks
        self.time_steps = time_steps
        self.n_features = n_features

        self.time_mixer = MultiScaleTimeMixing(time_steps, n_features, scales, activation, dropout)
        self.channel_fc = nn.Linear(n_features, 1)          # 指标混合最终收口：F -> 1
        out_len = self.time_mixer.out_len
        self.time_head = nn.Linear(out_len, 1)               # 时间分支预测头
        self.market_mix = MarketAwareStockMixing(n_stocks, market_hidden, activation,
                                                 layerscale_init=market_layerscale_init)
        self.market_head = nn.Linear(out_len, 1)              # 市场分支预测头

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, t, f = x.shape
        if n != self.n_stocks:
            raise ValueError(f"输入股票数 {n} 与模型初始化的 n_stocks={self.n_stocks} 不一致")
        if t != self.time_steps or f != self.n_features:
            raise ValueError(f"输入形状 (T={t}, F={f}) 与模型 (T={self.time_steps}, F={self.n_features}) 不一致")

        x2 = x.reshape(b * n, t, f)              # [B*N, T, F]
        y = self.time_mixer(x2)                   # [B*N, L, F]
        y = self.channel_fc(y).squeeze(-1)         # [B*N, L]   指标维收口
        y = y.reshape(b, n, -1)                    # [B, N, L]

        y_time = self.time_head(y).squeeze(-1)     # [B, N]  时间分支预测

        y_market_in = y.permute(0, 2, 1)           # [B, L, N]
        z = self.market_mix(y_market_in)           # [B, L, N]  市场感知混合
        z = z.permute(0, 2, 1)                     # [B, N, L]
        y_market = self.market_head(z).squeeze(-1) # [B, N]  市场分支预测

        return y_time + y_market                   # [B, N] 预测的次日收益率
