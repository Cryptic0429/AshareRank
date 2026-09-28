"""组合损失函数：MSE（回归）+ λ · Pairwise 排序损失（Rank Loss）。

与官方实现（`src/model.py::get_loss`）思路一致，重新推导注释如下：

对同一交易日内任意两只股票 i, j：
    pred_diff[i,j] = pred_i - pred_j
    gt_diff[i,j]   = gt_i   - gt_j
若模型排序与真实排序一致（即 pred_i > pred_j 当且仅当 gt_i > gt_j），则
pred_diff 与 gt_diff 同号，乘积为正，不产生惩罚；一旦排序颠倒（异号），
乘积为负，我们用 `relu(-pred_diff*gt_diff)` 对"颠倒的幅度"进行惩罚，这正是
标准的 pairwise ranking hinge loss，用于让模型学到「谁涨得多/跌得多」的
相对排序，这对于选股/多空组合策略往往比绝对数值的 MSE 更重要。

---- 2026-07-23 迭代（P0 优化）----
首轮训练发现：MSE 分支很容易通过"预测接近当日全市场平均收益"来降低数值误差
（因为个股次日收益里方差最大的部分往往是大盘/板块的共同波动），但这样做对
选股排序毫无帮助；同时 λ=0.1 时排序损失分量占比太小，不足以纠正这个偏好。
于是加入 `cross_sectional_norm`：在计算 MSE 之前，先对 pred / target 做
"逐日横截面标准化"（减去当日均值、除以当日标准差），把"今天大盘涨了多少"这个
共同项从损失里剥离出去，让 MSE 也去关注个股相对强弱，而不仅仅是排序损失。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _cross_sectional_standardize(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """逐日横截面标准化：对 batch 内每一天，只用有效样本（mask=1）计算
    当日均值/标准差，再做 z-score。这样可以把"当天大盘整体涨跌"这个所有股票
    共享的公共项从数值里剥离出去，只保留个股相对强弱的部分。

    x, mask: [B, N] -> 返回同形状的标准化结果。
    """
    mask_f = mask.float()
    n = mask_f.sum(dim=-1, keepdim=True).clamp_min(1.0)
    mean = (x * mask_f).sum(dim=-1, keepdim=True) / n
    var = ((x - mean) ** 2 * mask_f).sum(dim=-1, keepdim=True) / n
    std = var.clamp_min(1e-8).sqrt()
    return (x - mean) / std


def combined_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                  rank_weight: float = 0.1,
                  cross_sectional_norm: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """计算组合损失。

    Args:
        pred, target, mask: 形状均为 [B, N]。mask 中 1 表示该 (日, 股票) 样本
            有效（无停牌/无缺失），0 表示应忽略（停牌/缺失/新股未上市）。
        rank_weight: 排序损失权重 λ（论文默认 0.1；A股小样本实测建议调高到 0.5~1.0）。
        cross_sectional_norm: 是否在计算 MSE 前做逐日横截面标准化（P0 优化项，默认开启）。
            注意：Pairwise 排序损失的惩罚符号只取决于 pred_diff/gt_diff 的符号，
            对每天独立做的正标准化（除以正数 std）不会改变符号，因此排序损失在
            标准化前后是等价的，这里统一在标准化后的值上计算两个分量，代码更简洁。

    Returns:
        (total_loss, mse_loss, rank_loss) 均为标量 tensor。
    """
    mask_f = mask.float()
    n_valid = mask_f.sum().clamp_min(1.0)

    if cross_sectional_norm:
        pred_c = _cross_sectional_standardize(pred, mask)
        target_c = _cross_sectional_standardize(target, mask)
    else:
        pred_c, target_c = pred, target

    # ---- 1) MSE 回归损失（仅在有效样本上计算）----
    mse = ((pred_c - target_c) ** 2 * mask_f).sum() / n_valid

    # ---- 2) Pairwise 排序损失 ----
    pred_diff = pred_c.unsqueeze(-1) - pred_c.unsqueeze(-2)      # [B, N, N]：pred_i - pred_j
    gt_diff = target_c.unsqueeze(-1) - target_c.unsqueeze(-2)     # [B, N, N]：gt_i - gt_j
    pair_mask = mask_f.unsqueeze(-1) * mask_f.unsqueeze(-2)        # 两只股票都有效才计入该 pair
    n_pairs = pair_mask.sum().clamp_min(1.0)

    rank_penalty = F.relu(-(pred_diff * gt_diff)) * pair_mask
    rank_loss = rank_penalty.sum() / n_pairs

    total = mse + rank_weight * rank_loss
    return total, mse.detach(), rank_loss.detach()
