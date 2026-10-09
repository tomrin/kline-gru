"""监督损失函数库：MSE 之外的排序目标。

与 infonce-compare/losses.py 同源（自包含拷贝，不跨项目引用）。
实验结论（infonce-compare rolling_eval，600只/4切分）：pairwise_rank_loss 在
4/4 切分上 IC 优于 MSE，均值近 3 倍（+0.0680 vs +0.0231），是本项目的 IC 基座损失。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def pairwise_rank_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    margin: float = 0.0,
) -> torch.Tensor:
    """Pairwise ranking loss：直接优化 batch 内的相对排序（IC 的可微代理）。

    对 batch 内所有样本对 (i,j)：若 target_i > target_j，要求 pred_i - pred_j > margin。
        loss = mean( softplus(margin - (pred_i - pred_j) * sign(target_i - target_j)) )

    性质：只关心相对大小，对标签仿射变换不变；不受极端收益值支配梯度；
    batch 即截面，batch >= 256 时排序信号较准。
    """
    if pred.size(0) < 2:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    pd = pred.unsqueeze(1) - pred.unsqueeze(0)      # (B, B) pred_i - pred_j
    yd = target.unsqueeze(1) - target.unsqueeze(0)  # (B, B)
    sy = torch.sign(yd)
    mask = sy != 0  # 剔除收益相等的对（无排序信息）
    if int(mask.sum()) == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    pair_loss = F.softplus(margin - pd * sy)
    return pair_loss[mask].mean()
