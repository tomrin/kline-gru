"""InfoNCE 对比损失与联合损失封装。

InfoNCE 公式：
    L = - (1/|P|) * Σ_(i,j)∈P log[ exp(sim(z_i, z_j) / τ) / Σ_k exp(sim(z_i, z_k) / τ) ]

其中：
- P 是 batch 内的「正样本对」集合
- τ 是温度系数（越小分布越尖锐）
- 同 batch 内的所有非正样本充当负样本（隐式负样本）

正样本构造策略（金融时序友好）：
- 同一只股票、时间相邻的窗口视为「同一形态的不同视角」，互为正样本
- 时间隔太远（如 ≥ 30 交易日）的不算正样本（避免非平稳性造成假正样本）
- 不同股票的所有窗口互为负样本
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def info_nce_loss(
    emb: torch.Tensor,
    stock_ids: torch.Tensor,
    time_offsets: torch.Tensor,
    tau: float = 0.1,
    max_pos_time_diff: int = 5,
    directions: torch.Tensor | None = None,
) -> torch.Tensor:
    """对 batch 内嵌入做 InfoNCE 损失。

    参数
    ----------
    emb : (B, D) L2 归一化后的嵌入（外部已归一化也无所谓，本函数内部会再归一化）
    stock_ids : (B,) 每条样本的股票 id（int，长整型）
    time_offsets : (B,) 每条样本的窗口在时间线上的索引（越大越靠后，int，长整型）
        用于判断「紧邻时间窗」——差值 ≤ max_pos_time_diff 视为正样本
    tau : 温度系数
    max_pos_time_diff : 时间差上限（包含）；差值超过此值不算正样本
    directions : (B,) 每条样本的「涨/跌方向」（int，0/1），传 None 则用「同股票紧邻」策略
        传 0/1 张量时改用「同方向」策略（不限股票、不限时间）——这是 SimCLR 风格横截面对比

    返回
    -------
    标量 loss（对一个 batch 平均）
    """
    B = emb.size(0)
    if B < 2:
        return torch.tensor(0.0, device=emb.device, requires_grad=True)

    # 归一化（即使外部已归一化也安全）
    z = F.normalize(emb, dim=1, eps=1e-9)

    # 相似度矩阵 (B, B)
    sim = z @ z.t() / tau  # (B, B)

    # 正样本掩码
    diag = torch.eye(B, dtype=torch.bool, device=emb.device)
    if directions is not None:
        # SimCLR 风格：同方向为正样本（不限股票、不限时间）
        same_dir = directions.unsqueeze(0) == directions.unsqueeze(1)
        pos_mask = same_dir & ~diag
    else:
        # 时序对比风格：同股票 + 紧邻时间窗
        same_stock = stock_ids.unsqueeze(0) == stock_ids.unsqueeze(1)  # (B, B)
        tdiff = (time_offsets.unsqueeze(0) - time_offsets.unsqueeze(1)).abs()
        pos_mask = same_stock & (tdiff > 0) & (tdiff <= max_pos_time_diff)  # (B, B)
        pos_mask = pos_mask & ~diag

    # 没有任何正样本对时返回 0（不影响梯度）
    if pos_mask.sum() == 0:
        return torch.tensor(0.0, device=emb.device, requires_grad=True)

    # 对每个 anchor i：把它的正样本对的 sim 取均值当作分子 logit 的均值
    # 负样本：同 batch 内所有非正样本（包括不同股票 + 同股票但时间差过大的）
    # 用 logsumexp 形式：-log( mean_pos_exp / all_exp )
    exp_sim = torch.exp(sim - sim.max(dim=1, keepdim=True).values.detach())  # 数值稳定

    # 分子：每个 i 的正样本的 exp(sim) 均值
    # pos_mask[i] 是 (B,)，其和 = i 的正样本数
    pos_exp = exp_sim * pos_mask.float()
    pos_count = pos_mask.sum(dim=1).clamp(min=1).float()  # (B,) 防 0
    pos_mean = pos_exp.sum(dim=1) / pos_count  # (B,)

    # 分母：所有样本（i 的对 j 的所有）的 exp(sim) 和
    # 用 mask 屏蔽对角线自身（不作为负样本/正样本）
    neg_mask = ~diag  # 自身不算
    denom = (exp_sim * neg_mask.float()).sum(dim=1)  # (B,)

    # 损失 = -log(pos_mean / denom)
    loss = -(torch.log(pos_mean + 1e-12) - torch.log(denom + 1e-12))
    # 只对有正样本的 anchor 求平均
    has_pos = pos_mask.sum(dim=1) > 0
    if has_pos.sum() == 0:
        return torch.tensor(0.0, device=emb.device, requires_grad=True)
    return loss[has_pos].mean()


def joint_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    emb: torch.Tensor,
    stock_ids: torch.Tensor,
    time_offsets: torch.Tensor,
    mse_weight: float = 0.5,
    infonce_weight: float = 0.5,
    tau: float = 0.1,
    max_pos_time_diff: int = 5,
    directions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """MSE + InfoNCE 联合损失（baseline 用 mse_weight=1, infonce_weight=0 即可退化为纯 MSE）。"""
    mse = F.mse_loss(pred, target)
    nce = info_nce_loss(emb, stock_ids, time_offsets, tau=tau,
                        max_pos_time_diff=max_pos_time_diff, directions=directions)
    total = mse_weight * mse + infonce_weight * nce
    return total, {"mse": float(mse.item()), "infonce": float(nce.item()), "total": float(total.item())}
