"""BYOL（Bootstrap Your Own Latent）自监督模块 —— 无负样本对比学习，天然防坍缩。

与 InfoNCE 的对照：
- InfoNCE 依赖 batch 内负样本把不同样本「推开」；负样本太弱时模型把所有东西
  映射到同一点（坍缩）。
- BYOL 不依赖任何负样本：online 网络用 predictor 去「预测」target 网络的投影，
  通过 stop-gradient + 指数滑动平均（EMA）保持表示稳定，天生防坍缩。

结构：
    online:  encoder -> projector -> predictor
    target:  encoder -> projector（EMA 更新，stop-grad）

损失（对称双向，对归一化投影做 MSE）：
    L = 0.5 * [ ||norm(p1) - norm(z2)||^2 + ||norm(p2) - norm(z1)||^2 ]
    其中 p = online 的预测输出，z = target 的投影输出（detach）

K 线序列增强（特征已 z-score，0 = 中性）：
    - 时间 mask：随机置零 ~mask_prob 比例的时间步
    - 高斯噪声
    - 幅度缩放
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """投影/预测头：Linear -> BatchNorm1d -> ReLU -> Linear。"""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BYOL(nn.Module):
    """BYOL 包装器：持有一个在线编码器（含回归 head）+ 一个 EMA 目标编码器。

    用法：
        byol = BYOL(encoder, hidden_dim, proj_dim, ema_tau).to(device)
        pred, emb = byol.encode(x)            # 监督预测 + 报告用嵌入
        p1 = byol.online_proj_pred(x1)        # 视图1的在线预测输出
        p2 = byol.online_proj_pred(x2)        # 视图2的在线预测输出
        z1 = byol.target_proj(x1)             # 视图1的目标投影（detach）
        z2 = byol.target_proj(x2)             # 视图2的目标投影（detach）
        byol.update_target()                  # 每个训练 step 后调用
    """

    def __init__(self, encoder: nn.Module, hidden_dim: int = 32,
                 proj_dim: int = 32, ema_tau: float = 0.996):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ema_tau = ema_tau
        self.encoder = encoder                       # GruEncoder，forward -> (pred, emb)
        self.projector = MLP(hidden_dim, hidden_dim, proj_dim)
        self.predictor = MLP(proj_dim, hidden_dim, proj_dim)

        self.target_encoder = copy.deepcopy(encoder)
        self.target_projector = MLP(hidden_dim, hidden_dim, proj_dim)

        # 冻结 target（stop-gradient）
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        for p in self.target_projector.parameters():
            p.requires_grad_(False)

        # 初始化 target = online
        self._ema_update(tau=0.0)

    @torch.no_grad()
    def _ema_update(self, tau: float | None = None):
        tau = self.ema_tau if tau is None else tau
        for o, t in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            t.data.mul_(tau).add_(o.data, alpha=1.0 - tau)
        for o, t in zip(self.projector.parameters(), self.target_projector.parameters()):
            t.data.mul_(tau).add_(o.data, alpha=1.0 - tau)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """在线编码器：返回 (pred, emb)。用于监督损失与报告嵌入。"""
        return self.encoder(x)

    def online_proj_pred(self, x: torch.Tensor) -> torch.Tensor:
        """在线分支的预测输出 p（未归一化，供 byol_loss 内部归一化）。"""
        _, emb = self.encoder(x)
        z = self.projector(emb)
        return self.predictor(z)

    @torch.no_grad()
    def target_proj(self, x: torch.Tensor) -> torch.Tensor:
        """目标分支的投影输出 z（detach）。"""
        _, emb = self.target_encoder(x)
        return self.target_projector(emb)

    def update_target(self):
        self._ema_update()


def augment(x: torch.Tensor, mask_prob: float = 0.2, noise_std: float = 0.1,
            scale_range: float = 0.1) -> torch.Tensor:
    """K 线序列增强：(B, T, F) -> (B, T, F)。每次调用随机，天然产生不同视图。"""
    x = x.clone()
    # 时间 mask：随机置零部分时间步（特征已 z-score，0 = 中性）
    keep = (torch.rand(x.size(0), x.size(1), 1, device=x.device) > mask_prob).float()
    x = x * keep
    # 高斯噪声
    x = x + torch.randn_like(x) * noise_std
    # 幅度缩放
    scale = 1.0 + (torch.rand(x.size(0), 1, 1, device=x.device) - 0.5) * 2 * scale_range
    x = x * scale
    return x


def byol_loss(p1: torch.Tensor, p2: torch.Tensor, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """对称归一化 MSE：L = 0.5 * [ MSE(norm(p1), norm(z2)) + MSE(norm(p2), norm(z1)) ]。

    所有输入应是 (B, proj_dim) 的未归一化向量；内部做 L2 归一化。
    """
    p1 = F.normalize(p1, dim=1, eps=1e-9)
    p2 = F.normalize(p2, dim=1, eps=1e-9)
    z1 = F.normalize(z1, dim=1, eps=1e-9)
    z2 = F.normalize(z2, dim=1, eps=1e-9)
    return 0.5 * (F.mse_loss(p1, z2) + F.mse_loss(p2, z1))