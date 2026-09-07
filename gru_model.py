"""GRU 编码器（路线2）：把一段 K 线窗口编码为固定长度嵌入向量。

与 KNN（路线1）的对照，本质是「相似度度量」的不同：

- KNN  用 features.feature_vector() 得到 4*WINDOW 维「手工特征向量」，
       再用欧氏距离（固定度量）找最近邻。
- GRU  用 GruEncoder 把 (WINDOW, 4) 的特征序列编码为 hidden_dim 维「学到的嵌入」，
       用余弦距离找最近邻——这个度量随训练目标自动学习。

训练目标：预测窗口之后 FORWARD_DAYS 个交易日的超额收益（相对等权市场基准）。
于是嵌入空间里「距离近」≈「历史上后续走势相近」，这正是相似案例检索想要的
性质——与 KNN 手工度量相比，GRU 是「学出来的、对齐未来收益的相似度」。

设计取舍（见落地建议）：
- 用 GRU 而非 LSTM/Transformer：参数更少、收敛更快、小样本下更不易过拟合。
- 用最后一步隐藏态作为嵌入，而非取全序列池化：信息聚焦于序列末端。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from features import feature_sequences, zscore

FEATURE_KEYS = ("return", "range", "body", "volume")


def feature_matrix(rows: list[dict]) -> np.ndarray:
    """把一段 K 线窗口转为 (WINDOW, 4) 的特征序列。

    与 features.feature_vector 完全同口径（return/range/body/volume，逐列 Z-score），
    只是保留序列形态而非压平成向量，供 GRU 逐时间步消费。
    """
    seq = feature_sequences(rows)
    cols = [zscore(seq[k]) for k in FEATURE_KEYS]
    return np.asarray(cols, dtype=np.float32).T  # (WINDOW, 4)


class GruEncoder(nn.Module):
    """GRU 编码器：序列 -> 嵌入 -> 线性头预测未来超额收益。

    forward 返回 (pred, emb)：
    - pred : (B,)   未来超额收益预测（训练用监督信号）
    - emb  : (B, hidden_dim) 窗口嵌入（检索用，即「学到的形态表示」）
    """

    def __init__(self, input_dim: int = 4, hidden_dim: int = 32,
                 num_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out, _ = self.gru(x)                 # out: (B, T, H)
        last = out[:, -1, :]                 # 最后一步隐藏态 (B, H)
        emb = self.drop(last)                # (B, H)
        pred = self.head(emb).squeeze(-1)    # (B,)
        return pred, emb


def build_model(cfg: dict, device: torch.device) -> GruEncoder:
    """按 config 超参构造 GRU 编码器。"""
    return GruEncoder(
        input_dim=4,
        hidden_dim=int(cfg.get("GRU_HIDDEN", 32)),
        num_layers=int(cfg.get("GRU_LAYERS", 1)),
        dropout=float(cfg.get("GRU_DROPOUT", 0.3)),
    ).to(device)


@torch.no_grad()
def encode(model: GruEncoder, seq_np: np.ndarray, device: torch.device,
           batch_size: int = 2048) -> np.ndarray:
    """把 (n, WINDOW, 4) 的 numpy 序列批量编码为 (n, hidden_dim) 嵌入。"""
    model.eval()
    embs: list[np.ndarray] = []
    n = len(seq_np)
    for i in range(0, n, batch_size):
        x = torch.from_numpy(seq_np[i:i + batch_size]).to(device)
        _, emb = model(x)
        embs.append(emb.detach().cpu().numpy())
    if embs:
        return np.vstack(embs)
    return np.empty((0, model.hidden_dim), dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (m, d) 与 b: (n, d) 的余弦相似度矩阵 -> (m, n)。"""
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a @ b.T
