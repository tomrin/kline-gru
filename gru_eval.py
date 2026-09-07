"""GRU vs KNN 同口径 walk-forward 评估（路线2 vs 路线1 的诚实对比）。

思路：在时间上「held-out」的一段窗口（模型训练从未见过）里采样查询，对每个查询
分别用两种检索器找最相似历史先例，取「先例后续超额收益均值」作为预测，与实际
超额收益对比，输出秩相关 IC + 方向命中率。

关键：两种检索器使用 gru_train.py 保存的同一窗口集（gru_X.npy 与 knn_X.npy 一一
对应），且都遵守「只用查询窗口开始日之前的先例」约束，避免未来函数——这才是
两条路线能公平对比的前提。

用法：
    python gru_eval.py --n-queries 1000 --top 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from gru_model import cosine_similarity

ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = ROOT / "models"


def spearman(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def load_all():
    gru_X = np.load(MODEL_DIR / "gru_X.npy")
    knn_X = np.load(MODEL_DIR / "knn_X.npy")
    ends = np.load(MODEL_DIR / "ends.npy")
    with open(MODEL_DIR / "gru_meta.json", encoding="utf-8") as f:
        metas = json.load(f)
    return gru_X, knn_X, ends, metas


def neighbor_mean_excess(order, metas, ends, start_q, k, exclude_q):
    """沿 order（距离升序）取前 k 个「end < start_q 且非自身」的先例，返回其超额收益均值。"""
    vals = []
    for idx in order:
        if idx == exclude_q:
            continue
        if int(metas[idx]["end"]) >= start_q:
            continue
        exc = metas[idx]["excess"]
        if exc is not None:
            vals.append(exc)
        if len(vals) >= k:
            break
    return float(np.mean(vals)) if vals else None


def main() -> int:
    parser = argparse.ArgumentParser(description="GRU vs KNN walk-forward 评估")
    parser.add_argument("--n-queries", type=int, default=1000, help="采样查询窗口数")
    parser.add_argument("--top", type=int, default=5, help="先例个数 K")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    gru_X, knn_X, ends, metas = load_all()
    n = len(metas)
    print(f"[eval] 载入索引 {n} 窗口 | 嵌入维度 {gru_X.shape[1]} | 特征维度 {knn_X.shape[1]}")

    # held-out：时间上最后 test_frac 一段（与 gru_train.py 的 test 切片一致）
    train_frac = float(getattr(config, "GRU_TRAIN_FRAC", 0.7))
    val_frac = float(getattr(config, "GRU_VAL_FRAC", 0.15))
    test_start = int(n * (train_frac + val_frac))
    pool = np.arange(test_start, n)
    rng = np.random.default_rng(args.seed)
    qs = rng.choice(pool, size=min(args.n_queries, len(pool)), replace=False)
    print(f"[eval] held-out 范围 [{test_start}, {n})，采样 {len(qs)} 个查询，先例 K={args.top}")

    # 预归一化嵌入，加速余弦计算
    gru_n = gru_X / (np.linalg.norm(gru_X, axis=1, keepdims=True) + 1e-9)

    preds = {"gru": [], "knn": []}
    actuals = []
    t0 = time.perf_counter()
    for qi, q in enumerate(qs):
        start_q = int(metas[q]["start"])
        end_q = int(metas[q]["end"])
        actual = metas[q]["excess"]
        if actual is None:
            continue
        # 先例候选：end < start_q（严格早于目标窗口开始日）
        pos = int(np.searchsorted(ends, start_q, side="left"))
        if pos < args.top:
            continue

        # GRU：余弦距离
        sim = gru_n[:pos] @ gru_n[q]
        order_gru = np.argsort(-sim)
        p_gru = neighbor_mean_excess(order_gru, metas, ends, start_q, args.top, q)

        # KNN：欧氏距离
        diff = knn_X[:pos] - knn_X[q]
        dist = np.einsum("ij,ij->i", diff, diff)  # 平方欧氏，等价排序
        order_knn = np.argsort(dist)
        p_knn = neighbor_mean_excess(order_knn, metas, ends, start_q, args.top, q)

        if p_gru is not None and p_knn is not None:
            preds["gru"].append(p_gru)
            preds["knn"].append(p_knn)
            actuals.append(actual)

        if (qi + 1) % 200 == 0:
            print(f"[eval] 已处理 {qi + 1}/{len(qs)}，用时 {time.perf_counter() - t0:.1f}s", flush=True)

    actual = np.asarray(actuals, dtype=np.float64)
    print("\n" + "=" * 64)
    print(f"GRU vs KNN 相似案例检索预测能力（held-out {len(actual)} 个查询）")
    print("=" * 64)
    print(f"{'方法':<8} {'秩相关IC':>10} {'方向命中率':>10} {'预测MAE':>10}")
    for name in ("gru", "knn"):
        p = np.asarray(preds[name], dtype=np.float64)
        ic = spearman(p, actual)
        hit = float(np.mean((p > 0) == (actual > 0)))
        mae = float(np.mean(np.abs(p - actual)))
        print(f"{name.upper():<8} {ic:>+10.4f} {hit*100:>9.1f}% {mae:>10.5f}")

    print("\n解读：IC 越接近 +1 预测越有效，接近 0 说明形态对短期收益预测力有限（日线常态）。")
    print("两条路线差异通常都很小——这本身就是结论：不要对深度学习抱过高期望。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
