"""kline-gru 内 GRU 两个变体的对比评估。

读入：
  models/gru_baseline_X.npy    — 纯 MSE 训练的嵌入
  models/gru_X.npy             — 当前 gru.pt（InfoNCE 联合训练）的嵌入
  models/gru_meta.json         — 两个变体共用的窗口元信息
  models/ends.npy              — 各窗口结束日期（int YYYYMMDD）

输出：
  models/compare_gru.json      — 关键指标
  output/gru_compare.png       — 对比图（同股票紧邻 vs 跨股票相似度）
  output/gru_compare_report.md — 简短报告
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import config

MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def cosine_sim_matrix(a, b):
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a @ b.T


def gap_quality(emb: np.ndarray, stock: np.ndarray, tidx: np.ndarray) -> dict:
    """同股票紧邻窗相似度 vs 跨股票随机对相似度（infonce-compare 同一口径）。"""
    sims_pos = []; sims_neg = []
    rng = np.random.default_rng(0)
    n = len(emb)
    for s in np.unique(stock):
        mask = stock == s
        e = emb[mask]; t = tidx[mask]
        if len(e) < 2: continue
        order = np.argsort(t); es = e[order]
        cs = cosine_sim_matrix(es[:-1], es[1:])
        sims_pos.extend(np.diag(cs).tolist())
    for _ in range(5000):
        i, j = rng.integers(0, n, 2)
        if stock[i] == stock[j]: continue
        ea = emb[i] / (np.linalg.norm(emb[i]) + 1e-9)
        eb = emb[j] / (np.linalg.norm(emb[j]) + 1e-9)
        sims_neg.append(float(ea @ eb))
    return {
        "n_pairs_pos": len(sims_pos),
        "n_pairs_neg": len(sims_neg),
        "pos_mean": float(np.mean(sims_pos)) if sims_pos else 0.0,
        "pos_std":  float(np.std(sims_pos)) if sims_pos else 0.0,
        "neg_mean": float(np.mean(sims_neg)) if sims_neg else 0.0,
        "neg_std":  float(np.std(sims_neg)) if sims_neg else 0.0,
        "gap": (float(np.mean(sims_pos)) - float(np.mean(sims_neg))) if (sims_pos and sims_neg) else 0.0,
    }


def direction_gap(emb: np.ndarray, direction: np.ndarray) -> dict:
    """InfoNCE v4 训练目标对应的口径：同方向（涨/跌）vs 异方向相似度。"""
    sims_pos = []; sims_neg = []
    rng = np.random.default_rng(0)
    n = len(emb)
    for _ in range(5000):
        i, j = rng.integers(0, n, 2)
        if i == j: continue
        ea = emb[i] / (np.linalg.norm(emb[i]) + 1e-9)
        eb = emb[j] / (np.linalg.norm(emb[j]) + 1e-9)
        s = float(ea @ eb)
        if direction[i] == direction[j]:
            sims_pos.append(s)
        else:
            sims_neg.append(s)
    return {
        "n_pairs_pos": len(sims_pos),
        "n_pairs_neg": len(sims_neg),
        "pos_mean": float(np.mean(sims_pos)) if sims_pos else 0.0,
        "neg_mean": float(np.mean(sims_neg)) if sims_neg else 0.0,
        "gap": (float(np.mean(sims_pos)) - float(np.mean(sims_neg))) if (sims_pos and sims_neg) else 0.0,
    }


def main():
    base_emb = np.load(MODEL_DIR / "gru_baseline_X.npy")
    cur_emb = np.load(MODEL_DIR / "gru_X.npy")
    metas = json.load(open(MODEL_DIR / "gru_meta.json", encoding="utf-8"))
    ends = np.load(MODEL_DIR / "ends.npy")
    print(f"[load] baseline_emb {base_emb.shape}, current_emb {cur_emb.shape}, metas {len(metas)}")
    assert base_emb.shape == cur_emb.shape, "两个嵌入 shape 不一致"

    # 构造 stock / tidx / direction
    stock_id_map: dict[str, int] = {}
    stock_counter: dict[str, int] = {}
    stock = np.zeros(len(metas), dtype=np.int64)
    tidx = np.zeros(len(metas), dtype=np.int64)
    direction = np.zeros(len(metas), dtype=np.int64)
    for i, m in enumerate(metas):
        code = m["code"]
        if code not in stock_id_map:
            stock_id_map[code] = len(stock_id_map)
            stock_counter[code] = 0
        stock[i] = stock_id_map[code]
        tidx[i] = stock_counter[code]
        stock_counter[code] += 1
        exc = m.get("excess")
        if exc is not None:
            direction[i] = 1 if exc > 0 else 0
    print(f"[load] {len(stock_id_map)} unique stocks, {direction.sum()} up / {(1-direction).sum()} down")

    # gap（同股票紧邻）
    b_gap = gap_quality(base_emb, stock, tidx)
    c_gap = gap_quality(cur_emb, stock, tidx)
    # direction gap（涨/跌同向）
    b_dir = direction_gap(base_emb, direction)
    c_dir = direction_gap(cur_emb, direction)

    print(f"\n=== 同股票紧邻窗 gap ===")
    print(f"  baseline: pos={b_gap['pos_mean']:+.4f}  neg={b_gap['neg_mean']:+.4f}  gap={b_gap['gap']:+.4f}")
    print(f"  InfoNCE : pos={c_gap['pos_mean']:+.4f}  neg={c_gap['neg_mean']:+.4f}  gap={c_gap['gap']:+.4f}")
    print(f"\n=== 涨/跌同向 gap ===")
    print(f"  baseline: same_dir={b_dir['pos_mean']:+.4f}  diff_dir={b_dir['neg_mean']:+.4f}  gap={b_dir['gap']:+.4f}")
    print(f"  InfoNCE : same_dir={c_dir['pos_mean']:+.4f}  diff_dir={c_dir['neg_mean']:+.4f}  gap={c_dir['gap']:+.4f}")

    # 保存
    summary = {
        "baseline": {"adjacent_gap": b_gap, "direction_gap": b_dir},
        "infonce":  {"adjacent_gap": c_gap, "direction_gap": c_dir},
    }
    with open(MODEL_DIR / "compare_gru.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 画图
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    labels = ["baseline\n(纯 MSE)", "InfoNCE v4\n(MSE + 涨/跌同向 InfoNCE)"]
    pos_v = [b_gap["pos_mean"], c_gap["pos_mean"]]
    neg_v = [b_gap["neg_mean"], c_gap["neg_mean"]]
    x = np.arange(2); w = 0.35
    axes[0].bar(x - w/2, pos_v, w, label="同股票紧邻", color="#2ecc71")
    axes[0].bar(x + w/2, neg_v, w, label="跨股票随机对", color="#e74c3c")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels)
    axes[0].set_title("GRU 嵌入：同股票紧邻窗 vs 跨股票")
    axes[0].set_ylabel("余弦相似度")
    axes[0].legend(); axes[0].grid(axis="y", alpha=0.3)
    for i, (pv, nv) in enumerate(zip(pos_v, neg_v)):
        gap = pv - nv
        axes[0].annotate(f"gap={gap:+.4f}", (i, max(pv, nv)),
                         textcoords="offset points", xytext=(0, 8), ha="center", fontsize=10)
    pos_v2 = [b_dir["pos_mean"], c_dir["pos_mean"]]
    neg_v2 = [b_dir["neg_mean"], c_dir["neg_mean"]]
    axes[1].bar(x - w/2, pos_v2, w, label="同方向", color="#3498db")
    axes[1].bar(x + w/2, neg_v2, w, label="异方向", color="#f39c12")
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels)
    axes[1].set_title("GRU 嵌入：涨/跌同向 vs 异向")
    axes[1].set_ylabel("余弦相似度")
    axes[1].legend(); axes[1].grid(axis="y", alpha=0.3)
    for i, (pv, nv) in enumerate(zip(pos_v2, neg_v2)):
        gap = pv - nv
        axes[1].annotate(f"gap={gap:+.4f}", (i, max(pv, nv)),
                         textcoords="offset points", xytext=(0, 8), ha="center", fontsize=10)
    fig.tight_layout()
    out_png = OUTPUT_DIR / "gru_compare.png"
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"\n[save] {MODEL_DIR / 'compare_gru.json'}")
    print(f"[save] {out_png}")


if __name__ == "__main__":
    main()
