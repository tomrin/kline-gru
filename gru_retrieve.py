"""GRU 相似历史案例检索（路线2）：用学习到的嵌入做相似检索。

与 retrieve.py（KNN，路线1）同层级——都是「证据层相似案例检索」，区别只在
相似度度量：

- KNN：手工特征向量的欧氏距离（固定度量）
- GRU：GRU 编码器嵌入的余弦相似度（随训练目标——预测未来超额收益——学到）

用法：
    python gru_train.py                            # 先训练 + 建索引（一次性）
    python gru_retrieve.py 600519 --end-date 20260801 --top 5
    python gru_retrieve.py 600519 --end-date 20260801 --top 5 --before
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import cluster as cl
import chart
from gru_model import feature_matrix, cosine_similarity
from retrieve import parse_date, load_case_rows

ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = ROOT / "models" / "gru.pt"
GRU_X_PATH = ROOT / "models" / "gru_X.npy"
GRU_META_PATH = ROOT / "models" / "gru_meta.json"


def load_model():
    """加载 GRU 模型（返回可用于编码的 model 对象，cpu）。"""
    import torch
    from gru_model import GruEncoder
    if not MODEL_PATH.exists():
        raise FileNotFoundError("未找到 GRU 模型，请先运行 python gru_train.py")
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    model = GruEncoder(
        input_dim=int(ckpt.get("input_dim", 4)),
        hidden_dim=int(ckpt.get("hidden_dim", 32)),
        num_layers=int(ckpt.get("num_layers", 1)),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def load_index():
    if not GRU_X_PATH.exists() or not GRU_META_PATH.exists():
        raise FileNotFoundError("未找到 GRU 检索库，请先运行 python gru_train.py")
    X = np.load(GRU_X_PATH)
    with open(GRU_META_PATH, encoding="utf-8") as f:
        metas = json.load(f)
    return X, metas


@np.errstate(invalid="ignore", divide="ignore")
def encode_target(model, win_rows):
    import torch
    mat = feature_matrix(win_rows)
    with torch.no_grad():
        x = torch.from_numpy(mat[None]).float()
        _, emb = model(x)
    emb = emb.detach().numpy().astype(np.float32)
    emb = emb / (np.linalg.norm(emb) + 1e-9)
    return emb  # (1, hidden)


def retrieve(X, metas, emb, top, exclude_self=None, before_date=None):
    """余弦相似度检索，返回 [(相似度, meta), ...]。"""
    sims = cosine_similarity(emb, X)[0]  # (n,)
    order = np.argsort(-sims)
    out = []
    for idx in order:
        m = metas[idx]
        if exclude_self and (m["code"], m["start"], m["end"]) == exclude_self:
            continue
        if before_date and str(m["end"]) >= before_date:
            continue
        out.append((float(sims[idx]), m))
        if len(out) >= top:
            break
    return out


def render_retrieval(target_rows, results, data_dir, window, out_path, title):
    import matplotlib.pyplot as plt
    n = 1 + len(results)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.3 * n))
    if n == 1:
        axes = [axes]

    chart.draw_candles(axes[0], target_rows)
    axes[0].set_title(f"目标窗口：{title}", fontsize=12)
    axes[0].set_xticks(range(len(target_rows)))
    axes[0].set_xticklabels(chart._tick_labels(target_rows), fontsize=6, rotation=45)
    axes[0].grid(axis="y", alpha=0.25)

    for i, (sim, m) in enumerate(results, 1):
        case_rows = load_case_rows(data_dir, m, window)
        chart.draw_candles(axes[i], case_rows)
        fwd = m["forward"]
        exc = m["excess"]
        fwd_txt = f"{fwd*100:+.2f}%" if fwd is not None else "N/A"
        exc_txt = f"{exc*100:+.2f}%" if exc is not None else "N/A"
        axes[i].set_title(f"#{i} {m['code']} {m['name']} {m['start']}-{m['end']}  "
                          f"相似度{sim:.3f} | 前向{fwd_txt} 超额{exc_txt}", fontsize=10)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="GRU 相似历史案例检索")
    parser.add_argument("code", nargs="?", help="股票代码，例如 600519")
    parser.add_argument("--start-date", help="起始日期 YYYYMMDD")
    parser.add_argument("--end-date", help="结束日期 YYYYMMDD")
    parser.add_argument("--top", type=int, default=5, help="返回最相似 N 个")
    parser.add_argument("--no-exclude-self", action="store_true", help="不排除自我匹配")
    parser.add_argument("--before", action="store_true", help="只返回目标窗口之前的案例")
    args = parser.parse_args()

    data_dir = ROOT / config.KLINE_DIR
    window = config.WINDOW

    if not args.code or not args.end_date:
        print("错误: 检索需提供 股票代码 + --end-date（先运行 python gru_train.py 建库）")
        return 1

    code = args.code.zfill(6)
    end = parse_date(args.end_date)
    model = load_model()
    X, metas = load_index()

    loaded = cl.load_stock_file(data_dir / f"{code}.json")
    if loaded is None:
        print(f"错误: 未找到股票数据 {code}，请先运行 python fetch_data.py")
        return 1
    _, name, rows = loaded

    if args.start_date:
        start = parse_date(args.start_date)
        win_rows = [r for r in rows if start <= str(r["date"]) <= end]
        if len(win_rows) != window:
            print(f"错误: {start}~{end} 之间只有 {len(win_rows)} 个交易日，而窗口固定为 {window} 个。")
            return 1
    else:
        dates = [r["date"] for r in rows if str(r["date"]) <= end]
        if len(dates) < window:
            print(f"错误: 截至 {end} 只有 {len(dates)} 个交易日，不足 {window} 个。")
            return 1
        start = dates[-window]
        win_rows = [r for r in rows if start <= str(r["date"]) <= end]

    emb = encode_target(model, win_rows)
    exclude = None if args.no_exclude_self else (code, str(win_rows[0]["date"]), str(win_rows[-1]["date"]))
    before = start if args.before else None
    results = retrieve(X, metas, emb, args.top, exclude_self=exclude, before_date=before)

    print("\n" + "=" * 66)
    print(f"GRU 检索：{code} {name} | {start} 至 {end}（{len(win_rows)} 个交易日，最相似 {len(results)} 例）")
    print("=" * 66)
    print(f"{'#':>2} {'代码':<8} {'名称':<10} {'日期区间':<21} {'相似度':>7} {'前向收益':>9} {'超额收益':>9}")
    for rank, (sim, m) in enumerate(results, 1):
        fwd_txt = f"{m['forward']*100:+.2f}%" if m["forward"] is not None else "N/A"
        exc_txt = f"{m['excess']*100:+.2f}%" if m["excess"] is not None else "N/A"
        print(f"{rank:>2} {m['code']:<8} {m['name']:<10} {m['start']}-{m['end']:<9} "
              f"{sim:>7.3f} {fwd_txt:>9} {exc_txt:>9}")

    fwds = [m["forward"] for _, m in results if m["forward"] is not None]
    excs = [m["excess"] for _, m in results if m["excess"] is not None]
    if fwds:
        print(f"\n相似案例前向收益均值: {np.mean(fwds)*100:+.2f}%（绝对）")
    if excs:
        print(f"相似案例超额收益均值: {np.mean(excs)*100:+.2f}%（相对等权市场基准）")
    print("（以上为历史相似片段的描述性统计，不代表预测，仅供讲评/教学参考）")

    out_dir = ROOT / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    chart_path = render_retrieval(win_rows, results, data_dir, window,
                                  out_dir / f"gru_retrieve_{code}_{start}_{end}.png",
                                  f"{code} {name} {start}-{end}")
    print(f"\n[图表] {chart_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
