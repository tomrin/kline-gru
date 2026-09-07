"""批量截面扫描：某交易日全市场形态分类，服务「选股」。

给定一个结束日期，扫描 data/kline 下所有股票，取每只股票截至该日最近的
WINDOW 个交易日，用已训练的 K-means 预测其所属簇，并关联该簇的历史超额收益。

输出：
    1) 控制台：各簇当天股票数量分布 + 历史超额收益排名 + 跑赢簇名单
    2) output/scan_YYYYMMDD.csv：每只股票 → 簇 + 簇历史超额/绝对收益
    3) output/scan_YYYYMMDD.png：各簇当天数量柱状图（按历史超额收益着色）

用法：
    python scan.py --end-date 20260801
    python scan.py --end-date 20260801 --min-excess 0.1 --top-list 50
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import cluster as cl
import chart
from features import feature_vector

ROOT = Path(os.path.dirname(os.path.abspath(__file__)))


def parse_date(value: str) -> str:
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).strftime("%Y%m%d")
        except ValueError:
            pass
    raise ValueError(f"日期格式错误: {value}")


def load_model():
    import joblib
    path = ROOT / "models" / "kmeans.joblib"
    if not path.exists():
        raise FileNotFoundError("未找到模型，请先运行 python train.py")
    return joblib.load(path)


def scan(data_dir: Path, model, stats: dict[int, dict], window: int,
         end_date: str) -> list[dict]:
    """扫描全市场，返回每只股票的分类结果列表。"""
    centers = model.cluster_centers_
    results: list[dict] = []
    paths = sorted(data_dir.glob("*.json"))
    started = time.perf_counter()
    for pidx, path in enumerate(paths, 1):
        loaded = cl.load_stock_file(path)
        if loaded is None:
            continue
        code, name, rows = loaded
        upto = [r for r in rows if str(r["date"]) <= end_date]
        if len(upto) < window:
            continue
        win_rows = upto[-window:]
        vec = np.asarray(feature_vector(win_rows), dtype=np.float32).reshape(1, -1)
        cid = int(model.predict(vec)[0])
        dist = float(np.linalg.norm(centers[cid] - vec))
        s = stats[cid]
        results.append({
            "code": code,
            "name": name,
            "cluster": cid,
            "excess": s["avg_excess_return"],
            "excess_win": s["excess_win_rate"],
            "abs": s["avg_forward_return"],
            "win": s["win_rate"],
            "distance": dist,
        })
        if pidx % 1000 == 0:
            print(f"[scan] 已扫 {pidx}/{len(paths)}，命中 {len(results)}，"
                  f"用时 {time.perf_counter() - started:.1f}s", flush=True)
    print(f"[scan] 完成：扫 {len(paths)} 只，可用 {len(results)} 只，"
          f"用时 {time.perf_counter() - started:.1f}s")
    return results


def render_scan_chart(cluster_rows: list[tuple], out_path: Path) -> Path:
    """各簇当天股票数量柱状图，按历史超额收益着色。cluster_rows: (cid, count, excess, excess_win)。"""
    import matplotlib.pyplot as plt
    cids = [r[0] for r in cluster_rows]
    counts = [r[1] for r in cluster_rows]
    excess = [r[2] for r in cluster_rows]
    labels = [f"簇{cid}\n{(e*100 if e is not None else 0):+.2f}%" for cid, _, e, _ in cluster_rows]
    colors = [chart.UP_COLOR if (e is not None and e >= 0) else chart.DOWN_COLOR for _, _, e, _ in cluster_rows]

    fig, ax = plt.subplots(figsize=(max(8, len(cids) * 0.7), 5.5))
    bars = ax.bar(labels, counts, color=colors, alpha=0.9)
    ax.set_title("各簇当天股票数量（颜色=该簇历史超额收益红涨绿跌）", fontsize=13)
    ax.set_ylabel("当天股票数量")
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                str(cnt), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="批量截面扫描：全市场形态分类")
    parser.add_argument("--end-date", required=True, help="截止日期 YYYYMMDD")
    parser.add_argument("--min-excess", type=float, default=0.0,
                        help="跑赢簇阈值（超额收益≥该百分点，例如 0.1 表示 0.1%%），默认 0")
    parser.add_argument("--top-list", type=int, default=30, help="跑赢名单最多列出多少只")
    args = parser.parse_args()

    end_date = parse_date(args.end_date)
    data_dir = ROOT / config.KLINE_DIR
    model_pack = load_model()
    model = model_pack["model"]
    stats = {s["cluster_id"]: s for s in model_pack["stats"]}
    window = model_pack["window"]

    results = scan(data_dir, model, stats, window, end_date)
    if not results:
        print("错误: 没有可用的股票数据")
        return 1

    # 按簇聚合
    agg: dict[int, dict] = {}
    for r in results:
        cid = r["cluster"]
        if cid not in agg:
            s = stats[cid]
            agg[cid] = {"count": 0, "excess": s["avg_excess_return"],
                        "excess_win": s["excess_win_rate"]}
        agg[cid]["count"] += 1

    # 按历史超额收益降序
    order = sorted(agg.keys(), key=lambda c: (agg[c]["excess"] is None, agg[c]["excess"] or 0), reverse=True)

    print("\n" + "=" * 66)
    print(f"截面扫描结果：截至 {end_date}（窗口 {window} 个交易日，共 {len(results)} 只股票）")
    print("=" * 66)
    print(f"{'簇':>3} {'当天数':>7} {'占比':>7} {'历史超额均收':>12} {'超额胜率':>9}")
    for cid in order:
        a = agg[cid]
        exc = f"{a['excess']*100:+.2f}%" if a["excess"] is not None else "N/A"
        ewr = f"{a['excess_win']*100:.1f}%" if a["excess_win"] is not None else "N/A"
        pct = a["count"] / len(results) * 100
        print(f"{cid:>3} {a['count']:>7} {pct:>6.1f}% {exc:>12} {ewr:>9}")

    # 跑赢簇名单（excess 为小数，转换为百分点与阈值比较）
    winners = [r for r in results
               if r["excess"] is not None and r["excess"] * 100 >= args.min_excess]
    winners.sort(key=lambda r: (r["excess"] or 0, -r["distance"]), reverse=True)
    print(f"\n跑赢簇名单（簇历史超额收益 ≥ {args.min_excess:.2f}%，按超额降序，最多 {args.top_list} 只）：")
    for r in winners[: args.top_list]:
        print(f"  {r['code']} {r['name']:<8} 簇#{r['cluster']}  超额{r['excess']*100:+.2f}%  "
              f"距中心{r['distance']:.2f}")
    if len(winners) > args.top_list:
        print(f"  ... 共 {len(winners)} 只，完整见 CSV")

    # 写 CSV
    out_dir = ROOT / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"scan_{end_date}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["code", "name", "cluster", "excess_return", "excess_win_rate",
                    "abs_return", "win_rate", "distance"])
        for r in sorted(results, key=lambda x: ((x["excess"] is None, -(x["excess"] or 0), x["distance"]))):
            w.writerow([
                r["code"], r["name"], r["cluster"],
                f"{r['excess']:.6f}" if r["excess"] is not None else "",
                f"{r['excess_win']:.4f}" if r["excess_win"] is not None else "",
                f"{r['abs']:.6f}" if r["abs"] is not None else "",
                f"{r['win']:.4f}" if r["win"] is not None else "",
                f"{r['distance']:.4f}",
            ])

    # 画图
    cluster_rows = [(cid, agg[cid]["count"], agg[cid]["excess"], agg[cid]["excess_win"]) for cid in order]
    png_path = render_scan_chart(cluster_rows, out_dir / f"scan_{end_date}.png")

    print(f"\n[CSV]  {csv_path}")
    print(f"[图表] {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
