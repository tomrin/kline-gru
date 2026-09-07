"""训练 K-means 聚类模型。

用法：
    python train.py                 # 使用 config.py 默认参数
    python train.py --k 20 --window 10
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import cluster as cl
import chart

ROOT = Path(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    parser = argparse.ArgumentParser(description="训练 K 线形态 K-means 聚类模型")
    parser.add_argument("--k", type=int, default=config.K_CLUSTERS)
    parser.add_argument("--window", type=int, default=config.WINDOW)
    parser.add_argument("--step", type=int, default=config.SLIDE_STEP)
    parser.add_argument("--forward-days", type=int, default=config.FORWARD_DAYS)
    parser.add_argument("--max-windows", type=int, default=config.MAX_WINDOWS)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()

    data_dir = ROOT / config.KLINE_DIR
    result = cl.train(data_dir, args.k, args.window, args.step,
                      args.forward_days, args.max_windows, args.seed)

    # 生成可视化
    overview = chart.render_overview(result["stats"], cl.OUTPUT_DIR / "cluster_overview.png")
    excess = chart.render_excess_overview(result["stats"], cl.OUTPUT_DIR / "cluster_excess.png")
    centroids = chart.render_centroids(None, args.window, result["stats"],
                                       data_dir, cl.OUTPUT_DIR / "cluster_centroids.png")
    print(f"\n[图表] 簇收益总览(绝对): {overview}")
    print(f"[图表] 簇收益总览(超额): {excess}")
    print(f"[图表] 簇代表形态: {centroids}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
