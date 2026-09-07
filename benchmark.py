"""构建等权市场基准（从所有股票日线计算），用于计算超额收益。

无需额外 API：直接用 data/kline/ 下所有可用股票的日收益等权平均，
得到每日"市场平均收益"，再累积为市场指数。对齐个股窗口时，按实际交易日区间计算。
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import cluster as cl
import config

ROOT = Path(os.path.dirname(os.path.abspath(__file__)))


def build_market_benchmark(data_dir: Path, output_path: Path | None = None,
                           min_stocks: int = 100) -> dict[str, float]:
    """读取全部股票 JSON，计算每个交易日的等权收益率，保存并返回 {date: close_index}。"""
    if output_path is None:
        output_path = ROOT / config.BENCHMARK_FILE

    print("[benchmark] 正在从全部股票构建等权市场基准...")
    daily_returns: dict[str, list[float]] = defaultdict(list)
    paths = sorted(data_dir.glob("*.json"))
    for path in paths:
        loaded = cl.load_stock_file(path)
        if loaded is None:
            continue
        _, _, rows = loaded
        for i, r in enumerate(rows):
            c = float(r["close"])
            prev = float(rows[i - 1]["close"]) if i else c
            ret = c / prev - 1 if prev else 0.0
            daily_returns[str(r["date"])].append(ret)

    # 排序交易日，过滤样本数太少的日期
    sorted_dates = sorted(d for d, rets in daily_returns.items() if len(rets) >= min_stocks)
    index = 1.0
    series: list[dict[str, Any]] = []
    closes: dict[str, float] = {}
    for d in sorted_dates:
        avg_ret = float(np.mean(daily_returns[d]))
        index *= 1 + avg_ret
        closes[d] = index
        series.append({"date": d, "avg_return": avg_ret, "close": index})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"name": "等权市场基准", "source": "data/kline 全部股票等权", "series": series}, f, ensure_ascii=False, indent=2)

    print(f"[benchmark] 完成：共 {len(series)} 个交易日，最新指数 {index:.4f}")
    return closes


def load_market_benchmark(path: Path) -> dict[str, float]:
    """加载等权市场基准 close 序列。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {item["date"]: item["close"] for item in payload.get("series", [])}


def get_or_build_benchmark(data_dir: Path) -> dict[str, float]:
    """如果存在就加载，否则构建。"""
    path = ROOT / config.BENCHMARK_FILE
    if path.exists():
        try:
            return load_market_benchmark(path)
        except (OSError, json.JSONDecodeError):
            pass
    return build_market_benchmark(data_dir, path)
