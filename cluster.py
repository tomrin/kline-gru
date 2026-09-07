"""K-means 聚类核心：数据加载、窗口构建、训练、簇统计、存取。"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from features import feature_vector

MODEL_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "models"
OUTPUT_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "output"


def load_stock_file(path: Path) -> tuple[str, str, list[dict[str, Any]]] | None:
    """读取单个股票 JSON，返回 (code, name, rows)。"""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    rows = payload.get("kline", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return None
    required = ("date", "open", "high", "low", "close")
    rows = [r for r in rows if all(k in r for k in required)]
    rows.sort(key=lambda r: str(r["date"]))
    code = str(payload.get("code", path.stem)) if isinstance(payload, dict) else path.stem
    name = str(payload.get("name", code)) if isinstance(payload, dict) else code
    return code, name, rows


def forward_return(rows: list[dict[str, Any]], end_idx: int, forward_days: int) -> tuple[float | None, str | None, str | None]:
    """窗口末位索引为 end_idx，计算后续 forward_days 个交易日的收益。
    返回 (收益, 窗口结束日, 前向结束日)。"""
    target = end_idx + forward_days
    if target >= len(rows):
        return None, None, None
    c0 = float(rows[end_idx]["close"])
    c1 = float(rows[target]["close"])
    ret = (c1 / c0 - 1) if c0 else None
    return ret, str(rows[end_idx]["date"]), str(rows[target]["date"])


def forward_return_excess(rows: list[dict[str, Any]], end_idx: int, forward_days: int,
                          benchmark: dict[str, float]) -> tuple[float | None, float | None]:
    """计算超额收益 = 个股前向收益 - 等权市场基准同期收益。
    返回 (超额收益, 个股绝对收益)。"""
    fwd, end_date, fwd_date = forward_return(rows, end_idx, forward_days)
    if fwd is None or end_date not in benchmark or fwd_date not in benchmark:
        return None, fwd
    b0 = benchmark[end_date]
    b1 = benchmark[fwd_date]
    market_ret = (b1 / b0 - 1) if b0 else None
    if market_ret is None:
        return None, fwd
    return fwd - market_ret, fwd


def collect_samples(data_dir: Path, window: int, step: int, forward_days: int,
                    max_windows: int, seed: int):
    """遍历全部股票，收集窗口样本。

    返回 (X, fwd, fwd_excess, metas)：
    - X         : np.float32, shape (n, 4*window)
    - fwd       : np.float32, shape (n,)，缺失为 NaN（个股绝对收益）
    - fwd_excess: np.float32, shape (n,)，缺失为 NaN（相对等权市场基准超额收益）
    - metas     : list[(code, name, start_date, end_date)]
    """
    from benchmark import get_or_build_benchmark

    paths = sorted(data_dir.glob("*.json"))
    random.seed(seed)
    random.shuffle(paths)

    benchmark = get_or_build_benchmark(data_dir)
    if not benchmark:
        print("[collect] 警告：未构建出市场基准，超额收益将不可用")

    xs: list[np.ndarray] = []
    fwds: list[float] = []
    fwd_excesses: list[float] = []
    metas: list[tuple[str, str, str, str]] = []

    started = time.perf_counter()
    for pidx, path in enumerate(paths, 1):
        loaded = load_stock_file(path)
        if loaded is None:
            continue
        code, name, rows = loaded
        for i in range(0, len(rows) - window + 1, step):
            win = rows[i:i + window]
            vec = feature_vector(win)
            excess, fwd = forward_return_excess(rows, i + window - 1, forward_days, benchmark)
            xs.append(np.asarray(vec, dtype=np.float32))
            fwds.append(fwd if fwd is not None else np.nan)
            fwd_excesses.append(excess if excess is not None else np.nan)
            metas.append((code, name, str(win[0]["date"]), str(win[-1]["date"])))
            if len(xs) >= max_windows:
                break
        if len(xs) >= max_windows:
            print(f"[collect] 已达上限 {max_windows} 个窗口，停止收集（扫描到 {pidx}/{len(paths)} 只股票）", flush=True)
            break
        if pidx % 500 == 0:
            print(f"[collect] 已扫描 {pidx}/{len(paths)} 只股票，累计窗口 {len(xs)}，用时 {time.perf_counter() - started:.1f}s", flush=True)

    X = np.vstack(xs) if xs else np.empty((0, 4 * window), dtype=np.float32)
    fwd = np.asarray(fwds, dtype=np.float32)
    fwd_excess = np.asarray(fwd_excesses, dtype=np.float32)
    print(f"[collect] 完成：共 {len(metas)} 个窗口，用时 {time.perf_counter() - started:.1f}s")
    return X, fwd, fwd_excess, metas


def train(data_dir: Path | None = None, k: int | None = None, window: int | None = None,
          step: int | None = None, forward_days: int | None = None,
          max_windows: int | None = None, seed: int | None = None) -> dict[str, Any]:
    """训练 K-means，保存模型与簇统计，返回统计结果。"""
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import silhouette_score
    import joblib

    data_dir = data_dir or Path(os.path.dirname(os.path.abspath(__file__))) / config.KLINE_DIR
    k = k or config.K_CLUSTERS
    window = window or config.WINDOW
    step = step or config.SLIDE_STEP
    forward_days = forward_days or config.FORWARD_DAYS
    max_windows = max_windows or config.MAX_WINDOWS
    seed = seed if seed is not None else config.RANDOM_SEED

    print("=" * 60)
    print("K线形态 K-means 聚类训练")
    print(f"  窗口={window} 交易日 | 簇数 K={k} | 步长={step} | 前向={forward_days}日 | 上限={max_windows}")
    print("=" * 60)

    X, fwd, fwd_excess, metas = collect_samples(data_dir, window, step, forward_days, max_windows, seed)
    if len(X) < k:
        raise RuntimeError(f"样本数 {len(X)} 少于簇数 {k}，请增大数据量或减小 K")

    print(f"\n[训练] MiniBatchKMeans K={k} ...")
    model = MiniBatchKMeans(n_clusters=k, batch_size=4096, random_state=seed, n_init=3, max_iter=200)
    labels = model.fit_predict(X)

    # 簇统计
    stats = []
    centers = model.cluster_centers_
    for cid in range(k):
        mask = labels == cid
        size = int(mask.sum())

        def _avg(arr: np.ndarray) -> float | None:
            valid = arr[mask]
            valid = valid[~np.isnan(valid)]
            return float(np.mean(valid)) if len(valid) else None

        def _win_rate(arr: np.ndarray) -> float | None:
            valid = arr[mask]
            valid = valid[~np.isnan(valid)]
            return float(np.mean(valid > 0)) if len(valid) else None

        # 该簇中离中心最近的样本作为"代表形态"
        idxs = np.where(mask)[0]
        if len(idxs):
            dists = np.linalg.norm(X[idxs] - centers[cid], axis=1)
            rep_idx = int(idxs[int(np.argmin(dists))])
            rep_code, rep_name, rep_start, rep_end = metas[rep_idx]
        else:
            rep_code = rep_name = rep_start = rep_end = ""

        stats.append({
            "cluster_id": cid,
            "size": size,
            "size_pct": round(size / len(X) * 100, 2),
            "avg_forward_return": round(avg, 5) if (avg := _avg(fwd)) is not None else None,
            "win_rate": round(wr, 5) if (wr := _win_rate(fwd)) is not None else None,
            "avg_excess_return": round(exc, 5) if (exc := _avg(fwd_excess)) is not None else None,
            "excess_win_rate": round(ewr, 5) if (ewr := _win_rate(fwd_excess)) is not None else None,
            "rep_code": rep_code,
            "rep_name": rep_name,
            "rep_start": rep_start,
            "rep_end": rep_end,
        })

    # 按超额收益排序，更符合"跑赢市场"的解读
    stats_sorted = sorted(stats, key=lambda s: (s["avg_excess_return"] is None, s["avg_excess_return"] or 0), reverse=True)

    # silhouette 诊断（抽样评估，避免过慢）
    sil = None
    if len(X) > 6000:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), 6000, replace=False)
        sil = float(silhouette_score(X[idx], labels[idx]))
    else:
        sil = float(silhouette_score(X, labels))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump({"model": model, "centers": centers, "stats": stats,
                 "window": window, "k": k, "seed": seed}, MODEL_DIR / "kmeans.joblib")
    with open(MODEL_DIR / "cluster_stats.json", "w", encoding="utf-8") as f:
        json.dump({"window": window, "k": k, "forward_days": forward_days,
                   "silhouette": sil, "stats": stats}, f, ensure_ascii=False, indent=2)

    print(f"\n[完成] silhouette_score = {sil:.4f}（越接近 1 簇越紧致，接近 0 或负值说明簇结构弱）")
    print("\n簇统计（按相对等权市场基准的超额收益降序）：")
    print(f"{'簇':>3} {'样本数':>8} {'超额均收%':>10} {'超额胜率%':>10} {'绝对均收%':>10} {'绝对胜率%':>10}  代表形态")
    for s in stats_sorted:
        exc = f"{s['avg_excess_return']*100:+.2f}" if s["avg_excess_return"] is not None else "  NaN"
        ewr = f"{s['excess_win_rate']*100:.1f}" if s["excess_win_rate"] is not None else "NaN"
        avg = f"{s['avg_forward_return']*100:+.2f}" if s["avg_forward_return"] is not None else "  NaN"
        wr = f"{s['win_rate']*100:.1f}" if s["win_rate"] is not None else "NaN"
        rep = f"{s['rep_code']} {s['rep_start']}-{s['rep_end']}"
        print(f"{s['cluster_id']:>3} {s['size']:>8} {exc:>10} {ewr:>10} {avg:>10} {wr:>10}  {rep}")

    return {"silhouette": sil, "stats": stats_sorted, "n_samples": len(X)}
