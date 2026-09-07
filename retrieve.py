"""KNN 相似历史案例检索：给目标窗口找最相似的历史片段（对接 K 线模拟器讲评）。

与 K-means 的"判类别"互补：
- K-means（train.py/query.py）回答「这只股票这段形态属于哪个簇」——组别层面。
- KNN（本脚本）回答「最像哪些具体历史个案，那些个案之后怎么走的」——个案层面。

用法：
    # 首次：构建检索库（遍历全部历史窗口，一次性，约 1-2 分钟）
    python retrieve.py --build

    # 检索：某股票截至某日最近 WINDOW 日形态，最像哪些历史片段
    python retrieve.py 600519 --end-date 20260801 --top 5
    python retrieve.py 000001 --start-date 20240701 --end-date 20240719 --top 5

可选：
    --top N          返回最相似 N 个（默认 5）
    --no-exclude-self  保留与目标同股票同时间段的窗口（默认排除自我匹配）
    --before         只返回目标窗口开始日之前的案例（避免"用同期/未来当先例"）
    --max-windows    构建索引时的窗口上限（默认全量）
"""
from __future__ import annotations

import argparse
import json
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
X_PATH = ROOT / "models" / "knn_X.npy"
META_PATH = ROOT / "models" / "knn_meta.json"


def parse_date(value: str) -> str:
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).strftime("%Y%m%d")
        except ValueError:
            pass
    raise ValueError(f"日期格式错误: {value}")


def build_index(data_dir: Path, window: int, step: int, forward_days: int,
                max_windows: int | None) -> tuple[np.ndarray, list[dict]]:
    """遍历全部股票全部窗口，构建特征库 X + 元信息（含前向收益）。"""
    from benchmark import get_or_build_benchmark

    benchmark = get_or_build_benchmark(data_dir)
    xs: list[np.ndarray] = []
    metas: list[dict] = []
    paths = sorted(data_dir.glob("*.json"))
    started = time.perf_counter()

    for pidx, path in enumerate(paths, 1):
        loaded = cl.load_stock_file(path)
        if loaded is None:
            continue
        code, name, rows = loaded
        for i in range(0, len(rows) - window + 1, step):
            win = rows[i:i + window]
            vec = feature_vector(win)
            excess, fwd = cl.forward_return_excess(rows, i + window - 1, forward_days, benchmark)
            xs.append(np.asarray(vec, dtype=np.float32))
            metas.append({
                "code": code,
                "name": name,
                "start": str(win[0]["date"]),
                "end": str(win[-1]["date"]),
                "forward": fwd if fwd is not None else None,
                "excess": excess if excess is not None else None,
            })
            if max_windows and len(xs) >= max_windows:
                break
        if max_windows and len(xs) >= max_windows:
            print(f"[build] 已达上限 {max_windows} 个窗口（扫描到 {pidx}/{len(paths)} 只股票）", flush=True)
            break
        if pidx % 1000 == 0:
            print(f"[build] 已扫 {pidx}/{len(paths)} 只，累计窗口 {len(xs)}，"
                  f"用时 {time.perf_counter() - started:.1f}s", flush=True)

    X = np.vstack(xs) if xs else np.empty((0, 4 * window), dtype=np.float32)
    print(f"[build] 完成：共 {len(metas)} 个窗口，用时 {time.perf_counter() - started:.1f}s")
    return X, metas


def save_index(X: np.ndarray, metas: list[dict]) -> None:
    X_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(X_PATH, X)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False)
    print(f"[index] 已保存 {X_PATH} ({X.nbytes / 1e6:.1f} MB) 与 {META_PATH}")


def load_index() -> tuple[np.ndarray, list[dict]]:
    if not X_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError("未找到检索库，请先运行 python retrieve.py --build")
    X = np.load(X_PATH)
    with open(META_PATH, encoding="utf-8") as f:
        metas = json.load(f)
    return X, metas


def retrieve(X: np.ndarray, metas: list[dict], vec: np.ndarray, top: int,
             exclude_self: tuple[str, str, str] | None = None,
             before_date: str | None = None) -> list[tuple[float, dict]]:
    """在检索库中找最相似的 top 个窗口，返回 [(距离, meta), ...]。"""
    dists = np.linalg.norm(X - vec, axis=1)
    order = np.argsort(dists)
    out: list[tuple[float, dict]] = []
    for idx in order:
        m = metas[idx]
        if exclude_self and (m["code"], m["start"], m["end"]) == exclude_self:
            continue
        if before_date and str(m["end"]) >= before_date:
            continue
        out.append((float(dists[idx]), m))
        if len(out) >= top:
            break
    return out


def load_case_rows(data_dir: Path, m: dict, window: int) -> list[dict]:
    path = data_dir / f"{m['code']}.json"
    if not path.exists():
        return []
    loaded = cl.load_stock_file(path)
    if loaded is None:
        return []
    _, _, rows = loaded
    sel = [r for r in rows if m["start"] <= str(r.get("date", "")) <= m["end"]]
    return sel if len(sel) == window else []


def render_retrieval(target_rows: list[dict], results: list[tuple[float, dict]],
                     data_dir: Path, window: int, out_path: Path, title: str) -> Path:
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

    for i, (dist, m) in enumerate(results, 1):
        case_rows = load_case_rows(data_dir, m, window)
        chart.draw_candles(axes[i], case_rows)
        fwd = m["forward"]
        exc = m["excess"]
        fwd_txt = f"{fwd*100:+.2f}%" if fwd is not None else "N/A"
        exc_txt = f"{exc*100:+.2f}%" if exc is not None else "N/A"
        axes[i].set_title(f"#{i} {m['code']} {m['name']} {m['start']}-{m['end']}  "
                          f"距离{dist:.2f} | 前向{fwd_txt} 超额{exc_txt}", fontsize=10)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="KNN 相似历史案例检索")
    parser.add_argument("code", nargs="?", help="股票代码，例如 600519")
    parser.add_argument("--build", action="store_true", help="构建检索库")
    parser.add_argument("--start-date", help="起始日期 YYYYMMDD")
    parser.add_argument("--end-date", help="结束日期 YYYYMMDD")
    parser.add_argument("--top", type=int, default=5, help="返回最相似 N 个")
    parser.add_argument("--max-windows", type=int, default=None, help="构建索引时的窗口上限")
    parser.add_argument("--no-exclude-self", action="store_true", help="不排除自我匹配")
    parser.add_argument("--before", action="store_true", help="只返回目标窗口之前的案例")
    args = parser.parse_args()

    data_dir = ROOT / config.KLINE_DIR
    window = config.WINDOW

    if args.build:
        X, metas = build_index(data_dir, window, config.SLIDE_STEP,
                               config.FORWARD_DAYS, args.max_windows)
        save_index(X, metas)
        return 0

    if not args.code or not args.end_date:
        print("错误: 检索需提供 股票代码 + --end-date（或先 --build）")
        return 1

    code = args.code.zfill(6)
    end = parse_date(args.end_date)
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

    vec = np.asarray(feature_vector(win_rows), dtype=np.float32).reshape(1, -1)
    exclude = None if args.no_exclude_self else (code, str(win_rows[0]["date"]), str(win_rows[-1]["date"]))
    before = start if args.before else None  # 先例：案例窗口须完全早于目标窗口开始日
    results = retrieve(X, metas, vec, args.top, exclude_self=exclude, before_date=before)

    print("\n" + "=" * 66)
    print(f"KNN 检索：{code} {name} | {start} 至 {end}（{len(win_rows)} 个交易日，最相似 {len(results)} 例）")
    print("=" * 66)
    print(f"{'#':>2} {'代码':<8} {'名称':<10} {'日期区间':<21} {'距离':>7} {'前向收益':>9} {'超额收益':>9}")
    for rank, (dist, m) in enumerate(results, 1):
        fwd_txt = f"{m['forward']*100:+.2f}%" if m["forward"] is not None else "N/A"
        exc_txt = f"{m['excess']*100:+.2f}%" if m["excess"] is not None else "N/A"
        print(f"{rank:>2} {m['code']:<8} {m['name']:<10} {m['start']}-{m['end']:<9} "
              f"{dist:>7.2f} {fwd_txt:>9} {exc_txt:>9}")

    # 汇总相似案例的历史表现
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
                                  out_dir / f"retrieve_{code}_{start}_{end}.png",
                                  f"{code} {name} {start}-{end}")
    print(f"\n[图表] {chart_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
