"""K线图绘制（涨红跌绿，符合 A 股习惯）。"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

UP_COLOR = "#d03050"   # 涨 = 红
DOWN_COLOR = "#0f9d6b"  # 跌 = 绿

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "PingFang SC", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def draw_candles(ax, rows: list[dict[str, Any]]) -> None:
    """在给定 ax 上绘制蜡烛图。rows 含 open/high/low/close，按时间升序。"""
    for i, r in enumerate(rows):
        o = float(r["open"])
        h = float(r["high"])
        l = float(r["low"])
        c = float(r["close"])
        color = UP_COLOR if c >= o else DOWN_COLOR
        ax.plot([i, i], [l, h], color=color, linewidth=0.8, zorder=1)
        body_h = abs(c - o)
        if body_h < 1e-9:
            body_h = max(h - l, 1e-6) * 0.01
        ax.add_patch(Rectangle((i - 0.34, min(o, c)), 0.68, body_h,
                               facecolor=color, edgecolor=color, linewidth=0.5, zorder=2))


def _tick_labels(rows: list[dict[str, Any]]) -> list[str]:
    return [str(r.get("date", ""))[4:6] + "/" + str(r.get("date", ""))[6:8] for r in rows]


def _bar_chart(labels: list[str], values: list[float], sizes: list[int],
               title: str, ylabel: str, out_path: Path) -> Path:
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.6), 5.5))
    colors = [UP_COLOR if v >= 0 else DOWN_COLOR for v in values]
    bars = ax.bar(labels, values, color=colors, alpha=0.9)
    ax.axhline(0, color="#888", linewidth=0.8)
    ax.set_title(title, fontsize=13)
    ax.set_ylabel(ylabel)
    for bar, v, size in zip(bars, values, sizes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:+.2f}%\n(样本{size})", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def render_overview(stats: list[dict[str, Any]], out_path: Path) -> Path:
    """训练总览图：每个簇的样本占比 + 前向平均收益（红涨绿跌条形）。"""
    labels = [f"簇{s['cluster_id']}" for s in stats]
    avg = [s["avg_forward_return"] * 100 if s["avg_forward_return"] is not None else 0 for s in stats]
    sizes = [s["size"] for s in stats]
    return _bar_chart(labels, avg, sizes,
                      "各簇历史前向平均收益（未基准化，仅描述性）",
                      "前向平均收益 %", out_path)


def render_excess_overview(stats: list[dict[str, Any]], out_path: Path) -> Path:
    """训练总览图：每个簇相对等权市场基准的超额收益（红涨绿跌条形）。"""
    labels = [f"簇{s['cluster_id']}" for s in stats]
    avg = [s["avg_excess_return"] * 100 if s.get("avg_excess_return") is not None else 0 for s in stats]
    sizes = [s["size"] for s in stats]
    return _bar_chart(labels, avg, sizes,
                      "各簇相对等权市场基准的历史超额收益（红涨绿跌）",
                      "超额平均收益 %", out_path)


def render_query(target_rows: list[dict[str, Any]], rep_rows: list[dict[str, Any]],
                 cluster_id: int, out_path: Path, title: str = "") -> Path:
    """查询结果图：上半为目标窗口，下半为该簇代表形态。"""
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [1, 1]})

    ax1, ax2 = axes
    draw_candles(ax1, target_rows)
    ax1.set_title(f"目标窗口：{title}", fontsize=13)
    ax1.set_xticks(range(len(target_rows)))
    ax1.set_xticklabels(_tick_labels(target_rows), fontsize=7, rotation=45)
    ax1.grid(axis="y", alpha=0.25)

    draw_candles(ax2, rep_rows)
    ax2.set_title(f"所属簇 #{cluster_id} 的代表形态（离簇中心最近的真实K线）", fontsize=13)
    ax2.set_xticks(range(len(rep_rows)))
    ax2.set_xticklabels(_tick_labels(rep_rows), fontsize=7, rotation=45)
    ax2.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def render_centroids(centers, window: int, stats: list[dict[str, Any]],
                     data_dir: Path, out_path: Path) -> Path:
    """每个簇一张代表形态小图，拼成网格。centers 用于标注，实际画代表真实K线。"""
    n = len(stats)
    cols = 4
    rows_count = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_count, cols, figsize=(cols * 3.2, rows_count * 3.0))
    axes = axes.flatten() if n > 1 else [axes]
    for i, s in enumerate(stats):
        ax = axes[i]
        rep_rows = _load_rep_rows(s, data_dir, window)
        if rep_rows:
            draw_candles(ax, rep_rows)
        exc = s.get("avg_excess_return")
        exc_txt = f"超额{exc*100:+.2f}%" if exc is not None else "超额N/A"
        wr = s.get("excess_win_rate")
        wr_txt = f"胜率{wr*100:.0f}%" if wr is not None else "胜率N/A"
        ax.set_title(f"簇{s['cluster_id']} {exc_txt} {wr_txt}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle("各簇代表形态（涨红跌绿）", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def _load_rep_rows(s: dict[str, Any], data_dir: Path, window: int) -> list[dict[str, Any]]:
    code = s.get("rep_code")
    start = s.get("rep_start")
    end = s.get("rep_end")
    if not code:
        return []
    path = data_dir / f"{code}.json"
    if not path.exists():
        return []
    try:
        import json
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("kline", [])
    except (OSError, json.JSONDecodeError):
        return []
    sel = [r for r in rows if start <= str(r.get("date", "")) <= end]
    return sel if len(sel) == window else []
