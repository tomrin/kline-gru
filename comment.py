"""簇评语生成器：为 K-means 的每个形态簇各生成一段评语，
并以"窗口之后 3 个交易日"的历史表现做估算。

与训练保持严格一致：用相同的 shuffle seed、相同的滑窗步长、相同的窗口上限，
因此 predict 出的簇标签与训练时完全相同，3 日口径与已有 10 日口径同源。

用法：
    python comment.py                    # 用默认模型，生成评语
    python comment.py --forward 3        # 估算窗口（默认 3 个交易日）
    python comment.py --html             # 同时生成 HTML 报告
"""
from __future__ import annotations

import argparse
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
import cluster as cl
from features import feature_sequences, feature_vector

ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = ROOT / "models" / "kmeans.joblib"
OUTPUT_DIR = ROOT / "output"

# 涨红跌绿（A 股习惯）
UP = "#d03050"
UP_LIGHT = "#e26d84"
DOWN = "#0f9d6b"
DOWN_LIGHT = "#3ba97f"
NEUTRAL = "#8a8f98"


def load_model() -> dict[str, Any]:
    import joblib
    return joblib.load(MODEL_PATH)


def collect_and_predict(data_dir: Path, model: Any, window: int, step: int,
                        max_windows: int, seed: int, forward_short: int):
    """扫描全部股票，构建与训练一致的样本，返回 (labels, raw, fwd_abs, fwd_exc, metas)。"""
    from benchmark import get_or_build_benchmark

    benchmark = get_or_build_benchmark(data_dir)
    paths = sorted(data_dir.glob("*.json"))
    random.seed(seed)
    random.shuffle(paths)

    xs: list[np.ndarray] = []
    raws: list[np.ndarray] = []
    fwd_abs: list[float] = []
    fwd_exc: list[float] = []
    metas: list[tuple[str, str, str, str]] = []

    started = time.perf_counter()
    for pidx, path in enumerate(paths, 1):
        loaded = cl.load_stock_file(path)
        if loaded is None:
            continue
        code, name, rows = loaded
        for i in range(0, len(rows) - window + 1, step):
            win = rows[i:i + window]
            vec = feature_vector(win)
            seq = feature_sequences(win)
            raw: list[float] = []
            for key in ("return", "range", "body", "volume"):
                raw.extend(seq[key])
            exc, fwd = cl.forward_return_excess(rows, i + window - 1, forward_short, benchmark)
            xs.append(np.asarray(vec, dtype=np.float32))
            raws.append(np.asarray(raw, dtype=np.float32))
            fwd_abs.append(fwd if fwd is not None else np.nan)
            fwd_exc.append(exc if exc is not None else np.nan)
            metas.append((code, name, str(win[0]["date"]), str(win[-1]["date"])))
            if len(xs) >= max_windows:
                break
        if len(xs) >= max_windows:
            print(f"[scan] 已达上限 {max_windows} 窗口（{pidx}/{len(paths)} 只股票），与训练一致")
            break
        if pidx % 500 == 0:
            print(f"[scan] {pidx}/{len(paths)} 只，累计 {len(xs)} 窗口，{time.perf_counter() - started:.1f}s", flush=True)

    X = np.vstack(xs)
    raw = np.vstack(raws)
    fwd_abs = np.asarray(fwd_abs, dtype=np.float32)
    fwd_exc = np.asarray(fwd_exc, dtype=np.float32)
    print(f"[scan] 共 {len(metas)} 窗口，开始 predict ...", flush=True)
    labels = model.predict(X)
    return labels, raw, fwd_abs, fwd_exc, metas


def describe_pattern(ret, rng, body, vol) -> dict[str, Any]:
    """根据簇的原始特征均值序列，生成形态描述（形状层面，不判涨跌方向）。"""
    avg_range = float(np.mean(rng))

    # 波动水平
    if avg_range > 0.035:
        wave = "高波动"
    elif avg_range > 0.026:
        wave = "中等波动"
    else:
        wave = "低波动"

    # 量能趋势：末 5 日 vs 前 5 日
    vol_head = float(np.mean(vol[:5]))
    vol_tail = float(np.mean(vol[-5:]))
    ratio = (vol_tail / vol_head) if vol_head > 0 else 1.0
    if ratio > 1.06:
        vol_word = "末端放量"
    elif ratio < 0.94:
        vol_word = "末端缩量"
    else:
        vol_word = "量能平稳"

    # 价格节奏（return 序列形状：先抑后扬 / 冲高回落 / 平稳）
    seg = ret[1:]  # 忽略第 0 日（相对窗口外前一日）
    first = float(np.mean(seg[:7]))
    second = float(np.mean(seg[7:]))
    diff = second - first
    if diff > 0.0008:
        rhythm = "先抑后扬（后半段转强）"
    elif diff < -0.0008:
        rhythm = "冲高回落（后半段转弱）"
    else:
        rhythm = "走势平稳"

    return {
        "wave": wave,
        "amp_pct": avg_range * 100,
        "vol": vol_word,
        "vol_ratio": ratio,
        "rhythm": rhythm,
        "ret_diff_pct": diff * 100,
    }


def verdict(exc: float | None) -> tuple[str, str]:
    """3 日超额收益五档定性（幅度天然较小，阈值相应收紧）。"""
    if exc is None:
        return "中性", NEUTRAL
    if exc >= 0.0006:
        return "偏强", UP
    if exc >= 0.0002:
        return "略强", UP_LIGHT
    if exc > -0.0002:
        return "中性", NEUTRAL
    if exc > -0.0006:
        return "略弱", DOWN_LIGHT
    return "偏弱", DOWN


def review_line(v: str, d: dict[str, Any], avg_abs: float | None, avg_exc: float | None) -> str:
    """结合定性 + 节奏生成一句不矛盾的点评（5 档定性 × 3 种节奏）。"""
    rhythm = d["rhythm"]
    up = "先抑后扬" in rhythm      # 后半段转强
    down = "冲高回落" in rhythm    # 后半段转弱

    if v in ("偏强", "略强"):
        degree = "偏强" if v == "偏强" else "略强"
        if up:
            return f"短线{degree}，'先抑后扬'的后段转强与正超额方向一致，可适当关注。"
        if down:
            return f"短线{degree}，但形态属'冲高回落'，后段动能已现衰减，追涨需谨慎。"
        return f"短线{degree}，走势平稳，正超额或更多来自市场而非形态本身。"
    if v in ("偏弱", "略弱"):
        degree = "偏弱" if v == "偏弱" else "略弱"
        if up:
            return f"短线{degree}，'先抑后扬'的转强未兑现为超额，谨防诱多。"
        if down:
            return f"短线{degree}，'冲高回落'叠加负超额，短线弱势，回避追高。"
        return f"短线{degree}，形态平稳但超额为负，弱势震荡为主。"
    # 中性
    if up:
        return "方向不明，虽后半段转强但未兑现为超额，参考价值有限。"
    if down:
        return "方向不明，冲高回落节奏下后市更易回踩，观望为主。"
    return "方向不明，形态与超额均无显著倾向。"


def build_comments(labels, raw, fwd_abs, fwd_exc, metas, stats_by_id, k: int, window: int):
    raw_sum = np.zeros((k, 4 * window), dtype=np.float64)
    count = np.zeros(k, dtype=np.int64)
    for idx, lab in enumerate(labels):
        raw_sum[lab] += raw[idx]
        count[lab] += 1

    comments: list[dict[str, Any]] = []
    for cid in range(k):
        mask = labels == cid
        size = int(mask.sum())
        size_pct = size / len(labels) * 100

        def _avg(arr):
            v = arr[mask]
            v = v[~np.isnan(v)]
            return float(np.mean(v)) if len(v) else None

        def _win(arr):
            v = arr[mask]
            v = v[~np.isnan(v)]
            return float(np.mean(v > 0)) if len(v) else None

        mean = raw_sum[cid] / count[cid]
        ret = mean[0:window]
        rng = mean[window:2 * window]
        body = mean[2 * window:3 * window]
        vol = mean[3 * window:4 * window]

        d = describe_pattern(ret, rng, body, vol)
        avg_abs = _avg(fwd_abs)
        win_abs = _win(fwd_abs)
        avg_exc = _avg(fwd_exc)
        win_exc = _win(fwd_exc)
        v_word, v_color = verdict(avg_exc)
        rv = review_line(v_word, d, avg_abs, avg_exc)

        s = stats_by_id.get(cid, {})
        comments.append({
            "cluster_id": cid,
            "size": size,
            "size_pct": round(size_pct, 2),
            "desc": d,
            "avg_abs": avg_abs,
            "win_abs": win_abs,
            "avg_exc": avg_exc,
            "win_exc": win_exc,
            "verdict": v_word,
            "verdict_color": v_color,
            "review": rv,
            "rep_code": s.get("rep_code", ""),
            "rep_name": s.get("rep_name", ""),
            "rep_start": s.get("rep_start", ""),
            "rep_end": s.get("rep_end", ""),
        })

    order = sorted(comments, key=lambda c: (c["avg_exc"] is None, -(c["avg_exc"] or 0)))
    return comments, order


def fmt_pct(v: float | None) -> str:
    return f"{v*100:+.2f}%" if v is not None else "N/A"


def fmt_wr(v: float | None) -> str:
    return f"{v*100:.1f}%" if v is not None else "N/A"


def render_text(order, forward_short: int) -> str:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(f"K 线形态簇评语（K={len(order)}，窗口 15 交易日，估算后续 {forward_short} 个交易日）")
    lines.append("说明：形态基于簇内样本均值，3 日表现基于历史前向收益统计，属描述性、非预测。")
    lines.append("=" * 72)
    for c in order:
        d = c["desc"]
        lines.append("")
        lines.append(f"【簇 #{c['cluster_id']}】{c['verdict']}（样本 {c['size']} 个，占 {c['size_pct']}%）")
        lines.append(f"  形态：日均振幅 {d['amp_pct']:.1f}%、{d['vol']}、{d['rhythm']}。")
        lines.append(f"  后 {forward_short} 日估算：绝对均收 {fmt_pct(c['avg_abs'])}（胜率 {fmt_wr(c['win_abs'])}）；"
                     f"相对等权市场超额 {fmt_pct(c['avg_exc'])}（超额胜率 {fmt_wr(c['win_exc'])}）。")
        lines.append(f"  点评：{c['review']}")
        if c["rep_code"]:
            lines.append(f"  代表：{c['rep_code']} {c['rep_name']}（{c['rep_start']}-{c['rep_end']}）")
    return "\n".join(lines)


def render_html(order, forward_short: int, meta: dict[str, Any]) -> str:
    cards = []
    for c in order:
        vc = c["verdict_color"]
        d = c["desc"]
        rows_html = "".join(
            f"<tr><td>{k}</td><td style='color:{v}'>{vtext}</td></tr>"
            for k, v, vtext in [
                ("绝对均收", UP if (c['avg_abs'] or 0) >= 0 else DOWN, fmt_pct(c["avg_abs"])),
                ("绝对胜率", NEUTRAL, fmt_wr(c["win_abs"])),
                ("超额均收", UP if (c['avg_exc'] or 0) >= 0 else DOWN, fmt_pct(c["avg_exc"])),
                ("超额胜率", NEUTRAL, fmt_wr(c["win_exc"])),
            ]
        )
        rep = (f"{c['rep_code']} {c['rep_name']}（{c['rep_start']}-{c['rep_end']}）"
               if c["rep_code"] else "—")
        cards.append(f"""
        <div class="card">
          <div class="card-head">
            <span class="cid">簇 #{c['cluster_id']}</span>
            <span class="badge" style="background:{vc}">{c['verdict']}</span>
          </div>
          <div class="words">日均振幅 {d['amp_pct']:.1f}% · {d['vol']} · {d['rhythm']}</div>
          <table>
            {rows_html}
          </table>
          <div class="review">“{c['review']}”</div>
          <div class="meta">样本 {c['size']} 个（占 {c['size_pct']}%）</div>
          <div class="rep">代表形态：{rep}</div>
        </div>""")

    sil = meta.get("silhouette")
    sil_txt = f"{sil:.4f}" if sil is not None else "N/A"
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>K线形态簇评语（后{forward_short}日估算）</title>
<style>
  body {{ font-family: "Microsoft YaHei", "PingFang SC", sans-serif; background:#f5f6f8; color:#222; margin:0; padding:24px; }}
  h1 {{ font-size:22px; margin:0 0 6px; }}
  .sub {{ color:#666; font-size:13px; margin-bottom:14px; line-height:1.6; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(320px,1fr)); gap:14px; }}
  .card {{ background:#fff; border-radius:10px; padding:16px; box-shadow:0 1px 3px rgba(0,0,0,.08); }}
  .card-head {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; }}
  .cid {{ font-size:18px; font-weight:700; }}
  .badge {{ color:#fff; font-size:12px; padding:3px 10px; border-radius:12px; }}
  .words {{ font-size:14px; color:#333; margin-bottom:10px; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; margin-bottom:8px; }}
  td {{ padding:4px 0; border-bottom:1px solid #eee; }}
  td:first-child {{ color:#888; }}
  td:last-child {{ text-align:right; font-weight:600; }}
  .review {{ font-size:13px; color:#555; background:#fafafa; border-left:3px solid #ccc; padding:8px 10px; margin-bottom:8px; border-radius:0 6px 6px 0; }}
  .meta {{ font-size:12px; color:#999; }}
  .rep {{ font-size:12px; color:#666; margin-top:2px; }}
  .legend {{ font-size:12px; color:#888; margin-top:16px; line-height:1.7; }}
</style>
</head>
<body>
  <h1>K 线形态簇评语（后 {forward_short} 个交易日估算）</h1>
  <div class="sub">K = {meta.get('k')} · 窗口 = {meta.get('window')} 交易日 · 样本 = {meta.get('n_samples')} · silhouette = {sil_txt}<br>
  按 3 日超额收益降序排列。形态描述聚焦“振幅 / 量能 / 节奏”三要素；因聚类去趋势，不判涨跌方向。<br>
  注：各簇日均振幅集中在 3.8%~4.2%（A 股常态），波动水平并非簇间主要差异；簇间区分主要来自量能变化与价格节奏。</div>
  <div class="grid">{''.join(cards)}
  </div>
  <div class="legend">红 = 涨/跑赢（偏强/略强）· 绿 = 跌/跑输（偏弱/略弱）· 灰 = 中性。<br>
  本结果是对历史 K 线形态的分类与描述性统计，不构成投资建议或未来收益预测。</div>
</body>
</html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 K-means 各簇评语 + 后 N 日估算")
    parser.add_argument("--forward", type=int, default=3, help="估算窗口交易日数，默认 3")
    parser.add_argument("--html", action="store_true", help="同时生成 HTML 报告")
    args = parser.parse_args()

    data = load_model()
    model = data["model"]
    stats = data["stats"]
    window = data.get("window", config.WINDOW)
    k = data.get("k", config.K_CLUSTERS)
    seed = data.get("seed", config.RANDOM_SEED)
    step = config.SLIDE_STEP
    max_windows = config.MAX_WINDOWS

    stats_by_id = {s["cluster_id"]: s for s in stats}
    data_dir = ROOT / config.KLINE_DIR

    print("=" * 60)
    print(f"簇评语生成：K={k}，窗口={window}，估算后续 {args.forward} 个交易日")
    print("=" * 60)

    labels, raw, fwd_abs, fwd_exc, metas = collect_and_predict(
        data_dir, model, window, step, max_windows, seed, args.forward)

    comments, order = build_comments(labels, raw, fwd_abs, fwd_exc, metas, stats_by_id, k, window)

    excs = [c["avg_exc"] for c in comments if c["avg_exc"] is not None]
    if excs:
        print(f"\n[诊断] 3 日超额均收分布：min={min(excs)*100:+.3f}%  max={max(excs)*100:+.3f}%  中位={float(np.median(excs))*100:+.3f}%")

    text = render_text(order, args.forward)
    print("\n" + text)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    md_path = OUTPUT_DIR / "cluster_comments.md"
    md_path.write_text(text + "\n", encoding="utf-8")
    print(f"\n[输出] 文本评语已写入 {md_path}")

    stats_json = json.loads((ROOT / 'models' / 'cluster_stats.json').read_text(encoding='utf-8'))
    meta = {"k": k, "window": window, "n_samples": len(labels),
            "silhouette": stats_json.get("silhouette")}
    if args.html:
        html_path = OUTPUT_DIR / "cluster_comments.html"
        html_path.write_text(render_html(order, args.forward, meta), encoding="utf-8")
        print(f"[输出] HTML 报告已写入 {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
