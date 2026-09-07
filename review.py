"""讲评提示词生成器：把 K-means 判类别 + KNN 相似案例，组装成给 Kimi-K3 的讲评 prompt。

与 kline-training-camp 的"智能讲评"（逐根 K 线技术分析）互补：
本脚本聚焦"相似历史情况讲评"——告诉 Kimi 目标形态属于哪个簇、最像哪些历史个案、
那些个案之后怎么走的，让 Kimi 结合历史相似形态做复盘讲评。

用法：
    # 生成讲评 prompt（打印 + 存 txt，不调用 Kimi）
    python review.py 600519 --end-date 20260801 --top 5 --before

    # 生成后直接调用 Kimi-K3 出讲评
    python review.py 600519 --end-date 20260801 --top 5 --before --call

可选：
    --top N         相似案例数（默认 5）
    --before        只取目标窗口之前的先例（推荐）
    --call          生成 prompt 后直接调用 Kimi API
    --model         覆盖 config.KIMI_MODEL
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import cluster as cl
from features import feature_vector
from retrieve import load_index, retrieve

ROOT = Path(os.path.dirname(os.path.abspath(__file__)))


def parse_date(value: str) -> str:
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).strftime("%Y%m%d")
        except ValueError:
            pass
    raise ValueError(f"日期格式错误: {value}")


def load_kmeans_model():
    import joblib
    path = ROOT / "models" / "kmeans.joblib"
    if not path.exists():
        raise FileNotFoundError("未找到模型，请先运行 python train.py")
    return joblib.load(path)


def fmt_kline_rows(rows: list[dict]) -> list[str]:
    """把窗口 K 线格式化成可读文本行。"""
    lines = []
    for i, k in enumerate(rows, 1):
        change = ""
        if i > 1:
            prev = float(rows[i - 2]["close"])
            pct = (float(k["close"]) - prev) / prev * 100
            change = f" 涨跌:{pct:+.2f}%"
        vol = float(k.get("volume", 0))
        lines.append(f"第{i}根: 开={float(k['open']):.2f} 高={float(k['high']):.2f} "
                     f"低={float(k['low']):.2f} 收={float(k['close']):.2f} "
                     f"量={vol/1e4:.0f}万手{change}")
    return lines


def build_review_prompt(code: str, name: str, win_rows: list[dict],
                        cluster_info: dict, cases: list[tuple[float, dict]]) -> str:
    """组装讲评 prompt。"""
    L: list[str] = []
    L.append("你是一位资深 A 股技术形态分析师，擅长结合历史相似形态做复盘讲评。")
    L.append("")
    L.append("【任务背景】")
    L.append("这是 K 线形态研究工具（kline-cluster）的讲评请求。系统已用无监督聚类(K-means)和")
    L.append("最近邻检索(k-NN)，为下方目标形态找到了历史相似案例。请结合这些相似案例讲评，")
    L.append("重点回答：这段形态历史上像什么、那些相似案例之后通常怎么走。")
    L.append("")
    L.append(f"【目标股票与时间段】")
    L.append(f"代码：{code} {name}")
    L.append(f"时间：{win_rows[0]['date']} 至 {win_rows[-1]['date']}（{len(win_rows)} 个交易日）")
    L.append("")
    L.append("【目标窗口 K 线数据】")
    L.extend(fmt_kline_rows(win_rows))
    L.append("")
    L.append("【所属形态簇（K-means 分类结果）】")
    L.append(f"- 所属簇：#{cluster_info['cluster_id']}")
    L.append(f"- 距簇中心距离：{cluster_info['distance']:.2f}（越小越典型）")
    L.append(f"- 该簇历史超额收益（相对等权市场基准）：{cluster_info['excess']}")
    L.append(f"- 该簇超额胜率：{cluster_info['excess_win']}")
    L.append(f"- 该簇历史绝对收益：{cluster_info['abs']}")
    L.append(f"- 代表形态：{cluster_info['rep_name']}({cluster_info['rep_code']}) "
             f"{cluster_info['rep_start']}-{cluster_info['rep_end']}")
    L.append("")
    L.append(f"【最相似历史案例（k-NN 检索，前 {len(cases)} 例）】")
    for rank, (dist, m) in enumerate(cases, 1):
        fwd = f"{m['forward']*100:+.2f}%" if m["forward"] is not None else "N/A"
        exc = f"{m['excess']*100:+.2f}%" if m["excess"] is not None else "N/A"
        L.append(f"#{rank} {m['code']} {m['name']} {m['start']}-{m['end']}（形态距离{dist:.2f}）")
        L.append(f"   该案例后续 {config.FORWARD_DAYS} 日：绝对收益 {fwd}，超额收益 {exc}")
    L.append("")
    L.append("【讲评要求】")
    L.append("请严格基于以上数据输出以下内容（只做复盘讲评，不预测目标未来涨跌）：")
    L.append("1. 形态解读：目标窗口的形态特征（趋势、量价、实体、波动节奏）")
    L.append("2. 相似案例对比：历史相似案例与目标的共同点、差异点")
    L.append("3. 历史表现统计：相似案例\"之后怎么走\"的描述性统计（收益均值、正收益占比、分布）")
    L.append("4. 关键观察与风险提示")
    L.append("5. 训练要点：总结一条可复用的形态识别知识点")
    L.append("")
    L.append("重要：历史相似案例的后续走势是历史事实，不代表目标会复现；请在结尾附免责声明。")
    return "\n".join(L)


def call_kimi(prompt: str, model: str) -> str | None:
    """调用 Kimi API（OpenAI 兼容 /chat/completions），返回讲评文本。"""
    base = config.KIMI_BASE_URL.rstrip("/")
    if base.endswith("/chat/completions"):
        url = base
    else:
        url = base + "/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + config.KIMI_API_KEY,
    }
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是一位专业的 A 股技术形态分析师，擅长结合历史相似形态做客观、专业的复盘讲评。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 1,
        "max_completion_tokens": 8000,
    }
    req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"),
                                 headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=300) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    if "choices" in result and len(result["choices"]) > 0:
        content = (result["choices"][0]["message"].get("content") or "").strip()
        if not content:
            return None
        return content
    return json.dumps(result, ensure_ascii=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="讲评提示词生成器（对接 Kimi-K3）")
    parser.add_argument("code", help="股票代码，例如 600519")
    parser.add_argument("--start-date", help="起始日期 YYYYMMDD")
    parser.add_argument("--end-date", required=True, help="结束日期 YYYYMMDD")
    parser.add_argument("--top", type=int, default=5, help="相似案例数")
    parser.add_argument("--before", action="store_true", help="只取目标窗口之前的先例")
    parser.add_argument("--call", action="store_true", help="生成后直接调用 Kimi")
    parser.add_argument("--model", default=None, help="覆盖 config.KIMI_MODEL")
    args = parser.parse_args()

    code = args.code.zfill(6)
    end = parse_date(args.end_date)
    data_dir = ROOT / config.KLINE_DIR
    window = config.WINDOW

    # 加载 K-means 模型 + KNN 检索库
    model_pack = load_kmeans_model()
    model = model_pack["model"]
    stats = {s["cluster_id"]: s for s in model_pack["stats"]}
    X, metas = load_index()

    loaded = cl.load_stock_file(data_dir / f"{code}.json")
    if loaded is None:
        print(f"错误: 未找到股票数据 {code}，请先运行 python fetch_data.py")
        return 1
    _, name, rows = loaded

    # 选目标窗口
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

    # K-means 判类别
    cid = int(model.predict(vec)[0])
    centers = model.cluster_centers_
    dist = float(np.linalg.norm(centers[cid] - vec))
    s = stats[cid]
    def _pct(v, sign=False):
        return f"{v*100:+.2f}%" if v is not None else "N/A"
    cluster_info = {
        "cluster_id": cid,
        "distance": dist,
        "excess": _pct(s["avg_excess_return"]),
        "excess_win": _pct(s["excess_win_rate"]),
        "abs": _pct(s["avg_forward_return"]),
        "rep_name": s["rep_name"],
        "rep_code": s["rep_code"],
        "rep_start": s["rep_start"],
        "rep_end": s["rep_end"],
    }

    # KNN 找相似案例
    exclude = (code, str(win_rows[0]["date"]), str(win_rows[-1]["date"]))
    before = start if args.before else None
    cases = retrieve(X, metas, vec, args.top, exclude_self=exclude, before_date=before)

    # 组装 prompt
    prompt = build_review_prompt(code, name, win_rows, cluster_info, cases)

    out_dir = ROOT / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = out_dir / f"review_{code}_{start}_{end}_prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")

    print("=" * 66)
    print(f"讲评 prompt 已生成：{code} {name} | {start} 至 {end}")
    print(f"  所属簇 #{cid}，相似案例 {len(cases)} 例")
    print("=" * 66)
    print(prompt)
    print("\n" + "=" * 66)
    print(f"[prompt] 已保存: {prompt_path}")

    if args.call:
        model_name = args.model or config.KIMI_MODEL
        print(f"\n[call] 正在调用 Kimi（model={model_name}）...")
        try:
            reply = call_kimi(prompt, model_name)
        except Exception as e:
            print(f"[call] 调用失败: {e}")
            return 1
        reply_path = out_dir / f"review_{code}_{start}_{end}_kimi.txt"
        reply_path.write_text(reply or "", encoding="utf-8")
        print("\n" + "=" * 66)
        print("Kimi 讲评结果：")
        print("=" * 66)
        print(reply)
        print("\n" + "=" * 66)
        print(f"[结果] 已保存: {reply_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
