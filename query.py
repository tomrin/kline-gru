"""查询：某股票某时间段属于 K-means 的哪个簇。

用法（两种方式二选一）：
    1) 指定起止日期（范围内交易日数须等于 WINDOW）：
       python query.py 000001 --start-date 20240801 --end-date 20240821
    2) 只指定结束日期（取截至该日最近的 WINDOW 个交易日）：
       python query.py 000001 --end-date 20240821

可选：
    --top N   同时展示最接近的 N 个簇（默认 3）
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


def select_window(rows, start, end):
    """取 [start, end] 内的交易日。"""
    return [r for r in rows if start <= str(r["date"]) <= end]


def main() -> int:
    parser = argparse.ArgumentParser(description="查询股票某时间段所属的 K-means 簇")
    parser.add_argument("code", help="股票代码，例如 000001")
    parser.add_argument("--start-date", help="起始日期 YYYYMMDD")
    parser.add_argument("--end-date", help="结束日期 YYYYMMDD")
    parser.add_argument("--top", type=int, default=3, help="展示最接近的 N 个簇")
    args = parser.parse_args()

    code = args.code.zfill(6)
    if not args.end_date:
        print("错误: 至少需要 --end-date")
        return 1

    data_dir = ROOT / config.KLINE_DIR
    model_pack = load_model()
    model = model_pack["model"]
    stats = {s["cluster_id"]: s for s in model_pack["stats"]}
    window = model_pack["window"]

    loaded = cl.load_stock_file(data_dir / f"{code}.json")
    if loaded is None:
        print(f"错误: 未找到股票数据 {code}，请先运行 python fetch_data.py")
        return 1
    _, name, rows = loaded

    end = parse_date(args.end_date)
    if args.start_date:
        start = parse_date(args.start_date)
        win_rows = select_window(rows, start, end)
        if len(win_rows) != window:
            print(f"错误: {start}~{end} 之间只有 {len(win_rows)} 个交易日，"
                  f"而模型窗口固定为 {window} 个交易日。")
            print("提示: 调整日期范围，或改用 --end-date 单参数模式（自动取最近 N 个交易日）。")
            return 1
    else:
        # 取截至 end 的最近 window 个交易日
        dates = [r["date"] for r in rows if str(r["date"]) <= end]
        if len(dates) < window:
            print(f"错误: 截至 {end} 只有 {len(dates)} 个交易日，不足 {window} 个。")
            return 1
        start = dates[-window]
        win_rows = select_window(rows, start, end)
        if len(win_rows) != window:
            win_rows = [r for r in rows if start <= str(r["date"]) <= end][-window:]

    vec = np.asarray(feature_vector(win_rows), dtype=np.float32).reshape(1, -1)
    cluster_id = int(model.predict(vec)[0])
    dists = np.linalg.norm(model.cluster_centers_ - vec, axis=1)
    nearest = np.argsort(dists)[: args.top]

    s = stats[cluster_id]
    avg = s["avg_forward_return"]
    wr = s["win_rate"]
    exc = s["avg_excess_return"]
    ewr = s["excess_win_rate"]
    avg_txt = f"{avg*100:+.2f}%" if avg is not None else "N/A（样本前向数据不足）"
    wr_txt = f"{wr*100:.1f}%" if wr is not None else "N/A"
    exc_txt = f"{exc*100:+.2f}%" if exc is not None else "N/A（无市场基准或样本不足）"
    ewr_txt = f"{ewr*100:.1f}%" if ewr is not None else "N/A"

    print("\n" + "=" * 60)
    print(f"查询结果：{code} {name} | {win_rows[0]['date']} 至 {win_rows[-1]['date']}（{len(win_rows)} 个交易日）")
    print("=" * 60)
    print(f"所属簇        : #{cluster_id}")
    print(f"距离簇中心    : {dists[cluster_id]:.4f}（越小越典型）")
    print(f"簇样本数      : {s['size']}（占全部 {s['size_pct']}%）")
    print(f"历史超额均收  : {exc_txt}（相对等权市场基准，{config.FORWARD_DAYS} 个交易日）")
    print(f"超额胜率      : {ewr_txt}（超额收益>0 占比）")
    print(f"历史绝对均收  : {avg_txt}（{config.FORWARD_DAYS} 个交易日，未基准化）")
    print(f"绝对胜率      : {wr_txt}（前向收益>0 占比）")
    print(f"代表形态      : {s['rep_name']}({s['rep_code']}) {s['rep_start']}-{s['rep_end']}")
    print("\n最接近的簇（距离从小到大）:")
    for rank, cid in enumerate(nearest, 1):
        cs = stats[int(cid)]
        a = cs["avg_forward_return"]
        e = cs["avg_excess_return"]
        a_txt = f"{a*100:+.2f}%" if a is not None else "N/A"
        e_txt = f"{e*100:+.2f}%" if e is not None else "N/A"
        mark = " ◀ 命中" if int(cid) == cluster_id else ""
        print(f"  {rank}. 簇#{cid}  距离={dists[cid]:.4f}  超额均收={e_txt}  绝对均收={a_txt}  样本={cs['size']}{mark}")

    # 生成图表
    out_dir = ROOT / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    rep_rows = chart._load_rep_rows(s, data_dir, window)
    chart_path = out_dir / f"query_{code}_{start}_{end}.png"
    chart.render_query(win_rows, rep_rows, cluster_id, chart_path,
                       title=f"{code} {name} {start}-{end}")
    print(f"\n[图表] 查询结果图: {chart_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
