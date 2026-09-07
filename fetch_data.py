"""下载 A 股日线数据（Tushare），保存为 JSON（自包含，读取本项目 config.py）。

用法：
    python fetch_data.py                          # 全量下载（默认 3 年）
    python fetch_data.py --max-stocks 10 --interval 0.2   # 小规模试跑
    python fetch_data.py --years 5                # 下载 5 年
    python fetch_data.py --force                  # 强制重新下载
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / config.DATA_DIR
KLINE_DIR = DATA_DIR / config.KLINE_DIR


def get_token() -> str:
    token = os.environ.get("TUSHARE_TOKEN", "").strip()
    if token:
        return token
    return str(getattr(config, "TUSHARE_TOKEN", "")).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="下载 A 股日线数据（Tushare）")
    parser.add_argument("--years", type=int, default=3)
    parser.add_argument("--interval", type=float, default=1.5)
    parser.add_argument("--max-stocks", type=int, help="测试用限制；默认不限制")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    token = get_token()
    if not token or token == "你的Tushare Pro Token":
        print("错误: 请先在 config.py 中填入 TUSHARE_TOKEN")
        return 1
    try:
        import tushare as ts
        ts.set_token(token)
        pro = ts.pro_api()
        stock_frame = pro.stock_basic(exchange="", list_status="L", fields="ts_code,symbol,name")
    except ImportError:
        print("错误: ai_env 中未安装 tushare，请执行 pip install -r requirements.txt")
        return 1
    except Exception as error:
        print(f"错误: 获取股票列表失败: {error}")
        return 1

    stocks = []
    for row in stock_frame.to_dict("records"):
        code = str(row["symbol"]).zfill(6)
        if code.startswith(("0", "3", "6")):
            stocks.append({"code": code, "ts_code": row["ts_code"], "name": str(row["name"])})
    if args.max_stocks:
        stocks = stocks[: args.max_stocks]

    KLINE_DIR.mkdir(parents=True, exist_ok=True)
    end = date.today()
    start = end - timedelta(days=args.years * 366)

    success = 0
    total = 0
    for index, stock in enumerate(stocks, 1):
        output = KLINE_DIR / f"{stock['code']}.json"
        if output.exists() and not args.force:
            success += 1
            print(f"[{index}/{len(stocks)}] 跳过 {stock['code']}（已有数据）")
            continue
        try:
            frame = pro.daily(ts_code=stock["ts_code"],
                              start_date=start.strftime("%Y%m%d"),
                              end_date=end.strftime("%Y%m%d"))
            frame = frame.sort_values("trade_date")
            rows = [{"date": str(row["trade_date"]), "open": float(row["open"]),
                     "high": float(row["high"]), "low": float(row["low"]),
                     "close": float(row["close"]), "volume": float(row["vol"]),
                     "amount": float(row.get("amount", 0) or 0)}
                    for row in frame.to_dict("records")]
            if rows:
                output.write_text(json.dumps({"code": stock["code"], "name": stock["name"],
                                              "start_date": start.strftime("%Y%m%d"),
                                              "kline": rows}, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
                success += 1
                total += len(rows)
                print(f"[{index}/{len(stocks)}] 完成 {stock['code']} {stock['name']}（{len(rows)} 条）")
            else:
                print(f"[{index}/{len(stocks)}] 无数据 {stock['code']}")
        except Exception as error:
            print(f"[{index}/{len(stocks)}] 失败 {stock['code']}: {error}")
        time.sleep(max(0, args.interval))

    stock_list = []
    for stock in stocks:
        output = KLINE_DIR / f"{stock['code']}.json"
        if output.exists():
            try:
                count = len(json.loads(output.read_text(encoding="utf-8")).get("kline", []))
                if count >= 5:
                    stock_list.append({"code": stock["code"], "name": stock["name"], "count": count})
            except (OSError, json.JSONDecodeError):
                pass
    (DATA_DIR / "stock_list.json").write_text(
        json.dumps(stock_list, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"完成: {success}/{len(stocks)} 只成功，本次写入 {total} 条；当前可用 {len(stock_list)} 只")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
