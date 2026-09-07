"""
K线训练营 - 示例数据生成器
生成模拟K线数据，用于在无API时测试项目功能
"""
import json, os, math, random
from datetime import datetime, timedelta

# 自包含路径：以本脚本所在目录为基准
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KLINE_DIR = os.path.join(BASE_DIR, 'data', 'kline')
WEB_DATA_DIR = os.path.join(BASE_DIR, 'web', 'data')

os.makedirs(KLINE_DIR, exist_ok=True)
os.makedirs(WEB_DATA_DIR, exist_ok=True)

def generate_sample_kline(code, name, days=800):
    seed = hash(code) % 10000
    random.seed(seed)
    price = 10.0 + random.random() * 80
    trend = random.choice(['up', 'down', 'sideways', 'volatile'])
    trend_strength = random.uniform(0.0001, 0.003)
    kline = []
    date = datetime(2020, 1, 2)
    i = 0
    while date <= datetime(2025, 12, 31) and len(kline) < days:
        if date.weekday() >= 5:
            date += timedelta(days=1)
            continue
        trend_factor = 0
        if trend == 'up': trend_factor = trend_strength
        elif trend == 'down': trend_factor = -trend_strength
        elif trend == 'volatile': trend_factor = random.uniform(-0.005, 0.005)
        daily_change = random.gauss(0, 0.02) + trend_factor
        open_price = price * (1 + random.gauss(0, 0.006))
        close_price = open_price * (1 + daily_change)
        high_price = max(open_price, close_price) * (1 + abs(random.gauss(0, 0.01)))
        low_price = min(open_price, close_price) * (1 - abs(random.gauss(0, 0.01)))
        low_price = max(low_price, 0.5)
        volume = random.randint(500000, 50000000) * (1 + abs(daily_change) * 20)
        kline.append({
            'date': date.strftime('%Y%m%d'),
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': round(volume)
        })
        price = close_price
        date += timedelta(days=1)
        i += 1
    return kline

def main():
    stocks = [
        ('000001', '平安银行'), ('000002', '万科A'), ('000333', '美的集团'),
        ('000651', '格力电器'), ('000858', '五粮液'), ('002415', '海康威视'),
        ('002475', '立讯精密'), ('300750', '宁德时代'), ('600000', '浦发银行'),
        ('600036', '招商银行'), ('600104', '上汽集团'), ('600276', '恒瑞医药'),
        ('600519', '贵州茅台'), ('600887', '伊利股份'), ('601166', '兴业银行'),
        ('601318', '中国平安'), ('601398', '工商银行'), ('601857', '中国石油'),
        ('603259', '药明康德'), ('688981', '中芯国际'),
    ]
    stock_list = []
    for code, name in stocks:
        print(f'Generating {code} {name}...')
        kline = generate_sample_kline(code, name)
        data = {'code': code, 'name': name, 'kline': kline}
        with open(os.path.join(KLINE_DIR, f'{code}.json'), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        stock_list.append({'code': code, 'name': name, 'count': len(kline)})

    # Save stock list
    with open(os.path.join(WEB_DATA_DIR, 'stocks.json'), 'w', encoding='utf-8') as f:
        json.dump(stock_list, f, ensure_ascii=False, indent=2)
    with open(os.path.join(BASE_DIR, 'data', 'stock_list.json'), 'w', encoding='utf-8') as f:
        json.dump(stock_list, f, ensure_ascii=False, indent=2)

    print(f'Done! Generated {len(stock_list)} stocks')
    print(f'Kline data: {KLINE_DIR}')
    print(f'Stock list: {WEB_DATA_DIR}')

if __name__ == '__main__':
    main()