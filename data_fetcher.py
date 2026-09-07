"""
K线训练营 - 数据采集脚本
从 Tushare Pro / AKShare 获取A股历史K线数据，保存为JSON格式
"""

import json
import os
import time
import sys

# 加载配置
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# 确保数据目录存在
os.makedirs(config.DATA_DIR, exist_ok=True)
os.makedirs(config.KLINE_DIR, exist_ok=True)


# ==========================================
# 数据源：Tushare Pro
# ==========================================
# 预设的优质股票列表（涵盖各行业龙头，确保API可用性）
PRESET_STOCKS = [
    ('000001', '平安银行'), ('000002', '万科A'), ('000063', '中兴通讯'),
    ('000100', 'TCL科技'), ('000157', '中联重科'), ('000166', '申万宏源'),
    ('000333', '美的集团'), ('000338', '潍柴动力'), ('000425', '徐工机械'),
    ('000538', '云南白药'), ('000568', '泸州老窖'), ('000596', '古井贡酒'),
    ('000625', '长安汽车'), ('000651', '格力电器'), ('000661', '长春高新'),
    ('000725', '京东方A'), ('000768', '中航西飞'), ('000776', '广发证券'),
    ('000792', '盐湖股份'), ('000800', '一汽解放'), ('000858', '五粮液'),
    ('000938', '紫光股份'), ('000963', '华东医药'), ('002007', '华兰生物'),
    ('002049', '紫光国微'), ('002129', '中环股份'), ('002142', '宁波银行'),
    ('002179', '中航光电'), ('002230', '科大讯飞'), ('002236', '大华股份'),
    ('002241', '歌尔股份'), ('002304', '洋河股份'), ('002311', '海大集团'),
    ('002352', '顺丰控股'), ('002371', '北方华创'), ('002410', '广联达'),
    ('002415', '海康威视'), ('002459', '晶澳科技'), ('002466', '天齐锂业'),
    ('002475', '立讯精密'), ('002493', '荣盛石化'), ('002555', '三七互娱'),
    ('002594', '比亚迪'), ('002601', '龙佰集团'), ('002602', '世纪华通'),
    ('002648', '卫星化学'), ('002709', '天赐材料'), ('002714', '牧原股份'),
    ('002812', '恩捷股份'), ('002821', '凯莱英'), ('300014', '亿纬锂能'),
    ('300015', '爱尔眼科'), ('300033', '同花顺'), ('300059', '东方财富'),
    ('300122', '智飞生物'), ('300124', '汇川技术'), ('300274', '阳光电源'),
    ('300308', '中际旭创'), ('300347', '泰格医药'), ('300408', '三环集团'),
    ('300413', '芒果超媒'), ('300450', '先导智能'), ('300496', '中科创达'),
    ('300502', '新易盛'), ('300628', '亿联网络'), ('300661', '圣邦股份'),
    ('300676', '华大基因'), ('300724', '捷佳伟创'), ('300750', '宁德时代'),
    ('300751', '迈为股份'), ('300759', '康龙化成'), ('300760', '迈瑞医疗'),
    ('300763', '锦浪科技'), ('300782', '卓胜微'), ('300832', '新产业'),
    ('300896', '爱美客'), ('300919', '中伟股份'), ('300957', '贝泰妮'),
    ('300999', '金龙鱼'), ('301269', '华大九天'), ('600000', '浦发银行'),
    ('600009', '上海机场'), ('600010', '包钢股份'), ('600011', '华能国际'),
    ('600015', '华夏银行'), ('600016', '民生银行'), ('600019', '宝钢股份'),
    ('600028', '中国石化'), ('600029', '南方航空'), ('600030', '中信证券'),
    ('600031', '三一重工'), ('600036', '招商银行'), ('600048', '保利发展'),
    ('600050', '中国联通'), ('600061', '国投资本'), ('600085', '同仁堂'),
    ('600089', '特变电工'), ('600104', '上汽集团'), ('600111', '北方稀土'),
    ('600115', '中国东航'), ('600150', '中国船舶'), ('600176', '中国巨石'),
    ('600188', '兖矿能源'), ('600196', '复星医药'), ('600219', '南山铝业'),
    ('600276', '恒瑞医药'), ('600309', '万华化学'), ('600340', '华夏幸福'),
    ('600346', '恒力石化'), ('600362', '江西铜业'), ('600383', '金地集团'),
    ('600406', '国电南瑞'), ('600426', '华鲁恒升'), ('600436', '片仔癀'),
    ('600438', '通威股份'), ('600482', '中国动力'), ('600487', '亨通光电'),
    ('600489', '中金黄金'), ('600519', '贵州茅台'), ('600522', '中天科技'),
    ('600547', '山东黄金'), ('600570', '恒生电子'), ('600585', '海螺水泥'),
    ('600588', '用友网络'), ('600600', '青岛啤酒'), ('600660', '福耀玻璃'),
    ('600690', '海尔智家'), ('600703', '三安光电'), ('600741', '华域汽车'),
    ('600745', '闻泰科技'), ('600754', '锦江酒店'), ('600760', '中航沈飞'),
    ('600795', '国电电力'), ('600809', '山西汾酒'), ('600837', '海通证券'),
    ('600845', '宝信软件'), ('600887', '伊利股份'), ('600893', '航发动力'),
    ('600900', '长江电力'), ('600905', '三峡能源'), ('600918', '中泰证券'),
    ('600919', '江苏银行'), ('600926', '杭州银行'), ('600941', '中国移动'),
    ('600958', '东方证券'), ('600989', '宝丰能源'), ('600999', '招商证券'),
    ('601006', '大秦铁路'), ('601009', '南京银行'), ('601012', '隆基绿能'),
    ('601066', '中信建投'), ('601088', '中国神华'), ('601100', '恒立液压'),
    ('601111', '中国国航'), ('601117', '中国化学'), ('601127', '塞力斯'),
    ('601138', '工业富联'), ('601166', '兴业银行'), ('601168', '西部矿业'),
    ('601169', '北京银行'), ('601186', '中国铁建'), ('601211', '国泰君安'),
    ('601225', '陕西煤业'), ('601229', '上海银行'), ('601236', '红塔证券'),
    ('601238', '广汽集团'), ('601288', '农业银行'), ('601318', '中国平安'),
    ('601319', '中国人保'), ('601328', '交通银行'), ('601336', '新华保险'),
    ('601360', '三六零'), ('601377', '兴业证券'), ('601390', '中国中铁'),
    ('601398', '工商银行'), ('601601', '中国太保'), ('601607', '上海医药'),
    ('601615', '明阳智能'), ('601618', '中国中冶'), ('601628', '中国人寿'),
    ('601633', '长城汽车'), ('601658', '邮储银行'), ('601668', '中国建筑'),
    ('601669', '中国电建'), ('601688', '华泰证券'), ('601689', '拓普集团'),
    ('601698', '中国卫通'), ('601727', '上海电气'), ('601728', '中国电信'),
    ('601766', '中国中车'), ('601788', '光大证券'), ('601800', '中国交建'),
    ('601816', '京沪高铁'), ('601818', '光大银行'), ('601857', '中国石油'),
    ('601877', '正泰电器'), ('601878', '浙商证券'), ('601881', '中国银河'),
    ('601888', '中国中免'), ('601899', '紫金矿业'), ('601901', '方正证券'),
    ('601916', '浙商银行'), ('601919', '中远海控'), ('601939', '建设银行'),
    ('601985', '中国核电'), ('601988', '中国银行'), ('601989', '中国重工'),
    ('601990', '南京证券'), ('601995', '中金公司'),
    ('603259', '药明康德'), ('603288', '海天味业'), ('603501', '韦尔股份'),
    ('603799', '华友钴业'), ('603833', '欧派家居'), ('603899', '晨光股份'),
    ('603986', '兆易创新'), ('688008', '澜起科技'), ('688012', '中微公司'),
    ('688036', '传音控股'), ('688111', '金山办公'), ('688126', '沪硅产业'),
    ('688169', '石头科技'), ('688185', '康希诺'), ('688187', '时代电气'),
    ('688188', '柏楚电子'), ('688223', '晶科能源'), ('688256', '寒武纪'),
    ('688271', '联影医疗'), ('688303', '大全能源'), ('688347', '华虹公司'),
    ('688396', '华润微'), ('688536', '思瑞浦'), ('688561', '奇安信'),
    ('688599', '天合光能'), ('688660', '电气风电'), ('688728', '格科微'),
    ('688772', '珠海冠宇'), ('688776', '国光电气'), ('688981', '中芯国际'),
]

def fetch_via_tushare():
    """使用 Tushare Pro 获取数据（使用预设股票列表，确保兼容性）"""
    import tushare as ts

    token = getattr(config, 'TUSHARE_TOKEN', None)
    if not token or token == '你的Tushare Pro Token':
        print('[Tushare] 未配置 Token，跳过')
        return False, []

    print('[Tushare] 正在连接...')
    ts.set_token(token)
    pro = ts.pro_api()

    # 验证Token是否有效
    try:
        df = pro.daily(ts_code='000001.SZ', start_date='20240801', end_date='20240810')
        if df.empty:
            print('[Tushare] Token验证失败')
            return False, []
        print(f'[Tushare] Token验证成功')
    except Exception as e:
        print(f'[Tushare] Token验证失败: {e}')
        return False, []

    # 使用预设股票列表
    stocks = []
    for code, name in PRESET_STOCKS:
        suffix = 'SH' if code.startswith('6') or code.startswith('68') else 'SZ'
        stocks.append({
            'code': code,
            'ts_code': f'{code}.{suffix}',
            'name': name
        })

    print(f'[Tushare] 准备采集 {len(stocks)} 只股票')
    return True, stocks


# ==========================================
# 数据源：AKShare（开源免费，无需Token）
# ==========================================
def fetch_via_akshare():
    """使用 AKShare 获取数据"""
    import akshare as ak

    print('[AKShare] 获取股票列表...')

    # 获取A股股票列表
    stock_df = ak.stock_info_a_code_name()
    stocks = []
    for _, row in stock_df.iterrows():
        code = str(row['code'])
        # 过滤：只保留沪深主板、创业板、科创板
        if code.startswith('6') or code.startswith('0') or code.startswith('3'):
            stocks.append({
                'code': code,
                'name': row['name'],
                'ts_code': f"{code}.{'SH' if code.startswith('6') else 'SZ'}"
            })

    print(f'[AKShare] 获取到 {len(stocks)} 只股票')
    return True, stocks


def download_kline(stock, data_source='tushare'):
    """下载单只股票的K线数据"""
    code = stock['code']
    ts_code = stock.get('ts_code', '')
    name = stock['name']
    filepath = os.path.join(config.KLINE_DIR, f'{code}.json')

    # 如果已存在则跳过
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if len(data.get('kline', [])) > 100:
                print(f'  [跳过] {code} {name} (已存在)')
                return True
        except:
            pass

    try:
        if data_source == 'tushare':
            result = download_kline_tushare(ts_code)
        else:
            result = download_kline_akshare(code, name)

        if result:
            kline_data, start_date = result
            # 保存数据
            stock_data = {
                'code': code,
                'name': name,
                'start_date': start_date,
                'kline': kline_data
            }
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(stock_data, f, ensure_ascii=False, indent=2)
            print(f'  [OK] {code} {name} ({len(kline_data)}条K线)')
            return True
        else:
            print(f'  [失败] {code} {name}')
            return False
    except Exception as e:
        print(f'  [错误] {code} {name}: {e}')
        return False


def download_kline_tushare(ts_code):
    """通过 Tushare 下载K线"""
    import tushare as ts
    pro = ts.pro_api()

    end_date = time.strftime('%Y%m%d')
    years = config.HISTORY_YEARS
    start_year = int(time.strftime('%Y')) - years
    start_date = f'{start_year}0101'

    df = pro.daily(ts_code=ts_code,
                   start_date=start_date,
                   end_date=end_date)

    if df.empty:
        return None

    # 按日期升序排列
    df = df.sort_values('trade_date')

    kline = []
    for _, row in df.iterrows():
        kline.append({
            'date': row['trade_date'],
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': float(row['vol']),
            'amount': float(row['amount']) if 'amount' in row else 0
        })

    return kline, start_date


def download_kline_akshare(code, name):
    """通过 AKShare 下载K线"""
    import akshare as ak

    try:
        # 使用 AKShare 获取历史K线
        symbol = f"{code}"  # AKShare 使用纯数字代码
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                start_date=f"{time.strftime('%Y')-config.HISTORY_YEARS}0101",
                                end_date=time.strftime('%Y%m%d'),
                                adjust="qfq")  # 前复权

        if df.empty:
            return None

        kline = []
        for _, row in df.iterrows():
            kline.append({
                'date': str(row['日期']),
                'open': float(row['开盘']),
                'high': float(row['最高']),
                'low': float(row['最低']),
                'close': float(row['收盘']),
                'volume': float(row['成交量']),
                'amount': float(row['成交额']) if '成交额' in row else 0
            })

        return kline, f"{time.strftime('%Y')-config.HISTORY_YEARS}0101"
    except Exception as e:
        print(f'  AKShare获取失败: {e}')
        return None


def generate_stock_list_json(stocks):
    """生成前端用的股票列表JSON"""
    stock_list = []
    for s in stocks:
        code = s['code']
        filepath = os.path.join(config.KLINE_DIR, f'{code}.json')
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if len(data.get('kline', [])) >= config.TRAIN_KLINE_COUNT:
                    stock_list.append({
                        'code': code,
                        'name': s['name'],
                        'count': len(data['kline'])
                    })
            except:
                pass

    # 保存到前端数据目录
    web_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'web', 'data')
    os.makedirs(web_data_dir, exist_ok=True)

    output_path = os.path.join(web_data_dir, 'stocks.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stock_list, f, ensure_ascii=False, indent=2)

    # 也保存到 data 目录
    data_output = os.path.join(config.DATA_DIR, 'stock_list.json')
    with open(data_output, 'w', encoding='utf-8') as f:
        json.dump(stock_list, f, ensure_ascii=False, indent=2)

    print(f'\n股票列表已保存: {output_path}')
    print(f'可用股票数量: {len(stock_list)}')
    return stock_list


def main():
    """主函数"""
    print('=' * 50)
    print('K线训练营 - 数据采集工具')
    print('=' * 50)

    # 尝试使用 Tushare Pro
    stocks = []
    used_source = ''

    success, stocks = fetch_via_tushare()
    if success:
        used_source = 'tushare'
    else:
        print('\n尝试使用 AKShare...')
        success, stocks = fetch_via_akshare()
        if success:
            used_source = 'akshare'

    if not stocks:
        print('\n错误：无法获取股票列表！')
        print('请确保已安装依赖: pip install tushare akshare pandas')
        print('如果使用 Tushare，请先在 config.py 中配置 Token')
        sys.exit(1)

    print(f'\n使用数据源: {used_source}')
    print(f'开始下载K线数据 ({config.HISTORY_YEARS}年)...')

    # 限制股票数量，避免请求过多（免费版积分有限）
    max_stocks = min(len(stocks), 100)
    stocks = stocks[:max_stocks]

    success_count = 0
    for i, stock in enumerate(stocks):
        print(f'[{i+1}/{max_stocks}] ', end='')
        if download_kline(stock, used_source):
            success_count += 1
        # 请求间隔，避免被限流（Tushare免费版 50次/分钟）
        time.sleep(1.5)

    print(f'\n下载完成: {success_count}/{max_stocks} 只股票成功')

    # 生成股票列表
    stock_list = generate_stock_list_json(stocks)

    # 统计总K线数量
    total_kline = sum(s['count'] for s in stock_list)
    print(f'总K线数据量: {total_kline} 条')

    print('\n[完成] 数据采集完成！')
    print(f'打开 web/index.html 开始训练')


if __name__ == '__main__':
    main()