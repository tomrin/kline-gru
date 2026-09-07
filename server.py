"""
K线训练营 - HTTP服务器
支持多客户端并发、AI讲评接口、日志记录
"""

import http.server
import socketserver
import os
import json
import sys
import logging
import threading
from datetime import datetime
from logging.handlers import RotatingFileHandler

# 加载配置
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from engines import get_retriever, get_llm_provider

# ==========================================
# 日志配置
# ==========================================
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# 服务器日志
server_log = logging.getLogger('server')
server_log.setLevel(logging.INFO)
server_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, 'server.log'),
    maxBytes=1024 * 1024 * 5,  # 5MB
    backupCount=3,
    encoding='utf-8'
)
server_handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
server_log.addHandler(server_handler)
# 同时输出到控制台
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S'))
server_log.addHandler(console_handler)

# AI讲评日志
ai_log = logging.getLogger('ai_commentary')
ai_log.setLevel(logging.INFO)
ai_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, 'ai_commentary.log'),
    maxBytes=1024 * 1024 * 10,  # 10MB
    backupCount=5,
    encoding='utf-8'
)
ai_handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
ai_log.addHandler(ai_handler)

PORT = 8200
DIRECTORY = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(DIRECTORY, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# ==========================================
# 用户与记录系统
# ==========================================
USERS_FILE = os.path.join(DATA_DIR, 'users.json')
RECORDS_FILE = os.path.join(DATA_DIR, 'records.json')

def load_json(filepath, default=None):
    if default is None: default = {}
    if not os.path.exists(filepath):
        return default
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return default

def save_json(filepath, data):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_users():
    return load_json(USERS_FILE, {})

def save_users(users):
    save_json(USERS_FILE, users)

def get_records():
    return load_json(RECORDS_FILE, [])

def save_records(records):
    save_json(RECORDS_FILE, records)


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def log_message(self, format, *args):
        """将请求日志写入文件"""
        msg = format % args
        server_log.info('%s - %s', self.client_address[0], msg)

    def do_GET(self):
        """处理GET请求"""
        self.log_message('GET %s', self.path)
        # 三层引擎元信息（供前端显示各层引擎名与耗时）
        if self.path == '/api/engines':
            self._handle_engines()
            return
        # 根路径与 /index.html 直接重定向到训练营前端入口
        if self.path in ('/', '/index.html', '/index.htm'):
            self.send_response(302)
            self.send_header('Location', '/web/index.html')
            self.end_headers()
            return
        super().do_GET()

    def _handle_engines(self):
        """返回三层讲评引擎的元信息（规则引擎/相似形态检索/深度讲评）"""
        retriever = get_retriever()
        llm = get_llm_provider()
        response = {
            'success': True,
            'layers': {
                'rule': {'name': 'rule', 'label': '规则引擎', 'latency': '毫秒级'},
                'retriever': retriever.meta(),
                'llm': llm.meta(),
            }
        }
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.end_headers()
        self.wfile.write(json.dumps(response, ensure_ascii=False).encode('utf-8'))

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        # 静态资源强制 no-cache，避免浏览器缓存旧代码（去掉 query string 再判断）
        pure_path = self.path.split('?')[0]
        if pure_path.endswith(('.html', '.htm', '.js', '.css')):
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Expires', '0')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_POST(self):
        if self.path == '/api/commentary':
            self.log_message('POST /api/commentary')
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))

            kline_data = data.get('klineData', [])
            current_index = data.get('currentIndex', 0)
            position_info = data.get('positionInfo', '空仓')
            ai_learning = data.get('aiLearning', '')
            stock_code = data.get('stockCode', '') or ''

            # 编排：先证据层检索相似形态，再讲解层生成深度讲评（引擎由 config 选择，可插拔）
            retriever = get_retriever()
            llm = get_llm_provider()
            knn_cases = retriever.retrieve(kline_data, stock_code, top=5)
            result = llm.analyze(kline_data, current_index, position_info, ai_learning,
                                 stock_code, knn_cases)

            if result:
                response = {'success': True, 'content': result}
            else:
                response = {'success': False, 'message': 'AI分析服务暂不可用，请稍后重试'}

            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(response, ensure_ascii=False).encode('utf-8'))

        elif self.path == '/api/knn':
            # 秒级返回 KNN 相似历史形态（不依赖 Kimi，用于即时呈现）
            self.log_message('POST /api/knn')
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            kline_data = data.get('klineData', [])
            stock_code = data.get('stockCode', '') or ''
            cases = get_retriever().retrieve(kline_data, stock_code, top=5)
            if cases:
                response = {
                    'success': True,
                    'cases': [
                        {
                            'code': m.get('code', ''),
                            'name': m.get('name', ''),
                            'start': m.get('start', ''),
                            'end': m.get('end', ''),
                            'forward': m.get('forward', 0),
                            'excess': m.get('excess', 0),
                            'distance': round(d, 4)
                        }
                        for d, m in cases
                    ]
                }
            else:
                response = {'success': False, 'message': '检索库缺失或当前K线数据不足（需≥15根）'}
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(response, ensure_ascii=False).encode('utf-8'))

        elif self.path == '/api/register':
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            username = data.get('username', '').strip()
            if not username:
                response = {'success': False, 'message': '用户名不能为空'}
            else:
                users = get_users()
                if username in users:
                    response = {'success': False, 'message': '用户名已存在'}
                else:
                    users[username] = {'registerTime': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    save_users(users)
                    response = {'success': True, 'message': '注册成功', 'username': username}
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(response, ensure_ascii=False).encode('utf-8'))

        elif self.path == '/api/login':
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            username = data.get('username', '').strip()
            users = get_users()
            if username in users:
                response = {'success': True, 'message': '登录成功', 'username': username}
            else:
                response = {'success': False, 'message': '用户不存在，请先注册'}
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(response, ensure_ascii=False).encode('utf-8'))

        elif self.path == '/api/save_record':
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            records = get_records()
            record = {
                'id': len(records) + 1,
                'username': data.get('username', '匿名'),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'stockName': data.get('stockName', ''),
                'stockCode': data.get('stockCode', ''),
                'stockChange': data.get('stockChange', 0),
                'returnRate': data.get('returnRate', 0),
                'winRate': data.get('winRate', 0),
                'tradeCount': data.get('tradeCount', 0),
                'maxDrawdown': data.get('maxDrawdown', 0),
                'totalReturn': data.get('totalReturn', 0)
            }
            records.append(record)
            save_records(records)
            response = {'success': True, 'message': '保存成功'}
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(response, ensure_ascii=False).encode('utf-8'))

        elif self.path == '/api/records':
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            username = data.get('username', '')
            records = get_records()
            user_records = [r for r in records if r['username'] == username]
            response = {'success': True, 'records': user_records[-50:]}  # 最近50条
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(response, ensure_ascii=False).encode('utf-8'))

        elif self.path == '/api/leaderboard':
            records = get_records()
            # 按用户汇总
            user_stats = {}
            for r in records:
                u = r['username']
                if u not in user_stats:
                    user_stats[u] = {'totalReturn': 0, 'rounds': 0, 'wins': 0, 'totalTrades': 0}
                user_stats[u]['totalReturn'] += r.get('totalReturn', 0)
                user_stats[u]['rounds'] += 1
                if r.get('returnRate', 0) > 0:
                    user_stats[u]['wins'] += 1
                user_stats[u]['totalTrades'] += r.get('tradeCount', 0)

            leaderboard = []
            for u, s in user_stats.items():
                winRate = s['wins'] / s['rounds'] * 100 if s['rounds'] > 0 else 0
                leaderboard.append({
                    'username': u,
                    'totalReturn': round(s['totalReturn'], 2),
                    'rounds': s['rounds'],
                    'winRate': round(winRate, 1),
                    'totalTrades': s['totalTrades']
                })
            leaderboard.sort(key=lambda x: x['totalReturn'], reverse=True)
            response = {'success': True, 'leaderboard': leaderboard[:20]}
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(response, ensure_ascii=False).encode('utf-8'))

        else:
            self.send_response(404)
            self.end_headers()


if __name__ == '__main__':
    os.chdir(DIRECTORY)
    print('=' * 50)
    print('  K线训练营 - 服务器已启动')
    print('=' * 50)
    print('  前端地址: http://localhost:' + str(PORT) + '/web/index.html')
    print('  API接口:  http://localhost:' + str(PORT) + '/api/commentary')
    print('  日志文件: ' + os.path.join(LOG_DIR, 'server.log'))
    print('  AI讲评日志: ' + os.path.join(LOG_DIR, 'ai_commentary.log'))
    print('  多客户端模式: 已启用')
    print('  按 Ctrl+C 停止服务器')
    print('=' * 50)

    # 使用ThreadingTCPServer支持多客户端并发
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    httpd = socketserver.ThreadingTCPServer(('', PORT), Handler)
    httpd.socket.settimeout(None)  # 无超时限制

    # 后台预热相似形态检索库（不阻塞服务启动，用户首次讲评时大概率已就绪）
    def _warmup_retriever():
        get_retriever().warmup()
    threading.Thread(target=_warmup_retriever, daemon=True).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print('\n服务器已停止')
        httpd.server_close()