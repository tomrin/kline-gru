"""
讲评引擎抽象层（三层架构）

    底座层（规则引擎）  → 在前端本地实现（web/js/kline-analysis.js），毫秒级，不属于本模块
    证据层（Retriever）  → 相似历史形态检索，当前实现 KNN，可插拔其他技术路线（DTW / embedding / 簇中心等）
    讲解层（LLMProvider）→ 大模型深度讲评，OpenAI 兼容协议，可切换任意大模型（Kimi / DeepSeek / Qwen / GPT 等）

通过 config.RETRIEVER 与 config.LLM_PROVIDER 选择当前引擎；
新增引擎只需继承对应基类、在注册表中登记，并在 config 中切换。
"""

import os
import json
import time
import threading
import logging
import urllib.request

import numpy as np

import config
from features import feature_vector

_log = logging.getLogger('server')
_ai_log = logging.getLogger('ai_commentary')

_DIR = os.path.dirname(os.path.abspath(__file__))
KNN_X_PATH = os.path.join(_DIR, 'models', 'knn_X.npy')
KNN_META_PATH = os.path.join(_DIR, 'models', 'knn_meta.json')

GRU_MODEL_PATH = os.path.join(_DIR, 'models', 'gru.pt')
GRU_X_PATH = os.path.join(_DIR, 'models', 'gru_X.npy')
GRU_META_PATH = os.path.join(_DIR, 'models', 'gru_meta.json')

BYOL_MODEL_PATH = os.path.join(_DIR, 'models', 'byol.pt')
BYOL_X_PATH = os.path.join(_DIR, 'models', 'byol_X.npy')
BYOL_META_PATH = os.path.join(_DIR, 'models', 'byol_meta.json')


def _load_gru_model(path):
    """惰性加载 GRU 模型（只在启用 GRU 检索器时引入 torch）。"""
    import torch
    from gru_model import GruEncoder
    ckpt = torch.load(path, map_location='cpu')
    model = GruEncoder(
        input_dim=int(ckpt.get('input_dim', 4)),
        hidden_dim=int(ckpt.get('hidden_dim', 32)),
        num_layers=int(ckpt.get('num_layers', 1)),
    )
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model


# ==========================================
# 证据层：相似历史形态检索器
# ==========================================
class BaseRetriever:
    """相似历史形态检索器基类（证据层）。

    子类需实现 retrieve()，返回相似历史案例列表，格式：
        [(distance, meta_dict), ...]
    meta_dict 至少含 code/name/start/end/forward/excess 字段。
    """
    name = 'base'
    label = '相似形态检索'
    latency = '秒级'

    def retrieve(self, kline_data, code='', top=5):
        raise NotImplementedError

    def warmup(self):
        """服务启动时的预热钩子（可选），用于提前加载检索库。"""
        pass

    def meta(self):
        return {'name': self.name, 'label': self.label, 'latency': self.latency}


class KnnRetriever(BaseRetriever):
    """KNN 检索器：在特征库中做欧氏距离暴力检索，找最相似的历史先例。"""
    name = 'knn'
    label = 'KNN'
    latency = '约1秒'

    _index = None          # (X, metas)
    _lock = threading.Lock()

    def _load_index(self):
        if self._index is not None:
            return self._index
        with self._lock:
            if self._index is not None:
                return self._index
            if not os.path.exists(KNN_X_PATH) or not os.path.exists(KNN_META_PATH):
                _log.warning('KNN 检索库不存在（请先运行 python retrieve.py --build），讲评将不含相似形态')
                return None
            t0 = time.time()
            X = np.load(KNN_X_PATH)
            with open(KNN_META_PATH, encoding='utf-8') as f:
                metas = json.load(f)
            self._index = (X, metas)
            _log.info('KNN 检索库加载完成：%d 个历史窗口，用时 %.1fs', len(metas), time.time() - t0)
            return self._index

    def warmup(self):
        try:
            self._load_index()
        except Exception as e:
            _log.warning('KNN 检索库预热失败: %s', e)

    def retrieve(self, kline_data, code='', top=5):
        window = getattr(config, 'WINDOW', 15)
        if not kline_data or len(kline_data) < window:
            return None

        idx = self._load_index()
        if idx is None:
            return None
        X, metas = idx

        # 目标窗口：最近 window 根 K 线
        win_rows = kline_data[-window:]
        try:
            vec = np.asarray(feature_vector(win_rows), dtype=np.float32).reshape(1, -1)
        except Exception as e:
            _log.warning('KNN 特征计算失败: %s', e)
            return None

        start = str(win_rows[0].get('date', ''))
        end = str(win_rows[-1].get('date', ''))

        # 距离计算 + 过滤（排除自我匹配、只留目标窗口开始日之前的先例）
        dists = np.linalg.norm(X - vec, axis=1)
        order = np.argsort(dists)

        exclude = (code, start, end) if code else None
        out = []
        for i in order:
            m = metas[i]
            if exclude and (m.get('code'), str(m.get('start')), str(m.get('end'))) == exclude:
                continue
            if start and str(m.get('end', '')) >= start:
                continue
            out.append((float(dists[i]), m))
            if len(out) >= top:
                break
        return out


class GruRetriever(BaseRetriever):
    """GRU 检索器（路线2）：用学习到的嵌入做余弦相似检索，找最相似的历史先例。

    与 KnnRetriever 同层级，区别只在相似度度量：
    - KNN：手工特征向量欧氏距离（固定度量）
    - GRU：GRU 编码器嵌入余弦相似度（学习度量，训练目标是预测未来超额收益）

    需先运行 `python gru_train.py` 产出模型与索引。
    """
    name = 'gru'
    label = 'GRU'
    latency = '约1秒'

    _index = None          # (X, metas)
    _model = None
    _lock = threading.Lock()

    def _load_index(self):
        if self._index is not None:
            return self._index
        with self._lock:
            if self._index is not None:
                return self._index
            for path in (GRU_MODEL_PATH, GRU_X_PATH, GRU_META_PATH):
                if not os.path.exists(path):
                    _log.warning('GRU 检索库不完整（请先运行 python gru_train.py），讲评将不含相似形态')
                    return None
            t0 = time.time()
            X = np.load(GRU_X_PATH)
            with open(GRU_META_PATH, encoding='utf-8') as f:
                metas = json.load(f)
            model = _load_gru_model(GRU_MODEL_PATH)
            self._index = (X, metas)
            self._model = model
            _log.info('GRU 检索库加载完成：%d 个历史窗口，用时 %.1fs', len(metas), time.time() - t0)
            return self._index

    def warmup(self):
        try:
            self._load_index()
        except Exception as e:
            _log.warning('GRU 检索库预热失败: %s', e)

    def retrieve(self, kline_data, code='', top=5):
        window = getattr(config, 'WINDOW', 15)
        if not kline_data or len(kline_data) < window:
            return None

        idx = self._load_index()
        if idx is None:
            return None
        X, metas = idx
        model = self._model

        win_rows = kline_data[-window:]
        try:
            import torch
            from gru_model import feature_matrix, cosine_similarity
            mat = feature_matrix(win_rows)
            with torch.no_grad():
                x = torch.from_numpy(mat[None]).float()
                _, emb = model(x)
            emb = emb.detach().numpy().astype(np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-9)
        except Exception as e:
            _log.warning('GRU 特征计算失败: %s', e)
            return None

        start = str(win_rows[0].get('date', ''))
        end = str(win_rows[-1].get('date', ''))

        # 余弦相似度 + 过滤（排除自我匹配、只留目标窗口开始日之前的先例）
        sims = cosine_similarity(emb, X)[0]
        order = np.argsort(-sims)

        exclude = (code, start, end) if code else None
        out = []
        for i in order:
            m = metas[i]
            if exclude and (m.get('code'), str(m.get('start')), str(m.get('end'))) == exclude:
                continue
            if start and str(m.get('end', '')) >= start:
                continue
            out.append((float(sims[i]), m))
            if len(out) >= top:
                break
        return out


class ByolRetriever(BaseRetriever):
    """BYOL 检索器（路线3）：用 BYOL 自监督训练的 GRU 编码器嵌入做余弦相似检索。

    与 GruRetriever 同层级，区别只在训练方式：
    - GRU（路线2）：纯 MSE 监督
    - BYOL（路线3）：MSE 监督 + BYOL 自监督（无负样本、stop-gradient + EMA），
                  嵌入更稳健、防坍缩，跨股票检索把不同股票推得更正交

    需先运行 `python byol_train.py` 产出模型与索引。
    """
    name = 'byol'
    label = 'BYOL'
    latency = '约1秒'

    _index = None          # (X, metas)
    _model = None
    _lock = threading.Lock()

    def _load_index(self):
        if self._index is not None:
            return self._index
        with self._lock:
            if self._index is not None:
                return self._index
            for path in (BYOL_MODEL_PATH, BYOL_X_PATH, BYOL_META_PATH):
                if not os.path.exists(path):
                    _log.warning('BYOL 检索库不完整（请先运行 python byol_train.py），讲评将不含相似形态')
                    return None
            t0 = time.time()
            X = np.load(BYOL_X_PATH)
            with open(BYOL_META_PATH, encoding='utf-8') as f:
                metas = json.load(f)
            model = _load_gru_model(BYOL_MODEL_PATH)
            self._index = (X, metas)
            self._model = model
            _log.info('BYOL 检索库加载完成：%d 个历史窗口，用时 %.1fs', len(metas), time.time() - t0)
            return self._index

    def warmup(self):
        try:
            self._load_index()
        except Exception as e:
            _log.warning('BYOL 检索库预热失败: %s', e)

    def retrieve(self, kline_data, code='', top=5):
        window = getattr(config, 'WINDOW', 15)
        if not kline_data or len(kline_data) < window:
            return None

        idx = self._load_index()
        if idx is None:
            return None
        X, metas = idx
        model = self._model

        win_rows = kline_data[-window:]
        try:
            import torch
            from gru_model import feature_matrix, cosine_similarity
            mat = feature_matrix(win_rows)
            with torch.no_grad():
                x = torch.from_numpy(mat[None]).float()
                _, emb = model(x)
            emb = emb.detach().numpy().astype(np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-9)
        except Exception as e:
            _log.warning('BYOL 特征计算失败: %s', e)
            return None

        start = str(win_rows[0].get('date', ''))
        end = str(win_rows[-1].get('date', ''))

        # 余弦相似度 + 过滤（排除自我匹配、只留目标窗口开始日之前的先例）
        sims = cosine_similarity(emb, X)[0]
        order = np.argsort(-sims)

        exclude = (code, start, end) if code else None
        out = []
        for i in order:
            m = metas[i]
            if exclude and (m.get('code'), str(m.get('start')), str(m.get('end'))) == exclude:
                continue
            if start and str(m.get('end', '')) >= start:
                continue
            out.append((float(sims[i]), m))
            if len(out) >= top:
                break
        return out


# ==========================================
# 讲解层：大模型讲评器
# ==========================================
class BaseLLMProvider:
    """大模型讲评器基类（讲解层）。

    analyze() 返回生成的讲评文本（str）或 None（失败时由上层回退）。
    """
    name = 'base'
    label = '深度讲评'
    latency = '约30-40秒'

    def analyze(self, kline_data, current_index, position_info, ai_learning='',
                stock_code='', knn_cases=None):
        raise NotImplementedError

    def meta(self):
        return {'name': self.name, 'label': self.label, 'latency': self.latency}


class OpenAICompatProvider(BaseLLMProvider):
    """OpenAI 兼容协议的大模型讲评器。

    任何兼容 /chat/completions 的模型（Kimi、DeepSeek、Qwen、OpenAI、Claude 兼容层等）
    都能通过传入 base_url/model/api_key 直接使用，无需改代码。
    """
    name = 'openai-compat'
    latency = '约30-40秒'

    def __init__(self, name, label, base_url, model, api_key,
                 temperature=1, max_completion_tokens=8000, timeout=300):
        self.name = name
        self.label = label
        self.base_url = (base_url or '').rstrip('/')
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.timeout = timeout

    def analyze(self, kline_data, current_index, position_info, ai_learning='',
                stock_code='', knn_cases=None):
        _ai_log.info('--- AI讲评请求（provider=%s, model=%s）---', self.name, self.model)
        _ai_log.info('K线数量: %d根, 当前: 第%d根', len(kline_data), current_index + 1)
        _ai_log.info('持仓状态: %s', position_info)
        if knn_cases:
            _ai_log.info('相似历史案例: %d 例', len(knn_cases))

        prompt = build_prompt(kline_data, current_index, position_info, ai_learning, knn_cases)

        headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + self.api_key
        }
        data = {
            'model': self.model,
            'messages': [
                {'role': 'system', 'content': '你是一位专业的股票技术分析师，擅长K线形态识别、技术指标分析和交易策略制定。请基于用户提供的K线数据，给出客观、专业的技术分析。'},
                {'role': 'user', 'content': prompt}
            ],
            'temperature': self.temperature,
            'max_completion_tokens': self.max_completion_tokens
        }

        req = urllib.request.Request(
            self.base_url + '/chat/completions',
            data=json.dumps(data).encode('utf-8'),
            headers=headers,
            method='POST'
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode('utf-8'))
                if 'choices' in result and len(result['choices']) > 0:
                    content = result['choices'][0]['message'].get('content') or ''
                    content = content.strip()
                    if not content:
                        _ai_log.warning('AI返回正文为空（可能思考token耗尽被截断），完整响应: %s',
                                        json.dumps(result, ensure_ascii=False)[:500])
                        return None
                    _ai_log.info('--- AI响应 ---')
                    _ai_log.info('%s', content)
                    _ai_log.info('--- 响应结束 ---')
                    return content
                else:
                    _ai_log.warning('AI响应异常: %s', json.dumps(result, ensure_ascii=False))
                    return None
        except Exception as e:
            error_msg = '%s 大模型调用失败: %s' % (self.name, e)
            _log.error(error_msg)
            _ai_log.error(error_msg)
            return None


# ==========================================
# 注册表与工厂
# ==========================================
RETRIEVERS = {
    'knn': KnnRetriever,
    'gru': GruRetriever,
    'byol': ByolRetriever,
    # 在此登记其他证据层实现，例如：
    # 'dtw': DtwRetriever,
    # 'embedding': EmbeddingRetriever,
    # 'cluster': ClusterCentroidRetriever,
}


def get_retriever(name=None):
    """根据 config.RETRIEVER 返回检索器实例（默认 knn）。"""
    name = name or getattr(config, 'RETRIEVER', 'knn')
    cls = RETRIEVERS.get(name)
    if cls is None:
        _log.warning('未知检索器 %s，回退到 knn', name)
        cls = KnnRetriever
        name = 'knn'
    return cls()


def get_llm_provider(name=None):
    """根据 config.LLM_PROVIDER + config.LLM_PROVIDERS 返回大模型讲评器实例。

    切换大模型 = 在 config.LLM_PROVIDERS 中登记一条，并将 config.LLM_PROVIDER 指向它。
    """
    name = name or getattr(config, 'LLM_PROVIDER', 'kimi')
    providers = getattr(config, 'LLM_PROVIDERS', {}) or {}
    cfg = providers.get(name) or {}

    if not cfg:
        # 回退到旧版 KIMI_* 配置（向后兼容）
        cfg = {
            'base_url': getattr(config, 'KIMI_BASE_URL', 'https://api.kimi.com/coding/v1'),
            'model': getattr(config, 'KIMI_MODEL', 'kimi-k3'),
            'api_key': getattr(config, 'KIMI_API_KEY', ''),
        }

    return OpenAICompatProvider(
        name=name,
        label=cfg.get('label') or (cfg.get('model') or name),
        base_url=cfg.get('base_url', ''),
        model=cfg.get('model', name),
        api_key=cfg.get('api_key', ''),
        temperature=cfg.get('temperature', 1),
        max_completion_tokens=cfg.get('max_completion_tokens', 8000),
        timeout=cfg.get('timeout', 300),
    )


# ==========================================
# 提示词构建
# ==========================================
def build_prompt(kline_data, current_index, position_info, ai_learning='', knn_cases=None):
    """构建K线分析提示词"""
    # 获取前20根+当前K线
    start = max(0, current_index - 20)
    visible = kline_data[start:current_index + 1]
    cur = kline_data[current_index] if current_index < len(kline_data) else None

    # 计算最近7天的量价关系
    recent7 = kline_data[max(0, current_index - 6):current_index + 1]
    vol_up = 0
    vol_down = 0
    up_days = 0
    down_days = 0
    for i in range(1, len(recent7)):
        if recent7[i]['close'] >= recent7[i - 1]['close']:
            vol_up += recent7[i]['volume']
            up_days += 1
        else:
            vol_down += recent7[i]['volume']
            down_days += 1

    avg_vol_up = vol_up / up_days if up_days > 0 else 0
    avg_vol_down = vol_down / down_days if down_days > 0 else 0

    # 计算均线
    closes = [k['close'] for k in visible]
    ma5 = sum(closes[-5:]) / 5 if len(closes) >= 5 else None
    ma10 = sum(closes[-10:]) / 10 if len(closes) >= 10 else None
    ma20 = sum(closes[-20:]) / 20 if len(closes) >= 20 else None

    # 格式化K线数据
    lines = []
    lines.append('你作为资深高级交易员，请基于以下K线数据进行专业的技术分析。')
    lines.append('')
    lines.append('以下是最近' + str(len(visible)) + '根K线的详细数据（含成交量）：\n')

    for i, k in enumerate(visible):
        is_cur = (start + i) == current_index
        marker = ' ← 当前K线' if is_cur else ''
        change = ''
        if i > 0:
            prev_close = visible[i - 1]['close']
            pct = (k['close'] - prev_close) / prev_close * 100
            change = ' 涨跌:' + ('+' if pct >= 0 else '') + format(pct, '.2f') + '%'
        lines.append('第' + str(start + i + 1) + '根: 开=' + format(k['open'], '.2f') +
                     ' 高=' + format(k['high'], '.2f') +
                     ' 低=' + format(k['low'], '.2f') +
                     ' 收=' + format(k['close'], '.2f') +
                     ' 量=' + format(k['volume'] / 10000, '.0f') + '万手' +
                     change + marker)

    # 均线信息
    lines.append('\n【均线信息】')
    lines.append('MA5=' + (format(ma5, '.2f') if ma5 else 'N/A') +
                 ' MA10=' + (format(ma10, '.2f') if ma10 else 'N/A') +
                 ' MA20=' + (format(ma20, '.2f') if ma20 else 'N/A'))
    cur_close = cur['close'] if cur else 0
    if ma5 and ma10 and ma20:
        if cur_close > ma5 and ma5 > ma10 and ma10 > ma20:
            lines.append('均线多头排列，短期趋势偏多')
        elif cur_close < ma5 and ma5 < ma10 and ma10 < ma20:
            lines.append('均线空头排列，短期趋势偏空')
        else:
            lines.append('均线交叉状态，趋势不明朗')

    # 最近7天量价统计
    lines.append('\n【最近7天量价统计】')
    lines.append('上涨日数: ' + str(up_days) + '天, 下跌日数: ' + str(down_days) + '天')
    if up_days > 0 and down_days > 0:
        lines.append('上涨日均量: ' + format(avg_vol_up / 10000, '.0f') + '万手, 下跌日均量: ' + format(avg_vol_down / 10000, '.0f') + '万手')
        if avg_vol_down > avg_vol_up * 1.2:
            lines.append('下跌放量特征: 下跌日成交量明显大于上涨日，有吸筹或恐慌抛售可能')
        elif avg_vol_up > avg_vol_down * 1.2:
            lines.append('上涨放量特征: 上涨日成交量明显大于下跌日，有派筹或资金介入可能')

    lines.append('\n当前持仓状态：' + position_info)

    # 加入AI学习历史
    if ai_learning:
        lines.append('')
        lines.append(ai_learning)

    # 加入 KNN 相似历史案例（历史相似形态）
    if knn_cases:
        window = getattr(config, 'WINDOW', 15)
        fwd_days = getattr(config, 'FORWARD_DAYS', 10)
        target_rows = kline_data[-window:]
        lines.append('')
        lines.append('【历史相似形态（k-NN 检索结果）】')
        lines.append('系统已把当前"最近 ' + str(window) + ' 个交易日"的形态（'
                     + str(target_rows[0].get('date', '')) + ' 至 '
                     + str(target_rows[-1].get('date', '')) + '）'
                     + '在全市场历史窗口中做相似度检索，找到最相似的 ' + str(len(knn_cases)) + ' 个历史先例：')
        for rank, (dist, m) in enumerate(knn_cases, 1):
            fwd = m.get('forward')
            exc = m.get('excess')
            fwd_txt = ('%+.2f%%' % (fwd * 100)) if isinstance(fwd, (int, float)) else 'N/A'
            exc_txt = ('%+.2f%%' % (exc * 100)) if isinstance(exc, (int, float)) else 'N/A'
            lines.append('  #%d %s %s %s-%s（形态距离 %.2f）→ 后续 %d 日绝对收益 %s，超额收益 %s'
                         % (rank, m.get('code', ''), m.get('name', ''),
                            m.get('start', ''), m.get('end', ''), dist, fwd_days, fwd_txt, exc_txt))
        lines.append('（以上案例均为当前目标窗口之前的历史，仅作复盘参考，不代表目标未来会复现。）')
        lines.append('')
        lines.append('请在讲评输出中增加【历史相似形态】小节：先说明上述相似案例的形态共同点，')
        lines.append('再统计这些相似案例之后通常怎么走（收益均值、正收益占比），并注明仅为历史复盘参考。')

    lines.append('')
    lines.append('请严格按照以下格式输出分析结果，不要添加额外内容：')
    lines.append('')

    lines.append('''K线训练营盘面分析
仅做训练复盘参考，不构成实盘投资建议

当前盘面关键信息

均线系统
MA5={ma5}、MA10={ma10}、MA20={ma20}
现价 {price}，股价在 XX 日、XX 日均线 XX 方，大级别属于 XX 趋势。

K线与量价
（描述当前K线形态、量价关系、近期走势特征）

指标状态
MACD：（描述DIF/DEA位置、红绿柱变化）
KDJ：（描述K/D/J数值及位置）
RSI：（描述RSI数值及区域）

系统综合评估：多空 XX% 占优，整体偏 XX。

形态定性：（描述当前形态特征，如下跌中继/底部企稳/上升中继/顶部滞涨等）

操作建议：选择【买入/卖出/持股/观望】

理由
（1-3条理由，清晰说明为什么选择这个操作）

什么条件才考虑（买入/卖出）（训练用）
（列出具体的技术条件，如放量站上MAXX、KDJ金叉、MACD红柱放大等）

训练要点：（总结一条核心训练知识点）
''')
    lines.append('''【趋势判断】：短期中期趋势，均线状态
【量价配合】：成交量变化，识别主力行为+依据
【指标状态】MACD、KDJ、RSI现状，区分是反转信号还是中继信号
【关键价位】支撑位，压力位
【风险提示】列出盘面风险
【操作思路】二选一：买入/观望；如果要买入需要哪些共振确认条件。
已知股票最近N根K线数据：[把你的K线、收盘、高低点、成交量写在这里]
请严格基于给出的数据做技术分析：
1. 均线多空排列情况
2. 量价关系，识别可能的主力行为，每个判断给出证据
3. MACD、KDJ、RSI信号解读，区分高位/低位/中继信号
4. 关键压力、支撑
5. 当前不预测涨跌，给出：做多需要什么条件；什么信号出现确认风险离场
6. 当前训练操作选择：买入/观望/卖出。
重点区分洗盘和出货，洗盘特征是下跌缩量；出货特征是放量，高位放量优先视为派发。
只有价格、成交量、至少 2 个技术指标同时共振，才给出买入倾向；单一指标信号不作为进场理由。
''')

    return '\n'.join(lines)
