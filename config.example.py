# kline-studio（K线工作室）- 配置模板
# 复制为 config.py 并填入真实密钥

# ==========================================
# 数据源配置（Tushare Pro）
# 注册地址：https://tushare.pro
# ==========================================
TUSHARE_TOKEN = '在这里填入你的 Tushare Token'

# ==========================================
# 数据目录
# ==========================================
DATA_DIR = 'data'
KLINE_DIR = 'data/kline'
BENCHMARK_FILE = 'data/market_benchmark.json'

# ==========================================
# K-means 聚类参数（形态分析引擎）
# ==========================================
WINDOW = 15            # 每个形态片段包含多少个交易日
K_CLUSTERS = 16        # K-means 簇数
SLIDE_STEP = 3         # 滑窗步长（交易日）
FORWARD_DAYS = 10      # 统计簇"后续表现"的前向交易日数
MAX_WINDOWS = 200000   # 参与聚类的最大窗口数
RANDOM_SEED = 42

# ==========================================
# GRU 检索器参数（路线2：深度学习时序检索）
# 与 KNN 同层级，用 GRU 学习到的嵌入做相似案例检索
# ==========================================
GRU_HIDDEN = 32          # GRU 隐藏维（= 嵌入维）
GRU_LAYERS = 1           # GRU 层数
GRU_DROPOUT = 0.3        # dropout（防过拟合）
GRU_EPOCHS = 30          # 最大训练轮数（含早停）
GRU_PATIENCE = 6         # 早停耐心（连续 N 轮验证不改善则停）
GRU_BATCH_SIZE = 512     # 批大小
GRU_LR = 1e-3            # 学习率
GRU_WEIGHT_DECAY = 1e-4  # 权重衰减（L2 正则）
GRU_TRAIN_FRAC = 0.7     # 时间序列切分：训练集占比
GRU_VAL_FRAC = 0.15      # 时间序列切分：验证集占比（余下为测试集）

# ==========================================
# K线训练营参数
# ==========================================
STOCK_SCOPE = 'hs300'  # 'hs300' = 沪深300成分股, 'all' = 全部A股
HISTORY_YEARS = 3      # 历史数据采集年限
TRAIN_KLINE_COUNT = 30 # 训练使用的K线数量

# ==========================================
# 讲评引擎（三层架构）
#   底座层：规则引擎 —— 前端本地实现，毫秒级，无需配置
#   证据层：相似形态检索器（RETRIEVER）
#   讲解层：大模型深度讲评（LLM_PROVIDER）
# ==========================================

# 证据层：相似历史形态检索器（与 K-means「判类别」互补）
#   'knn' = 欧氏距离 KNN（路线1，手工特征 + 固定度量）
#   'gru' = GRU 学习嵌入 + 余弦相似度（路线2，学习度量）
# 切换：改 RETRIEVER 即可，二者同层级、可互换；gru 需先运行 python gru_train.py
RETRIEVER = 'knn'

# 讲解层：当前使用的大模型（须在下方 LLM_PROVIDERS 中登记）
LLM_PROVIDER = 'kimi'

# 讲解层：可用大模型清单（OpenAI 兼容协议，改 base_url/model/api_key 即可切换）
LLM_PROVIDERS = {
    'kimi': {
        'base_url': 'https://api.kimi.com/coding/v1',
        'model': 'kimi-k3',
        'api_key': '在这里填入你的 Kimi API Key',
        'temperature': 1,
        'max_completion_tokens': 8000,
        'timeout': 300,
    },
    # 切换其它大模型示例：在下方登记，并把 LLM_PROVIDER 改成对应名字
    # 'deepseek': {'base_url': 'https://api.deepseek.com/v1', 'model': 'deepseek-chat', 'api_key': '你的DeepSeek Key'},
    # 'qwen':     {'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1', 'model': 'qwen-plus', 'api_key': '你的通义Key'},
    # 'openai':   {'base_url': 'https://api.openai.com/v1', 'model': 'gpt-4o', 'api_key': '你的OpenAI Key'},
}

# ==========================================
# Kimi 讲评（旧字段，保留向后兼容，review.py 仍在使用）
# ==========================================
KIMI_API_KEY = '在这里填入你的 Kimi API Key'
KIMI_BASE_URL = 'https://api.kimi.com/coding/v1'
KIMI_MODEL = 'kimi-k3'
