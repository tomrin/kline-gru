/**
 * K线训练营 - 主应用入口（同花顺风格）
 */

const $ = id => document.getElementById(id);
let engine = null;
let chart = null;
let stockList = [];
let gameView = null;
var currentUser = localStorage.getItem('kline_user') || '';

// 持久化统计（跨局累计）
var gameStats = {
    totalProfit: 0,      // 总收益（按月累计）
    weeklyProfit: 0,     // 周收益（按周累计）
    winCount: 0,         // 盈利局数
    loseCount: 0,        // 亏损局数
    totalRounds: 0,      // 总训练局数
    lastRoundTime: null,  // 上次游戏时间

    // AI学习数据
    aiCorrect: 0,        // AI判断正确次数
    aiWrong: 0,          // AI判断错误次数
    aiHistory: []        // AI历史记录 [{advice, nextChange, correct, step, stock}]
};

// ============================================
// 初始化
// ============================================
async function init() {
    try {
        if (typeof echarts === 'undefined') {
            showMsg('ECharts图表库加载失败，请刷新重试', 'error');
            return;
        }
        try {
            engine = new GameEngine();
        } catch (e) {
            showMsg('游戏引擎初始化失败: ' + e.message, 'error');
            return;
        }
        const list = await loadStockList();
        if (!list || list.length === 0) {
            showMsg('没有找到股票数据，请先运行 data_fetcher.py', 'error');
            return;
        }
        stockList = list;
        chart = new KlineChart();
        bindEvents();
        initUser();
        loadEngines();
        showMsg('准备就绪，点击"开始训练"！', 'success');
    } catch (e) {
        showMsg('初始化失败: ' + (e.message || '未知错误'), 'error');
    }
}

/** 加载三层讲评引擎信息，动态显示各层当前引擎名与耗时 */
function loadEngines() {
    fetch('/api/engines')
        .then(function(r) { return r.json(); })
        .then(function(d) {
            if (!d || !d.success || !d.layers) return;
            var layers = d.layers;
            setLayerMeta('layerRetrieverMeta', layers.retriever);
            setLayerMeta('layerLLMMeta', layers.llm);
            // 初始化检索器切换按钮（KNN / GRU），高亮后端默认项
            if (d.retrievers && d.retrievers.length) {
                var def = null;
                for (var i = 0; i < d.retrievers.length; i++) {
                    if (d.retrievers[i].current) def = d.retrievers[i].name;
                }
                window._retriever = def || 'knn';
                setRetrieverSwitch(window._retriever);
            }
        })
        .catch(function(e) { console.warn('[engines] 加载引擎信息失败:', e); });
}

/** 高亮检索器切换按钮 + 记录当前选择（KNN / GRU） */
function setRetrieverSwitch(name) {
    window._retriever = name;
    var btns = document.querySelectorAll('#retrieverSwitch .retriever-btn');
    for (var i = 0; i < btns.length; i++) {
        if (btns[i].getAttribute('data-retriever') === name) {
            btns[i].classList.add('active');
        } else {
            btns[i].classList.remove('active');
        }
    }
}

/** 切换检索器并立即触发检索（KNN / GRU 结果各自独立保留，可同时对比） */
function switchRetriever(name) {
    setRetrieverSwitch(name);
    triggerRetrieval(name);
}

/** 取当前检索上下文（游戏状态 + K线数据）；未开始训练返回 null */
function getRetrievalContext() {
    var s = engine.state;
    if (!s || !s.stock || s.phase !== 'playing') {
        showMsg('请先开始训练', 'warn');
        return null;
    }
    return {
        s: s,
        allKline: [...s.stock.referenceKline, ...s.stock.kline.slice(0, s.visibleCount)]
    };
}

/** 触发指定检索器检索（结果渲染到对应区并保留，不影响另一路的结果） */
function triggerRetrieval(name) {
    var ctx = getRetrievalContext();
    if (ctx) requestRetriever(name, ctx.allKline, ctx.s);
}

/** 更新某层的引擎名+耗时标注 */
function setLayerMeta(id, meta) {
    if (!meta) return;
    var el = document.getElementById(id);
    if (el) el.textContent = (meta.label || meta.name || '') + ' · ' + (meta.latency || '');
}

async function loadStockList() {
    const paths = ['data/stocks.json', '../web/data/stocks.json', 'stocks.json'];
    for (const path of paths) {
        try {
            const resp = await fetch(path);
            if (resp.ok) {
                const data = await resp.json();
                if (data && data.length > 0) {
                    return data;
                }
            }
        } catch (e) {
            console.log('load failed:', path, e.message);
        }
    }
    return null;
}

// ============================================
// 游戏控制
// ============================================
async function startGame() {
    try {
        showMsg('正在抽取股票...', 'info');
        gameView = await engine.startGame(stockList);
        applyView(gameView);
        showMsg('游戏开始！', 'success');
        $('btnStart').disabled = true;
        $('welcomeOverlay').classList.add('hidden');
    } catch (e) {
        showMsg(e.message || '启动失败', 'error');
    }
}

function applyView(view) {
    if (!view) return;
    updateStats(view);
    updateInfoBar(view);
    updateButtons(view);
    updateStep(view);
    if (chart) {
        const signals = buildSignals();
        chart.render(view.allVisibleKline, view.referenceCount, view.mode, signals);
    }
    updateBottom(view);
    updateRightPanel(view);
}

function buildSignals() {
    const s = engine.state;
    if (!s || !s.stock || !s.decisions) return [];
    const signals = [];
    const refLen = s.stock.referenceKline ? s.stock.referenceKline.length : 0;
    for (const d of s.decisions) {
        if (d.action === '买入' || d.action === '卖出') {
            signals.push({
                index: refLen + d.step - 1,
                type: d.action === '买入' ? 'buy' : 'sell',
                price: d.price || 0
            });
        }
    }
    return signals;
}

// ============================================
// 操作处理
// ============================================
function handleBuy() {
    const result = engine.buy();
    if (result.success) {
        gameView = engine.getView();
        applyView(gameView);
        showMsg(result.msg, 'success');
        doAutoNext();
    } else {
        showMsg(result.msg, 'warn');
    }
}

function handleWait() {
    const result = engine.wait();
    if (result.success) {
        gameView = engine.getView();
        applyView(gameView);
        showMsg(result.msg, 'info');
        doAutoNext();
    } else {
        showMsg(result.msg, 'warn');
    }
}

function doAutoNext() {
    const result = engine.next();
    if (!result) { showMsg('游戏已结束', 'warn'); return; }
    if (result.error) { showMsg(result.msg, 'warn'); return; }
    if (result.phase === 'finished') {
        gameView = result.view;
        applyView(gameView);
        showSettlement(result.result);
        return;
    }
    gameView = result;
    applyView(gameView);
    try { analyzeCurrentKline(); } catch(e) {
        console.error('K线分析失败:', e);
        showMsg('分析异常: ' + e.message, 'warn');
    }
}

// ============================================
// 当前K线分析
// ============================================
function analyzeCurrentKline() {
    var s = engine.state;
    if (!s.stock) return;

    var allKline = [...s.stock.referenceKline, ...s.stock.kline.slice(0, s.visibleCount)];
    var currentIndex = s.stock.referenceKline.length + s.currentIndex;

    var curK = allKline[currentIndex];
    var prevK = currentIndex > 0 ? allKline[currentIndex - 1] : null;

    // 直接使用规则分析
    ruleBasedAnalysis(allKline, currentIndex, s, curK, prevK);
}

/** 手动触发 AI 深度讲评（Kimi 大模型 + KNN 相似历史形态检索） */
function triggerAIAnalysis() {
    console.log('[AI讲评] 按钮被点击');
    var s = engine.state;
    console.log('[AI讲评] engine.state=', s ? {phase: s.phase, hasStock: !!s.stock, currentIndex: s.currentIndex, visibleCount: s.visibleCount} : 'null');
    if (!s || !s.stock || s.phase !== 'playing') {
        showMsg('请先开始训练', 'warn');
        return;
    }
    var allKline = [...s.stock.referenceKline, ...s.stock.kline.slice(0, s.visibleCount)];
    var currentIndex = s.stock.referenceKline.length + s.currentIndex;
    console.log('[AI讲评] 发送请求, currentIndex=', currentIndex, 'allKline长度=', allKline.length);
    callAIAnalysis(allKline, currentIndex, s);
}

/** 调用AI分析 */
function callAIAnalysis(allKline, currentIndex, s) {
    console.log('[AI] callAIAnalysis 开始');
    var btn = document.getElementById('btnAIAnalysis');
    var posInfo = s.position > 0
        ? '持仓 ' + s.position + '股，成本 ' + s.avgCost.toFixed(2)
        : '空仓，现金 ' + (s.cash / 10000).toFixed(2) + '万';
    console.log('[AI] posInfo=', posInfo);

    // 加入AI学习历史，帮助AI自我改进
    var learningSummary = getAILearningSummary();
    console.log('[AI] learningSummary长度=', learningSummary ? learningSummary.length : 0);

    // 先用当前选中的检索器秒级检索相似历史形态并即时呈现（不等待 Kimi，1秒内可见）
    requestRetriever(window._retriever || 'knn', allKline, s);

    var payload = {
        klineData: allKline,
        currentIndex: currentIndex,
        positionInfo: posInfo,
        aiLearning: learningSummary,
        stockCode: s.stock.code,
        stockName: s.stock.name,
        retriever: window._retriever || 'knn'
    };
    console.log('[AI] payload stock=', payload.stockCode, payload.stockName);

    var data = JSON.stringify(payload);
    console.log('[AI] JSON data长度=', data.length);

    var xhr = new XMLHttpRequest();
    console.log('[AI] xhr 创建');
    xhr.open('POST', '/api/commentary', true);
    xhr.setRequestHeader('Content-Type', 'application/json');
    console.log('[AI] xhr open 完成');

    // 深度讲评层显示加载中
    document.getElementById('layerLLMBody').innerHTML =
        '<div class="ai-loading">🤖 大模型深度讲评生成中，约需 30~40 秒…</div>';
    if (btn) { btn.disabled = true; btn.classList.add('loading'); btn.textContent = '⏳ 分析中...'; }

    var finish = function() {
        if (btn) { btn.disabled = false; btn.classList.remove('loading'); btn.textContent = '🤖 深度讲评'; }
    };

    // AI失败时回退到本地规则分析，并明确提示用户
    var fallbackToRule = function(reason) {
        console.error('[AI] 回退到规则分析, 原因=', reason);
        showMsg('AI讲评失败（' + reason + '），已回退到本地规则分析', 'warn');
        finish();
        var curK = allKline[currentIndex];
        var prevK = currentIndex > 0 ? allKline[currentIndex - 1] : null;
        ruleBasedAnalysis(allKline, currentIndex, s, curK, prevK);
    };

    xhr.onreadystatechange = function() {
        console.log('[AI] readyState=', xhr.readyState, 'status=', xhr.status);
    };

    xhr.onload = function() {
        console.log('[AI] onload status=', xhr.status);
        finish();
        if (xhr.status === 200) {
            try {
                var resp = JSON.parse(xhr.responseText);
                console.log('[AI] response success=', resp.success, 'content长度=', resp.content ? resp.content.length : 0);
                if (resp.success && resp.content) {
                    displayAIContent(resp.content, s.currentIndex + 1);
                } else {
                    fallbackToRule('模型返回为空');
                }
            } catch(e) {
                console.error('[AI] parse error', e);
                fallbackToRule('响应解析失败');
            }
        } else {
            fallbackToRule('HTTP ' + xhr.status);
        }
    };

    xhr.onerror = function(e) {
        console.error('[AI] onerror', e);
        fallbackToRule('网络错误');
    };

    xhr.ontimeout = function() {
        console.error('[AI] ontimeout');
        fallbackToRule('请求超时');
    };

    console.log('[AI] xhr.send 前');
    xhr.send(data);
    console.log('[AI] xhr.send 已调用');
}

/** 指定检索器（knn/gru）检索相似历史形态，结果渲染到该检索器专属区域并保留 */
function requestRetriever(name, allKline, s) {
    name = name || 'knn';
    var isGru = (name === 'gru');
    var bodyEl = document.getElementById(isGru ? 'gruResultBody' : 'knnResultBody');
    var statusEl = document.getElementById(isGru ? 'gruStatus' : 'knnStatus');

    console.log('[Retriever] 开始检索, 检索器=', name);
    if (statusEl) { statusEl.textContent = '检索中...'; statusEl.className = 'rr-status running'; }
    if (bodyEl) bodyEl.innerHTML = '<div class="layer-placeholder">检索中…</div>';

    var payload = {
        klineData: allKline,
        stockCode: s.stock.code,
        retriever: name
    };
    var xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/knn', true);
    xhr.setRequestHeader('Content-Type', 'application/json');
    xhr.onload = function() {
        if (xhr.status === 200) {
            try {
                var resp = JSON.parse(xhr.responseText);
                if (resp.success && resp.cases && resp.cases.length > 0) {
                    console.log('[Retriever] ' + name + ' 检索到', resp.cases.length, '个案例');
                    window._cards = window._cards || {};
                    window._cards[name] = buildKNNCard(resp.cases, name);
                    if (bodyEl) bodyEl.innerHTML = window._cards[name];
                    if (statusEl) { statusEl.textContent = resp.cases.length + ' 例'; statusEl.className = 'rr-status done'; }
                } else {
                    console.warn('[Retriever] ' + name + ' 无结果:', resp.message);
                    if (bodyEl) bodyEl.innerHTML = '<div class="layer-placeholder">未检索到相似形态（' + (resp.message || '数据不足') + '）</div>';
                    if (statusEl) { statusEl.textContent = '无结果'; statusEl.className = 'rr-status fail'; }
                }
            } catch(e) {
                console.error('[Retriever] parse error', e);
                if (statusEl) { statusEl.textContent = '解析异常'; statusEl.className = 'rr-status fail'; }
            }
        } else {
            if (statusEl) { statusEl.textContent = '失败 ' + xhr.status; statusEl.className = 'rr-status fail'; }
        }
    };
    xhr.onerror = function(e) {
        console.error('[Retriever] onerror', e);
        if (statusEl) { statusEl.textContent = '网络错误'; statusEl.className = 'rr-status fail'; }
    };
    xhr.send(JSON.stringify(payload));
}

/** 生成相似历史形态卡片 HTML（纯卡片，不含 header）；name 决定底部相似度说明文案 */
function buildKNNCard(cases, name) {
    var html = '';

    var fwds = [];
    var html2 = '';
    for (var i = 0; i < cases.length; i++) {
        var c = cases[i];
        fwds.push(c.forward);
        var fwdPct = (c.forward * 100).toFixed(2) + '%';
        var fwdColor = c.forward >= 0 ? '#e74c3c' : '#2ecc71';
        var dateRange = fmtDate(c.start) + ' ~ ' + fmtDate(c.end);
        html2 += '<div class="knn-row">';
        html2 += '<span class="knn-rank">' + (i + 1) + '</span>';
        html2 += '<span class="knn-name">' + c.name + ' <em>(' + c.code + ')</em></span>';
        html2 += '<span class="knn-date">' + dateRange + '</span>';
        html2 += '<span class="knn-fwd" style="color:' + fwdColor + ';">后续 ' + fwdPct + '</span>';
        html2 += '</div>';
    }

    var sum = 0;
    for (var j = 0; j < fwds.length; j++) sum += fwds[j];
    var avg = sum / fwds.length;
    var posCnt = 0;
    for (var k = 0; k < fwds.length; k++) if (fwds[k] > 0) posCnt++;
    var avgColor = avg >= 0 ? '#e74c3c' : '#2ecc71';

    html += '<div class="knn-card">';
    html += '<div class="knn-stat">';
    html += '共 <b>' + cases.length + '</b> 个历史相似案例 · 后续10日均收益 <b style="color:' + avgColor + ';">' + (avg * 100).toFixed(2) + '%</b>';
    html += ' · 正收益占比 <b>' + posCnt + '/' + cases.length + '</b>';
    html += '</div>';
    html += html2;
    var note = (name === 'gru')
        ? '相似度按 GRU 学习嵌入的余弦相似度计算（越接近 1 越相似）；收益为历史复盘参考，不代表未来。'
        : '相似度按最近15根K线的形态特征（60维）计算（距离越小越相似）；收益为历史复盘参考，不代表未来。';
    html += '<div class="knn-note">' + note + '</div>';
    html += '</div>';
    return html;
}

/** 格式化日期 20230831 -> 2023-08-31 */
function fmtDate(s) {
    if (!s || s.length !== 8) return s;
    return s.slice(0, 4) + '-' + s.slice(4, 6) + '-' + s.slice(6, 8);
}

/** 显示AI返回的讲评内容（③ 深度讲评层） */
function displayAIContent(content, step) {
    var el = document.getElementById('layerLLMBody');
    var html = '';

    html += '<div class="comment-header">';
    html += '<span class="step-badge">第' + step + '根</span>';
    html += '<span class="pattern-badge neutral" style="background:rgba(52,152,219,0.2);color:#3498db;">AI分析</span>';
    html += '</div>';

    // 提取AI的推荐操作（用于后续学习评估）
    var aiAdvice = extractAIAdvice(content);
    if (aiAdvice && engine && engine.state) {
        saveAIRecommendation(aiAdvice, step, engine.state);
    }

    var paragraphs = content.split('\n');
    var currentSection = '';
    var sectionContent = [];

    for (var i = 0; i < paragraphs.length; i++) {
        var line = paragraphs[i].trim();
        if (!line) continue;

        // 检测各段落标题
        var section = '';
        var inlineContent = '';
        var bracketMatch = line.match(/^【(.+?)】\s*[:：]?\s*(.*)$/);
        if (bracketMatch) {
            // 方括号标题：【趋势判断】【量价配合】【关键价位】等，标题与内容可能同行
            var bt = bracketMatch[1];
            inlineContent = bracketMatch[2] || '';
            if (bt.indexOf('历史相似形态') >= 0) section = 'knn';
            else section = bt;
        } else if (line.indexOf('历史相似形态') >= 0) section = 'knn';
        else if (line.indexOf('均线系统') >= 0) section = 'ma';
        else if (line.indexOf('K线与量价') >= 0) section = 'kline';
        else if (line.indexOf('指标状态') >= 0) section = 'indicators';
        else if (line.indexOf('系统综合评估') >= 0) section = 'assessment';
        else if (line.indexOf('形态定性') >= 0) section = 'pattern';
        else if (line.indexOf('操作建议') >= 0) section = 'advice';
        else if (line.indexOf('理由') >= 0) section = 'reason';
        else if (line.indexOf('什么条件') >= 0) section = 'conditions';
        else if (line.indexOf('训练要点') >= 0) section = 'lesson';

        if (section) {
            if (currentSection && sectionContent.length > 0) {
                html += renderSection(currentSection, sectionContent);
            }
            currentSection = section;
            sectionContent = [];
            if (inlineContent) sectionContent.push(inlineContent);
        } else if (currentSection) {
            sectionContent.push(line);
        } else {
            // 开头部分（免责声明等）
            if (line.indexOf('仅做训练复盘') >= 0 || line.indexOf('不构成') >= 0) {
                html += '<div class="lesson-item note"><div class="lesson-content" style="color:#888;font-size:10px;">' + line + '</div></div>';
            } else if (line.indexOf('K线训练营盘面分析') >= 0) {
                html += '<div class="lesson-item pattern"><div class="lesson-title" style="font-size:13px;color:#f39c12;">K线训练营盘面分析</div></div>';
            } else {
                html += '<div class="lesson-content">' + line + '</div>';
            }
        }
    }

    // 处理最后一个段落
    if (currentSection && sectionContent.length > 0) {
        html += renderSection(currentSection, sectionContent);
    }

    // 如果没有任何段落被解析，直接显示全部内容
    if (html.indexOf('lesson-item') < 0) {
        html = '<div class="comment-header">';
        html += '<span class="step-badge">第' + step + '根</span>';
        html += '<span class="pattern-badge neutral">AI分析</span>';
        html += '</div>';
        html += '<div class="lesson-item pattern"><div class="lesson-content" style="white-space:pre-wrap;">' + content + '</div></div>';
    }

    el.innerHTML = html;
    el.scrollTop = 0;
}

/** 渲染各段落 */
function renderSection(section, lines) {
    // 历史相似形态：硬数据卡片已在证据层(②)独立呈现，讲解层仅保留大模型的文字解读
    var html = '';
    var titleMap = {
        'knn': '历史相似形态',
        'ma': '均线系统',
        'kline': 'K线与量价',
        'indicators': '指标状态',
        'assessment': '系统综合评估',
        'pattern': '形态定性',
        'advice': '操作建议',
        'reason': '理由',
        'conditions': '关注条件',
        'lesson': '训练要点'
    };
    var typeMap = {
        'knn': 'trend',
        'ma': 'pattern',
        'kline': 'pattern',
        'indicators': 'trend',
        'assessment': 'trend',
        'pattern': 'pattern',
        'advice': 'advice',
        'reason': 'note',
        'conditions': 'note',
        'lesson': 'praise'
    };

    var title = titleMap[section] || section;
    var type = typeMap[section] || 'note';

    html += '<div class="lesson-item ' + type + '"><div class="lesson-title">' + title + '</div>';
    for (var i = 0; i < lines.length; i++) {
        html += '<div class="lesson-content">' + lines[i] + '</div>';
    }
    html += '</div>';
    return html;
}

// ============================================
// AI学习系统
// ============================================

/** 从AI回复中提取推荐操作 */
function extractAIAdvice(content) {
    if (!content) return null;
    var lines = content.split('\n');
    for (var i = 0; i < lines.length; i++) {
        var line = lines[i].trim();
        // 匹配"操作建议：选择【买入】"或"操作建议：选择【观望】"等
        var match = line.match(/选择[【\[（](.+?)[】\]）]/);
        if (match) {
            var advice = match[1];
            if (advice.indexOf('买入') >= 0 || advice.indexOf('卖出') >= 0 ||
                advice.indexOf('持股') >= 0 || advice.indexOf('观望') >= 0) {
                return advice;
            }
        }
    }
    return null;
}

/** 保存AI推荐记录 */
function saveAIRecommendation(advice, step, gameState) {
    if (!gameState || !gameState.stock) return;
    var stockName = gameState.stock.name || '未知';
    var currentPrice = 0;
    if (gameState.stock.kline && gameState.stock.kline[gameState.currentIndex]) {
        currentPrice = gameState.stock.kline[gameState.currentIndex].close;
    }
    // 获取下一根K线的价格（用于评估）
    var nextPrice = 0;
    if (gameState.stock.kline && gameState.stock.kline[gameState.currentIndex + 1]) {
        nextPrice = gameState.stock.kline[gameState.currentIndex + 1].close;
    }

    gameStats.aiHistory.push({
        advice: advice,
        step: step,
        stock: stockName,
        price: currentPrice,
        nextPrice: nextPrice,
        evaluated: false,
        correct: false
    });
}

/** 评估AI推荐（在游戏结束后调用） */
function evaluateAIRecommendations() {
    var history = gameStats.aiHistory;
    // 只评估当前局的记录（最近30条以内）
    var start = Math.max(0, history.length - 35);
    for (var i = start; i < history.length; i++) {
        var h = history[i];
        if (h.evaluated) continue;
        if (h.nextPrice <= 0) continue;

        var change = (h.nextPrice - h.price) / h.price * 100;
        var correct = false;

        if (h.advice === '买入' && change > 0) correct = true;
        else if (h.advice === '卖出' && change < 0) correct = true;
        else if (h.advice === '持股' && change > 0) correct = true;
        else if (h.advice === '观望' && change < 0) correct = true;
        // 如果推荐与实际情况一致，也认为正确
        // 观望时下跌正确，持股时上涨正确

        h.evaluated = true;
        h.correct = correct;
        h.actualChange = change;

        if (correct) {
            gameStats.aiCorrect++;
        } else {
            gameStats.aiWrong++;
        }
    }
}

/** 获取AI成功率 */
function getAISuccessRate() {
    var total = gameStats.aiCorrect + gameStats.aiWrong;
    if (total === 0) return 0;
    return (gameStats.aiCorrect / total * 100);
}

/** 获取AI学习总结（用于提示词，让AI参考历史表现） */
function getAILearningSummary() {
    var total = gameStats.aiCorrect + gameStats.aiWrong;
    if (total === 0) return '';

    var rate = getAISuccessRate();
    var summary = '【AI历史表现】\n';
    summary += '历史判断次数: ' + total + ', 正确率: ' + rate.toFixed(1) + '%\n';

    // 统计各类建议的正确率
    var adviceStats = {};
    for (var i = 0; i < gameStats.aiHistory.length; i++) {
        var h = gameStats.aiHistory[i];
        if (!h.evaluated) continue;
        if (!adviceStats[h.advice]) {
            adviceStats[h.advice] = { correct: 0, total: 0 };
        }
        adviceStats[h.advice].total++;
        if (h.correct) adviceStats[h.advice].correct++;
    }

    for (var key in adviceStats) {
        var s = adviceStats[key];
        var r = s.total > 0 ? (s.correct / s.total * 100).toFixed(1) : 0;
        summary += key + ': ' + s.correct + '/' + s.total + '(' + r + '%)\n';
    }

    // 取最近3条正确分析作为参考
    var recentGood = [];
    for (var i = gameStats.aiHistory.length - 1; i >= 0 && recentGood.length < 3; i--) {
        if (gameStats.aiHistory[i].correct && gameStats.aiHistory[i].evaluated) {
            recentGood.push(gameStats.aiHistory[i]);
        }
    }
    if (recentGood.length > 0) {
        summary += '\n【近期正确判断参考】\n';
        for (var i = 0; i < recentGood.length; i++) {
            var g = recentGood[i];
            summary += '第' + g.step + '步推荐' + g.advice + '（' + g.stock + '），实际涨跌' + (g.actualChange >= 0 ? '+' : '') + g.actualChange.toFixed(2) + '%\n';
        }
    }

    return summary;
}

/** 规则分析（兜底） */
function ruleBasedAnalysis(allKline, currentIndex, s, curK, prevK) {
    try {
        if (typeof KlineAnalysis !== 'undefined') {
            var decision = { action: '分析', step: s.currentIndex + 1 };
            var context = { hasPosition: s.position > 0, avgCost: s.avgCost, cash: s.cash };
            var commentary = KlineAnalysis.comment(decision, allKline, currentIndex, context);
            displayKlineAnalysis(commentary, curK);
            return;
        }
    } catch(e) {
        console.error('KlineAnalysis error:', e);
    }

    // 最简兜底
    var prevClose = prevK ? prevK.close : (curK ? curK.open : 0);
    var changePct = prevClose ? (curK.close - prevClose) / prevClose * 100 : 0;
    var isUp = changePct >= 0;
    var el = document.getElementById('layerRuleBody');
    var html = '';
    html += '<div class="comment-header">';
    html += '<span class="step-badge">第' + (s.currentIndex + 1) + '根</span>';
    html += '<span class="pattern-badge ' + (isUp ? 'bullish' : 'bearish') + '">' + (isUp ? '阳线' : '阴线') + '</span>';
    html += '<span style="font-size:11px;color:#888;margin-left:4px;">' + (isUp ? '+' : '') + changePct.toFixed(2) + '%</span>';
    html += '</div>';
    html += '<div class="lesson-item pattern"><div class="lesson-title">当前K线</div>';
    html += '<div class="lesson-content">价格: ' + curK.close.toFixed(2) + ' 涨跌: ' + (isUp ? '+' : '') + changePct.toFixed(2) + '%</div>';
    html += '<div class="lesson-content">开: ' + curK.open.toFixed(2) + ' 高: ' + curK.high.toFixed(2) + ' 低: ' + curK.low.toFixed(2) + ' 收: ' + curK.close.toFixed(2) + '</div>';
    html += '</div>';
    html += '<div class="lesson-item advice"><div class="lesson-title">操作建议</div>';
    if (s.position > 0) {
        html += '<div class="lesson-content">持仓中，可考虑持股或卖出</div>';
    } else {
        html += '<div class="lesson-content">空仓中，可考虑买入或观望</div>';
    }
    html += '</div>';
    el.innerHTML = html;
    el.scrollTop = 0;
}
function displayKlineAnalysis(commentary, curK) {
    if (!commentary) return;
    var el = document.getElementById('layerRuleBody');

    var pattern = commentary.pattern || { name: '未知', type: 'neutral' };
    var analysis = commentary.analysis || [];
    var judgment = commentary.judgment || [];
    var prediction = commentary.prediction || [];
    var recommended = commentary.recommended || { action: '观望', reason: '' };
    var lessons = commentary.lessons || [];
    var changePct = commentary.changePct || 0;
    var isUp = changePct >= 0;

    var html = '';
    html += '<div class="comment-header">';
    html += '<span class="step-badge">第' + (commentary.step || 0) + '根</span>';
    html += '<span class="pattern-badge ' + pattern.type + '">' + pattern.name + '</span>';
    html += '<span style="font-size:11px;color:#888;margin-left:4px;">' + (isUp ? '+' : '') + changePct.toFixed(2) + '%</span>';
    html += '</div>';

    // 分析
    if (analysis.length > 0) {
        html += '<div class="lesson-item pattern"><div class="lesson-title">分析：当前状况</div>';
        for (var i = 0; i < analysis.length; i++) {
            html += '<div class="lesson-content">' + analysis[i] + '</div>';
        }
        html += '</div>';
    }

    // 判断
    if (judgment.length > 0) {
        html += '<div class="lesson-item trend"><div class="lesson-title">判断：综合评估</div>';
        for (var i = 0; i < judgment.length; i++) {
            html += '<div class="lesson-content">' + judgment[i] + '</div>';
        }
        html += '</div>';
    }

    // 预测
    if (prediction.length > 0) {
        html += '<div class="lesson-item advice"><div class="lesson-title">预测：走势研判</div>';
        for (var i = 0; i < prediction.length; i++) {
            html += '<div class="lesson-content">' + prediction[i] + '</div>';
        }
        html += '</div>';
    }

    // 操作建议
    if (recommended.reason) {
        html += '<div class="lesson-item note"><div class="lesson-title">操作参考</div>';
        html += '<div class="lesson-content">建议：' + recommended.action + '。' + recommended.reason + '</div>';
        html += '</div>';
    }

    // 知识点
    for (var i = 0; i < lessons.length; i++) {
        var lesson = lessons[i];
        html += '<div class="lesson-item ' + (lesson.type || 'note') + '">';
        html += '<div class="lesson-title">' + (lesson.title || '') + '</div>';
        html += '<div class="lesson-content">' + (lesson.content || '') + '</div>';
        html += '</div>';
    }

    el.innerHTML = html;
    el.scrollTop = 0;
}
function displayQuantReport(report, curK, changePct) {
    if (!report) return;
    var el = document.getElementById('layerRuleBody');
    var html = '';

    html += '<div class="comment-header">';
    html += '<span class="step-badge">第' + (report.step || 0) + '根</span>';
    var p = report.pattern || { name: '未知', type: 'neutral' };
    html += '<span class="pattern-badge ' + p.type + '">' + p.name + '</span>';
    html += '<span style="font-size:11px;color:#888;margin-left:4px;">' + (changePct >= 0 ? '+' : '') + changePct.toFixed(2) + '%</span>';
    html += '</div>';

    var s = report.snapshot || {};

    // 一、核心指标快照
    html += '<div class="lesson-item pattern"><div class="lesson-title">一、【核心指标快照】</div>';
    if (s.price) {
        html += '<div class="lesson-content">价格：' + s.price.close.toFixed(2) + '（' + (changePct >= 0 ? '+' : '') + changePct.toFixed(2) + '%）</div>';
        html += '<div class="lesson-content">均线排列：' + s.price.maArrange + '</div>';
        html += '<div class="lesson-content">MA5=' + (s.price.ma5 !== '-' ? s.price.ma5.toFixed(2) : '--') + ' MA10=' + (s.price.ma10 !== '-' ? s.price.ma10.toFixed(2) : '--') + ' MA20=' + (s.price.ma20 !== '-' ? s.price.ma20.toFixed(2) : '--') + '</div>';
    }
    if (s.macd) {
        var m = s.macd;
        var str = 'MACD: DIF=' + (m.dif !== '-' ? m.dif.toFixed(3) : '--') + ' DEA=' + (m.dea !== '-' ? m.dea.toFixed(3) : '--');
        if (m.cross === 'golden') str += ' 金叉';
        else if (m.cross === 'death') str += ' 死叉';
        else if (m.cross === '多头') str += ' 多头';
        else str += ' 空头';
        if (m.barDesc) str += ' | ' + m.barDesc;
        html += '<div class="lesson-content">' + str + '</div>';
    }
    if (s.rsi) {
        var r = s.rsi;
        html += '<div class="lesson-content">RSI(14): ' + (r.value !== '-' ? r.value.toFixed(1) : '--') + '（' + r.zone + '）</div>';
    }
    if (s.bollinger) {
        var b = s.bollinger;
        html += '<div class="lesson-content">布林带: 上轨=' + b.upper.toFixed(2) + ' 中轨=' + b.mid.toFixed(2) + ' 下轨=' + b.lower.toFixed(2) + '</div>';
        html += '<div class="lesson-content">价格位置：' + b.position + ' | 带宽=' + b.bandwidth.toFixed(1) + '%</div>';
    }
    if (s.volume) {
        html += '<div class="lesson-content">量能：' + s.volume.status + '（量比' + s.volume.ratio.toFixed(2) + '）</div>';
    }
    html += '</div>';

    // 二、趋势综合判断
    var t = report.trendJudge || {};
    html += '<div class="lesson-item trend"><div class="lesson-title">二、【趋势综合判断】（评分：' + (t.score || 0) + '）</div>';
    html += '<div class="lesson-content">短期（1-5日）：' + (t.shortTerm || '分析中') + '</div>';
    html += '<div class="lesson-content">中期（1-4周）：' + (t.midTerm || '分析中') + '</div>';
    if (t.reasons && t.reasons.length > 0) {
        html += '<div class="lesson-content" style="color:#888;">依据：' + t.reasons.join('、') + '</div>';
    }
    html += '</div>';

    // 三、操作建议
    var a = report.advice || {};
    html += '<div class="lesson-item advice"><div class="lesson-title">三、【操作建议】</div>';
    html += '<div class="lesson-content">策略：' + (a.strategy || '分析中') + '</div>';
    html += '<div class="lesson-content">买点：' + (a.buyPoint || '--') + '</div>';
    html += '<div class="lesson-content">止损：' + (a.stopLoss || '--') + '</div>';
    html += '<div class="lesson-content">仓位：' + (a.position || '--') + '</div>';
    html += '<div class="lesson-content">止盈：' + (a.takeProfit || '--') + '</div>';
    html += '</div>';

    // 四、关键风险预警
    if (report.warnings && report.warnings.length > 0) {
        html += '<div class="lesson-item note"><div class="lesson-title">四、【关键风险预警】</div>';
        for (var i = 0; i < report.warnings.length; i++) {
            html += '<div class="lesson-content">' + report.warnings[i] + '</div>';
        }
        html += '</div>';
    }

    el.innerHTML = html;
    el.scrollTop = 0;
}

function calcMA(data, n) {
    var result = [];
    for (var i = 0; i < data.length; i++) {
        if (i < n - 1) { result.push('-'); }
        else {
            var sum = 0;
            for (var j = i - n + 1; j <= i; j++) sum += data[j];
            result.push(sum / n);
        }
    }
    return result;
}

// ============================================
// UI更新
// ============================================
function updateStats(view) {
    // 使用持久化统计（跨局累计）
    var total = gameStats.winCount + gameStats.loseCount;
    var rate = total > 0 ? (gameStats.winCount / total * 100) : 0;

    $('statTotalProfit').textContent = gameStats.totalProfit >= 0
        ? '+' + gameStats.totalProfit.toFixed(2) : gameStats.totalProfit.toFixed(2);
    $('statTotalProfit').className = 'stat-value ' + (gameStats.totalProfit >= 0 ? 'up' : 'down');

    $('statWeeklyProfit').textContent = gameStats.weeklyProfit >= 0
        ? '+' + gameStats.weeklyProfit.toFixed(2) : gameStats.weeklyProfit.toFixed(2);
    $('statWeeklyProfit').className = 'stat-value ' + (gameStats.weeklyProfit >= 0 ? 'up' : 'down');

    $('statWinRate').textContent = rate.toFixed(1) + '%';
    $('statWinRate').className = 'stat-value ' + (rate >= 60 ? 'up' : (rate >= 30 ? 'neutral' : 'down'));

    $('statPlayCount').textContent = gameStats.totalRounds;

    var aiRate = getAISuccessRate();
    $('statAIAccuracy').textContent = aiRate.toFixed(1) + '%';
    $('statAIAccuracy').className = 'stat-value';
    $('statAIAccuracy').style.color = aiRate >= 60 ? '#2ecc71' : (aiRate >= 30 ? '#f39c12' : '#e74c3c');
}

function updateInfoBar(view) {
    $('ma5Label').textContent = view.ma5 && view.ma5 !== '-' ? 'MA5: ' + view.ma5.toFixed(2) : 'MA5: --';
    $('ma10Label').textContent = view.ma10 && view.ma10 !== '-' ? 'MA10: ' + view.ma10.toFixed(2) : 'MA10: --';
    $('ma20Label').textContent = view.ma20 && view.ma20 !== '-' ? 'MA20: ' + view.ma20.toFixed(2) : 'MA20: --';
    if (view.ohlc) {
        var o = view.ohlc;
        $('infoOpen').textContent = o.open.toFixed(2);
        $('infoHigh').textContent = o.high.toFixed(2);
        $('infoLow').textContent = o.low.toFixed(2);
        $('infoClose').textContent = o.close.toFixed(2);
        $('infoClose').className = o.change >= 0 ? 'up-val' : 'down-val';
        $('infoChange').textContent = (o.change >= 0 ? '+' : '') + o.change.toFixed(2);
        $('infoChange').className = o.change >= 0 ? 'up-val' : 'down-val';
        $('infoChangePct').textContent = (o.change >= 0 ? '+' : '') + o.changePct.toFixed(2) + '%';
        $('infoChangePct').className = o.change >= 0 ? 'up-val' : 'down-val';
    }
}

function updateButtons(view) {
    var buyBtn = $('btnBuy');
    var waitBtn = $('btnWait');
    if (view.operatedToday) {
        buyBtn.disabled = true;
        waitBtn.disabled = true;
        return;
    }
    if (view.hasPosition) {
        buyBtn.textContent = '卖出';
        buyBtn.className = 'action-btn buy-btn';
        waitBtn.textContent = '持股';
        waitBtn.className = 'action-btn wait-btn';
    } else {
        buyBtn.textContent = '买入';
        buyBtn.className = 'action-btn buy-btn';
        waitBtn.textContent = '观望';
        waitBtn.className = 'action-btn wait-btn';
    }
    buyBtn.disabled = false;
    waitBtn.disabled = false;
}

function updateStep(view) {
    $('roundStep').textContent = view.currentStep + ' / ' + view.totalSteps;
    $('progressFill').style.width = (view.currentStep / view.totalSteps * 100) + '%';
}

function updateBottom(view) {
    var rate = view.returnRate;
    $('gameProfit').textContent = (rate >= 0 ? '+' : '') + rate.toFixed(2) + '%';
    $('gameProfit').className = 'profit-value ' + (rate >= 0 ? '' : 'down');
    if (view.hasPosition) {
        $('positionText').textContent = '持仓 ' + view.position + '股 @ ' + view.avgCost.toFixed(2);
    } else {
        $('positionText').textContent = '空仓';
    }
    $('cashText').textContent = '现金: ' + (view.cash / 10000).toFixed(2) + '万';
}

function updateRightPanel(view) {
    if (!view || !view.ohlc) return;
    var o = view.ohlc;
    var isUp = o.change >= 0;
    var upCls = 'pv up';
    var downCls = 'pv down';
    $('rpOpen').textContent = o.open.toFixed(2);
    $('rpHigh').textContent = o.high.toFixed(2);
    $('rpLow').textContent = o.low.toFixed(2);
    $('rpClose').textContent = o.close.toFixed(2);
    $('rpClose').className = isUp ? upCls : downCls;
    $('rpChange').textContent = (isUp ? '+' : '') + o.change.toFixed(2);
    $('rpChange').className = isUp ? upCls : downCls;
    $('rpChangePct').textContent = (isUp ? '+' : '') + o.changePct.toFixed(2) + '%';
    $('rpChangePct').className = isUp ? upCls : downCls;
    $('rpMA5').textContent = view.ma5 && view.ma5 !== '-' ? view.ma5.toFixed(2) : '--';
    $('rpMA10').textContent = view.ma10 && view.ma10 !== '-' ? view.ma10.toFixed(2) : '--';
    $('rpMA20').textContent = view.ma20 && view.ma20 !== '-' ? view.ma20.toFixed(2) : '--';
    var allKline = view.allVisibleKline || [];
    if (allKline.length < 20) return;
    var closes = allKline.map(function(d) { return d.close; });
    var highs = allKline.map(function(d) { return d.high; });
    var lows = allKline.map(function(d) { return d.low; });
    var ema12 = calcEMASimple(closes, 12);
    var ema26 = calcEMASimple(closes, 26);
    var dif = ema12[ema12.length - 1];
    var dea = calcEMASimple(ema12.map(function(v, i) { return (v !== '-' && ema26[i] !== '-') ? v - ema26[i] : '-'; }), 9);
    var deaVal = dea[dea.length - 1];
    var macdVal = (dif !== '-' && deaVal !== '-') ? (dif - deaVal) * 2 : '-';
    $('rpDIF').textContent = dif !== '-' ? dif.toFixed(3) : '--';
    $('rpDEA').textContent = deaVal !== '-' ? deaVal.toFixed(3) : '--';
    $('rpMACD').textContent = macdVal !== '-' ? macdVal.toFixed(3) : '--';
    $('rpMACD').className = macdVal !== '-' && macdVal >= 0 ? upCls : downCls;
    var kValues = [], dValues = [];
    for (var i = 0; i < closes.length; i++) {
        if (i < 8) { kValues.push('-'); dValues.push('-'); continue; }
        var hh = -Infinity, ll = Infinity;
        for (var jj = i - 8; jj <= i; jj++) {
            hh = Math.max(hh, highs[jj]); ll = Math.min(ll, lows[jj]);
        }
        var rsv = hh !== ll ? (closes[i] - ll) / (hh - ll) * 100 : 50;
        var prevK = kValues[i - 1] !== '-' ? kValues[i - 1] : 50;
        kValues.push(2 / 3 * prevK + 1 / 3 * rsv);
        dValues.push(2 / 3 * (dValues[i - 1] !== '-' ? dValues[i - 1] : 50) + 1 / 3 * kValues[kValues.length - 1]);
    }
    var lastK = kValues[kValues.length - 1];
    var lastD = dValues[dValues.length - 1];
    var lastJ = (lastK !== '-' && lastD !== '-') ? 3 * lastK - 2 * lastD : '-';
    $('rpK').textContent = lastK !== '-' ? lastK.toFixed(2) : '--';
    $('rpD').textContent = lastD !== '-' ? lastD.toFixed(2) : '--';
    $('rpJ').textContent = lastJ !== '-' ? lastJ.toFixed(2) : '--';
    var rsiVals = calcRSISimple(closes, 6);
    var lastRSI = rsiVals[rsiVals.length - 1];
    $('rpRSI6').textContent = lastRSI !== '-' ? lastRSI.toFixed(2) : '--';
    $('rpRSI6').className = lastRSI !== '-' && lastRSI > 70 ? 'pv up' : (lastRSI !== '-' && lastRSI < 30 ? 'pv down' : 'pv');
}

function calcEMASimple(data, n) {
    if (data.length === 0) return [];
    var k = 2 / (n + 1);
    var ema = typeof data[0] === 'number' ? data[0] : 0;
    var result = [];
    for (var i = 0; i < data.length; i++) {
        if (data[i] === '-') { result.push('-'); continue; }
        if (i === 0) ema = data[0];
        else ema = data[i] * k + ema * (1 - k);
        result.push(ema);
    }
    return result;
}

function calcRSISimple(closes, n) {
    var rsi = [];
    for (var i = 0; i < closes.length; i++) {
        if (i < n) { rsi.push('-'); continue; }
        var gain = 0, loss = 0;
        for (var j = i - n + 1; j <= i; j++) {
            var diff = closes[j] - closes[j - 1];
            if (diff > 0) gain += diff; else loss -= diff;
        }
        var rs = loss !== 0 ? gain / loss : 100;
        rsi.push(100 - 100 / (1 + rs));
    }
    return rsi;
}

// ============================================
// 结算
// ============================================
function showSettlement(result) {
    if (!result) return;
    $('btnStart').disabled = false;

    // 更新持久化统计
    gameStats.totalRounds++;
    gameStats.totalProfit += result.totalReturn;
    gameStats.weeklyProfit += result.totalReturn;
    if (result.totalReturn > 0) gameStats.winCount++;
    else gameStats.loseCount++;

    // 评估AI推荐
    evaluateAIRecommendations();

    // 更新左侧统计显示
    updateStats(null);

    var html = '';

    html += '<div class="settlement-section"><h3>股票揭晓</h3>';
    html += '<div class="reveal-grid">';
    html += '<div><span class="label">股票名称</span><span class="val highlight">' + result.stockName + '</span></div>';
    html += '<div><span class="label">股票代码</span><span class="val">' + result.stockCode + '</span></div>';
    html += '<div><span class="label">时间区间</span><span class="val">' + result.klineRange.start + ' ~ ' + result.klineRange.end + '</span></div>';
    html += '<div><span class="label">区间涨跌</span><span class="val ' + (result.stockChangeRate >= 0 ? 'up' : 'down') + '">' + (result.stockChangeRate >= 0 ? '+' : '') + result.stockChangeRate.toFixed(2) + '%</span></div>';
    html += '</div></div>';

    html += '<div class="settlement-section"><h3>收益总结</h3>';
    html += '<div class="big-number ' + (result.totalReturnRate >= 0 ? 'up' : 'down') + '">' + (result.totalReturnRate >= 0 ? '+' : '') + result.totalReturnRate.toFixed(2) + '%</div>';
    html += '<div class="stats-grid">';
    html += '<div class="stat-item"><span class="label">你的收益率</span><span class="val ' + (result.totalReturnRate >= 0 ? 'up' : 'down') + '">' + (result.totalReturnRate >= 0 ? '+' : '') + result.totalReturnRate.toFixed(2) + '%</span></div>';
    html += '<div class="stat-item"><span class="label">买入持有</span><span class="val ' + (result.buyHoldReturnRate >= 0 ? 'up' : 'down') + '">' + (result.buyHoldReturnRate >= 0 ? '+' : '') + result.buyHoldReturnRate.toFixed(2) + '%</span></div>';
    html += '<div class="stat-item"><span class="label">交易胜率</span><span class="val">' + result.winRate.toFixed(0) + '%</span></div>';
    html += '<div class="stat-item"><span class="label">最大回撤</span><span class="val down">' + result.maxDrawdown.toFixed(2) + '%</span></div>';
    html += '</div></div>';

    html += '<div class="settlement-section"><h3>总评</h3>';
    var reviews = KlineAnalysis.finalReview(result);
    for (var i = 0; i < reviews.length; i++) {
        html += '<p>' + reviews[i] + '</p>';
    }
    html += '</div>';

    if (result.decisions && result.decisions.length > 0) {
        html += '<div class="settlement-section"><h3>操作记录</h3>';
        html += '<table class="trade-table">';
        html += '<tr><th>#</th><th>操作</th><th>价格</th><th>数量</th><th>收益</th></tr>';
        for (var i = 0; i < result.decisions.length; i++) {
            var d = result.decisions[i];
            html += '<tr><td>' + d.step + '</td>';
            html += '<td class="' + d.action + '">' + d.action + '</td>';
            html += '<td>' + (d.price || 0).toFixed(2) + '</td>';
            html += '<td>' + (d.shares ? d.shares + '股' : '-') + '</td>';
            html += '<td style="color:' + (d.profit && d.profit >= 0 ? '#e74c3c' : '#2ecc71') + '">' + (d.profit ? (d.profit >= 0 ? '+' : '') + d.profit.toFixed(2) : '-') + '</td>';
            html += '</tr>';
        }
        html += '</table></div>';
    }

    $('settlementContent').innerHTML = html;
    $('settlementModal').style.display = 'flex';
    renderAssetCurve(result.assetCurve, result.buyHoldReturnRate);
}

function renderAssetCurve(curve, buyHoldRate) {
    if (!curve || curve.length === 0) return;
    var container = document.getElementById('assetCurve');
    var ch = echarts.init(container);
    var steps = curve.map(function(c) { return '第' + c.step + '步'; });
    var assets = curve.map(function(c) { return c.asset; });
    var initAsset = curve[0] ? curve[0].asset : 100000;
    var firstPrice = curve[1] ? curve[1].price : 0;
    var buyHold = curve.map(function(c) {
        if (c.step === 0) return initAsset;
        return initAsset * (c.price / firstPrice);
    });

    ch.setOption({
        grid: { left: '8%', right: '5%', top: '10%', bottom: '5%' },
        xAxis: { type: 'category', data: steps, axisLabel: { color: '#888', fontSize: 10, interval: 4 } },
        yAxis: {
            type: 'value',
            axisLabel: { color: '#888', formatter: function(v) { return (v / 10000).toFixed(1) + '万'; } },
            splitLine: { lineStyle: { color: 'rgba(255,255,255,0.08)' } }
        },
        series: [
            { name: '你的资产', type: 'line', data: assets, smooth: true,
                lineStyle: { color: '#f39c12', width: 2 },
                areaStyle: { color: 'rgba(243,156,18,0.1)' },
                symbol: 'circle', symbolSize: 3 },
            { name: '买入持有', type: 'line', data: buyHold, smooth: true,
                lineStyle: { color: '#3498db', width: 1, type: 'dashed' }, symbol: 'none' }
        ]
    });
    ch.resize();
}

function closeSettlement() {
    $('settlementModal').style.display = 'none';
    document.getElementById('welcomeOverlay').classList.remove('hidden');
}

// ============================================
// 工具函数
// ============================================
function showMsg(msg, type) {
    if (!type) type = 'info';
    var el = $('message');
    el.textContent = msg;
    el.className = 'toast ' + type;
    el.style.display = 'block';
    clearTimeout(el._timer);
    el._timer = setTimeout(function() { el.style.display = 'none'; }, 3000);
}

// ============================================
// 用户系统
// ============================================
function initUser() {
    if (currentUser) {
        $('userId').textContent = currentUser;
    }
}

function openUserModal() {
    if (currentUser) {
        // 已登录，显示用户信息
        return;
    }
    $('userModal').style.display = 'flex';
    $('userNameInput').value = '';
    $('userMsg').textContent = '输入用户名登录或注册新用户';
    $('userNameInput').focus();
}

function closeUserModal() {
    $('userModal').style.display = 'none';
}

function doLogin() {
    var username = $('userNameInput').value.trim();
    if (!username) {
        $('userMsg').textContent = '请输入用户名';
        return;
    }

    var xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/login', true);
    xhr.setRequestHeader('Content-Type', 'application/json');
    xhr.onload = function() {
        if (xhr.status === 200) {
            var resp = JSON.parse(xhr.responseText);
            if (resp.success) {
                currentUser = username;
                localStorage.setItem('kline_user', username);
                $('userId').textContent = username;
                closeUserModal();
                showMsg('欢迎回来，' + username + '！', 'success');
            } else {
                // 用户不存在，自动注册
                doRegister(username);
            }
        }
    };
    xhr.send(JSON.stringify({username: username}));
}

function doRegister(username) {
    var xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/register', true);
    xhr.setRequestHeader('Content-Type', 'application/json');
    xhr.onload = function() {
        if (xhr.status === 200) {
            var resp = JSON.parse(xhr.responseText);
            if (resp.success) {
                currentUser = username;
                localStorage.setItem('kline_user', username);
                $('userId').textContent = username;
                closeUserModal();
                showMsg('注册成功，欢迎 ' + username + '！', 'success');
            } else {
                $('userMsg').textContent = resp.message;
            }
        }
    };
    xhr.send(JSON.stringify({username: username}));
}

function saveGameRecord(result) {
    if (!currentUser || !result) return;
    var xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/save_record', true);
    xhr.setRequestHeader('Content-Type', 'application/json');
    xhr.send(JSON.stringify({
        username: currentUser,
        stockName: result.stockName,
        stockCode: result.stockCode,
        stockChange: result.stockChangeRate,
        returnRate: result.totalReturnRate,
        winRate: result.winRate,
        tradeCount: result.tradeCount,
        maxDrawdown: result.maxDrawdown,
        totalReturn: result.totalReturn
    }));
}

function openRecords() {
    if (!currentUser) {
        showMsg('请先登录', 'warn');
        openUserModal();
        return;
    }
    $('recordsModal').style.display = 'flex';
    $('recordsContent').innerHTML = '<div style="text-align:center;color:#888;padding:30px;">加载中...</div>';

    var xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/records', true);
    xhr.setRequestHeader('Content-Type', 'application/json');
    xhr.onload = function() {
        if (xhr.status === 200) {
            var resp = JSON.parse(xhr.responseText);
            if (resp.success && resp.records) {
                displayRecords(resp.records);
            } else {
                $('recordsContent').innerHTML = '<div style="text-align:center;color:#888;padding:30px;">暂无记录</div>';
            }
        }
    };
    xhr.send(JSON.stringify({username: currentUser}));
}

function displayRecords(records) {
    if (!records || records.length === 0) {
        $('recordsContent').innerHTML = '<div style="text-align:center;color:#888;padding:30px;">暂无训练记录</div>';
        return;
    }
    var html = '<table class="trade-table"><tr><th>#</th><th>股票</th><th>收益率</th><th>涨跌</th><th>胜率</th><th>时间</th></tr>';
    for (var i = records.length - 1; i >= 0; i--) {
        var r = records[i];
        html += '<tr>';
        html += '<td>' + (i + 1) + '</td>';
        html += '<td>' + r.stockName + '</td>';
        html += '<td style="color:' + (r.returnRate >= 0 ? '#e74c3c' : '#2ecc71') + '">' + (r.returnRate >= 0 ? '+' : '') + r.returnRate.toFixed(2) + '%</td>';
        html += '<td style="color:' + (r.stockChange >= 0 ? '#e74c3c' : '#2ecc71') + '">' + (r.stockChange >= 0 ? '+' : '') + r.stockChange.toFixed(2) + '%</td>';
        html += '<td>' + (r.winRate || 0).toFixed(0) + '%</td>';
        html += '<td style="font-size:10px;color:#888;">' + (r.timestamp || '').slice(0, 10) + '</td>';
        html += '</tr>';
    }
    html += '</table>';
    $('recordsContent').innerHTML = html;
}

function closeRecordsModal() {
    $('recordsModal').style.display = 'none';
}

function openLeaderboard() {
    $('leaderboardModal').style.display = 'flex';
    $('leaderboardContent').innerHTML = '<div style="text-align:center;color:#888;padding:30px;">加载中...</div>';

    var xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/leaderboard', true);
    xhr.setRequestHeader('Content-Type', 'application/json');
    xhr.onload = function() {
        if (xhr.status === 200) {
            var resp = JSON.parse(xhr.responseText);
            if (resp.success && resp.leaderboard) {
                displayLeaderboard(resp.leaderboard);
            } else {
                $('leaderboardContent').innerHTML = '<div style="text-align:center;color:#888;padding:30px;">暂无数据</div>';
            }
        }
    };
    xhr.send();
}

function displayLeaderboard(leaderboard) {
    if (!leaderboard || leaderboard.length === 0) {
        $('leaderboardContent').innerHTML = '<div style="text-align:center;color:#888;padding:30px;">暂无数据</div>';
        return;
    }
    var html = '<table class="trade-table"><tr><th>排名</th><th>用户</th><th>总收益</th><th>局数</th><th>胜率</th><th>交易次数</th></tr>';
    for (var i = 0; i < leaderboard.length; i++) {
        var r = leaderboard[i];
        var medal = i === 0 ? '🥇' : (i === 1 ? '🥈' : (i === 2 ? '🥉' : i + 1));
        html += '<tr>';
        html += '<td>' + medal + '</td>';
        html += '<td>' + r.username + '</td>';
        html += '<td style="color:' + (r.totalReturn >= 0 ? '#e74c3c' : '#2ecc71') + '">' + (r.totalReturn >= 0 ? '+' : '') + r.totalReturn.toFixed(2) + '</td>';
        html += '<td>' + r.rounds + '</td>';
        html += '<td>' + r.winRate + '%</td>';
        html += '<td>' + r.totalTrades + '</td>';
        html += '</tr>';
    }
    html += '</table>';
    $('leaderboardContent').innerHTML = html;
}

function closeLeaderboardModal() {
    $('leaderboardModal').style.display = 'none';
}

// 保存记录（在结算时调用）
var origShowSettlement = showSettlement;
showSettlement = function(result) {
    if (origShowSettlement) origShowSettlement(result);
    saveGameRecord(result);
};

// ============================================
// 事件绑定
// ============================================
function bindEvents() {
    $('btnStart').addEventListener('click', startGame);
    $('btnStartBig').addEventListener('click', startGame);
    $('btnBuy').addEventListener('click', handleBuy);
    $('btnWait').addEventListener('click', handleWait);
    $('btnAIAnalysis').addEventListener('click', triggerAIAnalysis);
    document.addEventListener('keydown', function(e) {
        if (e.key === '1') { var b = $('btnBuy'); if (b) b.click(); }
        if (e.key === '2') { var w = $('btnWait'); if (w) w.click(); }
    });
    document.querySelectorAll('.ind-btn').forEach(function(btn) {
        btn.addEventListener('click', function() {
            document.querySelectorAll('.ind-btn').forEach(function(b) { b.classList.remove('active'); });
            this.classList.add('active');
            if (chart) chart.setIndicator(this.dataset.indicator);
        });
    });
    $('btnCloseModal').addEventListener('click', closeSettlement);
    $('btnCloseBtn').addEventListener('click', closeSettlement);
    $('btnPlayAgain').addEventListener('click', function() {
        closeSettlement();
        setTimeout(startGame, 100);
    });

    // 用户系统
    $('userCard').addEventListener('click', openUserModal);
    $('btnHistory').addEventListener('click', openRecords);
    $('btnRanking').addEventListener('click', openLeaderboard);

    // 检索器切换（KNN / GRU）
    document.querySelectorAll('#retrieverSwitch .retriever-btn').forEach(function(btn) {
        btn.addEventListener('click', function() {
            switchRetriever(this.getAttribute('data-retriever'));
        });
    });
}

// 启动
document.addEventListener('DOMContentLoaded', function() {
    init().catch(function(e) {
        showMsg('初始化失败: ' + (e.message || '未知错误'), 'error');
    });
});