/**
 * K线训练营 - 图表模块
 * 包含K线主图、成交量、MACD/KDJ/RSI/BOLL指标
 */

class KlineChart {
    constructor() {
        this.mainChart = echarts.init(document.getElementById('klineMainChart'), 'dark');
        this.volumeChart = echarts.init(document.getElementById('volumeChart'), 'dark');
        this.indicatorChart = echarts.init(document.getElementById('indicatorChart'), 'dark');
        this.currentIndicator = 'MACD';
        this.currentData = null;
        this.referenceCount = 0;
        this.mode = 'classic';

        window.addEventListener('resize', () => {
            this.mainChart.resize();
            this.volumeChart.resize();
            this.indicatorChart.resize();
        });
    }

    setIndicator(indicator) {
        this.currentIndicator = indicator;
        if (this.currentData) {
            this._renderIndicator(this.currentData, this.referenceCount);
        }
    }

    render(klineData, referenceCount = 0, mode = 'classic', signals = []) {
        if (!klineData || klineData.length === 0) return;
        this.currentData = klineData;
        this.referenceCount = referenceCount;
        this.mode = mode;

        this._renderMain(klineData, referenceCount, signals);
        this._renderVolume(klineData, referenceCount);
        this._renderIndicator(klineData, referenceCount);
    }

    /**
     * 渲染K线主图
     */
    _renderMain(klineData, referenceCount, signals = []) {
        const dates = klineData.map(d => d.date);
        const ohlc = klineData.map(d => [d.open, d.close, d.low, d.high]);
        const closes = klineData.map(d => d.close);

        const ma5 = this._calcMA(closes, 5);
        const ma10 = this._calcMA(closes, 10);
        const ma20 = this._calcMA(closes, 20);

        // 基准价（用于百分比显示）
        const baseIdx = Math.min(referenceCount, klineData.length - 1);
        const basePrice = klineData[baseIdx] ? klineData[baseIdx].close : klineData[0].close;

        // 构建买卖标记
        const markData = [];
        for (const sig of signals) {
            const idx = sig.index;
            if (idx < 0 || idx >= dates.length) continue;
            markData.push({
                name: sig.type === 'buy' ? 'B' : 'S',
                xAxis: dates[idx],
                yAxis: sig.price,
                label: {
                    formatter: sig.type === 'buy' ? 'B' : 'S',
                    color: sig.type === 'buy' ? '#e74c3c' : '#2ecc71',
                    fontSize: 16,
                    fontWeight: 'bold',
                    position: 'inside'
                },
                symbol: 'circle',
                symbolSize: 24,
                itemStyle: {
                    color: sig.type === 'buy' ? 'rgba(231,76,60,0.25)' : 'rgba(46,204,113,0.25)',
                    borderColor: sig.type === 'buy' ? '#e74c3c' : '#2ecc71',
                    borderWidth: 2
                }
            });
        }

        const option = {
            backgroundColor: 'transparent',
            animation: false,
            tooltip: {
                trigger: 'axis',
                axisPointer: { type: 'cross' },
                formatter: (params) => this._formatTooltip(params, klineData, referenceCount, basePrice)
            },
            grid: { left: '3%', right: '3%', top: '8%', bottom: '5%' },
            xAxis: {
                type: 'category',
                data: dates,
                axisLine: { lineStyle: { color: '#444' } },
                axisLabel: { color: '#888', fontSize: 10, formatter: () => '|', interval: 'auto' },
                splitLine: { show: false }
            },
            yAxis: {
                type: 'value',
                scale: true,
                splitLine: { lineStyle: { color: 'rgba(255,255,255,0.06)', type: 'dashed' } },
                axisLabel: {
                    color: '#888', fontSize: 10,
                    formatter: (v) => {
                        const pct = ((v - basePrice) / basePrice * 100);
                        return (pct > 0 ? '+' : '') + pct.toFixed(1) + '%';
                    }
                }
            },
            series: [
                {
                    name: 'K线',
                    type: 'candlestick',
                    data: ohlc,
                    itemStyle: {
                        color: '#e74c3c', color0: '#2ecc71',
                        borderColor: '#e74c3c', borderColor0: '#2ecc71'
                    },
                    markPoint: {
                        data: markData,
                        symbol: 'circle',
                        symbolSize: 24,
                        label: {
                            color: '#fff',
                            fontSize: 13,
                            fontWeight: 'bold'
                        }
                    },
                    z: 10
                },
                {
                    name: 'MA5', type: 'line', data: ma5,
                    smooth: true, lineStyle: { color: '#fff', width: 1 },
                    symbol: 'none', z: 5
                },
                {
                    name: 'MA10', type: 'line', data: ma10,
                    smooth: true, lineStyle: { color: '#f39c12', width: 1 },
                    symbol: 'none', z: 5
                },
                {
                    name: 'MA20', type: 'line', data: ma20,
                    smooth: true, lineStyle: { color: '#9b59b6', width: 1 },
                    symbol: 'none', z: 5
                }
            ]
        };

        // 参考区标记线
        if (referenceCount > 0 && referenceCount < dates.length) {
            option.series.push({
                type: 'line', data: [],
                markLine: {
                    silent: true, symbol: 'none',
                    lineStyle: { color: '#f39c12', type: 'dashed', width: 1 },
                    data: [{ xAxis: dates[referenceCount - 1], label: { show: false } }]
                },
                z: 1
            });
        }

        this.mainChart.setOption(option, true);
        this.mainChart.resize();
    }

    /**
     * 渲染成交量
     */
    _renderVolume(klineData, referenceCount) {
        const dates = klineData.map(d => d.date);
        const isUp = klineData.map((d, i) => {
            if (i === 0) return true;
            return d.close >= klineData[i - 1].close;
        });
        const volumes = klineData.map((d, i) => ({
            value: d.volume,
            itemStyle: { color: isUp[i] ? 'rgba(231,76,60,0.5)' : 'rgba(46,204,113,0.5)' }
        }));

        const option = {
            backgroundColor: 'transparent',
            animation: false,
            grid: { left: '3%', right: '3%', top: '5%', bottom: '5%' },
            xAxis: {
                type: 'category', data: dates,
                axisLine: { show: false },
                axisLabel: { show: false },
                splitLine: { show: false }
            },
            yAxis: {
                type: 'value',
                splitLine: { show: false },
                axisLabel: { show: false }
            },
            series: [{
                type: 'bar', data: volumes,
                barWidth: '60%', z: 3
            }]
        };

        this.volumeChart.setOption(option, true);
        this.volumeChart.resize();
    }

    /**
     * 渲染指标
     */
    _renderIndicator(klineData, referenceCount) {
        const closes = klineData.map(d => d.close);
        const highs = klineData.map(d => d.high);
        const lows = klineData.map(d => d.low);
        const volumes = klineData.map(d => d.volume);
        const dates = klineData.map(d => d.date);

        let series = [];
        let yAxis = [{
            type: 'value',
            splitLine: { lineStyle: { color: 'rgba(255,255,255,0.06)', type: 'dashed' } },
            axisLabel: { color: '#888', fontSize: 9 }
        }];

        switch (this.currentIndicator) {
            case 'MACD':
                series = this._calcMACD(closes, dates);
                break;
            case 'KDJ':
                series = this._calcKDJ(highs, lows, closes, dates);
                break;
            case 'RSI':
                series = this._calcRSI(closes, dates);
                break;
            case 'BOLL':
                const boll = this._calcBOLL(closes, dates);
                series = boll.series;
                yAxis = boll.yAxis;
                break;
            case 'W&R':
                series = this._calcWR(highs, lows, closes, dates);
                break;
            case 'BIAS':
                series = this._calcBIAS(closes, dates);
                break;
            default:
                series = this._calcMACD(closes, dates);
        }

        const option = {
            backgroundColor: 'transparent',
            animation: false,
            grid: { left: '3%', right: '3%', top: '8%', bottom: '5%' },
            xAxis: {
                type: 'category', data: dates,
                axisLine: { show: false },
                axisLabel: { show: false },
                splitLine: { show: false }
            },
            yAxis,
            series
        };

        this.indicatorChart.setOption(option, true);
        this.indicatorChart.resize();
    }

    // ============================================
    // 指标计算
    // ============================================

    _calcMACD(closes, dates) {
        const ema12 = this._EMA(closes, 12);
        const ema26 = this._EMA(closes, 26);
        const dif = ema12.map((v, i) => v !== '-' && ema26[i] !== '-' ? v - ema26[i] : '-');
        const dea = this._EMA(dif, 9);
        const macd = dif.map((v, i) => {
            if (v === '-' || dea[i] === '-') return '-';
            const val = (v - dea[i]) * 2;
            return { value: val, itemStyle: { color: val >= 0 ? '#e74c3c' : '#2ecc71' } };
        });

        return [
            { name: 'DIF', type: 'line', data: dif, lineStyle: { color: '#fff', width: 1 }, symbol: 'none', z: 5 },
            { name: 'DEA', type: 'line', data: dea, lineStyle: { color: '#f39c12', width: 1 }, symbol: 'none', z: 5 },
            { name: 'MACD', type: 'bar', data: macd, barWidth: '60%', z: 3 }
        ];
    }

    _calcKDJ(highs, lows, closes, dates) {
        const n = 9;
        const kValues = [], dValues = [], jValues = [];

        for (let i = 0; i < closes.length; i++) {
            if (i < n - 1) {
                kValues.push('-'); dValues.push('-'); jValues.push('-');
                continue;
            }
            let hh = -Infinity, ll = Infinity;
            for (let j = i - n + 1; j <= i; j++) {
                hh = Math.max(hh, highs[j]);
                ll = Math.min(ll, lows[j]);
            }
            const rsv = hh !== ll ? (closes[i] - ll) / (hh - ll) * 100 : 50;
            const prevK = kValues[i - 1] !== '-' ? kValues[i - 1] : 50;
            const prevD = dValues[i - 1] !== '-' ? dValues[i - 1] : 50;
            const k = 2 / 3 * prevK + 1 / 3 * rsv;
            const d = 2 / 3 * prevD + 1 / 3 * k;
            kValues.push(k);
            dValues.push(d);
            jValues.push(3 * k - 2 * d);
        }

        return [
            { name: 'K', type: 'line', data: kValues, lineStyle: { color: '#fff', width: 1 }, symbol: 'none', z: 5 },
            { name: 'D', type: 'line', data: dValues, lineStyle: { color: '#f39c12', width: 1 }, symbol: 'none', z: 5 },
            { name: 'J', type: 'line', data: jValues, lineStyle: { color: '#e74c3c', width: 1 }, symbol: 'none', z: 5 }
        ];
    }

    _calcRSI(closes, dates) {
        const n = 6;
        const rsi = [];
        for (let i = 0; i < closes.length; i++) {
            if (i < n) { rsi.push('-'); continue; }
            let gain = 0, loss = 0;
            for (let j = i - n + 1; j <= i; j++) {
                const diff = closes[j] - closes[j - 1];
                if (diff > 0) gain += diff;
                else loss -= diff;
            }
            const rs = loss !== 0 ? gain / loss : 100;
            rsi.push(100 - 100 / (1 + rs));
        }

        const rsi12 = this._calcRSI(closes, dates, 12);

        return [
            { name: 'RSI6', type: 'line', data: rsi, lineStyle: { color: '#fff', width: 1 }, symbol: 'none', z: 5 },
            ...(rsi12 ? [{ name: 'RSI12', type: 'line', data: rsi12, lineStyle: { color: '#f39c12', width: 1 }, symbol: 'none', z: 5 }] : [])
        ];
    }

    _calcBOLL(closes, dates) {
        const n = 20;
        const mid = [], upper = [], lower = [];

        for (let i = 0; i < closes.length; i++) {
            if (i < n - 1) {
                mid.push('-'); upper.push('-'); lower.push('-');
                continue;
            }
            let sum = 0;
            for (let j = i - n + 1; j <= i; j++) sum += closes[j];
            const avg = sum / n;
            let sqSum = 0;
            for (let j = i - n + 1; j <= i; j++) sqSum += (closes[j] - avg) ** 2;
            const std = Math.sqrt(sqSum / n);
            mid.push(avg);
            upper.push(avg + 2 * std);
            lower.push(avg - 2 * std);
        }

        return {
            series: [
                { name: 'MID', type: 'line', data: mid, lineStyle: { color: '#fff', width: 1 }, symbol: 'none', z: 5 },
                { name: 'UPPER', type: 'line', data: upper, lineStyle: { color: '#e74c3c', width: 1 }, symbol: 'none', z: 5 },
                { name: 'LOWER', type: 'line', data: lower, lineStyle: { color: '#2ecc71', width: 1 }, symbol: 'none', z: 5 }
            ],
            yAxis: [{
                type: 'value',
                scale: true,
                splitLine: { lineStyle: { color: 'rgba(255,255,255,0.06)', type: 'dashed' } },
                axisLabel: { color: '#888', fontSize: 9 }
            }]
        };
    }

    _calcWR(highs, lows, closes, dates) {
        const n = 14;
        const wr = [];
        for (let i = 0; i < closes.length; i++) {
            if (i < n - 1) { wr.push('-'); continue; }
            let hh = -Infinity, ll = Infinity;
            for (let j = i - n + 1; j <= i; j++) {
                hh = Math.max(hh, highs[j]);
                ll = Math.min(ll, lows[j]);
            }
            wr.push(hh !== ll ? (hh - closes[i]) / (hh - ll) * (-100) : -50);
        }
        return [{ name: 'W&R', type: 'line', data: wr, lineStyle: { color: '#fff', width: 1 }, symbol: 'none', z: 5 }];
    }

    _calcBIAS(closes, dates) {
        const bias1 = [], bias2 = [];
        for (let i = 0; i < closes.length; i++) {
            if (i < 5) { bias1.push('-'); bias2.push('-'); continue; }
            let sum5 = 0, sum10 = 0;
            for (let j = i - 5 + 1; j <= i; j++) sum5 += closes[j];
            for (let j = i - 10 + 1; j <= i; j++) sum10 += closes[j];
            bias1.push((closes[i] - sum5 / 5) / (sum5 / 5) * 100);
            bias2.push(i >= 10 ? (closes[i] - sum10 / 10) / (sum10 / 10) * 100 : '-');
        }
        return [
            { name: 'BIAS5', type: 'line', data: bias1, lineStyle: { color: '#fff', width: 1 }, symbol: 'none', z: 5 },
            { name: 'BIAS10', type: 'line', data: bias2, lineStyle: { color: '#f39c12', width: 1 }, symbol: 'none', z: 5 }
        ];
    }

    // ============================================
    // 工具方法
    // ============================================

    _EMA(data, n) {
        const result = [];
        if (data.length === 0) return result;
        const k = 2 / (n + 1);
        let ema = data[0] !== '-' ? data[0] : 0;
        for (let i = 0; i < data.length; i++) {
            if (data[i] === '-') { result.push('-'); continue; }
            if (i === 0) { ema = data[0]; }
            else { ema = data[i] * k + ema * (1 - k); }
            result.push(ema);
        }
        return result;
    }

    _calcMA(data, window) {
        const result = [];
        for (let i = 0; i < data.length; i++) {
            if (i < window - 1) { result.push('-'); }
            else {
                let sum = 0;
                for (let j = i - window + 1; j <= i; j++) sum += data[j];
                result.push(sum / window);
            }
        }
        return result;
    }

    _calcRSI(closes, dates, n = 6) {
        const rsi = [];
        for (let i = 0; i < closes.length; i++) {
            if (i < n) { rsi.push('-'); continue; }
            let gain = 0, loss = 0;
            for (let j = i - n + 1; j <= i; j++) {
                const diff = closes[j] - closes[j - 1];
                if (diff > 0) gain += diff; else loss -= diff;
            }
            const rs = loss !== 0 ? gain / loss : 100;
            rsi.push(100 - 100 / (1 + rs));
        }
        return rsi;
    }

    _formatTooltip(params, klineData, referenceCount, basePrice) {
        if (!params || params.length === 0) return '';
        const p = params[0];
        const idx = p.dataIndex;
        const d = klineData[idx];
        if (!d) return '';

        const prevClose = idx > 0 ? klineData[idx - 1].close : d.open;
        const change = d.close - prevClose;
        const changePct = (change / prevClose * 100);
        const isRef = idx < referenceCount;

        let html = `<div style="font-size:11px;color:#888;">`;
        html += isRef ? '参考区' : `第 ${idx - referenceCount + 1} 根`;
        html += `</div>`;
        html += `<div style="font-size:12px;margin:3px 0;">`;
        html += `开: ${((d.open - basePrice)/basePrice*100).toFixed(1)}% `;
        html += `高: ${((d.high - basePrice)/basePrice*100).toFixed(1)}% `;
        html += `低: ${((d.low - basePrice)/basePrice*100).toFixed(1)}% `;
        html += `收: ${((d.close - basePrice)/basePrice*100).toFixed(1)}%`;
        html += `</div>`;
        const color = change >= 0 ? '#e74c3c' : '#2ecc71';
        html += `<div style="font-size:12px;color:${color};">`;
        html += `涨跌: ${changePct >= 0 ? '+' : ''}${changePct.toFixed(2)}%`;
        html += `</div>`;
        return html;
    }

    resize() {
        this.mainChart.resize();
        this.volumeChart.resize();
        this.indicatorChart.resize();
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = KlineChart;
}