/**
 * K线训练营 - 简化版游戏引擎
 * 仅保留买入/观望操作，持仓时买入=卖出
 */

class GameEngine {
    constructor() {
        this.reset();
    }

    reset() {
        this.state = {
            phase: 'idle',
            stock: null,
            windowStart: 0,
            currentIndex: 0,
            visibleCount: 1,
            initCash: 100000,
            cash: 100000,
            position: 0,       // 持仓股数
            avgCost: 0,         // 持仓均价
            trades: [],
            decisions: [],
            operatedToday: false,
            klineCount: 30,
            round: 0,           // 训练局数
            totalProfit: 0,     // 总收益
            weeklyProfit: 0,    // 周收益
            winCount: 0,        // 盈利次数
            loseCount: 0,       // 亏损次数
            result: null
        };
    }

    /**
     * 开始新一局
     */
    async startGame(stockList, klineCount = 30) {
        if (!stockList || stockList.length === 0) {
            throw new Error('没有可用的股票数据');
        }
        this.reset();
        this.state.klineCount = klineCount;
        this.state.round++;

        // 随机选股
        const stockData = await this._pickRandomStock(stockList);
        if (!stockData) throw new Error('无法加载股票数据');

        const kline = stockData.kline;
        const maxStart = Math.max(20, kline.length - klineCount - 20);
        const windowStart = 20 + Math.floor(Math.random() * Math.max(1, maxStart - 20 + 1));

        this.state.stock = {
            code: stockData.code,
            name: stockData.name,
            allKline: kline,
            kline: kline.slice(windowStart, windowStart + klineCount),
            referenceKline: kline.slice(windowStart - 20, windowStart)
        };
        this.state.windowStart = windowStart;
        this.state.currentIndex = 0;
        this.state.visibleCount = 1;
        this.state.phase = 'playing';

        // 缓存数据
        if (!window.__stockDataCache) window.__stockDataCache = {};
        window.__stockDataCache[stockData.code] = stockData;

        return this.getView();
    }

    async _pickRandomStock(stockList) {
        if (!window.__stockDataCache) window.__stockDataCache = {};

        const shuffled = [...stockList].sort(() => Math.random() - 0.5);
        for (const stock of shuffled) {
            let data = window.__stockDataCache[stock.code];
            if (!data) {
                data = await this._loadStockData(stock.code);
                if (data) window.__stockDataCache[stock.code] = data;
            }
            if (data && data.kline && data.kline.length >= this.state.klineCount + 40) {
                return data;
            }
        }

        // 兜底
        for (const stock of stockList) {
            let data = window.__stockDataCache[stock.code];
            if (!data) {
                data = await this._loadStockData(stock.code);
                if (data) window.__stockDataCache[stock.code] = data;
            }
            if (data && data.kline) return data;
        }
        return null;
    }

    async _loadStockData(code) {
        try {
            const resp = await fetch(`../data/kline/${code}.json`);
            if (!resp.ok) return null;
            return await resp.json();
        } catch (e) {
            return null;
        }
    }

    /**
     * 获取当前视图
     */
    getView() {
        const s = this.state;
        if (!s.stock) return null;

        const ref = s.stock.referenceKline || [];
        const kline = s.stock.kline || [];
        const visible = kline.slice(0, s.visibleCount);
        const allVisible = [...ref, ...visible];
        const cur = kline[s.currentIndex];
        const prevClose = s.currentIndex > 0
            ? kline[s.currentIndex - 1].close
            : (ref.length > 0 ? ref[ref.length - 1].close : (cur ? cur.open : 0));

        const marketValue = s.position * (cur ? cur.close : 0);
        const totalAssets = s.cash + marketValue;
        const returnRate = s.initCash > 0 ? (totalAssets - s.initCash) / s.initCash * 100 : 0;

        // 计算本局收益
        const gameProfit = s.trades.length > 0
            ? s.trades.reduce((sum, t) => sum + (t.profit || 0), 0)
            : marketValue - s.position * s.avgCost;

        // 计算MA值
        const closes = allVisible.map(d => d.close);
        const ma5 = this._calcMA(closes, 5);
        const ma10 = this._calcMA(closes, 10);
        const ma20 = this._calcMA(closes, 20);

        return {
            phase: s.phase,
            currentStep: s.currentIndex + 1,
            totalSteps: s.klineCount,
            visibleCount: s.visibleCount,
            allVisibleKline: allVisible,
            referenceCount: ref.length,
            cash: s.cash,
            position: s.position,
            avgCost: s.avgCost,
            initCash: s.initCash,
            marketValue,
            totalAssets,
            returnRate,
            gameProfit,
            currentKline: cur,
            prevClose,
            hasPosition: s.position > 0,
            operatedToday: s.operatedToday,
            trades: s.trades,
            decisions: s.decisions,
            ma5: ma5[ma5.length - 1],
            ma10: ma10[ma10.length - 1],
            ma20: ma20[ma20.length - 1],
            totalProfit: s.totalProfit,
            weeklyProfit: s.weeklyProfit,
            winCount: s.winCount,
            loseCount: s.loseCount,
            round: s.round,
            stockHidden: true,
            stockCode: s.stock.code,
            stockName: s.stock.name,
            // 当前K线详细信息
            ohlc: cur ? {
                open: cur.open, high: cur.high,
                low: cur.low, close: cur.close,
                change: cur.close - prevClose,
                changePct: prevClose > 0 ? (cur.close - prevClose) / prevClose * 100 : 0
            } : null
        };
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

    /**
     * 买入操作
     * 空仓时买入，持仓时卖出
     */
    buy() {
        const s = this.state;
        if (s.phase !== 'playing') return { success: false, msg: '游戏未开始' };
        if (s.operatedToday) return { success: false, msg: '本根K线已操作过' };

        const kline = s.stock.kline[s.currentIndex];
        if (!kline) return { success: false, msg: '数据异常' };

        const price = kline.close;

        if (s.position > 0) {
            // 持仓中 → 卖出
            const revenue = s.position * price;
            const profit = (price - s.avgCost) * s.position;
            const profitRate = (price - s.avgCost) / s.avgCost * 100;

            s.cash += revenue;
            s.totalProfit += profit;
            if (profit > 0) s.winCount++;
            else s.loseCount++;

            s.trades.push({
                type: 'sell', step: s.currentIndex + 1, price,
                shares: s.position, amount: revenue,
                profit, profitRate, cash: s.cash
            });
            s.decisions.push({
                step: s.currentIndex + 1, action: '卖出', price,
                shares: s.position, profit, profitRate,
                cash: s.cash, position: 0
            });

            s.position = 0;
            s.avgCost = 0;
            s.operatedToday = true;

            let msg = `卖出 ${s.trades[s.trades.length-1].shares}股`;
            if (profit >= 0) msg += `，盈利 ${profit.toFixed(2)}元 🎉`;
            else msg += `，亏损 ${Math.abs(profit).toFixed(2)}元`;
            return { success: true, msg };
        } else {
            // 空仓 → 买入
            let shares = Math.floor(s.cash / price / 100) * 100;
            if (shares <= 0) return { success: false, msg: '资金不足（至少需买100股）' };

            const cost = shares * price;
            s.cash -= cost;
            s.position = shares;
            s.avgCost = price;
            s.operatedToday = true;

            s.trades.push({
                type: 'buy', step: s.currentIndex + 1, price,
                shares, amount: cost, cash: s.cash
            });
            s.decisions.push({
                step: s.currentIndex + 1, action: '买入', price,
                shares, cost, cash: s.cash, position: shares
            });

            return { success: true, msg: `买入 ${shares}股，成本 ${(cost).toFixed(2)}元` };
        }
    }

    /**
     * 观望操作
     */
    wait() {
        const s = this.state;
        if (s.phase !== 'playing') return { success: false, msg: '游戏未开始' };
        if (s.operatedToday) return { success: false, msg: '本根K线已操作过' };

        s.operatedToday = true;
        const action = s.position > 0 ? '持股' : '观望';
        s.decisions.push({
            step: s.currentIndex + 1, action,
            price: s.stock.kline[s.currentIndex].close,
            position: s.position, cash: s.cash
        });
        return { success: true, msg: s.position > 0 ? '持股观望' : '空仓观望' };
    }

    /**
     * 下一步
     */
    next() {
        const s = this.state;
        if (s.phase !== 'playing') return null;
        if (!s.operatedToday) return { error: true, msg: '请先做出决策再继续' };

        if (s.currentIndex >= s.klineCount - 1) {
            return this._settle();
        }

        s.currentIndex++;
        s.visibleCount = s.currentIndex + 1;
        s.operatedToday = false;
        return this.getView();
    }

    /**
     * 结算
     */
    _settle() {
        const s = this.state;
        const stock = s.stock;
        const first = stock.kline[0];
        const last = stock.kline[s.klineCount - 1];

        // 最终资产
        const finalAssets = s.cash + s.position * last.close;
        const totalReturn = finalAssets - s.initCash;
        const totalReturnRate = totalReturn / s.initCash * 100;
        const buyHoldReturn = (last.close - first.close) / first.close * 100;

        // 资产曲线 & 最大回撤
        let maxDrawdown = 0, peakAsset = s.initCash;
        let curCash = s.initCash, curPos = 0;
        const assetCurve = [{ step: 0, asset: s.initCash }];

        for (let i = 0; i < s.klineCount; i++) {
            const k = stock.kline[i];
            const dec = s.decisions.find(d => d.step === i + 1);
            if (dec) {
                if (dec.action === '买入') { curCash = dec.cash; curPos = dec.position || 0; }
                else if (dec.action === '卖出') { curCash = dec.cash; curPos = 0; }
            }
            const asset = curCash + curPos * k.close;
            if (asset > peakAsset) peakAsset = asset;
            const dd = (peakAsset - asset) / peakAsset * 100;
            if (dd > maxDrawdown) maxDrawdown = dd;
            assetCurve.push({ step: i + 1, asset, cash: curCash, pos: curPos, price: k.close });
        }

        // 如果最终持仓，计入总收益
        if (s.position > 0) {
            const profit = (last.close - s.avgCost) * s.position;
            if (profit > 0) s.winCount++;
            else s.loseCount++;
        }

        // 胜率
        const totalTrades = s.winCount + s.loseCount;
        const winRate = totalTrades > 0 ? s.winCount / totalTrades * 100 : 0;

        s.phase = 'finished';
        s.result = {
            stockCode: stock.code, stockName: stock.name,
            klineRange: { start: first.date, end: last.date },
            firstPrice: first.close, lastPrice: last.close,
            stockChangeRate: buyHoldReturn,
            initialAsset: s.initCash, finalAsset: finalAssets,
            totalReturn, totalReturnRate, buyHoldReturnRate: buyHoldReturn,
            tradeCount: totalTrades, winCount: s.winCount,
            winRate, maxDrawdown,
            trades: s.trades, decisions: s.decisions, assetCurve
        };

        // 更新累计统计
        s.totalProfit += totalReturn;
        s.weeklyProfit += totalReturn;

        return { phase: 'finished', result: s.result, view: this.getView() };
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = GameEngine;
}