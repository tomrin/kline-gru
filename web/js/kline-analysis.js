/**
 * K线训练营 - K线形态分析 & 智能讲评系统
 * 识别K线形态、分析趋势，对用户操作进行讲评
 */

class KlineAnalysis {
    static comment(userDecision, klineData, currentIndex, context) {
        if (!klineData || !Array.isArray(klineData) || klineData.length === 0) {
            return this._defaultCommentary(userDecision, '数据不足');
        }
        if (currentIndex < 0 || currentIndex >= klineData.length) {
            return this._defaultCommentary(userDecision, 'K线索引异常');
        }

        const curK = klineData[currentIndex];
        const prevK = currentIndex > 0 ? klineData[currentIndex - 1] : null;
        const prev2K = currentIndex > 1 ? klineData[currentIndex - 2] : null;
        const prev3K = currentIndex > 2 ? klineData[currentIndex - 3] : null;
        const prev4K = currentIndex > 3 ? klineData[currentIndex - 4] : null;

        const recentKlines = klineData.slice(Math.max(0, currentIndex - 20), currentIndex + 1);
        const recent7 = klineData.slice(Math.max(0, currentIndex - 6), currentIndex + 1);
        const recent20 = recentKlines; // 20天数据用于阶段分析

        const pattern = this._recognizePattern(curK, prevK, prev2K, prev3K, prev4K);
        const trend = this._analyzeTrend(recentKlines);
        const volumeAnalysis = this._analyzeVolume(klineData, currentIndex);
        const indicatorAnalysis = this._analyzeIndicators(recentKlines);
        const trapDetection = this._detectTraps(recent7, recent20, trend, volumeAnalysis, indicatorAnalysis);
        const recommendation = this._recommendAdvanced(pattern, trend, volumeAnalysis, indicatorAnalysis, trapDetection, context);

        const prevClose = prevK ? prevK.close : curK.open;
        const changePct = prevClose ? (curK.close - prevClose) / prevClose * 100 : 0;

        const commentary = this._generateCommentaryEnhanced(
            userDecision, pattern, trend, volumeAnalysis, indicatorAnalysis, trapDetection, recommendation,
            curK, prevK, prevClose, changePct, context, recent7
        );
        return commentary;
    }

    static _recognizePattern(cur, prev, prev2, prev3, prev4) {
        if (!cur) return { name: '未知', type: 'neutral', level: 0 };

        const open = Number(cur.open);
        const close = Number(cur.close);
        const high = Number(cur.high);
        const low = Number(cur.low);

        const body = Math.abs(close - open);
        const upperShadow = high - Math.max(open, close);
        const lowerShadow = Math.min(open, close) - low;
        const totalRange = high - low;

        if (totalRange === 0) {
            return { name: '一字线', type: 'neutral', level: 0 };
        }

        const bodyRatio = body / totalRange;
        const upperRatio = upperShadow / totalRange;
        const lowerRatio = lowerShadow / totalRange;
        const isBullish = close >= open;

        const prevClose = prev ? Number(prev.close) : open;
        const changePct = prevClose ? (close - prevClose) / prevClose * 100 : 0;

        let pattern = { name: '', type: 'neutral', level: 0 };

        // 一字线
        if (bodyRatio < 0.01 && upperRatio < 0.01 && lowerRatio < 0.01) {
            if (isBullish) {
                pattern = { name: '一字涨停', type: 'bullish', level: 5 };
            } else {
                pattern = { name: '一字跌停', type: 'bearish', level: 5 };
            }
            return pattern;
        }

        // 光头光脚
        if (upperRatio < 0.03 && lowerRatio < 0.03 && bodyRatio > 0.7) {
            if (isBullish) {
                pattern = { name: '光头光脚大阳线', type: 'bullish', level: 4 };
            } else {
                pattern = { name: '光头光脚大阴线', type: 'bearish', level: 4 };
            }
            return pattern;
        }

        // 十字星
        if (bodyRatio < 0.08) {
            if (upperRatio > 0.45 && lowerRatio > 0.45) {
                pattern = { name: '长腿十字星', type: 'neutral', level: 2 };
            } else if (upperRatio > 0.45 && lowerRatio < 0.1) {
                pattern = { name: '倒T字线', type: 'bearish', level: 2 };
            } else if (lowerRatio > 0.45 && upperRatio < 0.1) {
                pattern = { name: 'T字线', type: 'bullish', level: 2 };
            } else {
                pattern = { name: '十字星', type: 'neutral', level: 1 };
            }
            return pattern;
        }

        // 锤子线
        if (lowerRatio > 0.5 && bodyRatio < 0.35 && upperRatio < 0.2) {
            pattern = { name: '锤子线', type: 'bullish', level: 3 };
            return pattern;
        }

        // 倒锤子
        if (upperRatio > 0.5 && bodyRatio < 0.35 && lowerRatio < 0.2) {
            pattern = { name: '倒锤子线', type: 'bearish', level: 3 };
            return pattern;
        }

        // 纺锤线
        if (bodyRatio < 0.35 && upperRatio > 0.2 && lowerRatio > 0.2) {
            pattern = { name: '纺锤线', type: 'neutral', level: 1 };
            return pattern;
        }

        // 大阳线/大阴线
        if (bodyRatio > 0.6) {
            if (isBullish) {
                pattern = { name: '大阳线', type: 'bullish', level: 3 };
            } else {
                pattern = { name: '大阴线', type: 'bearish', level: 3 };
            }
            return pattern;
        }

        // 中阳线/中阴线
        if (bodyRatio >= 0.3) {
            if (isBullish) {
                pattern = { name: '中阳线', type: 'bullish', level: 1 };
            } else {
                pattern = { name: '中阴线', type: 'bearish', level: 1 };
            }
            return pattern;
        }

        // 小阳线/小阴线
        if (isBullish) {
            pattern = { name: '小阳线', type: 'bullish', level: 0 };
        } else {
            pattern = { name: '小阴线', type: 'bearish', level: 0 };
        }

        return pattern;
    }

    static _analyzeTrend(recentKlines) {
        if (recentKlines.length < 5) {
            return { direction: 'unknown', strength: 0, description: '数据不足，无法判断趋势' };
        }

        const closes = recentKlines.map(k => k.close);
        const len = closes.length;

        const xMean = (len - 1) / 2;
        const yMean = closes.reduce((a, b) => a + b, 0) / len;

        let num = 0, den = 0;
        for (let i = 0; i < len; i++) {
            num += (i - xMean) * (closes[i] - yMean);
            den += (i - xMean) * (i - xMean);
        }

        const slope = den !== 0 ? num / den : 0;
        const avgPrice = yMean;
        const slopePct = avgPrice !== 0 ? slope / avgPrice * 100 : 0;

        const ma5 = closes.slice(-5).reduce((a, b) => a + b, 0) / Math.min(5, closes.length);
        const ma10 = closes.slice(-10).reduce((a, b) => a + b, 0) / Math.min(10, closes.length);

        const maxPrice = Math.max(...closes);
        const minPrice = Math.min(...closes);
        const volatility = avgPrice !== 0 ? (maxPrice - minPrice) / avgPrice * 100 : 0;

        let direction = 'unknown';
        let strength = 0;
        let description = '';

        if (slopePct > 1) {
            direction = 'up';
            strength = Math.min(5, Math.abs(slopePct));
            description = '上升趋势';
            if (ma5 > ma10) {
                description += '，均线多头排列';
                strength += 1;
            }
        } else if (slopePct < -1) {
            direction = 'down';
            strength = Math.min(5, Math.abs(slopePct));
            description = '下降趋势';
            if (ma5 < ma10) {
                description += '，均线空头排列';
                strength += 1;
            }
        } else if (volatility < 5) {
            direction = 'sideways';
            strength = 0;
            description = '横盘震荡';
        } else {
            direction = 'sideways';
            strength = 1;
            description = '区间震荡';
        }

        const shortTerm = closes.slice(-3);
        const momentum = shortTerm.length >= 2
            ? (shortTerm[shortTerm.length - 1] - shortTerm[0]) / shortTerm[0] * 100
            : 0;

        return { direction, strength, slopePct, description, momentum, ma5, ma10, volatility };
    }

    static _analyzeVolume(klineData, currentIndex) {
        const result = { status: 'normal', description: '量能正常', ratio: 0, details: [] };

        if (!klineData || currentIndex < 0 || currentIndex >= klineData.length) return result;

        const cur = klineData[currentIndex];
        if (!cur || !cur.volume) return result;

        const start = Math.max(0, currentIndex - 10);
        let sumVol = 0, count = 0;
        for (let i = start; i < currentIndex; i++) {
            if (klineData[i] && klineData[i].volume) {
                sumVol += klineData[i].volume;
                count++;
            }
        }
        const avgVol = count > 0 ? sumVol / count : cur.volume;
        const ratio = avgVol > 0 ? cur.volume / avgVol : 1;
        result.ratio = ratio;

        const prevClose = currentIndex > 0 ? klineData[currentIndex - 1].close : cur.open;
        const isUp = cur.close >= prevClose;
        const changePct = prevClose > 0 ? (cur.close - prevClose) / prevClose * 100 : 0;

        if (ratio > 2) {
            if (isUp && changePct > 3) {
                result.status = '放量上攻';
                result.description = '成交量是均量的' + ratio.toFixed(1) + '倍，放量大涨，多头强势';
                result.details.push('放量上攻，买盘积极');
            } else if (!isUp && Math.abs(changePct) > 3) {
                result.status = '放量下跌';
                result.description = '成交量是均量的' + ratio.toFixed(1) + '倍，放量大跌，空头强势';
                result.details.push('放量下跌，抛压沉重');
            } else {
                result.status = '放量震荡';
                result.description = '成交量是均量的' + ratio.toFixed(1) + '倍，多空分歧加大';
                result.details.push('放量但价格变动不大，注意变盘');
            }
        } else if (ratio > 1.5) {
            result.status = '温和放量';
            result.description = '成交量是均量的' + ratio.toFixed(1) + '倍，市场活跃度提升';
            result.details.push(isUp ? '量价配合良好' : '放量滞涨，需警惕');
        } else if (ratio < 0.5) {
            result.status = '缩量';
            result.description = '成交量仅为均量的' + (ratio * 100).toFixed(0) + '%，市场交投冷清';
            result.details.push(isUp ? '缩量上涨，上涨动能不足' : '缩量下跌，杀跌动能减弱');
        } else {
            result.status = '量能正常';
            result.description = '成交量与近期均量持平';
        }

        if (currentIndex >= 3) {
            const prev3 = klineData.slice(Math.max(0, currentIndex - 3), currentIndex);
            const volTrend = prev3.reduce((sum, k) => sum + (k.volume || 0), 0) / prev3.length;
            const priceTrend = prev3.reduce((sum, k) => sum + k.close, 0) / prev3.length;

            if (cur.close > priceTrend * 1.05 && cur.volume < volTrend * 0.7) {
                result.details.push('价创新高而量能萎缩，可能存在顶背离');
            }
            if (cur.close < priceTrend * 0.95 && cur.volume > volTrend * 1.5) {
                result.details.push('价创新低同时放量，可能是底部恐慌性抛售');
            }
        }

        return result;
    }

    static _analyzeIndicators(klineData) {
        const result = { macd: null, kdj: null, rsi: null, signals: [] };

        if (!klineData || klineData.length < 20) return result;

        const closes = klineData.map(d => d.close);
        const highs = klineData.map(d => d.high);
        const lows = klineData.map(d => d.low);

        // MACD
        const ema12 = this._calcEMA(closes, 12);
        const ema26 = this._calcEMA(closes, 26);
        const dif = ema12.map((v, i) => (v !== '-' && ema26[i] !== '-') ? v - ema26[i] : '-');
        const dea = this._calcEMA(dif, 9);
        const macdVal = dif.map((v, i) => (v !== '-' && dea[i] !== '-') ? (v - dea[i]) * 2 : '-');

        const lastDIF = dif[dif.length - 1];
        const lastDEA = dea[dea.length - 1];
        const lastMACD = macdVal[macdVal.length - 1];
        const prevDIF = dif.length > 1 ? dif[dif.length - 2] : '-';
        const prevDEA = dea.length > 1 ? dea[dea.length - 2] : '-';

        result.macd = { dif: lastDIF, dea: lastDEA, macd: lastMACD };

        if (lastDIF !== '-' && lastDEA !== '-') {
            if (lastDIF > lastDEA && prevDIF !== '-' && prevDIF <= prevDEA) {
                result.macd.cross = 'golden';
                result.macd.crossDesc = 'MACD金叉，看涨信号';
                result.signals.push({ type: 'macd_golden', text: 'MACD金叉，DIF上穿DEA，看涨信号' });
            } else if (lastDIF < lastDEA && prevDIF !== '-' && prevDIF >= prevDEA) {
                result.macd.cross = 'death';
                result.macd.crossDesc = 'MACD死叉，看跌信号';
                result.signals.push({ type: 'macd_death', text: 'MACD死叉，DIF下穿DEA，看跌信号' });
            } else if (lastDIF > lastDEA) {
                result.macd.status = '多头';
                result.macd.crossDesc = 'DIF在DEA之上，MACD多头状态';
            } else {
                result.macd.status = '空头';
                result.macd.crossDesc = 'DIF在DEA之下，MACD空头状态';
            }
            if (lastMACD !== '-' && macdVal.length > 1 && macdVal[macdVal.length - 2] !== '-') {
                if (lastMACD > 0 && lastMACD > macdVal[macdVal.length - 2]) {
                    result.macd.barDesc = '红柱放大，多头动能增强';
                } else if (lastMACD > 0 && lastMACD < macdVal[macdVal.length - 2]) {
                    result.macd.barDesc = '红柱缩小，多头动能减弱';
                } else if (lastMACD < 0 && lastMACD < macdVal[macdVal.length - 2]) {
                    result.macd.barDesc = '绿柱放大，空头动能增强';
                } else if (lastMACD < 0 && lastMACD > macdVal[macdVal.length - 2]) {
                    result.macd.barDesc = '绿柱缩小，空头动能减弱';
                }
            }
        }

        // KDJ
        const kValues = [], dValues = [];
        for (let i = 0; i < closes.length; i++) {
            if (i < 8) { kValues.push('-'); dValues.push('-'); continue; }
            let hh = -Infinity, ll = Infinity;
            for (let j = i - 8; j <= i; j++) {
                hh = Math.max(hh, highs[j]); ll = Math.min(ll, lows[j]);
            }
            const rsv = hh !== ll ? (closes[i] - ll) / (hh - ll) * 100 : 50;
            const prevK = kValues[i - 1] !== '-' ? kValues[i - 1] : 50;
            kValues.push(2 / 3 * prevK + 1 / 3 * rsv);
            dValues.push(2 / 3 * (dValues[i - 1] !== '-' ? dValues[i - 1] : 50) + 1 / 3 * kValues[kValues.length - 1]);
        }
        const lastK = kValues[kValues.length - 1];
        const lastD = dValues[dValues.length - 1];
        result.kdj = { k: lastK, d: lastD };

        if (lastK !== '-' && lastD !== '-') {
            if (lastK > 80 && lastD > 80) {
                result.kdj.status = '超买';
                result.kdj.desc = 'KDJ处于超买区（>80），短期可能有回调风险';
                result.signals.push({ type: 'kdj_overbought', text: 'KDJ超买（K=' + lastK.toFixed(1) + '），注意回调' });
            } else if (lastK < 20 && lastD < 20) {
                result.kdj.status = '超卖';
                result.kdj.desc = 'KDJ处于超卖区（<20），短期可能有反弹机会';
                result.signals.push({ type: 'kdj_oversold', text: 'KDJ超卖（K=' + lastK.toFixed(1) + '），关注反弹' });
            } else {
                result.kdj.status = '中性';
                result.kdj.desc = 'KDJ处于中性区域';
            }
            if (kValues.length > 2 && dValues.length > 2) {
                const prevK2 = kValues[kValues.length - 2];
                const prevD2 = dValues[dValues.length - 2];
                if (prevK2 !== '-' && prevD2 !== '-') {
                    if (lastK > lastD && prevK2 <= prevD2) {
                        result.kdj.cross = 'golden';
                        result.signals.push({ type: 'kdj_golden', text: 'KDJ金叉，短线看涨' });
                    } else if (lastK < lastD && prevK2 >= prevD2) {
                        result.kdj.cross = 'death';
                        result.signals.push({ type: 'kdj_death', text: 'KDJ死叉，短线看跌' });
                    }
                }
            }
        }

        // RSI
        const rsi6 = this._calcRSIOnly(closes, 6);
        const lastRSI = rsi6[rsi6.length - 1];
        result.rsi = { rsi6: lastRSI };

        if (lastRSI !== '-' && lastRSI !== undefined) {
            if (lastRSI > 70) {
                result.rsi.status = '超买';
                result.rsi.desc = 'RSI(6)=' + lastRSI.toFixed(1) + '，处于超买区（>70）';
                result.signals.push({ type: 'rsi_overbought', text: 'RSI超买（' + lastRSI.toFixed(1) + '），警惕回调' });
            } else if (lastRSI < 30) {
                result.rsi.status = '超卖';
                result.rsi.desc = 'RSI(6)=' + lastRSI.toFixed(1) + '，处于超卖区（<30）';
                result.signals.push({ type: 'rsi_oversold', text: 'RSI超卖（' + lastRSI.toFixed(1) + '），关注反弹' });
            } else {
                result.rsi.status = '中性';
            }
        }

        return result;
    }

    static _calcEMA(data, n) {
        if (data.length === 0) return [];
        const k = 2 / (n + 1);
        let ema = typeof data[0] === 'number' ? data[0] : 0;
        const result = [];
        for (let i = 0; i < data.length; i++) {
            if (data[i] === '-') { result.push('-'); continue; }
            if (i === 0) ema = data[0];
            else ema = data[i] * k + ema * (1 - k);
            result.push(ema);
        }
        return result;
    }

    static _calcRSIOnly(closes, n) {
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

    static _detectPhase(recent7, recent20, volume, indicator) {
        const result = { phase: '中性', phaseType: 'neutral', traps: [], tests: [], details: [] };
        if (!recent7 || recent7.length < 5) return result;

        const closes = recent7.map(d => d.close);
        const highs = recent7.map(d => d.high);
        const lows = recent7.map(d => d.low);
        const volumes = recent7.map(d => d.volume || 0);
        const avgVol = volumes.reduce((a, b) => a + b, 0) / volumes.length;

        // 20天数据
        const c20 = recent20 ? recent20.map(d => d.close) : closes;
        const h20 = recent20 ? recent20.map(d => d.high) : highs;
        const l20 = recent20 ? recent20.map(d => d.low) : lows;
        const v20 = recent20 ? recent20.map(d => d.volume || 0) : volumes;

        const priceChange = (closes[closes.length - 1] - closes[0]) / closes[0] * 100;
        const highest = Math.max(...highs);
        const lowest = Math.min(...lows);
        const range = (highest - lowest) / lowest * 100;

        // 20天涨跌幅
        const change20 = c20.length > 1 ? (c20[c20.length - 1] - c20[0]) / c20[0] * 100 : 0;
        const high20 = Math.max(...h20);
        const low20 = Math.min(...l20);
        const range20 = (high20 - low20) / low20 * 100;

        let upVol = 0, downVol = 0, upDays = 0, downDays = 0;
        for (let i = 1; i < recent7.length; i++) {
            if (closes[i] >= closes[i - 1]) {
                upVol += volumes[i]; upDays++;
            } else {
                downVol += volumes[i]; downDays++;
            }
        }

        // ===== 一、建仓类 =====

        // 打压建仓（挖坑吸筹）：快速砸盘+大阴线+恐慌放量+不再创新低+缩量企稳
        if (recent7.length >= 5) {
            var hasBigDrop = false, hasStabilize = false;
            for (var i = 1; i < recent7.length; i++) {
                var drop = (recent7[i].close - recent7[i-1].close) / recent7[i-1].close * 100;
                if (drop < -4 && recent7[i].volume > avgVol * 1.5) hasBigDrop = true;
            }
            // 最近2根不再创新低
            var last2Low = Math.min(recent7[recent7.length-1].low, recent7[recent7.length-2].low);
            var prevLow = Math.min(...lows.slice(0, lows.length-2));
            if (last2Low >= prevLow * 0.99) hasStabilize = true;
            // 最近缩量
            var last2AvgVol = (volumes[volumes.length-1] + volumes[volumes.length-2]) / 2;
            if (hasBigDrop && hasStabilize && last2AvgVol < avgVol * 0.8) {
                result.phase = '打压建仓（挖坑吸筹）';
                result.phaseType = 'accumulation';
                result.details.push('快速砸盘杀跌后缩量企稳，不再创新低，典型打压建仓特征');
            }
        }

        // 拉高建仓：小幅抬升+持续温和放量+回调缩量+不回踩启动低点
        if (result.phaseType === 'neutral' && recent7.length >= 7) {
            var upCount = 0, upVolTotal = 0, downVolTotal = 0;
            for (var i = 1; i < recent7.length; i++) {
                if (closes[i] > closes[i-1]) { upCount++; upVolTotal += volumes[i]; }
                else { downVolTotal += volumes[i]; }
            }
            var startPrice = closes[0];
            var endPrice = closes[closes.length - 1];
            var risePct = (endPrice - startPrice) / startPrice * 100;
            var minPrice = Math.min(...lows);
            var volRatio = downVolTotal > 0 ? upVolTotal / downVolTotal : 3;

            if (upCount >= 4 && risePct > 2 && risePct < 15 && volRatio > 1.3 && minPrice >= startPrice * 0.98) {
                result.phase = '拉高建仓';
                result.phaseType = 'accumulation';
                result.details.push('价格小幅抬升，上涨放量回调缩量，不回踩启动低点，拉高建仓特征');
            }
        }

        // 震荡磨底（磨筹）：窄幅箱体+涨跌幅度小+成交量持续萎缩
        if (result.phaseType === 'neutral' && range20 < 12 && Math.abs(change20) < 5) {
            var volDeclining = true;
            var half = Math.floor(v20.length / 2);
            var avgVol1 = v20.slice(0, half).reduce((a,b) => a+b, 0) / half;
            var avgVol2 = v20.slice(-half).reduce((a,b) => a+b, 0) / half;
            if (avgVol2 < avgVol1 * 0.7) {
                result.phase = '震荡磨底（磨筹）';
                result.phaseType = 'accumulation';
                result.details.push('长期窄幅震荡，成交量持续萎缩，磨筹特征明显');
            }
        }

        // 普通吸筹
        if (result.phaseType === 'neutral' && range < 8 && priceChange > -5 && priceChange < 3) {
            if (downDays > 0 && upDays > 0 && downVol / downDays > upVol / upDays * 1.2) {
                result.phase = '吸筹阶段';
                result.phaseType = 'accumulation';
                result.details.push('近7天震荡中下跌放量、上涨缩量，典型吸筹特征');
            }
        }

        // ===== 二、洗盘类 =====

        // 浮筹清洗（短线洗盘）：上涨后回调+缩量+关键支撑止跌
        if (result.phaseType === 'neutral' && recent7.length >= 5) {
            var roseFirst = false, droppedLater = false, volShrink = false;
            var mid = Math.floor(recent7.length / 2);
            var firstHalfRise = (closes[mid] - closes[0]) / closes[0] * 100;
            var secondHalfDrop = (closes[closes.length-1] - closes[mid]) / closes[mid] * 100;
            if (firstHalfRise > 3) roseFirst = true;
            if (secondHalfDrop < -1) droppedLater = true;
            var dropVol = volumes.slice(mid).reduce((a,b) => a+b, 0) / (recent7.length - mid);
            var riseVol = volumes.slice(0, mid).reduce((a,b) => a+b, 0) / mid;
            if (dropVol < riseVol * 0.7) volShrink = true;
            if (roseFirst && droppedLater && volShrink) {
                result.phase = '浮筹清洗（短线洗盘）';
                result.phaseType = 'wash';
                result.details.push('上涨后缩量回调，抛压枯竭，短线洗盘特征');
            }
        }

        // 长上影洗盘（仙人试盘）：冲高回落+量不大+次日收回
        if (result.phaseType === 'neutral' && recent7.length >= 3) {
            var last = recent7[recent7.length - 1];
            var prev2 = recent7[recent7.length - 2];
            var prev3 = recent7[recent7.length - 3];
            var upShadow = prev2.high - Math.max(prev2.open, prev2.close);
            var shadowRatio = prev2.high - prev2.low > 0 ? upShadow / (prev2.high - prev2.low) : 0;
            if (shadowRatio > 0.5 && prev2.volume < avgVol * 1.1 && last.close > prev2.high * 0.98) {
                result.phase = '长上影洗盘（仙人试盘）';
                result.phaseType = 'wash';
                result.details.push('长上影缩量，次日快速收回，仙人试盘洗盘特征');
            }
        }

        // 破位假摔（黄金坑）：跌破支撑+放量有限+快速拉回
        if (result.phaseType === 'neutral' && recent7.length >= 5) {
            var breakLow = false, recover = false;
            var supportLevel = Math.min(...closes.slice(0, Math.floor(recent7.length/2)));
            for (var i = Math.floor(recent7.length/2); i < recent7.length; i++) {
                if (lows[i] < supportLevel * 0.97) breakLow = true;
            }
            if (closes[closes.length-1] > supportLevel * 1.01) recover = true;
            if (breakLow && recover) {
                result.phase = '破位假摔（黄金坑）';
                result.phaseType = 'wash';
                result.details.push('跌破支撑后快速收回，假破位洗盘特征');
            }
        }

        // ===== 三、拉升类 =====

        // 边拉边洗：涨一段就震荡+不深度回调+横盘代替回调
        if (result.phaseType === 'neutral' && recent7.length >= 7) {
            var totalRise = (closes[closes.length-1] - closes[0]) / closes[0] * 100;
            var maxDrop = 0;
            for (var i = 1; i < recent7.length; i++) {
                var dd = (closes[i] - Math.max(...closes.slice(0, i))) / Math.max(...closes.slice(0, i)) * 100;
                if (dd < maxDrop) maxDrop = dd;
            }
            if (totalRise > 3 && totalRise < 15 && maxDrop > -4) {
                result.phase = '边拉边洗';
                result.phaseType = 'markup';
                result.details.push('边拉升边震荡，回调不深，以横盘代替回调');
            }
        }

        // 锁仓拉升：缩量上涨
        if (result.phaseType === 'neutral' && recent7.length >= 5) {
            var riseVolumes = [];
            for (var i = 1; i < recent7.length; i++) {
                if (closes[i] > closes[i-1]) riseVolumes.push(volumes[i]);
            }
            var avgRiseVol = riseVolumes.length > 0 ? riseVolumes.reduce((a,b) => a+b, 0) / riseVolumes.length : 0;
            if (priceChange > 5 && avgRiseVol < avgVol * 0.6 && avgRiseVol > 0) {
                result.phase = '锁仓拉升';
                result.phaseType = 'markup';
                result.details.push('缩量上涨，筹码高度锁定，锁仓拉升特征');
            }
        }

        // ===== 四、出货/派筹类 =====

        // 高位震荡出货：高位箱体+成交量维持高位
        if (result.phaseType === 'neutral' && range20 > 5 && range20 < 15 && Math.abs(change20) < 5) {
            var avgVol20 = v20.reduce((a,b) => a+b, 0) / v20.length;
            var half20 = Math.floor(v20.length / 2);
            var vol1 = v20.slice(0, half20).reduce((a,b) => a+b, 0) / half20;
            var vol2 = v20.slice(-half20).reduce((a,b) => a+b, 0) / half20;
            if (vol2 > vol1 * 0.8 && avgVol20 > 1000000) {
                result.phase = '高位震荡出货';
                result.phaseType = 'distribution';
                result.details.push('高位长时间箱体震荡，成交量维持高位，注意派发风险');
            }
        }

        // 缓跌出货（阴跌派筹）：连续小阴小阳慢慢往下
        if (result.phaseType === 'neutral' && recent7.length >= 5) {
            var smallCandles = 0;
            for (var i = 1; i < recent7.length; i++) {
                var bodyRatio = Math.abs(closes[i] - recent7[i].open) / (recent7[i].high - recent7[i].low + 0.01);
                if (bodyRatio < 0.4 && closes[i] < closes[i-1]) smallCandles++;
            }
            if (smallCandles >= 3 && priceChange < 0 && priceChange > -8) {
                result.phase = '缓跌出货（阴跌派筹）';
                result.phaseType = 'distribution';
                result.details.push('连续小阴线阴跌，每天有资金出逃，阴跌派筹特征');
            }
        }

        // 跳水出货（断崖派发）：连续大阴线+放量暴跌
        if (recent7.length >= 3) {
            var bigDrops = 0;
            for (var i = 1; i < recent7.length; i++) {
                var d = (closes[i] - closes[i-1]) / closes[i-1] * 100;
                if (d < -3 && volumes[i] > avgVol * 1.3) bigDrops++;
            }
            if (bigDrops >= 2) {
                result.phase = '跳水出货（断崖派发）';
                result.phaseType = 'distribution';
                result.details.push('连续大阴线放量暴跌，不计成本出逃');
            }
        }

        // 普通派筹
        if (result.phaseType === 'neutral' && range < 8 && priceChange > -5 && priceChange < 3) {
            if (upDays > 0 && downDays > 0 && upVol / upDays > downVol / downDays * 1.2) {
                result.phase = '派筹阶段';
                result.phaseType = 'distribution';
                result.details.push('近7天震荡中上涨放量、下跌缩量，需警惕派筹行为');
            }
        }

        // ===== 五、其他经典盘口行为 =====

        // 对倒拉升：成交量急剧放大+长上影
        if (result.phaseType === 'neutral' && recent7.length >= 3) {
            var last2 = recent7[recent7.length - 1];
            var upShadow2 = last2.high - Math.max(last2.open, last2.close);
            var totalRange2 = last2.high - last2.low;
            if (totalRange2 > 0 && upShadow2 / totalRange2 > 0.4 && last2.volume > avgVol * 2) {
                result.tests.push('对倒拉升');
                result.details.push('成交量急剧放大且留长上影，可能存在对倒拉升行为');
            }
        }

        // 缩量反弹（弱势反弹）：下跌途中反弹无量
        if (result.phaseType === 'neutral' && recent7.length >= 5) {
            var firstHalfDrop2 = (closes[Math.floor(recent7.length/2)] - closes[0]) / closes[0] * 100;
            var lastHalfRise = (closes[closes.length-1] - closes[Math.floor(recent7.length/2)]) / closes[Math.floor(recent7.length/2)] * 100;
            if (firstHalfDrop2 < -3 && lastHalfRise > 0 && volumes[volumes.length-1] < avgVol * 0.7) {
                result.phase = '缩量反弹（弱势反弹）';
                result.phaseType = 'neutral';
                result.details.push('下跌后反弹无量，弱势反弹，警惕继续下跌');
            }
        }

        // 放量止跌：持续下跌后放量不再创新低
        if (result.phaseType === 'neutral' && recent7.length >= 5) {
            var dropped = false;
            for (var i = 1; i < recent7.length; i++) {
                if (closes[i] < closes[i-1]) dropped = true;
            }
            var lastVol = volumes[volumes.length-1];
            var lastClose = closes[closes.length-1];
            var prevLowest = Math.min(...closes.slice(0, closes.length-1));
            if (dropped && lastVol > avgVol * 1.3 && lastClose >= prevLowest * 0.99) {
                result.phase = '放量止跌';
                result.phaseType = 'neutral';
                result.details.push('持续下跌后放量不再创新低，恐慌筹码释放，可能阶段性见底');
            }
        }

        // 回踩确认
        if (result.phaseType === 'neutral' && recent7.length >= 5) {
            var firstHalfRise2 = (closes[Math.floor(recent7.length/2)] - closes[0]) / closes[0] * 100;
            if (firstHalfRise2 > 3) {
                var retrace = (closes[closes.length-1] - Math.max(...closes.slice(0, Math.floor(recent7.length/2)))) / Math.max(...closes.slice(0, Math.floor(recent7.length/2))) * 100;
                if (retrace > -3 && retrace < 0) {
                    result.tests.push('回踩确认');
                    result.details.push('突破后回踩支撑，缩量止跌，回踩确认特征');
                }
            }
        }

        // 向上测试 / 向下测试（基于3天数据）
        if (recent7.length >= 3) {
            const last3 = recent7.slice(-3);
            const l3Highs = last3.map(d => d.high);
            const l3Lows = last3.map(d => d.low);
            const l3Closes = last3.map(d => d.close);
            if (l3Highs[0] > l3Closes[0] * 1.03 && l3Closes[2] < l3Closes[0]) {
                result.tests.push('向上测试');
                result.details.push('出现向上测试：价格快速拉升后回落，测试上方抛压');
            }
            if (l3Lows[0] < l3Closes[0] * 0.97 && l3Closes[2] > l3Closes[0]) {
                result.tests.push('向下测试');
                result.details.push('出现向下测试：价格快速下探后收回，测试下方支撑');
            }
        }

        // 诱多 / 诱空（基于4天数据）
        if (recent7.length >= 4) {
            const mid = recent7.slice(1, -1);
            const midLows = mid.map(d => d.low);
            const midHighs = mid.map(d => d.high);
            const last = recent7[recent7.length - 1];

            const recentLow = Math.min(...midLows);
            if (last.low <= recentLow * 0.98 && last.close > recentLow * 0.99) {
                result.traps.push('诱空');
                result.details.push('识别诱空信号：价格跌破近期低点后快速收回，空头陷阱');
            }

            const recentHigh = Math.max(...midHighs);
            if (last.high >= recentHigh * 1.02 && last.close < recentHigh * 1.01) {
                result.traps.push('诱多');
                result.details.push('识别诱多信号：价格突破近期高点后快速回落，多头陷阱');
            }
        }

        // MACD辅助判断
        if (indicator && indicator.macd) {
            const macd = indicator.macd;
            if (result.phaseType === 'neutral' && macd.status === '空头' && macd.barDesc && macd.barDesc.includes('缩小')) {
                result.phase = '可能吸筹';
                result.phaseType = 'accumulation';
                result.details.push('MACD绿柱缩小，空头动能减弱，可能是吸筹末期');
            }
            if (result.phaseType === 'neutral' && macd.status === '多头' && macd.barDesc && macd.barDesc.includes('缩小')) {
                result.phase = '可能派筹';
                result.phaseType = 'distribution';
                result.details.push('MACD红柱缩小，多头动能减弱，需警惕派筹');
            }
        }

        // 量能辅助判断
        if (avgVol > 0 && volume && volume.ratio) {
            if (volume.ratio > 1.8 && result.phaseType === 'neutral') {
                if (priceChange > 0) {
                    result.phase = '放量上攻';
                    result.details.push('近期放量上攻，资金积极介入');
                } else if (priceChange < -2) {
                    result.phase = '放量下跌';
                    result.details.push('近期放量下跌，资金出逃明显');
                }
            }
        }

        return result;
    }

    static _detectTraps(recent7, recent20, trend, volume, indicator) {
        const recentKlines = recent7;
        const phase = this._detectPhase(recent7, recent20, volume, indicator);
        return {
            hasTrap: phase.traps.length > 0,
            traps: phase.traps,
            phase: phase.phase,
            phaseType: phase.phaseType,
            tests: phase.tests,
            details: phase.details
        };
    }

    static _recommendAdvanced(pattern, trend, volume, indicator, trapDetection, context) {
        const { hasPosition } = context;
        let action = '观望';
        let reasons = [];
        let confidence = 0;
        let buySignals = 0, sellSignals = 0;

        if (pattern.type === 'bullish') {
            buySignals += pattern.level;
            reasons.push('形态看涨: ' + pattern.name);
        } else if (pattern.type === 'bearish') {
            sellSignals += pattern.level;
            reasons.push('形态看跌: ' + pattern.name);
        }

        if (trend.direction === 'up') {
            buySignals += 2;
            reasons.push('趋势向上');
        } else if (trend.direction === 'down') {
            sellSignals += 2;
            reasons.push('趋势向下');
        }

        if (volume.status === '放量上攻') {
            buySignals += 3;
            reasons.push('放量上攻，量价配合');
        } else if (volume.status === '放量下跌') {
            sellSignals += 3;
            reasons.push('放量下跌，抛压沉重');
        } else if (volume.status === '放量震荡') {
            sellSignals += 1;
            reasons.push('放量滞涨，分歧加大');
        } else if (volume.status === '缩量') {
            if (trend.direction === 'up') {
                sellSignals += 1;
                reasons.push('缩量上涨，动能不足');
            } else {
                buySignals += 1;
                reasons.push('缩量下跌，杀跌衰竭');
            }
        }

        if (indicator && indicator.signals) {
            for (const sig of indicator.signals) {
                if (sig.type.includes('golden') || sig.type.includes('oversold')) {
                    buySignals += 2;
                    reasons.push(sig.text);
                } else if (sig.type.includes('death') || sig.type.includes('overbought')) {
                    sellSignals += 2;
                    reasons.push(sig.text);
                }
            }
        }

        if (indicator && indicator.macd) {
            const macd = indicator.macd;
            if (macd.barDesc && macd.barDesc.includes('放大')) {
                if (macd.barDesc.includes('多头')) buySignals += 1;
                else if (macd.barDesc.includes('空头')) sellSignals += 1;
            }
        }

        // 陷阱检测信号
        if (trapDetection) {
            if (trapDetection.hasTrap) {
                if (trapDetection.traps.includes('诱空')) {
                    buySignals += 3;
                    reasons.push('诱空陷阱识别，空头陷阱后看涨');
                }
                if (trapDetection.traps.includes('诱多')) {
                    sellSignals += 3;
                    reasons.push('诱多陷阱识别，多头陷阱后看跌');
                }
            }
            if (trapDetection.phaseType === 'accumulation') {
                buySignals += 2;
                reasons.push('当前处于吸筹阶段，主力资金在低位收集筹码');
            } else if (trapDetection.phaseType === 'distribution') {
                sellSignals += 2;
                reasons.push('当前处于派筹阶段，主力资金在派发筹码');
            }
            if (trapDetection.tests && trapDetection.tests.length > 0) {
                for (const test of trapDetection.tests) {
                    if (test === '向上测试') {
                        sellSignals += 1;
                        reasons.push('向上测试失败，上方抛压较重');
                    } else if (test === '向下测试') {
                        buySignals += 1;
                        reasons.push('向下测试获得支撑，下方承接有力');
                    }
                }
            }
        }

        const netScore = buySignals - sellSignals;

        if (hasPosition) {
            if (netScore <= -3) {
                action = '卖出';
                confidence = Math.min(5, Math.abs(netScore));
                reasons.push('综合信号偏空，建议卖出规避风险');
            } else if (netScore >= 3) {
                action = '持股';
                confidence = Math.min(5, netScore);
                reasons.push('综合信号偏多，持股待涨');
            } else {
                action = '持股';
                confidence = 2;
                reasons.push('信号中性，持股观望');
            }
        } else {
            if (netScore >= 3) {
                action = '买入';
                confidence = Math.min(5, netScore);
                reasons.push('综合信号偏多，可考虑买入');
            } else if (netScore <= -3) {
                action = '观望';
                confidence = Math.min(5, Math.abs(netScore));
                reasons.push('综合信号偏空，建议观望');
            } else {
                action = '观望';
                confidence = 2;
                reasons.push('信号不明确，等待更清晰的机会');
            }
        }

        return { action, reason: reasons.join('；'), confidence, buySignals, sellSignals, netScore };
    }

    static _generateCommentaryEnhanced(userDecision, pattern, trend, volume, indicator, trapDetection, recommendation,
                                        curK, prevK, prevClose, changePct, context, recent7) {
        const { hasPosition, avgCost, cash } = context;

        const analysis = this._generateAnalysis(pattern, trend, volume, indicator, trapDetection, curK, prevK, changePct);
        const judgment = this._generateJudgment(pattern, trend, volume, indicator, trapDetection, recommendation, hasPosition);
        const prediction = this._generatePrediction(pattern, trend, volume, indicator, curK, prevK, trapDetection);
        const supportResistance = this._calcSupportResistance(trend, curK);

        const c = {
            step: userDecision.step || 0,
            userAction: userDecision.action,
            pattern: pattern,
            changePct: changePct,
            trend: trend,
            volume: volume,
            indicator: indicator,
            trapDetection: trapDetection,
            recommended: recommendation,
            analysis: analysis,
            judgment: judgment,
            prediction: prediction,
            supportResistance: supportResistance,
            comparison: null,
            lessons: [],
            summary: ''
        };

        const isMatch = userDecision.action === recommendation.action;
        c.comparison = {
            isMatch: isMatch,
            userAction: userDecision.action,
            recommendedAction: recommendation.action,
            evaluation: isMatch
                ? '你的选择和推荐操作一致，判断正确！'
                : '你的选择和推荐操作不同，来看看为什么'
        };

        this._generateLessonsEnhanced(c, curK, prevK, prevClose, trapDetection);
        c.summary = this._buildSummaryEnhanced(c, trapDetection);

        return c;
    }

    static _generateAnalysis(pattern, trend, volume, indicator, trapDetection, curK, prevK, changePct) {
        const parts = [];

        if (trapDetection) {
            if (trapDetection.details && trapDetection.details.length > 0) {
                for (const d of trapDetection.details) {
                    parts.push('阶段分析: ' + d);
                }
            }
            if (trapDetection.traps && trapDetection.traps.length > 0) {
                for (const t of trapDetection.traps) {
                    parts.push('陷阱识别: 识别' + t + '信号');
                }
            }
            if (trapDetection.tests && trapDetection.tests.length > 0) {
                for (const t of trapDetection.tests) {
                    parts.push('测试行为: 出现' + t);
                }
            }
        }

        if (pattern.name && pattern.name !== '未知') {
            const signal = pattern.type === 'bullish' ? '看涨' : pattern.type === 'bearish' ? '看跌' : '中性';
            parts.push('K线形态: ' + pattern.name + '（' + signal + '信号，强度' + pattern.level + '/5）');
        }

        if (trend.direction === 'up') {
            parts.push('趋势: 上升趋势，强度' + trend.strength.toFixed(1) + '/5');
            if (trend.momentum > 0) {
                parts.push('近期动量: 向上（+' + trend.momentum.toFixed(1) + '%）');
            }
        } else if (trend.direction === 'down') {
            parts.push('趋势: 下降趋势，强度' + trend.strength.toFixed(1) + '/5');
            if (trend.momentum < 0) {
                parts.push('近期动量: 向下（' + trend.momentum.toFixed(1) + '%）');
            }
        } else {
            parts.push('趋势: 横盘震荡，波动率' + (trend.volatility || 0).toFixed(1) + '%');
        }

        if (volume && volume.status !== 'normal') {
            parts.push('量能: ' + volume.status + '（' + volume.description + '）');
        } else if (volume) {
            parts.push('量能: 正常水平');
        }

        if (indicator && indicator.macd) {
            const macd = indicator.macd;
            if (macd.cross === 'golden') {
                parts.push('MACD: 金叉信号，DIF上穿DEA，趋势转多');
            } else if (macd.cross === 'death') {
                parts.push('MACD: 死叉信号，DIF下穿DEA，趋势转空');
            } else if (macd.status === '多头') {
                parts.push('MACD: 多头排列，DIF在DEA之上');
            } else if (macd.status === '空头') {
                parts.push('MACD: 空头排列，DIF在DEA之下');
            }
            if (macd.barDesc) {
                parts.push('MACD柱: ' + macd.barDesc);
            }
        }

        if (indicator && indicator.kdj) {
            const kdj = indicator.kdj;
            if (kdj.status === '超买') {
                parts.push('KDJ: 超买区域（K=' + kdj.k.toFixed(1) + '），短期或有回调需求');
            } else if (kdj.status === '超卖') {
                parts.push('KDJ: 超卖区域（K=' + kdj.k.toFixed(1) + '），短期或有反弹机会');
            }
            if (kdj.cross === 'golden') {
                parts.push('KDJ: K线上穿D线，短线金叉信号');
            } else if (kdj.cross === 'death') {
                parts.push('KDJ: K线下穿D线，短线死叉信号');
            }
        }

        if (indicator && indicator.rsi) {
            const rsi = indicator.rsi;
            if (rsi.status === '超买') {
                parts.push('RSI: 超买区（' + rsi.rsi6.toFixed(1) + '），短期上涨过度');
            } else if (rsi.status === '超卖') {
                parts.push('RSI: 超卖区（' + rsi.rsi6.toFixed(1) + '），短期下跌过度');
            }
        }

        if (Math.abs(changePct) > 0.5) {
            parts.push('涨跌: ' + (changePct >= 0 ? '+' : '') + changePct.toFixed(2) + '%');
        }

        return parts;
    }

    static _generateJudgment(pattern, trend, volume, indicator, trapDetection, recommendation, hasPosition) {
        const parts = [];

        let bullPower = 0, bearPower = 0;

        if (pattern.type === 'bullish') bullPower += pattern.level;
        else if (pattern.type === 'bearish') bearPower += pattern.level;

        if (trend.direction === 'up') bullPower += Math.ceil(trend.strength);
        else if (trend.direction === 'down') bearPower += Math.ceil(trend.strength);

        if (volume && volume.status === '放量上攻') bullPower += 3;
        else if (volume && volume.status === '放量下跌') bearPower += 3;
        else if (volume && volume.status === '缩量' && trend.direction === 'down') bullPower += 1;

        if (indicator && indicator.macd) {
            if (indicator.macd.cross === 'golden') bullPower += 3;
            else if (indicator.macd.cross === 'death') bearPower += 3;
            else if (indicator.macd.status === '多头') bullPower += 1;
            else if (indicator.macd.status === '空头') bearPower += 1;
        }

        const totalPower = bullPower + bearPower;
        if (totalPower > 0) {
            const bullRatio = (bullPower / totalPower * 100).toFixed(0);
            if (bullRatio > 60) {
                parts.push('综合判断: 多方占优（多方' + bullRatio + '% vs 空方' + (100 - bullRatio) + '%），整体偏多');
            } else if (bullRatio < 40) {
                parts.push('综合判断: 空方占优（多方' + bullRatio + '% vs 空方' + (100 - bullRatio) + '%），整体偏空');
            } else {
                parts.push('综合判断: 多空均衡（多方' + bullRatio + '% vs 空方' + (100 - bullRatio) + '%），等待方向选择');
            }
        }

        if (recommendation) {
            if (recommendation.confidence >= 4) {
                parts.push('信号强度: 强烈（信心指数' + recommendation.confidence + '/5），信号明确');
            } else if (recommendation.confidence >= 2) {
                parts.push('信号强度: 中等（信心指数' + recommendation.confidence + '/5），可作为参考');
            } else {
                parts.push('信号强度: 较弱（信心指数' + recommendation.confidence + '/5），需进一步确认');
            }
        }

        if (indicator && indicator.kdj && indicator.kdj.status === '超买') {
            parts.push('风险提示: KDJ超买，短期回调风险较大，注意控制仓位');
        } else if (indicator && indicator.rsi && indicator.rsi.status === '超买') {
            parts.push('风险提示: RSI超买，短期上涨过度，追高需谨慎');
        } else if (trend.direction === 'down' && trend.strength > 3) {
            parts.push('风险提示: 下降趋势强劲，逆势操作风险较大');
        }

        return parts;
    }

    static _generatePrediction(pattern, trend, volume, indicator, curK, prevK, trapDetection) {
        const parts = [];
        if (!curK || !prevK) return ['数据不足，无法预测'];

        const prevClose = prevK.close;
        const curClose = curK.close;
        const curHigh = curK.high;
        const curLow = curK.low;

        let direction = '震荡';
        let confidence = 0;
        let reasons = [];

        if (pattern.type === 'bullish' && pattern.level >= 3) {
            direction = '看涨';
            confidence += 2;
            reasons.push('出现强势看涨形态');
        } else if (pattern.type === 'bearish' && pattern.level >= 3) {
            direction = '看跌';
            confidence += 2;
            reasons.push('出现强势看跌形态');
        }

        if (trend.direction === 'up' && trend.strength > 2) {
            if (direction !== '看跌') direction = '看涨';
            confidence += 1;
            reasons.push('上升趋势延续');
        } else if (trend.direction === 'down' && trend.strength > 2) {
            if (direction !== '看涨') direction = '看跌';
            confidence += 1;
            reasons.push('下降趋势延续');
        }

        if (indicator && indicator.macd) {
            if (indicator.macd.cross === 'golden') {
                if (direction !== '看跌') direction = '看涨';
                confidence += 2;
                reasons.push('MACD金叉信号');
            } else if (indicator.macd.cross === 'death') {
                if (direction !== '看涨') direction = '看跌';
                confidence += 2;
                reasons.push('MACD死叉信号');
            }
        }

        if (indicator && indicator.kdj) {
            if (indicator.kdj.cross === 'golden') {
                if (direction !== '看跌') direction = '看涨';
                confidence += 1;
                reasons.push('KDJ金叉');
            } else if (indicator.kdj.cross === 'death') {
                if (direction !== '看涨') direction = '看跌';
                confidence += 1;
                reasons.push('KDJ死叉');
            }
            if (indicator.kdj.status === '超买' && direction !== '看涨') {
                direction = '看跌';
                confidence += 1;
                reasons.push('KDJ超买，回调压力');
            } else if (indicator.kdj.status === '超卖' && direction !== '看跌') {
                direction = '看涨';
                confidence += 1;
                reasons.push('KDJ超卖，反弹需求');
            }
        }

        if (volume && volume.status === '放量上攻') {
            if (direction !== '看跌') direction = '看涨';
            confidence += 1;
            reasons.push('放量上攻，动能充足');
        } else if (volume && volume.status === '放量下跌') {
            if (direction !== '看涨') direction = '看跌';
            confidence += 1;
            reasons.push('放量下跌，抛压沉重');
        } else if (volume && volume.status === '缩量' && trend.direction === 'up') {
            if (direction === '看涨') confidence -= 1;
            reasons.push('缩量上涨，需警惕动能衰竭');
        }

        // 陷阱检测对预测的影响
        if (trapDetection) {
            if (trapDetection.traps && trapDetection.traps.includes('诱空')) {
                direction = '看涨';
                confidence += 2;
                reasons.push('诱空陷阱后看涨');
            }
            if (trapDetection.traps && trapDetection.traps.includes('诱多')) {
                direction = '看跌';
                confidence += 2;
                reasons.push('诱多陷阱后看跌');
            }
        }

        if (confidence >= 3) {
            parts.push('下一根预测: ' + direction + '（信心较高，信号数' + confidence + '）');
        } else if (confidence >= 1) {
            parts.push('下一根预测: ' + direction + '（信号偏弱，仅' + confidence + '个信号）');
        } else {
            parts.push('下一根预测: 方向不明，等待市场选择');
        }

        if (reasons.length > 0) {
            parts.push('依据: ' + reasons.join('、'));
        }

        const atr = Math.abs(curHigh - curLow);
        parts.push('阻力位: 上方 ' + (curClose + atr * 0.8).toFixed(2) + ' / ' + (curClose + atr * 1.5).toFixed(2));
        parts.push('支撑位: 下方 ' + (curClose - atr * 0.8).toFixed(2) + ' / ' + (curClose - atr * 1.5).toFixed(2));

        return parts;
    }

    static _calcSupportResistance(trend, curK) {
        if (!curK) return { support: [], resistance: [] };
        const atr = curK.high - curK.low;
        return {
            support: [
                (curK.close - atr).toFixed(2),
                (curK.close - atr * 2).toFixed(2)
            ],
            resistance: [
                (curK.close + atr).toFixed(2),
                (curK.close + atr * 2).toFixed(2)
            ]
        };
    }

    static _generateLessonsEnhanced(c, curK, prevK, prevClose, trapDetection) {
        const lessons = [];
        const pattern = c.pattern;
        const trend = c.trend;
        const volume = c.volume;
        const indicator = c.indicator;
        const changePct = c.changePct;
        const isMatch = c.comparison && c.comparison.isMatch;

        // 形态知识点
        if (pattern.name && pattern.name !== '未知') {
            const patternKnowledge = {
                '十字星': '十字星表示多空力量均衡，是趋势可能反转的信号。出现在高位需警惕见顶，低位则可能见底。',
                '长腿十字星': '上下影线都很长，说明盘中多空争夺激烈，分歧巨大，往往是变盘的前兆。',
                '锤子线': '锤子线出现在下跌趋势中，下影线表明下方有强支撑，是可能的底部反转信号。需要次日阳线确认。',
                '倒锤子线': '倒锤子线出现在上涨趋势中，上影线表明上方有抛压，是可能的顶部反转信号。',
                '大阳线': '大阳线表明买方力量强劲，收盘价接近最高价，后市看涨。',
                '大阴线': '大阴线表明卖方力量强劲，收盘价接近最低价，后市看跌。',
                '光头光脚大阳线': '开盘即最低，收盘即最高，买方完全主导市场，是非常强势的信号。',
                '光头光脚大阴线': '开盘即最高，收盘即最低，卖方完全主导市场，是非常弱势的信号。',
                '纺锤线': '实体小、影线长，说明市场动能减弱，可能进入盘整或反转。',
                'T字线': '下影线很长，说明下方有强力支撑，是可能的止跌信号。',
                '倒T字线': '上影线很长，说明上方有强力抛压，是可能的见顶信号。',
                '一字涨停': '开盘即涨停并封死，说明买方极度强势，通常有利好消息驱动。',
                '一字跌停': '开盘即跌停并封死，说明卖方极度恐慌，通常有利空消息驱动。',
                '看涨吞没': '阳线实体完全覆盖前一根阴线，是强烈的看涨反转信号。',
                '看跌吞没': '阴线实体完全覆盖前一根阳线，是强烈的看跌反转信号。',
                '早晨之星': '由大阴线+小实体+大阳线组成，是可靠的底部反转信号。中间的星线表示下跌动能衰竭。',
                '黄昏之星': '由大阳线+小实体+大阴线组成，是可靠的顶部反转信号。中间的星线表示上涨动能衰竭。',
                '三只乌鸦': '连续三根大阴线，顶部反转信号，后市看跌。',
                '红三兵': '连续三根大阳线，底部反转信号，后市看涨。',
                '乌云盖顶': '阳线后高开低走，阴线深入前阳线实体，顶部反转信号。',
                '穿刺线': '阴线后低开高走，阳线深入前阴线实体，底部反转信号。'
            };
            if (patternKnowledge[pattern.name]) {
                lessons.push({
                    type: 'pattern',
                    title: '形态识别：' + pattern.name,
                    content: patternKnowledge[pattern.name]
                });
            }
        }

        // 量价分析
        if (volume && volume.status !== 'normal') {
            let volTitle = '量价分析：' + volume.status;
            let volContent = volume.description;
            if (volume.details && volume.details.length > 0) {
                volContent += '（' + volume.details.join('；') + '）';
            }
            lessons.push({ type: 'advice', title: volTitle, content: volContent });
        }

        // 指标分析
        if (indicator) {
            if (indicator.macd) {
                const macd = indicator.macd;
                if (macd.crossDesc) {
                    lessons.push({
                        type: 'trend',
                        title: 'MACD指标：' + (macd.cross === 'golden' ? '金叉' : macd.cross === 'death' ? '死叉' : '状态'),
                        content: macd.crossDesc + (macd.barDesc ? '，' + macd.barDesc : '')
                    });
                }
            }
            if (indicator.kdj && indicator.kdj.desc) {
                lessons.push({
                    type: 'trend',
                    title: 'KDJ指标：' + (indicator.kdj.status || ''),
                    content: indicator.kdj.desc
                });
            }
            if (indicator.rsi && indicator.rsi.desc) {
                lessons.push({
                    type: 'trend',
                    title: 'RSI指标',
                    content: indicator.rsi.desc
                });
            }
            if (indicator.signals && indicator.signals.length > 0) {
                for (const sig of indicator.signals) {
                    lessons.push({
                        type: 'note',
                        title: '信号提示',
                        content: sig.text
                    });
                }
            }
        }

        // 操作建议
        if (!isMatch) {
            if (c.recommended.reason) {
                lessons.push({
                    type: 'advice',
                    title: '操作建议',
                    content: '推荐操作：' + c.recommended.action + '。' + c.recommended.reason
                });
            }
        } else {
            lessons.push({
                type: 'praise',
                title: '判断正确',
                content: '你的操作与综合信号分析一致，判断准确！'
            });
        }

        c.lessons = lessons;
    }

    static _buildSummaryEnhanced(c, trapDetection) {
        const pattern = c.pattern;
        const trend = c.trend;
        const volume = c.volume;
        const indicator = c.indicator;
        const isMatch = c.comparison && c.comparison.isMatch;

        let summary = '【第' + c.step + '根K线讲评】\n';
        summary += '形态：' + pattern.name + '（' + (pattern.type === 'bullish' ? '看涨' : pattern.type === 'bearish' ? '看跌' : '中性') + '信号）\n';
        summary += '趋势：' + trend.description + '\n';

        if (volume && volume.status !== 'normal') {
            summary += '量能：' + volume.description + '\n';
        }

        if (indicator && indicator.macd && indicator.macd.crossDesc) {
            summary += 'MACD：' + indicator.macd.crossDesc + '\n';
        }

        if (trapDetection && trapDetection.phase && trapDetection.phaseType !== 'neutral') {
            summary += '阶段：' + trapDetection.phase + '\n';
        }

        summary += '\n你的操作：' + c.userAction + '\n';
        summary += '推荐操作：' + c.recommended.action + '\n';

        if (c.recommended.reason) {
            summary += '\n理由：' + c.recommended.reason + '\n';
        }

        if (!isMatch) {
            summary += '\n你的选择和推荐操作不同。综合形态、量价、指标等多维度信号，推荐操作更符合当前市场特征。';
        } else {
            summary += '\n你的判断与多维度分析一致，继续保持！';
        }

        return summary;
    }

    static finalReview(result) {
        const reviews = [];

        const rate = result.totalReturnRate;
        if (rate > 30) {
            reviews.push('收益率超过30%，表现优秀！你对K线形态的判断非常准确。');
        } else if (rate > 10) {
            reviews.push('收益率超过10%，表现良好。你的K线分析能力已经相当不错了。');
        } else if (rate > 0) {
            reviews.push('正收益，值得肯定。继续练习可以进一步提升。');
        } else if (rate > -10) {
            reviews.push('小幅亏损。建议复盘一下亏损的交易，看看是哪些形态判断出了问题。');
        } else {
            reviews.push('亏损较大。K线分析需要更多练习，建议重点关注反转形态的识别和趋势判断。');
        }

        const diff = rate - result.buyHoldReturnRate;
        if (diff > 10) {
            reviews.push('你的操作比简单的买入持有策略多赚了 ' + diff.toFixed(1) + '%，说明你的主动交易创造了价值！');
        } else if (diff > 0) {
            reviews.push('你的操作略优于买入持有策略（+' + diff.toFixed(1) + '%），主动交易有一定效果。');
        } else if (diff > -10) {
            reviews.push('你的操作略逊于买入持有策略（' + diff.toFixed(1) + '%），可能交易过于频繁了。');
        } else {
            reviews.push('你的操作明显不如买入持有策略（' + diff.toFixed(1) + '%），建议减少不必要的交易。');
        }

        if (result.winRate > 60) {
            reviews.push('交易胜率 ' + result.winRate.toFixed(0) + '%，说明你的择时能力很强。');
        } else if (result.winRate > 40) {
            reviews.push('交易胜率 ' + result.winRate.toFixed(0) + '%，属于正常水平。');
        } else {
            reviews.push('交易胜率 ' + result.winRate.toFixed(0) + '%，建议设置止损位，控制单笔亏损。');
        }

        if (result.maxDrawdown < 5) {
            reviews.push('最大回撤仅 ' + result.maxDrawdown.toFixed(1) + '%，风险控制做得很好。');
        } else if (result.maxDrawdown < 15) {
            reviews.push('最大回撤 ' + result.maxDrawdown.toFixed(1) + '%，在可控范围内。');
        } else {
            reviews.push('最大回撤 ' + result.maxDrawdown.toFixed(1) + '%，回撤偏大，建议注意仓位管理。');
        }

        reviews.push('\n本次训练股票：' + result.stockName + '（' + result.stockCode + '）');
        reviews.push('时间区间：' + result.klineRange.start + ' 至 ' + result.klineRange.end);
        reviews.push('期间股价涨跌：' + (result.stockChangeRate >= 0 ? '+' : '') + result.stockChangeRate.toFixed(2) + '%');

        return reviews;
    }

    static _defaultCommentary(userDecision, reason) {
        return {
            step: userDecision ? (userDecision.step || 0) : 0,
            userAction: userDecision ? (userDecision.action || '未知') : '未知',
            pattern: { name: '未知', type: 'neutral', level: 0 },
            changePct: 0,
            trend: { direction: 'unknown', strength: 0, description: reason },
            recommended: { action: '观望', reason: reason, confidence: 0 },
            comparison: {
                isMatch: true,
                userAction: userDecision ? (userDecision.action || '未知') : '未知',
                recommendedAction: '观望',
                evaluation: reason
            },
            lessons: [{
                type: 'note',
                title: '提示',
                content: reason
            }],
            summary: '[讲评] ' + reason
        };
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = KlineAnalysis;
}