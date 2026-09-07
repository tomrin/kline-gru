/**
 * K线训练营 - 前端配置
 */

const CONFIG = {
    // 初始资金
    initCash: 100000,

    // 默认K线数量
    defaultKlineCount: 30,

    // 参考K线数量（训练前展示的"前情回顾"）
    referenceCount: 20,

    // 游戏模式
    modes: {
        classic: { label: '经典模式', desc: 'K线 + 成交量 + 均线' },
        hard: { label: '困难模式', desc: '仅K线，无成交量无均线' },
        free: { label: '自由模式', desc: '不限K线数，可随时结束' }
    },

    // 图表配置
    chart: {
        backgroundColor: '#1a1a2e',
        textColor: '#e0e0e0',
        upColor: '#ef5350',       // 阳线颜色（红）
        downColor: '#26a69a',     // 阴线颜色（绿）
        volumeUpColor: 'rgba(239, 83, 80, 0.5)',
        volumeDownColor: 'rgba(38, 166, 154, 0.5)',
        ma5Color: '#fdd835',
        ma10Color: '#42a5f5',
        ma20Color: '#ab47bc'
    },

    // 数据路径
    dataPath: 'data/stocks.json'
};