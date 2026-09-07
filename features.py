"""形态特征工程：把一段 K 线窗口转为固定长度特征向量。

口径与 stock-pattern-matcher 保持一致：
- return : 日收盘收益率
- range  : 日内振幅 (high-low)/close
- body   : 实体比例 (close-open)/close
- volume : 相对成交量 volume/窗口均量

每条序列在窗口内做 Z-score 标准化后拼接，得到 4*WINDOW 维向量。
Z-score 的作用是"去量纲 + 去趋势水平"，让聚类聚焦于形态节奏而非涨跌幅度。
"""
from __future__ import annotations

from math import sqrt
from typing import Any


def feature_sequences(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    """rows: 含 open/high/low/close/volume 的字典列表，按时间升序。"""
    closes = [float(r["close"]) for r in rows]
    volumes = [max(float(r.get("volume", 0)), 0.0) for r in rows]
    avg_vol = sum(volumes) / len(volumes) if volumes else 1.0

    ret: list[float] = []
    rng: list[float] = []
    body: list[float] = []
    vol: list[float] = []
    for i, r in enumerate(rows):
        c = closes[i]
        prev = closes[i - 1] if i else c
        ret.append(c / prev - 1 if prev else 0.0)
        rng.append((float(r["high"]) - float(r["low"])) / c if c else 0.0)
        body.append((c - float(r["open"])) / c if c else 0.0)
        vol.append(volumes[i] / avg_vol if avg_vol else 0.0)
    return {"return": ret, "range": rng, "body": body, "volume": vol}


def zscore(values: list[float]) -> list[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    deviation = sqrt(sum((v - mean) ** 2 for v in values) / len(values))
    return [(v - mean) / deviation if deviation > 1e-12 else 0.0 for v in values]


def feature_vector(rows: list[dict[str, Any]]) -> list[float]:
    """把一段窗口转为固定长度特征向量（4*WINDOW 维）。"""
    seq = feature_sequences(rows)
    vec: list[float] = []
    for key in ("return", "range", "body", "volume"):
        vec.extend(zscore(seq[key]))
    return vec
