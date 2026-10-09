# kline-gru（K线工作室 · 学习型检索版）

基于 [kline-studio](../kline-studio)（K-means 形态聚类 + k-NN 相似检索 + K线训练营）的独立衍生项目：
**在不改动 kline-studio 的前提下**，把「学习型相似度」作为与 KNN 同层级的检索器融合进来，
完成与 KNN 相同的「相似历史案例检索」工作内容。

> 项目完全自包含，代码与数据均在 `F:\project\kline-gru` 内，不跨项目引用。

---

## 一句话定位

| 技术路线 | 相似度度量 | 检索器 | 入口脚本 |
|---|---|---|---|
| 路线1 · KNN | 手工特征向量的**欧氏距离**（固定度量） | `KnnRetriever` | `retrieve.py` |
| 路线2 · GRU | GRU 学习嵌入的**余弦相似度**（有监督学习度量） | `GruRetriever` | `gru_train.py` + `gru_retrieve.py` |
| 路线3 · BYOL | BYOL 自监督嵌入的**余弦相似度**（自监督+排序损失） | `ByolRetriever` | `byol_train.py` |

三者的差异本质是「相似度怎么来」：KNN 用 `4×WINDOW` 维手工特征做固定欧氏度量；GRU 把
`(WINDOW, 4)` 序列喂给 GRU 编码器、用**最后一步隐藏态**作为嵌入；BYOL 在同一编码器上叠加
**无负样本自监督分支**（online/target 双网络 + EMA + stop-gradient）。嵌入空间的"距离近"
由此从「人为定义」变成「由训练目标塑造」——近 ≈ 历史上后续走势相近。

---

## 项目结构

```
kline-gru/
├── config.py / config.example.py   # 配置（Tushare/Kimi token、K-means、GRU、BYOL 超参、RETRIEVER 切换）
├── requirements.txt
│
├── # ── 路线3 · BYOL（自监督检索，当前最优）──
├── byol.py             # BYOL 双网络（online/target + projector/predictor）+ 序列增强 + 对称损失
├── losses.py           # pairwise 排序损失（IC 基座）
├── byol_train.py       # 监督损失（rank/mse）+ BYOL 联合训练 + 建索引
│
├── # ── 路线2 · GRU（有监督检索）──
├── gru_model.py        # GRU 编码器 + 特征矩阵 + 余弦相似度
├── gru_train.py        # walk-forward 训练 + 建 GRU/KNN 索引（一次性）
├── gru_retrieve.py     # GRU 相似案例检索 CLI（镜像 retrieve.py）
├── gru_eval.py         # GRU vs KNN 同口径 walk-forward 评估
│
├── # ── 路线1 · KNN 与形态分析引擎（原 kline-studio）──
├── features.py         # 特征工程（return/range/body/volume 窗口 Z-score）
├── cluster.py / benchmark.py / chart.py / fetch_data.py
├── train.py / query.py / scan.py / retrieve.py / review.py / comment.py
│
├── # ── K线训练营（原 kline-training-camp）──
├── server.py / data_fetcher.py / generate_sample.py / web/
│
├── data/               # kline/（全市场日线）、market_benchmark.json、stock_list.json 等
├── models/             # byol.pt / gru.pt / knn_X.npy / *_X.npy / *_meta.json / kmeans.joblib
├── output/             # 图表 / 报告 / 讲评结果
└── logs/
```

---

## 快速开始

```powershell
conda activate ai_env
cd F:\project\kline-gru
pip install -r requirements.txt
```

> 硬件：本机 Lenovo P53 + Quadro T1000（4GB）。小模型下 GPU/CPU 均可；全量 120 万窗口建议 GPU，
> `--max-windows` 可快速冒烟。

---

## 路线3 · BYOL（推荐 · 当前最优）

### 1. 训练 + 建索引

```powershell
python byol_train.py                          # 全量（约 120 万窗口，6~8 分钟）
python byol_train.py --max-windows 80000      # 冒烟 / 快速验证
python byol_train.py --sup-loss mse           # 切回纯 MSE 监督（默认 rank）
python byol_train.py --epochs 20 --hidden 64  # 调整训练规模
```

一次运行完成：收集全市场窗口 → 按**窗口结束日期**做时间序列切分 → 联合损失训练
（`BYOL_MSE_WEIGHT × 监督损失 + BYOL_WEIGHT × BYOL 自监督`）→ 编码全部窗口为嵌入 → 保存检索库。

- **监督损失默认 `rank`**（pairwise 排序损失），选模口径随之改为 **验证集 IC 最大化**
- 产出：`byol.pt`、`byol_X.npy`、`byol_meta.json`、`byol_train_meta.json`

### 2. 检索

```powershell
python -c "from engines import get_retriever; r=get_retriever('byol'); print(r.retrieve(kline, '600519', top=5))"
# 或在 config.py 里设 RETRIEVER='byol'，训练营界面点「BYOL」按钮直接检索
```

### 3. 实测结果（全市场 1,204,302 窗口）

```
监督损失：0.5*RANK + 0.5*BYOL        早停 @ epoch 11（最优 epoch 5，val_ic=+0.0706）
测试集： IC = +0.0616  方向命中 51.0%
```

| 方案（4 切分滚动验证均值） | IC 均值 | IC 标准差 | gap（嵌入质量） |
|---|---|---|---|
| baseline（纯 MSE） | +0.0231 | ±0.0288 | 0.110 |
| byol（MSE + BYOL） | +0.0279 | ±0.0168 | 0.172 |
| rank（纯排序损失） | +0.0680 | ±0.0208 | 0.210 |
| **byol_rank（排序 + BYOL）** | **+0.0788** | ±0.0219 | **0.256** |
| byol_rank_cs（+截面标签） | +0.0758 | **±0.0116** | 0.259 |

**结论：监督目标对齐评估指标（排序损失替 MSE）是最大的 IC 杠杆**（近三倍），BYOL 在其上
叠加嵌入质量的正交收益。追求跨期稳定时用截面标签版（方差近减半，风格切换段更强）。

---

## 路线2 · GRU（有监督检索）

```powershell
python gru_train.py                       # 训练 + 建索引（约 120 万窗口）
python gru_train.py --max-windows 80000   # 冒烟
python gru_retrieve.py 600519 --end-date 20260801 --top 5
python gru_retrieve.py 600519 --end-date 20260801 --top 5 --before   # 只看历史先例
python gru_eval.py --n-queries 1000 --top 5                          # GRU vs KNN 评估
```

输出与 `retrieve.py` 同构：目标窗口 + 最相似 N 例（代码/名称/日期/相似度/前向收益/超额收益）
+ 汇总均值 + 对比图。产出 `gru.pt`、`gru_X.npy`、`gru_meta.json`、`gru_train_meta.json`。

---

## 路线1 · KNN / 形态分析引擎（原 kline-studio）

```powershell
python retrieve.py --build                 # 首次构建 KNN 检索库
python train.py                            # K-means 训练
python query.py 600519 --end-date 20260801 # 判类别
python scan.py --end-date 20260801         # 截面扫描
python review.py 600519 --end-date 20260801 --top 5 --call  # 智能讲评
```

三个检索器在 `engines.py` 的注册表中同层级，通过 `config.RETRIEVER` 一键切换
（`'knn'` / `'gru'` / `'byol'`），都接入 K线训练营的「证据层」，前端可逐个点击、结果并列对比。

---

## 关键设计决策

1. **用 GRU 而非 LSTM/Transformer**：参数更少、收敛更快，小样本（A 股日线）下更不易过拟合；
   Transformer 对数据量更贪婪，不适合直接跑日线。
2. **监督目标对齐评估指标**：要 IC（排序能力）就用 pairwise 排序损失，而不是 MSE。
   MSE 优化点预测、被极端收益主导；排序损失仿射不变、在风格切换段更稳。
3. **无负样本防坍缩**：对比学习（InfoNCE）在"时间紧邻为正样本"的伪假设下会整体坍缩
   （gap≈0.0007）；BYOL 用 EMA target + stop-gradient 天然防坍缩，gap 达 baseline 的 180%。
4. **时间序列切分而非随机 K 折**：按窗口结束日期切 train/val/test，避免用未来信息训练。
5. **强正则防过拟合**：dropout + 权重衰减 + 早停 + 小隐藏维（默认 32）。

---

## 关键说明与限制

1. **检索结果都是描述性统计，不是交易信号。** 已做等权市场基准化（超额收益），但 K 线形态
   在短周期（3~10 日）上的预测力天然有限，IC 通常在 0.02~0.08 量级。
2. **单一切分不可信**：任何单次时间切分的 IC 都可能是运气，须用多切分滚动验证裁决
   （见 `infonce-compare/rolling_eval.py`）。本项目一度出现过"单切分 IC +0.0499"被滚动验证
   证伪的情况。
3. **下游检索质量实测**（生产索引、3000 query、无泄漏候选、K=5、含随机基线）：

   | 检索器 | IC(query收益, 命中均收益) | 符号一致率 |
   |---|---|---|
   | BYOL | **+0.0338** | 51.6% |
   | GRU | +0.0298 | 51.3% |
   | 随机基线 | +0.0197 | 51.4% |
   | KNN | −0.0060 | 51.7% |

   学习型嵌入的检索质量优于手工特征（BYOL 唯一明显跑赢随机基线，KNN 甚至低于随机）；
   但绝对水平弱——**检索结果应作为"参考案例"，而非预测信号**。
4. **市场风格切换**：参数化模型在风格漂移时易失效；KNN 永远用「最近历史」做参照，天然更自适应。
   三者可互补：GRU/BYOL 提供学习到的相似度，KNN 提供可解释先例。
5. **GPU 非必需**：当前规模 CPU 亦可，GPU 主要加速全量训练与嵌入编码。
