# kline-gru（K线工作室 · GRU 检索版）

基于 [kline-studio](F:\project\kline-studio)（K-means 形态聚类 + k-NN 相似检索 + K线训练营）
的独立衍生项目，**在不改动 kline-studio 的前提下**，把「深度学习时序路线（GRU）」作为与
KNN 同层级的检索器融合进来，完成与 KNN 相同的「相似历史案例检索」工作内容。

> 两个项目完全自包含、互不依赖；本项目的代码与数据均在 `F:\project\kline-gru` 内。

---

## 一句话定位

| 技术路线 | 相似度度量 | 检索器 | 入口脚本 |
|---|---|---|---|
| 路线1 · KNN | 手工特征向量的**欧氏距离**（固定度量） | `KnnRetriever` | `retrieve.py` |
| 路线2 · GRU | GRU 学习嵌入的**余弦相似度**（学习度量） | `GruRetriever` | `gru_train.py` + `gru_retrieve.py` |

两者的差异本质是「相似度」不同：KNN 用 4×WINDOW 维手工特征做固定欧氏度量；GRU 把
(WINDOW, 4) 特征序列喂给 GRU 编码器，用**最后一步隐藏态**作为嵌入，相似度随训练目标
（预测未来超额收益）**自动学习**。于是嵌入空间里「距离近」≈「历史上后续走势相近」。

---

## 项目结构

```
kline-gru/
├── config.py / config.example.py   # 配置（含 Tushare/Kimi token、K-means、GRU 超参、RETRIEVER 切换）
├── requirements.txt
│
├── # ── 路线2 · GRU（本项目新增）──
├── gru_model.py        # GRU 编码器 + 特征矩阵 + 余弦相似度
├── gru_train.py        # walk-forward 训练 + 建 GRU/KNN 索引（一次性）
├── gru_retrieve.py     # GRU 相似案例检索 CLI（镜像 retrieve.py）
├── gru_eval.py         # GRU vs KNN 同口径 walk-forward 评估
│
├── # ── 路线1 · KNN（原 kline-studio）──
├── features.py         # 特征工程（return/range/body/volume 窗口 Z-score）
├── cluster.py          # K-means 聚类 + 簇统计
├── benchmark.py        # 等权市场基准（超额收益用）
├── chart.py            # K 线图绘制（涨红跌绿）
├── fetch_data.py       # 数据下载（Tushare，全市场）
├── train.py / query.py / scan.py / retrieve.py / review.py / comment.py
│
├── # ── K线训练营（原 kline-training-camp）──
├── server.py / data_fetcher.py / generate_sample.py / web/
│
├── data/               # kline/（5194 只）、market_benchmark.json、stock_list.json 等
├── models/             # gru.pt / gru_X.npy / gru_meta.json / knn_X.npy / kmeans.joblib 等
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

> 硬件：本机 Lenovo P53 + Quadro T1000（4GB）。GRU 训练在小模型下 GPU/CPU 均可，
> 全量 120 万窗口建议 GPU；`--max-windows` 可快速冒烟。

---

## 路线2 · GRU：三步走

### 1. 训练 + 建索引（一次性）

```powershell
python gru_train.py                       # 全量（约 120 万窗口，几分钟）
python gru_train.py --max-windows 80000   # 冒烟/快速验证
python gru_train.py --device cpu          # 无 GPU
```

一次运行完成：收集全市场窗口 → 按**窗口结束日期**做时间序列切分（train/val/test）
→ 训练 GRU（早停 + 权重衰减 + dropout）→ 编码全部窗口为嵌入 → 保存检索库。

产出（`models/`）：`gru.pt`（模型）、`gru_X.npy`（嵌入索引）、`gru_meta.json`（元信息）、
`knn_X.npy`/`knn_meta.json`（同窗口集扁平特征，供同口径对比）、`gru_train_meta.json`（指标）。

### 2. 相似案例检索（与 KNN 相同的用法）

```powershell
python gru_retrieve.py 600519 --end-date 20260801 --top 5
python gru_retrieve.py 600519 --end-date 20260801 --top 5 --before   # 只看历史先例
```

输出与 `retrieve.py` 同构：目标窗口 + 最相似 N 例（代码/名称/日期/相似度/前向收益/超额收益）
+ 汇总均值 + 对比图 `output/gru_retrieve_*.png`。

### 3. 诚实评估（GRU vs KNN）

```powershell
python gru_eval.py --n-queries 1000 --top 5
```

在时间上 held-out 的一段窗口里采样查询，两种检索器各找 K 个历史先例，用「先例后续
超额收益均值」预测实际收益，输出**秩相关 IC** + **方向命中率** + **预测 MAE**。

---

## 路线1 · KNN / 形态分析引擎（原 kline-studio，不变）

```powershell
python retrieve.py --build                 # 首次构建 KNN 检索库
python train.py                            # K-means 训练
python query.py 600519 --end-date 20260801 # 判类别
python scan.py --end-date 20260801         # 截面扫描
python review.py 600519 --end-date 20260801 --top 5 --call  # 智能讲评
```

KNN 检索器与 GRU 检索器在 `engines.py` 的注册表中同层级，通过 `config.RETRIEVER`
一键切换（`'knn'` / `'gru'`），两者都接入 K线训练营的「证据层」。

---

## 关键设计决策（对应落地建议）

1. **用 GRU 而非 LSTM/Transformer**：参数更少、收敛更快，小样本（A 股日线）下更不易
   过拟合；Transformer 对数据量更贪婪，不适合直接跑日线。
2. **时间序列切分而非随机 K 折**：按窗口结束日期切 train/val/test，避免用未来信息训练，
   否则回测好看、实盘拉胯。
3. **强正则防过拟合**：dropout + 权重衰减 + 早停 + 小隐藏维（默认 32）。
4. **学习型相似度对齐目标**：GRU 训练目标是「预测未来超额收益」，嵌入因此对齐后续走势，
   与 KNN「固定度量 + 手工特征」形成对照——`gru_eval.py` 用来诚实验证谁更好。

---

## 关键说明与限制

1. **GRU/KNN 的检索结果都是描述性统计，不是交易信号。** 已做等权市场基准化（超额收益），
   但 K 线形态在短周期（3~10 日）上的预测力天然有限，IC 通常接近 0——这本身是重要结论，
   不要对深度学习抱过高期望（见 `gru_eval.py` 的解读）。
2. **不要用 CNN-LSTM 跑日线**：CNN 空间局部性先验对 K 线是伪先验，叠加 LSTM 是双重过拟合。
3. **市场风格切换**：GRU 是参数化规律，风格漂移时易失效；KNN 永远用「最近历史」做参照，
   天然更自适应。二者可互补：GRU 提供学习到的相似度，KNN 提供可解释先例。
4. **GPU 非必需**：当前规模 CPU 亦可，GPU 主要加速全量训练与嵌入编码。
