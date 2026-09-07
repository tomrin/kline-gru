"""GRU 路线训练 + 建检索索引（路线2）。

一次运行完成五件事：
1. 遍历全市场窗口，收集 (WINDOW,4) 特征序列 + 前向超额收益 + 扁平特征 + 元信息；
2. 按「窗口结束日期」做时间序列切分（train/val/test），避免用未来信息训练；
3. 训练 GRU 编码器（预测未来超额收益；早停 + 权重衰减 + dropout 防过拟合）；
4. 把全部窗口编码为嵌入，保存 GRU 检索库（gru_X.npy + gru_meta.json）；
5. 同时保存同一窗口集的扁平特征为 KNN 检索库（knn_X.npy + knn_meta.json），
   供 gru_eval.py 做「GRU vs KNN」同口径对比。

用法：
    python gru_train.py                       # 全量（约 1.2M 窗口，需几分钟）
    python gru_train.py --max-windows 80000   # 冒烟/快速验证
    python gru_train.py --device cpu          # 无 GPU 时

产出（models/）：
    gru.pt             GRU 模型权重
    gru_X.npy          GRU 嵌入索引（n × hidden）
    gru_meta.json      窗口元信息（code/name/start/end/forward/excess）
    knn_X.npy          扁平特征（n × 4*WINDOW），供 KNN 同口径对比
    knn_meta.json      同上（与 gru_meta 同窗口集）
    ends.npy           各窗口结束日期（int YYYYMMDD，供 eval 快速「先例」过滤）
    gru_train_meta.json 训练超参与评估指标
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import cluster as cl
from features import feature_vector
from gru_model import feature_matrix, build_model, encode

ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "gru.pt"
GRU_X_PATH = MODEL_DIR / "gru_X.npy"
GRU_META_PATH = MODEL_DIR / "gru_meta.json"
KNN_X_PATH = MODEL_DIR / "knn_X.npy"
KNN_META_PATH = MODEL_DIR / "knn_meta.json"
ENDS_PATH = MODEL_DIR / "ends.npy"
TRAIN_META_PATH = MODEL_DIR / "gru_train_meta.json"


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """秩相关（信息系数 IC），安全处理常量/空数组。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def collect_windows(data_dir: Path, window: int, step: int, forward_days: int,
                    max_windows: int | None, seed: int):
    """遍历全市场窗口，返回 (seq, flat, excess, metas)。"""
    from benchmark import get_or_build_benchmark
    benchmark = get_or_build_benchmark(data_dir)

    seqs: list[np.ndarray] = []
    flats: list[np.ndarray] = []
    excesses: list[float] = []
    metas: list[dict] = []

    paths = sorted(data_dir.glob("*.json"))
    rng = random.Random(seed)
    rng.shuffle(paths)  # 打乱股票顺序，避免 --max-windows 只采到字母序靠前的股票

    started = time.perf_counter()
    for pidx, path in enumerate(paths, 1):
        loaded = cl.load_stock_file(path)
        if loaded is None:
            continue
        code, name, rows = loaded
        for i in range(0, len(rows) - window + 1, step):
            win = rows[i:i + window]
            seq = feature_matrix(win)
            flat = np.asarray(feature_vector(win), dtype=np.float32)
            excess, fwd = cl.forward_return_excess(rows, i + window - 1, forward_days, benchmark)
            seqs.append(seq)
            flats.append(flat)
            excesses.append(excess if excess is not None else np.nan)
            metas.append({
                "code": code, "name": name,
                "start": str(win[0]["date"]), "end": str(win[-1]["date"]),
                "forward": fwd if fwd is not None else None,
                "excess": excess if excess is not None else None,
            })
            if max_windows and len(seqs) >= max_windows:
                break
        if max_windows and len(seqs) >= max_windows:
            print(f"[collect] 已达上限 {max_windows}（扫描 {pidx}/{len(paths)} 只股票）", flush=True)
            break
        if pidx % 1000 == 0:
            print(f"[collect] 已扫 {pidx}/{len(paths)} 只，累计 {len(seqs)} 窗口，"
                  f"用时 {time.perf_counter() - started:.1f}s", flush=True)

    seq = np.asarray(seqs, dtype=np.float32) if seqs else np.empty((0, window, 4), np.float32)
    flat = np.vstack(flats) if flats else np.empty((0, 4 * window), np.float32)
    excess = np.asarray(excesses, dtype=np.float32)
    print(f"[collect] 完成：共 {len(metas)} 个窗口，用时 {time.perf_counter() - started:.1f}s")
    return seq, flat, excess, metas


def sort_by_end(seq, flat, excess, metas):
    """按窗口结束日期升序排序（时间对齐，供切分 + 先例过滤）。"""
    order = np.argsort([m["end"] for m in metas], kind="stable")
    return seq[order], flat[order], excess[order], [metas[i] for i in order]


def time_split(n: int, train_frac: float, val_frac: float):
    """时间序列切分（已按时间排序）：返回 (train_idx, val_idx, test_idx)。"""
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return np.arange(n_train), np.arange(n_train, n_train + n_val), np.arange(n_train + n_val, n)


def train_gru(seq, excess, train_idx, val_idx, cfg, device):
    """训练 GRU（早停），返回 (model, y_mean, y_std)。标签为标准化后的超额收益。"""
    import torch
    import torch.nn as nn
    import torch.optim as optim

    torch.manual_seed(int(cfg.get("RANDOM_SEED", 42)))
    model = build_model(cfg, device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(cfg.get("GRU_LR", 1e-3)),
        weight_decay=float(cfg.get("GRU_WEIGHT_DECAY", 1e-4)),
    )
    loss_fn = nn.MSELoss()

    # 只用有前向超额收益（标签非 NaN）的窗口做监督
    tr_mask = ~np.isnan(excess[train_idx])
    va_mask = ~np.isnan(excess[val_idx])
    Xtr = seq[train_idx][tr_mask]
    ytr = excess[train_idx][tr_mask].astype(np.float32)
    Xva = seq[val_idx][va_mask]
    yva = excess[val_idx][va_mask].astype(np.float32)

    if len(Xtr) == 0 or len(Xva) == 0:
        raise RuntimeError("训练/验证集为空，请增大数据量或检查前向收益是否可计算")

    # 标签标准化（用训练集统计量），使 loss 尺度稳定
    y_mean = float(np.mean(ytr))
    y_std = float(np.std(ytr)) or 1.0
    ytr_n = (ytr - y_mean) / y_std
    yva_n = (yva - y_mean) / y_std

    batch = int(cfg.get("GRU_BATCH_SIZE", 512))
    epochs = int(cfg.get("GRU_EPOCHS", 30))
    patience = int(cfg.get("GRU_PATIENCE", 6))

    best_val = float("inf")
    best_state = None
    best_epoch = -1

    print(f"[train] 样本 train={len(Xtr)} val={len(Xva)} | hidden={cfg.get('GRU_HIDDEN', 32)} "
          f"layers={cfg.get('GRU_LAYERS', 1)} dropout={cfg.get('GRU_DROPOUT', 0.3)} "
          f"epochs={epochs} batch={batch} lr={cfg.get('GRU_LR', 1e-3)}", flush=True)

    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(len(Xtr))
        total = 0.0
        for i in range(0, len(Xtr), batch):
            idx = perm[i:i + batch]
            xb = torch.from_numpy(Xtr[idx]).to(device)
            yb = torch.from_numpy(ytr_n[idx]).to(device)
            pred, _ = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(idx)
        train_loss = total / len(Xtr)

        model.eval()
        with torch.no_grad():
            pred, _ = model(torch.from_numpy(Xva).to(device))
            val_loss = float(loss_fn(pred, torch.from_numpy(yva_n).to(device)).item())

        print(f"[train] epoch {ep + 1:>3}/{epochs}  train_mse={train_loss:.6f}  val_mse={val_loss:.6f}",
              flush=True)
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif ep + 1 - best_epoch >= patience:
            print(f"[train] 早停 @ epoch {ep + 1}（最优 epoch {best_epoch}）", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, y_mean, y_std


def evaluate(model, seq, excess, idx, device, y_mean, y_std, batch_size=4096):
    """在给定索引的窗口上评估（预测 vs 实际超额收益），返回指标 dict。"""
    import torch
    mask = ~np.isnan(excess[idx])
    if mask.sum() < 3:
        return {"n": int(mask.sum()), "mse": None, "mae": None, "ic": None, "hit": None}
    X = seq[idx][mask]
    y = excess[idx][mask].astype(np.float32)
    preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i + batch_size]).to(device)
            p, _ = model(xb)
            preds.append(p.detach().cpu().numpy())
    pred = np.concatenate(preds) * y_std + y_mean  # 反标准化
    mse = float(np.mean((pred - y) ** 2))
    mae = float(np.mean(np.abs(pred - y)))
    ic = spearman(pred, y)
    hit = float(np.mean((pred > 0) == (y > 0)))
    return {"n": int(len(y)), "mse": mse, "mae": mae, "ic": ic, "hit": hit}


def main() -> int:
    parser = argparse.ArgumentParser(description="GRU 路线训练 + 建检索索引")
    parser.add_argument("--max-windows", type=int, default=None, help="窗口上限（默认全量）")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖 config.GRU_EPOCHS")
    parser.add_argument("--hidden", type=int, default=None, help="覆盖 config.GRU_HIDDEN")
    parser.add_argument("--dropout", type=float, default=None, help="覆盖 config.GRU_DROPOUT")
    parser.add_argument("--device", default="auto", help="auto / cuda / cpu")
    args = parser.parse_args()

    cfg = dict(vars(config))
    cfg["GRU_EPOCHS"] = args.epochs or cfg.get("GRU_EPOCHS", 30)
    cfg["GRU_HIDDEN"] = args.hidden or cfg.get("GRU_HIDDEN", 32)
    cfg["GRU_DROPOUT"] = args.dropout if args.dropout is not None else cfg.get("GRU_DROPOUT", 0.3)

    import torch
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[device] 使用 {device} "
          f"({'cuda:' + torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})", flush=True)

    data_dir = ROOT / config.KLINE_DIR
    window = config.WINDOW
    train_frac = float(cfg.get("GRU_TRAIN_FRAC", 0.7))
    val_frac = float(cfg.get("GRU_VAL_FRAC", 0.15))

    print("=" * 64)
    print("GRU 路线训练（窗口结束日期时间切分，防未来信息泄漏）")
    print(f"  窗口={window} | 步长={config.SLIDE_STEP} | 前向={config.FORWARD_DAYS}日 | "
          f"train/val/test={train_frac}/{val_frac}/{1 - train_frac - val_frac:.2f}")
    print("=" * 64)

    seq, flat, excess, metas = collect_windows(
        data_dir, window, config.SLIDE_STEP, config.FORWARD_DAYS, args.max_windows, config.RANDOM_SEED)

    # 时间排序 + 切分
    seq, flat, excess, metas = sort_by_end(seq, flat, excess, metas)
    train_idx, val_idx, test_idx = time_split(len(metas), train_frac, val_frac)
    print(f"[split] 按结束日期切分：train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    model, y_mean, y_std = train_gru(seq, excess, train_idx, val_idx, cfg, device)

    # 测试集评估（held-out 时间切片，诚实指标）
    print("\n[eval] 测试集（时间上最后一段，模型从未见过）：")
    metrics = evaluate(model, seq, excess, test_idx, device, y_mean, y_std)
    ic_txt = f"{metrics['ic']:+.4f}" if metrics["ic"] is not None else "N/A"
    mse_txt = f"{metrics['mse']:.6f}" if metrics["mse"] is not None else "N/A"
    hit_txt = f"{metrics['hit']*100:.1f}%" if metrics["hit"] is not None else "N/A"
    print(f"  样本={metrics['n']} | MSE={mse_txt} | MAE={metrics['mae'] or 0:.5f} | "
          f"IC(秩相关)={ic_txt} | 方向命中={hit_txt}")
    print("  注：IC 接近 0 是日线形态的常态，见 README「限制」说明")

    # 编码全部窗口 -> 嵌入索引
    print("\n[encode] 编码全部窗口为嵌入...", flush=True)
    t0 = time.perf_counter()
    embs = encode(model, seq, device)
    print(f"[encode] 完成 {embs.shape} 用时 {time.perf_counter() - t0:.1f}s")

    # 保存模型与索引
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "hidden_dim": model.hidden_dim,
                "num_layers": model.num_layers, "input_dim": 4,
                "y_mean": y_mean, "y_std": y_std}, MODEL_PATH)
    np.save(GRU_X_PATH, embs.astype(np.float32))
    np.save(KNN_X_PATH, flat.astype(np.float32))
    np.save(ENDS_PATH, np.asarray([int(m["end"]) for m in metas], dtype=np.int32))

    def _dump(path, obj):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)

    _dump(GRU_META_PATH, metas)
    _dump(KNN_META_PATH, metas)  # 同一窗口集，供 KNN 同口径对比

    train_meta = {
        "route": "gru",
        "objective": "predict forward excess return",
        "window": window, "step": config.SLIDE_STEP, "forward_days": config.FORWARD_DAYS,
        "n_windows": len(metas),
        "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
        "hidden_dim": model.hidden_dim, "num_layers": model.num_layers,
        "test": metrics,
        "note": "GRU 嵌入检索库；knn_X.npy/knn_meta.json 为同窗口集扁平特征，供 gru_eval.py 对比",
    }
    _dump(TRAIN_META_PATH, train_meta)

    print("\n[完成] 已保存：")
    print(f"  {MODEL_PATH}          模型权重")
    print(f"  {GRU_X_PATH.name}   GRU 嵌入索引 {embs.shape}")
    print(f"  {GRU_META_PATH.name} 窗口元信息 {len(metas)}")
    print(f"  {KNN_X_PATH.name}   KNN 扁平特征（同窗口集）")
    print(f"  {TRAIN_META_PATH.name} 训练/评估元信息")
    print("\n下一步：python gru_retrieve.py 600519 --end-date 20260801 --top 5")
    print("        python gru_eval.py --n-queries 1000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
