"""BYOL 路线训练 + 建检索索引（路线3：无负样本自监督）。

与 gru_train.py 同层级，但 GRU 编码器额外走 BYOL 自监督分支：
- MSE 分支：保持预测未来超额收益的能力（监督头）
- BYOL 分支：online/target 双网络 + stop-gradient + EMA，让嵌入更稳健、防坍缩

用法：
    python byol_train.py                       # 全量（约 1.2M 窗口，BYOL 约 3× 纯 MSE 耗时）
    python byol_train.py --max-windows 80000   # 冒烟/快速验证
    python byol_train.py --device cpu          # 无 GPU 时

产出（models/）：
    byol.pt              在线 GRU 编码器权重（与 gru.pt 同结构）
    byol_X.npy           BYOL 嵌入索引（n × hidden）
    byol_meta.json       窗口元信息（与 gru_meta.json 同窗口集）
    byol_train_meta.json 训练超参与评估指标

注：本脚本复用 gru_train.py 的 collect_windows / sort_by_end / time_split / spearman，
仅训练循环、编码与保存路径不同。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from gru_model import build_model
from gru_train import collect_windows, sort_by_end, time_split, spearman
from byol import BYOL, augment, byol_loss
from losses import pairwise_rank_loss

ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "byol.pt"
BYOL_X_PATH = MODEL_DIR / "byol_X.npy"
BYOL_META_PATH = MODEL_DIR / "byol_meta.json"
TRAIN_META_PATH = MODEL_DIR / "byol_train_meta.json"


def train_byol(seq, excess, train_idx, val_idx, cfg, device):
    """训练 BYOL 包装的 GRU（早停），返回 (byol, y_mean, y_std)。"""
    import torch
    import torch.nn.functional as F

    torch.manual_seed(int(cfg.get("RANDOM_SEED", 42)))
    encoder = build_model(cfg, device)
    byol = BYOL(
        encoder,
        hidden_dim=int(cfg.get("GRU_HIDDEN", 32)),
        proj_dim=int(cfg.get("BYOL_PROJ_DIM", 32)),
        ema_tau=float(cfg.get("BYOL_EMA_TAU", 0.996)),
    ).to(device)
    # 只优化 online 分支（target 已 requires_grad=False）
    online_params = (list(byol.encoder.parameters())
                     + list(byol.projector.parameters())
                     + list(byol.predictor.parameters()))
    optimizer = torch.optim.Adam(
        online_params,
        lr=float(cfg.get("GRU_LR", 1e-3)),
        weight_decay=float(cfg.get("GRU_WEIGHT_DECAY", 1e-4)),
    )

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

    batch = int(cfg.get("BYOL_BATCH_SIZE", cfg.get("GRU_BATCH_SIZE", 512)))
    epochs = int(cfg.get("BYOL_EPOCHS", cfg.get("GRU_EPOCHS", 30)))
    patience = int(cfg.get("GRU_PATIENCE", 6))
    mse_w = float(cfg.get("BYOL_MSE_WEIGHT", 1.0 - float(cfg.get("BYOL_WEIGHT", 0.5))))
    byol_w = float(cfg.get("BYOL_WEIGHT", 0.5))
    mask_prob = float(cfg.get("BYOL_MASK_PROB", 0.2))
    noise_std = float(cfg.get("BYOL_NOISE_STD", 0.1))
    scale_range = float(cfg.get("BYOL_SCALE_RANGE", 0.1))
    sup_loss_name = str(cfg.get("BYOL_SUP_LOSS", "rank")).lower()
    if sup_loss_name not in ("mse", "rank"):
        raise ValueError(f"BYOL_SUP_LOSS 只支持 'mse'/'rank'，收到 {sup_loss_name}")
    use_rank = (sup_loss_name == "rank")

    # 选模口径与优化目标对齐：rank 按 val IC 最大化；mse 按 val MSE 最小化
    best_val = -float("inf") if use_rank else float("inf")
    best_state = None
    best_epoch = -1

    print(f"[train] 样本 train={len(Xtr)} val={len(Xva)} | hidden={cfg.get('GRU_HIDDEN', 32)} "
          f"epochs={epochs} batch={batch} loss={mse_w}*{sup_loss_name.upper()} + {byol_w}*BYOL | "
          f"ema_tau={cfg.get('BYOL_EMA_TAU', 0.996)} proj_dim={cfg.get('BYOL_PROJ_DIM', 32)} "
          f"选模=val_{'ic' if use_rank else 'mse'}",
          flush=True)

    for ep in range(epochs):
        byol.train()
        perm = np.random.permutation(len(Xtr))
        total = total_mse = total_byol = 0.0
        n_used = 0
        for i in range(0, len(Xtr), batch):
            idx = perm[i:i + batch]
            if len(idx) < 2:
                continue  # BYOL 的 BN + 对称损失至少需要 2 个样本
            xb = torch.from_numpy(Xtr[idx]).to(device)
            yb = torch.from_numpy(ytr_n[idx]).to(device)
            # 两个增强视图
            x1 = augment(xb, mask_prob, noise_std, scale_range)
            x2 = augment(xb, mask_prob, noise_std, scale_range)
            # 监督分支：原始 x 的预测（rank 或 mse）
            pred, _ = byol.encode(xb)
            sup = pairwise_rank_loss(pred, yb) if use_rank else F.mse_loss(pred, yb)
            # 自监督分支：两个视图的 BYOL 对齐
            p1 = byol.online_proj_pred(x1)
            p2 = byol.online_proj_pred(x2)
            z1 = byol.target_proj(x1)
            z2 = byol.target_proj(x2)
            b = byol_loss(p1, p2, z1, z2)
            loss = mse_w * sup + byol_w * b
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            byol.update_target()
            total += float(loss.item()) * len(idx)
            total_mse += float(sup.item()) * len(idx)
            total_byol += float(b.item()) * len(idx)
            n_used += len(idx)
        train_loss = total / max(n_used, 1)
        train_mse = total_mse / max(n_used, 1)
        train_byol_v = total_byol / max(n_used, 1)

        # val（rank 按 IC 选模；mse 按 MSE 选模）
        byol.eval()
        with torch.no_grad():
            pred, _ = byol.encode(torch.from_numpy(Xva).to(device))
            pred_np = pred.detach().cpu().numpy()
        if use_rank:
            val_score = spearman(pred_np, yva_n)
            better = val_score > best_val + 1e-6
            val_txt = f"val_ic={val_score:+.4f}"
        else:
            val_score = float(F.mse_loss(pred, torch.from_numpy(yva_n).to(device)).item())
            better = val_score < best_val - 1e-6
            val_txt = f"val_mse={val_score:.6f}"

        print(f"[train] epoch {ep + 1:>3}/{epochs}  train total={train_loss:.4f}  "
              f"sup={train_mse:.4f}  byol={train_byol_v:.4f}  {val_txt}",
              flush=True)
        if better:
            best_val = val_score
            best_epoch = ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in byol.encoder.state_dict().items()}
        elif ep + 1 - best_epoch >= patience:
            print(f"[train] 早停 @ epoch {ep + 1}（最优 epoch {best_epoch}）", flush=True)
            break

    if best_state is not None:
        byol.encoder.load_state_dict(best_state)
    return byol, y_mean, y_std


def evaluate_model(byol, seq, excess, idx, device, y_mean, y_std, batch_size=4096, use_rank=False):
    """评估 BYOL 编码器的监督预测能力（IC, hit）。
    use_rank=True 时预测值无刻度含义：MSE/MAE 仅供参考，hit 按预测中位数为零点计算。"""
    import torch
    import torch.nn.functional as F
    mask = ~np.isnan(excess[idx])
    if mask.sum() < 3:
        return {"n": int(mask.sum()), "mse": None, "mae": None, "ic": None, "hit": None}
    X = seq[idx][mask]
    y = excess[idx][mask].astype(np.float32)
    preds = []
    byol.eval()
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i + batch_size]).to(device)
            p, _ = byol.encode(xb)
            preds.append(p.detach().cpu().numpy())
    raw_pred = np.concatenate(preds)
    pred = raw_pred * y_std + y_mean  # 反标准化（rank 模型下仅影响 MSE/MAE 读数）
    mse = float(np.mean((pred - y) ** 2))
    mae = float(np.mean(np.abs(pred - y)))
    ic = spearman(pred, y)  # Spearman 对单调变换不变，rank 模型下依然有效
    if use_rank:
        zero = float(np.median(raw_pred))  # 排序模型的零点无意义，用中位数近似
        hit = float(np.mean((raw_pred > zero) == (y > 0)))
    else:
        hit = float(np.mean((pred > 0) == (y > 0)))
    return {"n": int(len(y)), "mse": mse, "mae": mae, "ic": ic, "hit": hit}


def encode_byol(byol, seq_np, device, batch_size=4096):
    """把 (n, WINDOW, 4) 的 numpy 序列批量编码为 (n, hidden_dim) BYOL 嵌入。"""
    import torch
    byol.eval()
    embs: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(seq_np), batch_size):
            xb = torch.from_numpy(seq_np[i:i + batch_size]).to(device)
            _, e = byol.encode(xb)
            embs.append(e.detach().cpu().numpy())
    if embs:
        return np.vstack(embs)
    return np.empty((0, byol.hidden_dim), dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="BYOL 路线训练 + 建检索索引")
    parser.add_argument("--max-windows", type=int, default=None, help="窗口上限（默认全量）")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖 config.BYOL_EPOCHS")
    parser.add_argument("--hidden", type=int, default=None, help="覆盖 config.GRU_HIDDEN")
    parser.add_argument("--device", default="auto", help="auto / cuda / cpu")
    parser.add_argument("--sup-loss", choices=["mse", "rank"], default=None,
                        help="监督损失（默认 config.BYOL_SUP_LOSS='rank'）：rank=pairwise 排序损失（IC 基座），mse=旧版点预测")
    args = parser.parse_args()

    cfg = dict(vars(config))
    cfg["GRU_EPOCHS"] = args.epochs or cfg.get("BYOL_EPOCHS") or cfg.get("GRU_EPOCHS", 30)
    cfg["GRU_HIDDEN"] = args.hidden or cfg.get("GRU_HIDDEN", 32)
    cfg["BYOL_SUP_LOSS"] = args.sup_loss or cfg.get("BYOL_SUP_LOSS", "rank")

    import torch
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[device] 使用 {device}", flush=True)

    data_dir = ROOT / config.KLINE_DIR
    window = config.WINDOW
    train_frac = float(cfg.get("GRU_TRAIN_FRAC", 0.7))
    val_frac = float(cfg.get("GRU_VAL_FRAC", 0.15))

    print("=" * 64)
    print(f"BYOL 路线训练（{str(cfg['BYOL_SUP_LOSS']).upper()} 监督 + BYOL 自监督，无负样本、stop-grad + EMA 防坍缩）")
    print(f"  窗口={window} | 步长={config.SLIDE_STEP} | 前向={config.FORWARD_DAYS}日 | "
          f"train/val/test={train_frac}/{val_frac}/{1 - train_frac - val_frac:.2f}")
    print("=" * 64)

    seq, flat, excess, metas = collect_windows(
        data_dir, window, config.SLIDE_STEP, config.FORWARD_DAYS,
        args.max_windows, config.RANDOM_SEED)
    seq, flat, excess, metas = sort_by_end(seq, flat, excess, metas)
    train_idx, val_idx, test_idx = time_split(len(metas), train_frac, val_frac)
    print(f"[split] 按结束日期切分：train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    byol, y_mean, y_std = train_byol(seq, excess, train_idx, val_idx, cfg, device)

    # 测试集评估
    use_rank = str(cfg["BYOL_SUP_LOSS"]).lower() == "rank"
    print("\n[eval] 测试集（时间上最后一段，模型从未见过）：")
    metrics = evaluate_model(byol, seq, excess, test_idx, device, y_mean, y_std, use_rank=use_rank)
    ic_txt = f"{metrics['ic']:+.4f}" if metrics["ic"] is not None else "N/A"
    mse_txt = f"{metrics['mse']:.6f}" if metrics["mse"] is not None else "N/A"
    hit_txt = f"{metrics['hit']*100:.1f}%" if metrics["hit"] is not None else "N/A"
    print(f"  样本={metrics['n']} | MSE={mse_txt} | MAE={metrics['mae'] or 0:.5f} | "
          f"IC(秩相关)={ic_txt} | 方向命中={hit_txt}")
    if use_rank:
        print("  注：rank 模式下 MSE/MAE 无刻度含义，hit 按预测中位数为零点估算，看 IC 即可")
    else:
        print("  注：IC 接近 0 是日线形态的常态")

    # 编码全部窗口 -> BYOL 嵌入索引
    print("\n[encode] 编码全部窗口为 BYOL 嵌入...", flush=True)
    t0 = time.perf_counter()
    embs = encode_byol(byol, seq, device)
    print(f"[encode] 完成 {embs.shape} 用时 {time.perf_counter() - t0:.1f}s")

    # 保存
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    import torch as _torch
    _torch.save({"state_dict": byol.encoder.state_dict(), "hidden_dim": byol.hidden_dim,
                 "num_layers": byol.encoder.num_layers, "input_dim": 4,
                 "y_mean": y_mean, "y_std": y_std}, MODEL_PATH)
    np.save(BYOL_X_PATH, embs.astype(np.float32))

    def _dump(path, obj):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)

    _dump(BYOL_META_PATH, metas)

    train_meta = {
        "route": "byol",
        "objective": f"{str(cfg['BYOL_SUP_LOSS']).upper()} + BYOL (no negatives, stop-grad + EMA)",
        "sup_loss": str(cfg["BYOL_SUP_LOSS"]).lower(),
        "window": window, "step": config.SLIDE_STEP, "forward_days": config.FORWARD_DAYS,
        "n_windows": len(metas),
        "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
        "hidden_dim": byol.hidden_dim, "num_layers": byol.encoder.num_layers,
        "byol_weight": cfg.get("BYOL_WEIGHT", 0.5),
        "byol_ema_tau": cfg.get("BYOL_EMA_TAU", 0.996),
        "byol_proj_dim": cfg.get("BYOL_PROJ_DIM", 32),
        "byol_mask_prob": cfg.get("BYOL_MASK_PROB", 0.2),
        "byol_noise_std": cfg.get("BYOL_NOISE_STD", 0.1),
        "byol_scale_range": cfg.get("BYOL_SCALE_RANGE", 0.1),
        "test": metrics,
        "note": "BYOL 嵌入检索库；与 gru_train.py 同窗口集同切分；模型权重只存 online encoder（与 gru.pt 同结构，可互加载）",
    }
    _dump(TRAIN_META_PATH, train_meta)

    print("\n[完成] 已保存：")
    print(f"  {MODEL_PATH.name}              BYOL 编码器（仅 online encoder）")
    print(f"  {BYOL_X_PATH.name}            BYOL 嵌入索引 {embs.shape}")
    print(f"  {BYOL_META_PATH.name}          窗口元信息 {len(metas)}")
    print(f"  {TRAIN_META_PATH.name}        训练/评估元信息")
    print("\n下一步：启动 server.py，在前端「③ 相似形态」区选择 BYOL 检索器")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())