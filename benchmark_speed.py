"""速度基准：CPU MiniBatchKMeans vs GPU(PyTorch) K-means。

用于量化「当前规模/更大规模」下 GPU 是否值得用。用法：
    python benchmark_speed.py --n 200000 --d 60 --k 16
"""
from __future__ import annotations

import argparse
import time

import numpy as np


def kmeans_gpu(X_np: np.ndarray, k: int, iters: int = 100, seed: int = 42, tol: float = 1e-4):
    """PyTorch CUDA 版 Lloyd K-means，返回 (labels, centers, 实际迭代数)。"""
    import torch
    X = torch.as_tensor(X_np, device="cuda", dtype=torch.float32)
    n, d = X.shape
    g = torch.Generator(device="cuda").manual_seed(seed)
    perm = torch.randperm(n, generator=g, device="cuda")
    centers = X[perm[:k]].clone()
    labels = None
    actual_iters = 0
    for it in range(iters):
        dist = torch.cdist(X, centers)  # n x k
        new_labels = dist.argmin(dim=1)
        oh = torch.nn.functional.one_hot(new_labels, k).to(torch.float32)  # n x k
        counts = oh.sum(dim=0)  # k
        sums = oh.T @ X  # k x d
        new_centers = sums / counts.clamp(min=1).unsqueeze(1)
        shift = torch.norm(new_centers - centers).item()
        centers = new_centers
        labels = new_labels
        actual_iters += 1
        if shift < tol:
            break
    return labels.cpu().numpy(), centers.cpu().numpy(), actual_iters


def main() -> int:
    parser = argparse.ArgumentParser(description="CPU vs GPU K-means 速度基准")
    parser.add_argument("--n", type=int, default=200000)
    parser.add_argument("--d", type=int, default=60)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--cpu-only", action="store_true", help="只跑 CPU，跳过 GPU")
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (args.n, args.d)).astype(np.float32)
    print(f"数据规模: {args.n:,} 样本 × {args.d} 维，K={args.k}，占用 {X.nbytes/1e6:.1f} MB")
    print("-" * 60)

    # ---- CPU ----
    from sklearn.cluster import MiniBatchKMeans
    t0 = time.perf_counter()
    MiniBatchKMeans(n_clusters=args.k, batch_size=4096, random_state=42,
                    n_init=3, max_iter=200).fit_predict(X)
    t_cpu = time.perf_counter() - t0
    print(f"CPU  MiniBatchKMeans : {t_cpu:6.2f} s")

    # ---- GPU ----
    if not args.cpu_only:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _, _, iters = kmeans_gpu(X, args.k)
            torch.cuda.synchronize()
            t_gpu = time.perf_counter() - t0
            print(f"GPU  PyTorch Lloyd    : {t_gpu:6.2f} s  (迭代 {iters} 次)")
            print("-" * 60)
            print(f"GPU 加速比（仅训练）: {t_cpu / t_gpu:.2f}x")
        else:
            print("GPU 不可用，跳过")

    print("\n说明：此基准只测「纯训练」耗时，不含特征构建/基准构建（那些仍在 CPU 上）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
