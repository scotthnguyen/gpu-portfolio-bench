"""
GPU-accelerated Monte Carlo VaR via PyTorch.
Key idea: replace the for-path loop with a single batched tensor op on CUDA.
"""
import time

import numpy as np
import torch


def simulate_gbm_gpu(
    returns: np.ndarray,
    weights: np.ndarray,
    n_paths: int = 100_000,
    horizon: int = 1,
    device: str = "cuda",
    seed: int = 42,
) -> torch.Tensor:
    """
    Returns 1-D tensor of simulated portfolio returns on `device`.
    """
    dev = torch.device(device)
    torch.manual_seed(seed)

    mu = torch.tensor(returns.mean(axis=0), dtype=torch.float32, device=dev)
    cov_np = np.cov(returns.T) + np.eye(returns.shape[1]) * 1e-8
    cov = torch.tensor(cov_np, dtype=torch.float32, device=dev)
    w = torch.tensor(weights, dtype=torch.float32, device=dev)

    L = torch.linalg.cholesky(cov)
    dt = horizon / 252.0

    # Single tensor op: (n_paths, n_assets) — no Python loop
    z = torch.randn(n_paths, returns.shape[1], device=dev)
    correlated = z @ L.T
    sim_returns = mu * dt + correlated * (dt ** 0.5)

    return sim_returns @ w


def compute_var_cvar_gpu(
    returns: np.ndarray,
    weights: np.ndarray,
    n_paths: int = 100_000,
    confidence: float = 0.95,
    horizon: int = 1,
    device: str = "cuda",
    seed: int = 42,
) -> dict:
    if not torch.cuda.is_available() and device == "cuda":
        raise RuntimeError("CUDA not available. Pass device='cpu' or run on a GPU instance.")

    # Warm-up pass to exclude CUDA kernel launch latency from timing
    if device == "cuda":
        _ = simulate_gbm_gpu(returns[:10], weights, n_paths=1000, device=device, seed=0)
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    portfolio_returns = simulate_gbm_gpu(returns, weights, n_paths, horizon, device, seed)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    threshold = torch.quantile(portfolio_returns, 1 - confidence).item()
    tail = portfolio_returns[portfolio_returns <= threshold]
    cvar = tail.mean().item() if tail.numel() > 0 else threshold

    gpu_mem_mb = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else 0.0

    return {
        "device": device,
        "n_paths": n_paths,
        "confidence": confidence,
        "VaR": float(-threshold),
        "CVaR": float(-cvar),
        "elapsed_sec": elapsed,
        "throughput_paths_per_sec": n_paths / elapsed,
        "gpu_mem_mb": gpu_mem_mb,
    }
