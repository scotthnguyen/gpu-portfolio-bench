"""
Phase 5 stretch: hand-written CuPy raw CUDA kernel for Monte Carlo path generation.
Shows understanding one level below PyTorch — directly writing GPU thread logic.

The kernel generates one simulated portfolio return per CUDA thread,
replacing both the Python loop (CPU) and PyTorch's batched matmul (GPU).
"""
import time

import numpy as np

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

# Each CUDA thread simulates one independent GBM path.
# Shared inputs: mu, chol (lower-triangular Cholesky factor), weights, dt.
# Output: one portfolio return per thread.
_GBM_KERNEL_CODE = r"""
#include <curand_kernel.h>

extern "C" __global__
void gbm_portfolio_kernel(
    const double* __restrict__ mu,
    const double* __restrict__ chol,
    const double* __restrict__ weights,
    double* __restrict__ out,
    const int n_assets,
    const double dt,
    const unsigned long long seed,
    const int n_paths
) {
    int path_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (path_id >= n_paths) return;

    // Per-thread RNG state (XORWOW generator — fast, good statistical quality)
    curandState_t state;
    curand_init(seed, (unsigned long long)path_id, 0, &state);

    // Draw n_assets standard normals
    // Apply Cholesky factor row-by-row to correlate them
    double portfolio_return = 0.0;
    for (int a = 0; a < n_assets; a++) {
        double correlated = 0.0;
        for (int j = 0; j <= a; j++) {
            double z = curand_normal_double(&state);
            correlated += chol[a * n_assets + j] * z;
        }
        double asset_return = mu[a] * dt + correlated * sqrt(dt);
        portfolio_return += weights[a] * asset_return;
    }

    out[path_id] = portfolio_return;
}
"""


def _compile_kernel():
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not installed. pip install cupy-cuda12x")
    import os
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    return cp.RawKernel(
        _GBM_KERNEL_CODE,
        "gbm_portfolio_kernel",
        options=("--std=c++14", f"-I{cuda_home}/include"),
        backend="nvrtc",
    )


_KERNEL = None


def simulate_gbm_cupy_kernel(
    returns: np.ndarray,
    weights: np.ndarray,
    n_paths: int = 100_000,
    horizon: int = 1,
    seed: int = 42,
) -> "cp.ndarray":
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = _compile_kernel()

    mu = returns.mean(axis=0)
    cov = np.cov(returns.T) + np.eye(returns.shape[1]) * 1e-8
    L = np.linalg.cholesky(cov)
    dt = horizon / 252.0
    n_assets = returns.shape[1]

    mu_gpu = cp.asarray(mu, dtype=cp.float64)
    chol_gpu = cp.asarray(L, dtype=cp.float64)
    w_gpu = cp.asarray(weights, dtype=cp.float64)
    out_gpu = cp.empty(n_paths, dtype=cp.float64)

    threads = 256
    blocks = (n_paths + threads - 1) // threads
    _KERNEL(
        (blocks,), (threads,),
        (mu_gpu, chol_gpu, w_gpu, out_gpu, n_assets, dt, np.uint64(seed), n_paths),
    )
    cp.cuda.Stream.null.synchronize()
    return out_gpu


def compute_var_cvar_cupy(
    returns: np.ndarray,
    weights: np.ndarray,
    n_paths: int = 100_000,
    confidence: float = 0.95,
    horizon: int = 1,
    seed: int = 42,
) -> dict:
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available")

    # Warm-up to trigger JIT compile before timed region
    _ = simulate_gbm_cupy_kernel(returns, weights, n_paths=100, seed=0)

    t0 = time.perf_counter()
    port_returns = simulate_gbm_cupy_kernel(returns, weights, n_paths, horizon, seed)
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - t0

    threshold = float(cp.percentile(port_returns, (1 - confidence) * 100).get())
    tail = port_returns[port_returns <= threshold]
    cvar = float(tail.mean().get()) if tail.size > 0 else threshold

    mem_pool = cp.get_default_memory_pool()

    return {
        "device": "cupy_kernel",
        "n_paths": n_paths,
        "confidence": confidence,
        "VaR": float(-threshold),
        "CVaR": float(-cvar),
        "elapsed_sec": elapsed,
        "throughput_paths_per_sec": n_paths / elapsed,
        "gpu_mem_mb": mem_pool.used_bytes() / 1e6,
    }
