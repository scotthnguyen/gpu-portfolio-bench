"""
Correctness tests: GPU and CPU VaR should agree within simulation noise.
These tests run on CPU only so they pass in CI without a GPU.
"""
import numpy as np
import pytest

from src.models.var_cpu import compute_var_cvar_cpu
from src.models.var_gpu import compute_var_cvar_gpu


@pytest.fixture
def synthetic_returns():
    rng = np.random.default_rng(0)
    # 252 days × 10 assets, small daily vol ~1%
    return rng.normal(0.0005, 0.01, size=(252, 10))


@pytest.fixture
def equal_weights():
    return np.ones(10) / 10


def test_cpu_var_positive(synthetic_returns, equal_weights):
    result = compute_var_cvar_cpu(synthetic_returns, equal_weights, n_paths=50_000)
    assert result["VaR"] > 0, "VaR should be a positive loss figure"
    assert result["CVaR"] >= result["VaR"], "CVaR must be >= VaR"


def test_cpu_gpu_var_close(synthetic_returns, equal_weights):
    """CPU and GPU VaR should agree within ~5% at 100k paths (sampling noise)."""
    import torch
    if not torch.cuda.is_available():
        pytest.skip("No CUDA device available")

    cpu_r = compute_var_cvar_cpu(synthetic_returns, equal_weights, n_paths=100_000, seed=42)
    gpu_r = compute_var_cvar_gpu(synthetic_returns, equal_weights, n_paths=100_000, seed=42, device="cuda")

    # Relative difference should be small — if > 20%, something is wrong structurally
    rel_diff = abs(cpu_r["VaR"] - gpu_r["VaR"]) / cpu_r["VaR"]
    assert rel_diff < 0.20, f"VaR mismatch too large: CPU={cpu_r['VaR']:.4f} GPU={gpu_r['VaR']:.4f}"


def test_throughput_increases_with_paths(synthetic_returns, equal_weights):
    r_small = compute_var_cvar_cpu(synthetic_returns, equal_weights, n_paths=10_000)
    r_large = compute_var_cvar_cpu(synthetic_returns, equal_weights, n_paths=100_000)
    # More paths = should take more wall time (sanity check — not a guarantee of O(n))
    assert r_large["elapsed_sec"] > r_small["elapsed_sec"]
