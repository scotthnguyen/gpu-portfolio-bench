"""
CPU baseline: Monte Carlo Value-at-Risk and CVaR via NumPy.
Uses Geometric Brownian Motion to simulate portfolio price paths.
"""
import time
import numpy as np


def simulate_gbm_cpu(
    returns: np.ndarray,
    weights: np.ndarray,
    n_paths: int = 100_000,
    horizon: int = 1,
    seed: int = 42,
) -> np.ndarray:
    """
    Returns array of simulated portfolio returns, shape (n_paths,).
    Each path: draw one T-step return vector per asset, apply weights.
    """
    rng = np.random.default_rng(seed)
    mu = returns.mean(axis=0)
    cov = np.cov(returns.T)
    n_assets = len(mu)

    # Cholesky decomposition for correlated draws
    L = np.linalg.cholesky(cov + np.eye(n_assets) * 1e-8)

    # Draw uncorrelated standard normals: (n_paths, n_assets)
    z = rng.standard_normal((n_paths, n_assets))
    # Correlate: (n_paths, n_assets)
    correlated = z @ L.T

    # Scale by annualized drift and vol for one-day horizon
    dt = horizon / 252.0
    sim_returns = mu * dt + correlated * np.sqrt(dt)

    # Portfolio return per path
    portfolio_returns = sim_returns @ weights
    return portfolio_returns


def compute_var_cvar_cpu(
    returns: np.ndarray,
    weights: np.ndarray,
    n_paths: int = 100_000,
    confidence: float = 0.95,
    horizon: int = 1,
    seed: int = 42,
) -> dict:
    t0 = time.perf_counter()
    portfolio_returns = simulate_gbm_cpu(returns, weights, n_paths, horizon, seed)
    elapsed = time.perf_counter() - t0

    threshold = np.percentile(portfolio_returns, (1 - confidence) * 100)
    tail = portfolio_returns[portfolio_returns <= threshold]
    cvar = tail.mean() if len(tail) > 0 else threshold

    return {
        "device": "cpu",
        "n_paths": n_paths,
        "confidence": confidence,
        "VaR": float(-threshold),
        "CVaR": float(-cvar),
        "elapsed_sec": elapsed,
        "throughput_paths_per_sec": n_paths / elapsed,
    }
