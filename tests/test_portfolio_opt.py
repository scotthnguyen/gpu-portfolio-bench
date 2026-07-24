"""
Tests for Markowitz portfolio optimizer.
Validates min-variance weights, max-Sharpe weights, and efficient frontier shape.
"""
import numpy as np
import pytest

from src.models.portfolio_opt import (
    EfficientFrontier,
    build_efficient_frontier,
    max_sharpe_weights,
    minimum_variance_weights,
)

RNG = np.random.default_rng(0)
N_ASSETS = 5
N_OBS = 252


@pytest.fixture(scope="module")
def synthetic_returns():
    mu = RNG.uniform(0.0002, 0.001, N_ASSETS)
    A = RNG.standard_normal((N_OBS, N_ASSETS))
    cov = (A.T @ A) / N_OBS + np.eye(N_ASSETS) * 1e-4
    L = np.linalg.cholesky(cov)
    return RNG.standard_normal((N_OBS, N_ASSETS)) @ L.T + mu


def test_min_variance_weights_sum_to_one(synthetic_returns):
    cov = np.cov(synthetic_returns.T)
    w = minimum_variance_weights(cov)
    assert w is not None, "Solver returned None"
    assert abs(w.sum() - 1.0) < 1e-4
    assert (w >= -1e-4).all()


def test_max_sharpe_weights_sum_to_one(synthetic_returns):
    mu = synthetic_returns.mean(axis=0) * 252
    cov = np.cov(synthetic_returns.T) * 252
    w = max_sharpe_weights(mu, cov)
    assert w is not None, "Solver returned None"
    assert abs(w.sum() - 1.0) < 1e-4
    assert (w >= -1e-4).all()


def test_efficient_frontier_shape(synthetic_returns):
    asset_names = [f"A{i}" for i in range(N_ASSETS)]
    ef: EfficientFrontier = build_efficient_frontier(
        synthetic_returns, asset_names, n_points=10, use_gpu=False
    )
    assert isinstance(ef, EfficientFrontier)
    assert len(ef.volatilities) > 0
    assert len(ef.volatilities) == len(ef.expected_returns)
    assert all(v > 0 for v in ef.volatilities)


def test_efficient_frontier_risk_return_tradeoff(synthetic_returns):
    asset_names = [f"A{i}" for i in range(N_ASSETS)]
    ef = build_efficient_frontier(synthetic_returns, asset_names, n_points=20, use_gpu=False)
    vols = np.array(ef.volatilities)
    rets = np.array(ef.expected_returns[: len(ef.volatilities)])
    # Higher return targets should generally have higher volatility
    assert vols.max() > vols.min()
    assert rets.max() > rets.min()
