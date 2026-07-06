"""
Markowitz mean-variance portfolio optimization.
Solves for minimum-variance and max-Sharpe weights.
GPU-accelerated covariance computation via PyTorch when available.
"""
import numpy as np
import torch
import cvxpy as cp
from dataclasses import dataclass


@dataclass
class EfficientFrontier:
    weights: list[np.ndarray]       # weight vectors along the frontier
    expected_returns: list[float]
    volatilities: list[float]
    sharpe_ratios: list[float]
    asset_names: list[str]


def compute_covariance_gpu(returns: np.ndarray, device: str = "cuda") -> np.ndarray:
    """Compute covariance matrix on GPU for large asset universes."""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    R = torch.tensor(returns, dtype=torch.float32, device=dev)
    R_centered = R - R.mean(dim=0)
    cov = (R_centered.T @ R_centered) / (R.shape[0] - 1)
    return cov.cpu().numpy()


def minimum_variance_weights(cov: np.ndarray) -> np.ndarray:
    n = cov.shape[0]
    w = cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov)))
    constraints = [cp.sum(w) == 1, w >= 0]
    cp.Problem(objective, constraints).solve(solver=cp.CLARABEL)
    return w.value


def max_sharpe_weights(
    mu: np.ndarray,
    cov: np.ndarray,
    risk_free: float = 0.05 / 252,
) -> np.ndarray:
    n = len(mu)
    excess = mu - risk_free
    w = cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov)))
    constraints = [excess @ w == 1, w >= 0]
    cp.Problem(objective, constraints).solve(solver=cp.CLARABEL)
    raw = w.value
    return raw / raw.sum()


def build_efficient_frontier(
    returns: np.ndarray,
    asset_names: list[str],
    n_points: int = 50,
    use_gpu: bool = True,
) -> EfficientFrontier:
    mu = returns.mean(axis=0) * 252
    cov_daily = compute_covariance_gpu(returns, device="cuda") if use_gpu else np.cov(returns.T)
    cov = cov_daily * 252
    n = len(mu)

    min_ret, max_ret = mu.min(), mu.max()
    target_returns = np.linspace(min_ret, max_ret, n_points)

    weights_list, vols, sharpes = [], [], []
    for target in target_returns:
        w = cp.Variable(n)
        obj = cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov)))
        cons = [mu @ w == target, cp.sum(w) == 1, w >= 0]
        prob = cp.Problem(obj, cons)
        prob.solve(solver=cp.CLARABEL)
        if w.value is None:
            continue
        wv = w.value
        vol = float(np.sqrt(wv @ cov @ wv))
        sr = float((target - 0.05) / vol) if vol > 0 else 0.0
        weights_list.append(wv)
        vols.append(vol)
        sharpes.append(sr)

    return EfficientFrontier(
        weights=weights_list,
        expected_returns=list(target_returns[:len(weights_list)]),
        volatilities=vols,
        sharpe_ratios=sharpes,
        asset_names=asset_names,
    )
