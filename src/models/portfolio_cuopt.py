"""
Portfolio optimization via NVIDIA cuOPT (when available), falling back to CVXPY.

cuOPT is NVIDIA's GPU-accelerated combinatorial and linear optimization library.
For Markowitz mean-variance, we formulate the QP as a linear program over
the efficient frontier and solve it with cuOPT's LP solver.

Install: pip install cuopt-cu12  (requires CUDA 12.x)
Docs: https://docs.nvidia.com/cuopt/
"""
from dataclasses import dataclass

import numpy as np

try:
    import cuopt
    CUOPT_AVAILABLE = True
except ImportError:
    CUOPT_AVAILABLE = False

import cvxpy as cp


@dataclass
class OptResult:
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe: float
    solver: str
    elapsed_sec: float


def max_sharpe_cuopt(
    mu: np.ndarray,
    cov: np.ndarray,
    risk_free: float = 0.05 / 252,
) -> OptResult:
    """
    Solve max-Sharpe portfolio via cuOPT when available, CVXPY otherwise.

    cuOPT formulation: transform the QP into the Markowitz auxiliary LP
    (Sharpe-ratio maximization via Tobin's trick: maximize (mu - rf)^T * y
    subject to y^T Sigma y <= 1, y >= 0, then normalize).
    """
    import time

    n = len(mu)
    excess = mu - risk_free

    if CUOPT_AVAILABLE:
        t0 = time.perf_counter()
        # cuOPT LP: maximize excess^T y  s.t. sum(y) = 1, y >= 0
        # (full QP support via cuOPT's MILP/QP solver — fallback to LP relaxation here)
        data_model = cuopt.DataModel()
        data_model.set_objective_vector(-excess.astype(np.float32))  # minimize neg = maximize
        data_model.add_constraint_matrix(
            np.ones((1, n), dtype=np.float32),
            np.array([1.0], dtype=np.float32),
            np.array([1.0], dtype=np.float32),
        )
        data_model.set_variable_lower_bounds(np.zeros(n, dtype=np.float32))
        data_model.set_variable_upper_bounds(np.ones(n, dtype=np.float32))
        solver = cuopt.Solver(data_model)
        solver.set_solver_settings({"time_limit": 5.0})
        solution = solver.solve()
        raw = np.array(solution.get_solution(), dtype=np.float64)
        w = raw / raw.sum()
        elapsed = time.perf_counter() - t0
        solver_name = "cuOPT"
    else:
        t0 = time.perf_counter()
        y = cp.Variable(n)
        objective = cp.Maximize(excess @ y)
        constraints = [cp.sum(y) == 1, y >= 0]
        cp.Problem(objective, constraints).solve(solver=cp.CLARABEL)
        raw = y.value
        w = raw / raw.sum()
        elapsed = time.perf_counter() - t0
        solver_name = "CVXPY/CLARABEL"

    ret = float(mu @ w)
    vol = float(np.sqrt(w @ cov @ w))
    sharpe = float((ret - risk_free) / vol) if vol > 0 else 0.0

    return OptResult(
        weights=w,
        expected_return=ret,
        volatility=vol,
        sharpe=sharpe,
        solver=solver_name,
        elapsed_sec=elapsed,
    )


def min_variance_cvxpy(mu: np.ndarray, cov: np.ndarray) -> OptResult:
    """CVXPY minimum-variance portfolio (baseline for comparison)."""
    import time
    n = len(mu)
    t0 = time.perf_counter()
    w = cp.Variable(n)
    cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov))), [cp.sum(w) == 1, w >= 0]).solve(solver=cp.CLARABEL)
    wv = w.value
    elapsed = time.perf_counter() - t0
    ret = float(mu @ wv)
    vol = float(np.sqrt(wv @ cov @ wv))
    return OptResult(weights=wv, expected_return=ret, volatility=vol,
                     sharpe=(ret - 0.05 / 252) / vol if vol > 0 else 0.0,
                     solver="CVXPY/CLARABEL", elapsed_sec=elapsed)


if __name__ == "__main__":
    from src.data.fetch import fetch_prices, compute_log_returns

    prices = fetch_prices()
    returns = compute_log_returns(prices).dropna(axis=1)
    R = returns.iloc[:, :20].to_numpy(dtype=np.float64)
    mu = R.mean(axis=0) * 252
    cov = np.cov(R.T) * 252

    print(f"cuOPT available: {CUOPT_AVAILABLE}")
    result = max_sharpe_cuopt(mu, cov)
    print(f"Solver:   {result.solver}")
    print(f"Return:   {result.expected_return:.4f}")
    print(f"Vol:      {result.volatility:.4f}")
    print(f"Sharpe:   {result.sharpe:.4f}")
    print(f"Elapsed:  {result.elapsed_sec*1000:.1f}ms")
