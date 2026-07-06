"""
End-to-end pipeline: LSTM-predicted returns → Markowitz optimizer.
This is the "full ML lifecycle" story: train → infer → optimize.
"""
from pathlib import Path

import numpy as np
import torch

from src.models.forecaster import load_model, predict_next_day
from src.models.portfolio_opt import compute_covariance_gpu, max_sharpe_weights

CHECKPOINT_PATH = str(Path(__file__).parent.parent.parent / "results" / "checkpoints" / "lstm_best.pt")


def forecast_weights(
    recent_returns: np.ndarray,
    asset_names: list[str],
    checkpoint_path: str = CHECKPOINT_PATH,
    device: str | None = None,
) -> dict:
    """
    Given the last `window` days of returns, use the LSTM to predict
    next-day returns and solve for max-Sharpe weights.

    Returns a dict with predicted returns and optimal weights.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, ckpt = load_model(checkpoint_path, device=device)
    window = ckpt["window"]

    if recent_returns.shape[0] < window:
        raise ValueError(f"Need at least {window} days of returns, got {recent_returns.shape[0]}")

    predicted_returns = predict_next_day(model, recent_returns[-window:], device=device)
    # Annualize single-day prediction
    mu_annual = predicted_returns * 252

    cov = compute_covariance_gpu(recent_returns, device=device) * 252
    weights = max_sharpe_weights(mu_annual, cov)

    return {
        "asset_names": asset_names,
        "predicted_daily_returns": predicted_returns,
        "predicted_annual_returns": mu_annual,
        "weights": weights,
        "device": device,
    }
