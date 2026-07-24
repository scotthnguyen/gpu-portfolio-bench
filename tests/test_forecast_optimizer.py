"""
Tests for the LSTM → Markowitz pipeline.
Verifies that forecast_weights returns valid allocations given a checkpoint.
"""
from pathlib import Path

import numpy as np
import pytest

CHECKPOINT_PATH = Path(__file__).parent.parent / "results" / "checkpoints" / "lstm_best.pt"


@pytest.mark.skipif(not CHECKPOINT_PATH.exists(), reason="No trained checkpoint — run forecaster first")
def test_forecast_weights_shape_and_sum():
    from src.models.forecast_optimizer import forecast_weights

    rng = np.random.default_rng(0)
    import torch
    ckpt = torch.load(str(CHECKPOINT_PATH), map_location="cpu")
    n_assets = ckpt["n_assets"]
    asset_names = ckpt["asset_names"]
    window = ckpt["window"]

    # Simulate recent returns matching the checkpoint's asset universe
    recent = rng.standard_normal((window + 5, n_assets)).astype(np.float32) * 0.01

    result = forecast_weights(recent, asset_names, checkpoint_path=str(CHECKPOINT_PATH), device="cpu")

    assert "weights" in result
    assert "predicted_daily_returns" in result
    assert len(result["weights"]) == n_assets
    assert abs(result["weights"].sum() - 1.0) < 1e-4
    assert (result["weights"] >= -1e-4).all()
    assert result["predicted_daily_returns"].shape == (n_assets,)


@pytest.mark.skipif(not CHECKPOINT_PATH.exists(), reason="No trained checkpoint — run forecaster first")
def test_forecast_weights_insufficient_history_raises():
    import torch

    from src.models.forecast_optimizer import forecast_weights
    ckpt = torch.load(str(CHECKPOINT_PATH), map_location="cpu")
    asset_names = ckpt["asset_names"]
    window = ckpt["window"]

    too_short = np.zeros((window - 1, ckpt["n_assets"]), dtype=np.float32)
    with pytest.raises(ValueError, match="at least"):
        forecast_weights(too_short, asset_names, checkpoint_path=str(CHECKPOINT_PATH), device="cpu")
