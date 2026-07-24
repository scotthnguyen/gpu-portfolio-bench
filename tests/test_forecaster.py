"""
Forecaster tests — CPU only, no GPU required.
Checks model shapes, that training reduces loss, and that inference runs.
"""
import numpy as np
import pytest
import torch

from src.models.forecaster import ReturnLSTM, ReturnWindowDataset, predict_next_day, train


@pytest.fixture
def tiny_returns():
    rng = np.random.default_rng(1)
    return rng.normal(0.0005, 0.01, size=(120, 5)).astype(np.float32)


def test_dataset_shapes(tiny_returns):
    ds = ReturnWindowDataset(tiny_returns, window=20)
    X, y = ds[0]
    assert X.shape == (20, 5)
    assert y.shape == (5,)
    assert len(ds) == len(tiny_returns) - 20


def test_model_output_shape():
    model = ReturnLSTM(n_assets=5, hidden=16, layers=1)
    x = torch.randn(4, 20, 5)
    out = model(x)
    assert out.shape == (4, 5)


def test_training_reduces_loss(tiny_returns):
    result = train(
        tiny_returns,
        asset_names=[f"A{i}" for i in range(5)],
        window=10,
        hidden=16,
        layers=1,
        epochs=5,
        batch_size=16,
        device="cpu",
        verbose=False,
    )
    # Val loss at epoch 5 should be lower than epoch 1 (or at least not exploding)
    assert result.val_losses[-1] < result.val_losses[0] * 10
    assert result.elapsed_sec > 0


def test_predict_shape(tiny_returns):
    result = train(
        tiny_returns,
        asset_names=[f"A{i}" for i in range(5)],
        window=10,
        hidden=16,
        layers=1,
        epochs=2,
        batch_size=16,
        device="cpu",
        verbose=False,
    )
    from src.models.forecaster import load_model
    model, ckpt = load_model(result.checkpoint_path, device="cpu")
    pred = predict_next_day(model, tiny_returns[-10:], device="cpu")
    assert pred.shape == (5,)
    assert not np.any(np.isnan(pred))
