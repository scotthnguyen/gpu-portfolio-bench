"""
LSTM return forecaster: predicts next-day log return for each asset
from a rolling window of historical returns.

Full ML lifecycle: data → train → eval → inference → feed into optimizer.
"""
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

CHECKPOINT_DIR = Path(__file__).parent.parent.parent / "results" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ── Dataset ───────────────────────────────────────────────────────────────────

class ReturnWindowDataset(Dataset):
    """
    Sliding-window dataset over log returns.
    X: (window_size, n_assets), y: (n_assets,) — next-day returns.
    """
    def __init__(self, returns: np.ndarray, window: int = 20):
        self.window = window
        R = torch.tensor(returns, dtype=torch.float32)
        self.X = torch.stack([R[i : i + window] for i in range(len(R) - window)])
        self.y = R[window:]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Model ─────────────────────────────────────────────────────────────────────

class ReturnLSTM(nn.Module):
    def __init__(self, n_assets: int, hidden: int = 64, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_assets,
            hidden_size=hidden,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden, n_assets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, n_assets) → take last hidden state
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


# ── Training ──────────────────────────────────────────────────────────────────

@dataclass
class TrainResult:
    train_losses: list[float]
    val_losses: list[float]
    checkpoint_path: str
    elapsed_sec: float
    device: str


def train(
    returns: np.ndarray,
    asset_names: list[str],
    window: int = 20,
    hidden: int = 64,
    layers: int = 2,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    val_split: float = 0.15,
    device: str | None = None,
    verbose: bool = True,
) -> TrainResult:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    # Chronological train/val split — no shuffling to avoid look-ahead
    split = int(len(returns) * (1 - val_split))
    train_ds = ReturnWindowDataset(returns[:split], window)
    val_ds = ReturnWindowDataset(returns[split:], window)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    n_assets = returns.shape[1]
    model = ReturnLSTM(n_assets, hidden, layers).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    best_val, best_epoch = float("inf"), 0
    train_losses, val_losses = [], []
    checkpoint_path = str(CHECKPOINT_DIR / "lstm_best.pt")

    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for X, y in train_dl:
            X, y = X.to(dev), y.to(dev)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item() * len(X)
        train_loss = running / len(train_ds)

        model.eval()
        vrunning = 0.0
        with torch.no_grad():
            for X, y in val_dl:
                X, y = X.to(dev), y.to(dev)
                vrunning += criterion(model(X), y).item() * len(X)
        val_loss = vrunning / len(val_ds)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_loss": best_val,
                    "n_assets": n_assets,
                    "asset_names": asset_names,
                    "window": window,
                    "hidden": hidden,
                    "layers": layers,
                },
                checkpoint_path,
            )

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(
                f"  epoch {epoch:3d}/{epochs}  train={train_loss:.6f}  val={val_loss:.6f}"
                + (" ← best" if epoch == best_epoch else "")
            )

    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"\nTraining done in {elapsed:.1f}s on {device}. Best val loss: {best_val:.6f} @ epoch {best_epoch}")

    import csv
    loss_path = CHECKPOINT_DIR.parent / "train_losses.csv"
    with open(loss_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        for i, (tl, vl) in enumerate(zip(train_losses, val_losses), 1):
            writer.writerow({"epoch": i, "train_loss": tl, "val_loss": vl})

    return TrainResult(
        train_losses=train_losses,
        val_losses=val_losses,
        checkpoint_path=checkpoint_path,
        elapsed_sec=elapsed,
        device=device,
    )


# ── Inference ─────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: str | None = None) -> tuple[ReturnLSTM, dict]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = ReturnLSTM(ckpt["n_assets"], ckpt["hidden"], ckpt["layers"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def predict_next_day(
    model: ReturnLSTM,
    recent_returns: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """
    recent_returns: (window, n_assets) — most recent `window` days.
    Returns predicted next-day log returns, shape (n_assets,).
    """
    x = torch.tensor(recent_returns, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x)
    return pred.squeeze(0).cpu().numpy()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from src.data.fetch import fetch_prices, compute_log_returns

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--assets", type=int, default=10)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    prices = fetch_prices()
    returns = compute_log_returns(prices).dropna(axis=1)
    assets = list(returns.columns[:args.assets])
    R = returns[assets].to_numpy(dtype=np.float32)

    print(f"Training on {len(assets)} assets × {len(R)} days  device={args.device or 'auto'}")
    result = train(R, assets, epochs=args.epochs, hidden=args.hidden, device=args.device)
    print(f"Checkpoint saved → {result.checkpoint_path}")
