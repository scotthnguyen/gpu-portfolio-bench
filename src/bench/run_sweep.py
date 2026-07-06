"""
Benchmark harness: sweeps over (n_assets, n_paths, device) configurations.
Logs runtime, throughput, and GPU memory to results/benchmark_sweep.csv.
Run: python -m src.bench.run_sweep [--device cpu|cuda|both] [--quick]
"""
import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np

from src.data.fetch import fetch_prices, compute_log_returns
from src.models.var_cpu import compute_var_cvar_cpu
from src.models.var_gpu import compute_var_cvar_gpu

try:
    import pynvml
    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
except Exception:
    PYNVML_AVAILABLE = False

RESULTS_PATH = Path(__file__).parent.parent.parent / "results" / "benchmark_sweep.csv"

ASSET_COUNTS = [10, 50, 100, 500]
PATH_COUNTS = [10_000, 100_000, 1_000_000, 10_000_000]

ASSET_COUNTS_QUICK = [10, 50]
PATH_COUNTS_QUICK = [10_000, 100_000]

FIELDNAMES = [
    "timestamp", "device", "n_assets", "n_paths",
    "VaR", "CVaR", "elapsed_sec", "throughput_paths_per_sec",
    "gpu_mem_mb", "gpu_util_pct",
]


def poll_gpu_util() -> float:
    if not PYNVML_AVAILABLE:
        return -1.0
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    return float(util.gpu)


def run_sweep(devices: list[str], quick: bool = False) -> list[dict]:
    asset_counts = ASSET_COUNTS_QUICK if quick else ASSET_COUNTS
    path_counts = PATH_COUNTS_QUICK if quick else PATH_COUNTS

    prices = fetch_prices()
    returns_full = compute_log_returns(prices).dropna(axis=1)
    all_assets = list(returns_full.columns)

    rows = []
    total = len(asset_counts) * len(path_counts) * len(devices)
    idx = 0

    for n_assets in asset_counts:
        assets = all_assets[:n_assets]
        if len(assets) < n_assets:
            print(f"  Only {len(assets)} assets available, skipping n_assets={n_assets}")
            continue
        returns_np = returns_full[assets].to_numpy(dtype=np.float64)
        weights = np.ones(n_assets) / n_assets

        for n_paths in path_counts:
            for device in devices:
                idx += 1
                print(f"[{idx}/{total}] device={device} n_assets={n_assets} n_paths={n_paths:,}", flush=True)

                gpu_util = poll_gpu_util()

                try:
                    if device == "cpu":
                        result = compute_var_cvar_cpu(returns_np, weights, n_paths=n_paths)
                        result["gpu_mem_mb"] = 0.0
                        result["gpu_util_pct"] = -1.0
                    else:
                        result = compute_var_cvar_gpu(returns_np, weights, n_paths=n_paths, device=device)
                        result["gpu_util_pct"] = gpu_util
                except Exception as e:
                    print(f"  SKIPPED: {e}")
                    continue

                row = {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "n_assets": n_assets,
                    **result,
                }
                rows.append(row)
                print(
                    f"  VaR={result['VaR']:.4f}  elapsed={result['elapsed_sec']:.3f}s  "
                    f"throughput={result['throughput_paths_per_sec']:,.0f} paths/s"
                    + (f"  gpu_mem={result.get('gpu_mem_mb', 0):.1f}MB" if device != "cpu" else "")
                )

    return rows


def save_results(rows: list[dict]):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if RESULTS_PATH.exists() else "w"
    with open(RESULTS_PATH, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if mode == "w":
            writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows → {RESULTS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="both", choices=["cpu", "cuda", "both"])
    parser.add_argument("--quick", action="store_true", help="Small sweep for smoke-testing")
    args = parser.parse_args()

    import torch
    if args.device == "both":
        devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
    else:
        devices = [args.device]

    rows = run_sweep(devices, quick=args.quick)
    save_results(rows)
