"""
RAPIDS cuDF variant of the data pipeline.
Drops in for fetch.py when cuDF is available — GPU-accelerated ETL.

cuDF mirrors the pandas API so the rest of the codebase stays unchanged;
the speedup shows up in the data-prep stage rather than the simulation stage.
"""
from pathlib import Path

import numpy as np

try:
    import cudf
    import cudf.pandas  # noqa: F401 — activates pandas compatibility layer
    CUDF_AVAILABLE = True
except ImportError:
    CUDF_AVAILABLE = False

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).parent.parent.parent / "results" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SP500_SUBSET = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "UNH", "JNJ",
    "JPM", "V", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "PEP", "KO",
    "AVGO", "COST", "WMT", "MCD", "BAC", "DIS", "CSCO", "ACN", "TMO", "ABT",
    "LIN", "DHR", "VZ", "ADBE", "CRM", "NFLX", "TXN", "NEE", "WFC", "PM",
    "AMD", "RTX", "QCOM", "HON", "UPS", "AMGN", "LOW", "SBUX", "IBM", "GS",
]


def fetch_prices_gpu(
    tickers: list[str] = SP500_SUBSET,
    start: str = "2018-01-01",
    end: str | None = None,
    force_refresh: bool = False,
) -> "cudf.DataFrame | pd.DataFrame":
    """
    Same interface as fetch.fetch_prices but returns a cuDF DataFrame when
    cuDF is available, falling back to pandas otherwise.
    """
    cache_path = CACHE_DIR / "prices_gpu.parquet"

    if cache_path.exists() and not force_refresh:
        if CUDF_AVAILABLE:
            cached = cudf.read_parquet(str(cache_path))
            last_date = cached.index.max()
        else:
            cached = pd.read_parquet(cache_path)
            last_date = cached.index.max()

        new_pd = yf.download(tickers, start=str(last_date), end=end, auto_adjust=True, progress=False)["Close"]
        new_pd = new_pd[new_pd.index > last_date]

        if not new_pd.empty:
            if CUDF_AVAILABLE:
                new = cudf.from_pandas(new_pd)
                prices = cudf.concat([cached, new]).sort_index()
            else:
                prices = pd.concat([cached, new_pd]).sort_index()
            prices = prices[~prices.index.duplicated(keep="last")]
        else:
            prices = cached
    else:
        raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
        prices_pd = raw["Close"]
        if CUDF_AVAILABLE:
            prices = cudf.from_pandas(prices_pd)
        else:
            prices = prices_pd

    if CUDF_AVAILABLE:
        prices.to_parquet(str(cache_path))
    else:
        prices.to_parquet(cache_path)

    return prices


def compute_log_returns_gpu(
    prices: "cudf.DataFrame | pd.DataFrame",
) -> "cudf.DataFrame | pd.DataFrame":
    """
    GPU-accelerated log-return computation.
    cuDF shift + log ops run on-device — no data transfer needed.
    """
    if CUDF_AVAILABLE and isinstance(prices, cudf.DataFrame):
        import cupy as cp
        prices_cupy = cp.asarray(prices.values, dtype=cp.float64)
        log_returns = cp.log(prices_cupy[1:] / prices_cupy[:-1])
        result = cudf.DataFrame(
            log_returns.get(),
            columns=prices.columns,
            index=prices.index[1:],
        )
        return result.dropna(axis=1)

    # pandas fallback
    import numpy as np
    returns = np.log(prices / prices.shift(1)).dropna()
    return returns


def benchmark_etl(n_runs: int = 5) -> dict:
    """
    Compare wall-clock time for pandas vs cuDF ETL on the same data.
    Returns timing dict for the benchmark sweep.
    """
    import time

    # pandas baseline
    from src.data.fetch import fetch_prices, compute_log_returns
    t0 = time.perf_counter()
    for _ in range(n_runs):
        prices_pd = fetch_prices(force_refresh=False)
        compute_log_returns(prices_pd)
    pandas_elapsed = (time.perf_counter() - t0) / n_runs

    result = {"pandas_etl_sec": pandas_elapsed, "cudf_etl_sec": None, "speedup": None}

    if not CUDF_AVAILABLE:
        return result

    t0 = time.perf_counter()
    for _ in range(n_runs):
        prices_gpu = fetch_prices_gpu(force_refresh=False)
        compute_log_returns_gpu(prices_gpu)
    cudf_elapsed = (time.perf_counter() - t0) / n_runs

    result["cudf_etl_sec"] = cudf_elapsed
    result["speedup"] = pandas_elapsed / cudf_elapsed
    return result


if __name__ == "__main__":
    print(f"cuDF available: {CUDF_AVAILABLE}")
    prices = fetch_prices_gpu()
    returns = compute_log_returns_gpu(prices)
    print(f"Backend: {'cuDF (GPU)' if CUDF_AVAILABLE else 'pandas (CPU)'}")
    print(f"Shape: {returns.shape}")

    if CUDF_AVAILABLE:
        timings = benchmark_etl(n_runs=3)
        print(f"pandas ETL: {timings['pandas_etl_sec']:.3f}s")
        print(f"cuDF ETL:   {timings['cudf_etl_sec']:.3f}s")
        print(f"Speedup:    {timings['speedup']:.1f}×")
