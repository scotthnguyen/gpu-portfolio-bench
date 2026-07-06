"""
Fetches historical price data via yfinance and caches to Parquet.
Incremental refresh: only downloads new rows if cache already exists.
"""
import pandas as pd
import yfinance as yf
from pathlib import Path

CACHE_DIR = Path(__file__).parent.parent.parent / "results" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SP500_SUBSET = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "UNH", "JNJ",
    "JPM", "V", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "PEP", "KO",
    "AVGO", "COST", "WMT", "MCD", "BAC", "DIS", "CSCO", "ACN", "TMO", "ABT",
    "LIN", "DHR", "VZ", "ADBE", "CRM", "NFLX", "TXN", "NEE", "WFC", "PM",
    "AMD", "RTX", "QCOM", "HON", "UPS", "AMGN", "LOW", "SBUX", "IBM", "GS",
]


def fetch_prices(
    tickers: list[str] = SP500_SUBSET,
    start: str = "2018-01-01",
    end: str | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    cache_path = CACHE_DIR / "prices.parquet"

    if cache_path.exists() and not force_refresh:
        cached = pd.read_parquet(cache_path)
        last_date = cached.index.max()
        new = yf.download(tickers, start=last_date, end=end, auto_adjust=True, progress=False)["Close"]
        new = new[new.index > last_date]
        if not new.empty:
            prices = pd.concat([cached, new]).sort_index()
            prices = prices[~prices.index.duplicated(keep="last")]
        else:
            prices = cached
    else:
        raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
        prices = raw["Close"]

    prices.to_parquet(cache_path)
    return prices


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    import numpy as np
    returns = np.log(prices / prices.shift(1)).dropna()
    return returns


if __name__ == "__main__":
    prices = fetch_prices()
    returns = compute_log_returns(prices)
    print(f"Fetched {len(prices)} days × {len(prices.columns)} assets")
    print(f"Returns shape: {returns.shape}")
    print(returns.tail(3))
