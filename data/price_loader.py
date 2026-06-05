"""
Stock price loader.

Downloads daily adjusted-close price history from Yahoo Finance and builds
a tidy long-format panel stored at data/prices/prices.parquet.

Why adjusted close?
  Raw close prices contain artificial jumps on split and dividend dates.
  adj_close corrects for both, so computed returns reflect actual investor
  experience rather than accounting mechanics.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

# Allow `from config.universe import …` when run as `python data/price_loader.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.universe import get_universe

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TICKERS: list[str] = ["AAPL", "MSFT", "XOM", "JPM", "WMT"]

DEFAULT_START = "2015-01-01"
DEFAULT_END   = "2024-12-31"

PRICES_DIR  = Path(__file__).resolve().parent / "prices"
PRICES_PATH = PRICES_DIR / "prices.parquet"

# How close the cached date boundaries must be to the requested range before
# we consider it "covered" — accommodates non-trading days at range edges.
_TOLERANCE = pd.Timedelta(days=7)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Single-ticker download
# ---------------------------------------------------------------------------

def _download_ticker(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """
    Fetch adjusted daily prices for one ticker via yfinance.

    Returns a DataFrame with columns [date, ticker, adj_close],
    index-reset and tz-stripped, or None if yfinance returns nothing.
    """
    try:
        raw = yf.Ticker(ticker).history(
            start=start,
            end=end,
            auto_adjust=True,   # all OHLC columns become split+dividend adjusted
            actions=False,      # exclude Dividends / Stock Splits columns
        )
    except Exception as exc:
        log.warning("yfinance error for %s: %s", ticker, exc)
        return None

    if raw is None or raw.empty:
        log.warning("No data returned for %s", ticker)
        return None

    # Strip timezone — Ticker.history() returns a tz-aware DatetimeIndex.
    # .date gives plain Python date objects; pd.to_datetime re-wraps tz-naive.
    raw.index = pd.to_datetime(raw.index.date)
    raw.index.name = "date"

    df = raw[["Close"]].rename(columns={"Close": "adj_close"}).copy()
    df = df[df["adj_close"].notna()].copy()
    df["ticker"] = ticker
    df = df.reset_index()[["date", "ticker", "adj_close"]]

    log.info(
        "  %-5s  %5d rows  (%s → %s)",
        ticker, len(df),
        df["date"].min().date(),
        df["date"].max().date(),
    )
    return df


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _load_cache() -> pd.DataFrame:
    if PRICES_PATH.exists():
        df = pd.read_parquet(PRICES_PATH)
        df["date"] = pd.to_datetime(df["date"])
        return df
    return pd.DataFrame(columns=["date", "ticker", "adj_close", "daily_return"])


def _is_covered(cache: pd.DataFrame, ticker: str, start: str, end: str) -> bool:
    """
    True if the cache already contains this ticker's data for the full
    requested range (within _TOLERANCE for non-trading days at the edges).
    """
    if cache.empty or ticker not in cache["ticker"].values:
        return False
    sub       = cache[cache["ticker"] == ticker]["date"]
    req_start = pd.Timestamp(start)
    req_end   = pd.Timestamp(end)
    return sub.min() <= req_start + _TOLERANCE and sub.max() >= req_end - _TOLERANCE


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------

def _add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append daily_return: simple pct-change from prior trading day,
    computed independently per ticker so returns don't bleed across names.
    """
    df = df.sort_values(["ticker", "date"]).copy()
    df["daily_return"] = (
        df.groupby("ticker", sort=False)["adj_close"].pct_change()
    )
    return df


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def run(
    tickers: list[str] | None = None,
    start: str                = DEFAULT_START,
    end: str                  = DEFAULT_END,
) -> pd.DataFrame:
    """
    Download prices for all tickers, merge with the on-disk cache,
    compute per-ticker daily returns, and persist to prices.parquet.

    Re-running is idempotent: tickers whose date range is already covered
    are read from cache rather than re-fetched.

    Returns the complete panel DataFrame.
    """
    if tickers is None:
        tickers = get_universe()
    PRICES_DIR.mkdir(parents=True, exist_ok=True)
    cache = _load_cache()

    cached_tickers   = [t for t in tickers if     _is_covered(cache, t, start, end)]
    download_tickers = [t for t in tickers if not _is_covered(cache, t, start, end)]

    if cached_tickers:
        log.info("Already cached (skipping download): %s", ", ".join(cached_tickers))

    fresh: list[pd.DataFrame] = []
    for ticker in download_tickers:
        log.info("Downloading %s …", ticker)
        frame = _download_ticker(ticker, start, end)
        if frame is not None:
            fresh.append(frame)

    # Pull only the relevant tickers from cache to avoid dragging in data
    # for tickers outside the current request.
    cached_rows = (
        cache[cache["ticker"].isin(cached_tickers)].drop(columns=["daily_return"], errors="ignore")
        if not cache.empty
        else pd.DataFrame()
    )

    pieces = [p for p in [cached_rows, *fresh] if not p.empty]

    if not pieces:
        log.error("No price data available for any requested ticker.")
        return pd.DataFrame(
            columns=["date", "ticker", "adj_close", "daily_return"]
        )

    panel = (
        pd.concat(pieces, ignore_index=True)
        .drop_duplicates(subset=["date", "ticker"])
        .dropna(subset=["adj_close"])
    )
    panel["date"] = pd.to_datetime(panel["date"])

    panel = _add_returns(panel)
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)

    panel.to_parquet(PRICES_PATH, index=False)
    log.info("Saved → %s  (%d rows, %d tickers)", PRICES_PATH.name, len(panel), panel["ticker"].nunique())

    return panel


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = run(start=DEFAULT_START, end=DEFAULT_END)

    divider = "─" * 52
    print(f"\n{divider}")
    print(f"Total rows     : {len(df):>10,}")
    print(f"Unique tickers : {df['ticker'].nunique():>10}")
    print(f"Date range     : {df['date'].min().date()}  →  {df['date'].max().date()}")
    print(divider)
