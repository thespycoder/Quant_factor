"""
Point-in-time join engine.

Computes forward returns for each SEC filing, anchored strictly on
filing_date (the date the document became public on EDGAR).

Lookahead-avoidance guarantee
------------------------------
Every price used in any forward-return calculation satisfies:

    price_date  >=  t0_entry_date  >=  filing_date

where t0_entry_date is the first trading day in the price panel that is
on or after filing_date.  Three invariants protect this:

  1. We use filing_date, NEVER report_date.  report_date is the fiscal
     period the filing COVERS; the document often appears weeks later.
     Trading on report_date would use information before it was public.

  2. We snap forward to the next trading day so weekend/holiday filings
     never borrow a price from before the filing landed.

  3. The entry price itself (t0_price = adj_close at t0) is the close on
     the first day the market COULD have reacted to the filing.  No price
     observation before t0 enters any calculation.
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.store import load_filings_index, load_prices

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_OUT_PATH = Path(__file__).resolve().parent / "forward_returns.parquet"

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
# Core function
# ---------------------------------------------------------------------------

def build_forward_returns(
    form_type: str        = "10-K",
    horizons:  tuple[int, ...] = (5, 21, 63),
) -> pd.DataFrame:
    """
    Build a point-in-time forward-return table — one row per filing.

    Parameters
    ----------
    form_type :
        SEC form type to include, e.g. ``"10-K"`` or ``"10-Q"``.
    horizons :
        Trading-day horizons for forward returns.
        Default ``(5, 21, 63)`` ≈ 1 week, 1 month, 1 quarter.

    Returns
    -------
    DataFrame with columns::

        filing_date, ticker, accession_number, form_type,
        t0_entry_date, fwd_ret_<h>d  (one column per horizon)

    saved to backtest/forward_returns.parquet.

    Algorithm
    ---------
    For each filing:

    1. Parse filing_date as the public release date.
    2. Find t0: index of the first date in that ticker's price series
       that is >= filing_date, using ``np.searchsorted``.
    3. Entry price  = adj_close[t0_idx].
    4. Forward price h = adj_close[t0_idx + h]  (if within bounds).
    5. fwd_ret_h  = forward_price / entry_price - 1.
       NaN when t0_idx + h >= length of the price series.
    """
    ret_cols = [f"fwd_ret_{h}d" for h in horizons]

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    log.info("Loading filings index (form_type=%s) …", form_type)
    try:
        filings = load_filings_index(form_type=form_type)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{exc}\nRun `python data/edgar_downloader.py` first."
        ) from exc

    filings["filing_date"] = pd.to_datetime(filings["filing_date"])
    log.info("  %d filings across %d tickers", len(filings), filings["ticker"].nunique())

    log.info("Loading price panel …")
    try:
        prices = load_prices()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{exc}\nRun `python data/price_loader.py` first."
        ) from exc

    prices["date"] = pd.to_datetime(prices["date"])
    log.info(
        "  %d price rows, %d tickers  (%s → %s)",
        len(prices),
        prices["ticker"].nunique(),
        prices["date"].min().date(),
        prices["date"].max().date(),
    )

    # ------------------------------------------------------------------
    # 2. Build per-ticker price arrays (sorted, numpy for searchsorted)
    # ------------------------------------------------------------------
    # Using numpy arrays (not pandas) for the inner loop makes the
    # index-arithmetic fast and keeps the lookahead logic explicit.
    ticker_dates:  dict[str, np.ndarray] = {}
    ticker_closes: dict[str, np.ndarray] = {}

    for ticker, grp in prices.groupby("ticker", sort=False):
        grp_s = grp.sort_values("date")
        ticker_dates[ticker]  = grp_s["date"].values.astype("datetime64[D]")
        ticker_closes[ticker] = grp_s["adj_close"].values.astype(np.float64)

    price_universe  = set(ticker_dates)
    filing_universe = set(filings["ticker"].unique())
    no_price        = filing_universe - price_universe

    if no_price:
        n_skip = int((filings["ticker"].isin(no_price)).sum())
        log.warning(
            "%d ticker(s) with filings have no price data → %d filing(s) skipped: %s",
            len(no_price), n_skip, ", ".join(sorted(no_price)),
        )

    # ------------------------------------------------------------------
    # 3. Process per ticker (vectorised t0 snap, scalar forward prices)
    # ------------------------------------------------------------------
    results: list[dict] = []
    n_no_price = 0   # filings dropped — ticker absent from price panel
    n_past_end = 0   # filings dropped — filing_date after last price date

    for ticker, t_filings in filings.groupby("ticker", sort=False):
        if ticker not in price_universe:
            n_no_price += len(t_filings)
            continue

        dates  = ticker_dates[ticker]    # datetime64[D], ascending
        closes = ticker_closes[ticker]   # float64, aligned with dates
        n      = len(dates)

        # Vectorised snap: one searchsorted call for all filings of this ticker.
        # side="left" → index of the first date >= filing_date.
        fd_arr  = t_filings["filing_date"].values.astype("datetime64[D]")
        t0_idxs = np.searchsorted(dates, fd_arr, side="left")

        for i, (_, row) in enumerate(t_filings.iterrows()):
            t0_idx = int(t0_idxs[i])

            # Guard: filing_date is beyond the last available price
            if t0_idx >= n:
                n_past_end += 1
                results.append({
                    "filing_date":      row["filing_date"],
                    "ticker":           ticker,
                    "accession_number": row["accession_number"],
                    "form_type":        row["form_type"],
                    "t0_entry_date":    pd.NaT,
                    **{c: np.nan for c in ret_cols},
                })
                continue

            # All prices used from here satisfy price_date >= t0 >= filing_date
            t0_date  = pd.Timestamp(dates[t0_idx])
            t0_price = closes[t0_idx]

            fwd: dict[str, float] = {}
            for h, col in zip(horizons, ret_cols):
                fwd_idx = t0_idx + h
                # NaN when forward date would run past end of price history
                fwd[col] = (
                    float(closes[fwd_idx]) / t0_price - 1.0
                    if fwd_idx < n
                    else np.nan
                )

            results.append({
                "filing_date":      row["filing_date"],
                "ticker":           ticker,
                "accession_number": row["accession_number"],
                "form_type":        row["form_type"],
                "t0_entry_date":    t0_date,
                **fwd,
            })

    # ------------------------------------------------------------------
    # 4. Assemble DataFrame
    # ------------------------------------------------------------------
    col_order = [
        "filing_date", "ticker", "accession_number", "form_type",
        "t0_entry_date", *ret_cols,
    ]
    out = (
        pd.DataFrame(results, columns=col_order)
        .sort_values(["ticker", "filing_date"])
        .reset_index(drop=True)
    )

    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(_OUT_PATH, index=False)
    log.info("Saved → %s  (%d rows)", _OUT_PATH.name, len(out))

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    total = len(out)
    w     = 62

    print(f"\n{'-' * w}")
    print(f" Forward-return build summary")
    print(f"{'-' * w}")
    print(f"  Form type           : {form_type}")
    print(f"  Filings in output   : {total:>6,}")
    print(f"  Skipped (no prices) : {n_no_price:>6,}")
    print(f"  Skipped (past data) : {n_past_end:>6,}")
    print()
    print(f"  {'Horizon':<16}  {'Complete':>8}  {'NaN (data ends)':>15}")
    print(f"  {'-'*16}  {'-'*8}  {'-'*15}")
    for col in ret_cols:
        n_ok  = int(out[col].notna().sum())
        n_nan = total - n_ok
        print(f"  {col:<16}  {n_ok:>8,}  {n_nan:>15,}")
    print(f"{'-' * w}")

    return out


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = build_forward_returns(form_type="10-K", horizons=(5, 21, 63))
    print()
    print(
        df[["ticker", "filing_date", "t0_entry_date", "fwd_ret_5d", "fwd_ret_21d", "fwd_ret_63d"]]
        .head(10)
        .to_string(index=False)
    )
