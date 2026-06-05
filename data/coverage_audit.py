"""
Data coverage audit — inspects what is already on disk.
Does NOT download or modify anything.

Checks every ticker in config/universe.parquet against:
  - data/prices/prices.parquet      (price coverage + history depth)
  - data/raw_filings/filings_index.parquet  (filing coverage)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT         = Path(__file__).resolve().parent.parent
UNIVERSE_PATH = _ROOT / "config"   / "universe.parquet"
PRICES_PATH   = _ROOT / "data"     / "prices"       / "prices.parquet"
FILINGS_PATH  = _ROOT / "data"     / "raw_filings"  / "filings_index.parquet"

# A ticker whose earliest price date falls within the first month of the
# window is considered to have FULL history.  Anything later is PARTIAL
# (IPO'd mid-period, data gap, or survivorship effect).
_FULL_HISTORY_CUTOFF = pd.Timestamp("2015-01-31")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(path: Path, label: str) -> pd.DataFrame | None:
    """Load a parquet file, returning None (with a message) if absent."""
    if not path.exists():
        print(f"  [not found] {label}: {path.relative_to(_ROOT)}")
        return None
    return pd.read_parquet(path)


def _fmt_tickers(tickers: list[str], indent: int = 4, per_line: int = 8) -> str:
    """Format a ticker list into wrapped lines for readability."""
    pad = " " * indent
    lines = []
    for i in range(0, len(tickers), per_line):
        lines.append(pad + "  ".join(f"{t:<8}" for t in tickers[i : i + per_line]))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Audit sections
# ---------------------------------------------------------------------------

def _audit_prices(
    universe: set[str],
) -> tuple[list[str], list[str], list[str], pd.Series]:
    """
    Returns (full, partial, missing, min_date_series).
    min_date_series is empty when prices.parquet is absent.
    """
    total = len(universe)
    df = _load(PRICES_PATH, "Price panel")

    if df is None:
        print("  Run `python data/price_loader.py` to populate price data.\n")
        return [], [], sorted(universe), pd.Series(dtype="datetime64[ns]")

    df["date"] = pd.to_datetime(df["date"])
    min_dates = df.groupby("ticker")["date"].min()

    present = universe & set(min_dates.index)
    missing = sorted(universe - present)
    full    = sorted(t for t in present if min_dates[t] <= _FULL_HISTORY_CUTOFF)
    partial = sorted(t for t in present if min_dates[t] >  _FULL_HISTORY_CUTOFF)

    print(f"  Present in panel : {len(present):>4} of {total}")
    print(f"  Full history     : {len(full):>4}  (earliest date ≤ 2015-01-31)")
    print(f"  Partial history  : {len(partial):>4}  (earliest date > 2015-01-31)")
    print(f"  Missing entirely : {len(missing):>4}")

    if partial:
        print(f"\n  Partial-history tickers — {len(partial)} ticker(s):")
        for t in partial:
            print(f"    {t:<8}  starts {min_dates[t].date()}")

    if missing:
        print(f"\n  Missing tickers — {len(missing)} ticker(s):")
        print(_fmt_tickers(missing))

    return full, partial, missing, min_dates


def _audit_filings(universe: set[str]) -> tuple[list[str], list[str]]:
    """
    Returns (with_filings, without_filings).
    """
    total = len(universe)
    df = _load(FILINGS_PATH, "Filings index")

    if df is None:
        print("  Run `python data/edgar_downloader.py` to populate filings.\n")
        return [], sorted(universe)

    in_index     = universe & set(df["ticker"].unique())
    with_filings = sorted(in_index)
    no_filings   = sorted(universe - in_index)

    filing_counts = (
        df[df["ticker"].isin(universe)]
        .groupby("ticker")
        .size()
        .rename("count")
    )

    print(f"  With ≥1 filing   : {len(with_filings):>4} of {total}")
    print(f"  Zero filings     : {len(no_filings):>4}")

    if no_filings:
        print(f"\n  Tickers with no filings — {len(no_filings)} ticker(s):")
        print(_fmt_tickers(no_filings))

    return with_filings, no_filings


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------

def run_audit() -> None:
    wide = 62
    print("=" * wide)
    print(" DATA COVERAGE AUDIT")
    print("=" * wide)

    # Universe
    universe_df = _load(UNIVERSE_PATH, "Universe")
    if universe_df is None:
        print(
            "\nNo universe file found. "
            "Run `python config/universe.py` first."
        )
        return

    universe = set(universe_df["ticker"].tolist())
    total    = len(universe)
    print(f"\n  Universe: {total} tickers across "
          f"{universe_df['sector'].nunique()} sectors\n")

    # 1. Price coverage
    print("─" * wide)
    print(" 1. PRICE COVERAGE")
    print("─" * wide)
    full, partial, price_missing, _ = _audit_prices(universe)

    # 2. Filing coverage
    print()
    print("─" * wide)
    print(" 2. FILING COVERAGE")
    print("─" * wide)
    with_filings, filing_missing = _audit_filings(universe)

    # Summary
    print()
    print("=" * wide)
    print(" SUMMARY")
    print("=" * wide)
    print(
        f"  {len(full)} of {total} tickers have full price history, "
        f"{len(partial)} have partial, {len(price_missing)} are missing."
    )
    print(
        f"  {len(with_filings)} of {total} have filings, "
        f"{len(filing_missing)} have none."
    )
    print("=" * wide)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_audit()
