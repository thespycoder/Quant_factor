"""
Phase 1 pipeline test.

Runs the full data-collection pipeline on the first 20 tickers of the
universe to catch bugs and configuration issues before the full pull.

Usage:  python data/run_pipeline_test.py
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

# Project root on sys.path so all sibling packages resolve correctly
# whether the script is run from the project root or from data/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.universe import get_universe_df
from data.edgar_downloader import run as edgar_run
from data.price_loader import run as price_run
# Import audit helpers directly so we can pass a restricted ticker set
# instead of always re-reading config/universe.parquet.
from data.coverage_audit import _audit_prices, _audit_filings
from data.store import load_filings_index, load_prices

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TEST_SIZE = 20
_W = 64   # output line width

# ---------------------------------------------------------------------------
# Logging  (configured once here; basicConfig calls in imported modules are no-ops)
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _rule(char: str = "─") -> None:
    print(char * _W)

def _step(n: int, title: str) -> None:
    print(f"\n{'─' * _W}")
    print(f" STEP {n}: {title}")
    print("─" * _W)

# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step1_select_universe() -> list[str]:
    """Load universe, return the first TEST_SIZE tickers."""
    _step(1, "SELECT TEST SUBSET")
    try:
        universe_df = get_universe_df()
    except Exception as exc:
        log.error("Cannot load universe: %s", exc)
        print("\n  Run `python config/universe.py` first.")
        sys.exit(1)

    test_df = universe_df.head(TEST_SIZE).copy()
    tickers = test_df["ticker"].tolist()

    print(f"\n  {TEST_SIZE} tickers selected (first {TEST_SIZE} rows of universe):\n")
    for _, row in test_df.iterrows():
        print(f"    {row['ticker']:<10}  {row['sector']}")

    return tickers


def step2_edgar(tickers: list[str]) -> None:
    """Run the EDGAR downloader for the test subset."""
    _step(2, "EDGAR DOWNLOADER  (10-K only, 2015–2024)")
    try:
        edgar_run(tickers=tickers, form_types=["10-K"], start_year=2015, end_year=2024)
    except Exception as exc:
        log.error("EDGAR downloader raised an unexpected error: %s", exc)
        log.error("Continuing to price loader …")


def step3_prices(tickers: list[str]) -> None:
    """Run the price loader for the test subset."""
    _step(3, "PRICE LOADER  (2015–2024)")
    try:
        price_run(tickers=tickers, start="2015-01-01", end="2024-12-31")
    except Exception as exc:
        log.error("Price loader raised an unexpected error: %s", exc)
        log.error("Continuing to coverage audit …")


def step4_audit(test_set: set[str]) -> tuple[
    list[str], list[str], list[str],   # price: full, partial, missing
    list[str], list[str],              # filings: with, without
]:
    """Run coverage audit restricted to the test ticker set."""
    _step(4, "COVERAGE AUDIT  (test subset only)")

    print(f"\n  Price coverage ({len(test_set)} tickers)")
    print("  " + "·" * 42)
    full, partial, price_missing, _ = _audit_prices(test_set)

    print(f"\n  Filing coverage ({len(test_set)} tickers)")
    print("  " + "·" * 42)
    with_filings, filing_missing = _audit_filings(test_set)

    return full, partial, price_missing, with_filings, filing_missing


def step5_report(
    tickers: list[str],
    full: list[str],
    partial: list[str],
    price_missing: list[str],
    with_filings: list[str],
    filing_missing: list[str],
) -> None:
    """Print the final PASS/FAIL summary report."""
    print(f"\n{'═' * _W}")
    print(f" PHASE 1 PIPELINE TEST — PASS/FAIL REPORT")
    print(f"{'═' * _W}")
    print(f"\n  Tickers tested : {TEST_SIZE}")

    # -- Filings --
    print(f"\n  FILINGS")
    n_with = len(with_filings)
    if filing_missing:
        print(f"  ✗  {n_with} / {TEST_SIZE} tickers have ≥1 filing")
        print(f"     Zero-filing tickers : {', '.join(filing_missing)}")
    else:
        print(f"  ✓  {n_with} / {TEST_SIZE} tickers have ≥1 filing")
    try:
        n_filings = len(load_filings_index(tickers=tickers))
        print(f"     Total filings in index : {n_filings:,}")
    except FileNotFoundError:
        print("     (filings index not found)")

    # -- Prices --
    print(f"\n  PRICES")
    n_prices = len(full) + len(partial)
    if price_missing:
        print(f"  ✗  {n_prices} / {TEST_SIZE} tickers have price data")
        print(f"     Missing tickers : {', '.join(price_missing)}")
    else:
        print(f"  ✓  {n_prices} / {TEST_SIZE} tickers have price data")
    if partial:
        print(f"     Partial history : {', '.join(partial)}")
    try:
        prices_df = load_prices(tickers=tickers)
        print(f"     Total price rows : {len(prices_df):>8,}")
        print(f"     Date range       : "
              f"{prices_df['date'].min().date()}  →  {prices_df['date'].max().date()}")
    except FileNotFoundError:
        print("     (price panel not found)")

    # -- Overall verdict --
    passed = (not filing_missing) and (not price_missing)
    print()
    _rule("═")
    if passed:
        print(f"  PASS — all {TEST_SIZE} tickers have both filings and price data.")
    else:
        gaps: list[str] = []
        if filing_missing:
            gaps.append(f"{len(filing_missing)} ticker(s) missing filings")
        if price_missing:
            gaps.append(f"{len(price_missing)} ticker(s) missing prices")
        print(f"  FAIL — {'; '.join(gaps)}.")
    _rule("═")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    tickers = step1_select_universe()
    step2_edgar(tickers)
    step3_prices(tickers)
    full, partial, price_missing, with_filings, filing_missing = step4_audit(set(tickers))
    step5_report(tickers, full, partial, price_missing, with_filings, filing_missing)


if __name__ == "__main__":
    main()
