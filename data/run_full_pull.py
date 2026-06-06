"""
Full data pull — runs the entire data-collection pipeline on all ~150
universe tickers.

Usage:  python data/run_full_pull.py

Re-running is safe: the EDGAR downloader skips already-saved filing texts,
and the price loader skips tickers whose date range is already cached.
"""

from __future__ import annotations

import sys
import time
import logging
from pathlib import Path

# Project root on sys.path so all package imports resolve correctly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Universe / config
from config.universe import get_universe_df

# EDGAR lower-level functions — imported directly so we can:
#   (a) fetch the CIK map exactly once instead of once per ticker, and
#   (b) load/save the filings index exactly once rather than 150 times.
from data.edgar_downloader import (
    build_ticker_cik_map,
    get_filing_records,
    download_filing,
    load_index  as _edgar_load_index,
    save_index  as _edgar_save_index,
    _INDEX_COLS as _EDGAR_INDEX_COLS,
)

# Price lower-level functions — same reasoning: load cache once, save once.
from data.price_loader import (
    _download_ticker,
    _is_covered,
    _load_cache,
    _add_returns,
    PRICES_DIR,
    PRICES_PATH,
)

# Audit helpers and final store access
from data.coverage_audit import _audit_prices, _audit_filings
from data.store import load_filings_index, load_prices, save_prices

import pandas as pd

# ---------------------------------------------------------------------------
# Logging  (basicConfig calls in imported modules become no-ops after this)
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_W = 64   # output line width

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_eta(elapsed_s: float, done: int, total: int) -> str:
    """Linear ETA estimate: 'calculating…' until the first ticker completes."""
    if done == 0:
        return "calculating…"
    remaining = total - done
    eta_s = int(elapsed_s / done * remaining)
    h, rem = divmod(eta_s, 3600)
    m, s   = divmod(rem, 60)
    if h:
        return f"~{h}h {m:02d}m"
    if m:
        return f"~{m}m {s:02d}s"
    return f"~{s}s"


def _fmt_elapsed(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _rule(char: str = "─") -> None:
    print(char * _W)


def _step(n: int, title: str) -> None:
    print(f"\n{'─' * _W}")
    print(f" STEP {n}: {title}")
    print("─" * _W)


def _fmt_tickers(tickers: list[str], per_line: int = 8) -> str:
    lines = []
    for i in range(0, len(tickers), per_line):
        lines.append("    " + "  ".join(f"{t:<8}" for t in tickers[i : i + per_line]))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Step 1 — load universe
# ---------------------------------------------------------------------------

def step1_load_universe() -> list[str]:
    _step(1, "LOAD UNIVERSE")
    try:
        df = get_universe_df()
    except Exception as exc:
        log.error("Cannot load universe: %s", exc)
        print("  Run `python config/universe.py` first.")
        sys.exit(1)

    tickers = df["ticker"].tolist()
    sectors = df["sector"].nunique()
    print(f"\n  {len(tickers)} tickers across {sectors} GICS sectors.\n")
    breakdown = df.groupby("sector").size().sort_values(ascending=False)
    for sector, count in breakdown.items():
        print(f"  {sector:<42}  {count:>3}")
    return tickers


# ---------------------------------------------------------------------------
# Step 2 — EDGAR downloader
# ---------------------------------------------------------------------------

def step2_edgar(
    tickers: list[str],
    form_types: list[str] = None,
    start_year: int = 2015,
    end_year:   int = 2024,
) -> list[str]:
    """
    Download 10-K filings for all tickers.  Fetches the CIK map and loads
    the filings index exactly once; saves the index once at the end.
    Returns a list of tickers that encountered errors.
    """
    if form_types is None:
        form_types = ["10-K"]

    total = len(tickers)
    _step(2, f"EDGAR DOWNLOADER  ({', '.join(form_types)}, {start_year}–{end_year}) — {total} tickers")

    log.info("Fetching CIK map …")
    ticker_cik       = build_ticker_cik_map()
    index_df         = _edgar_load_index()
    known_accessions = set(index_df["accession_number"].tolist())
    new_rows: list[dict] = []
    errors:   list[str]  = []

    t0 = time.time()
    for i, ticker in enumerate(tickers, 1):
        ticker = ticker.upper()
        elapsed = time.time() - t0
        eta     = _fmt_eta(elapsed, i - 1, total)
        log.info("[%d/%d]  %s  (ETA %s)", i, total, ticker, eta)

        cik = ticker_cik.get(ticker)
        if not cik:
            log.warning("  No CIK found for %s — skipping", ticker)
            errors.append(ticker)
            continue

        try:
            records = get_filing_records(cik, form_types, start_year, end_year)
        except Exception as exc:
            log.error("  Metadata fetch failed for %s: %s", ticker, exc)
            errors.append(ticker)
            continue

        log.info("  %d filing(s) matched", len(records))
        for rec in records:
            download_filing(ticker, rec)
            if rec["accession_number"] not in known_accessions:
                new_rows.append({"ticker": ticker, **rec})
                known_accessions.add(rec["accession_number"])

    if new_rows:
        new_df   = pd.DataFrame(new_rows)[_EDGAR_INDEX_COLS]
        index_df = pd.concat([index_df, new_df], ignore_index=True)

    _edgar_save_index(index_df)

    elapsed_total = time.time() - t0
    log.info(
        "EDGAR complete in %s — %d new row(s) added, %d error(s).",
        _fmt_elapsed(elapsed_total), len(new_rows), len(errors),
    )
    if errors:
        log.warning("EDGAR errors on: %s", ", ".join(errors))
    return errors


# ---------------------------------------------------------------------------
# Step 3 — price loader
# ---------------------------------------------------------------------------

def step3_prices(
    tickers:   list[str],
    start: str = "2015-01-01",
    end:   str = "2024-12-31",
) -> list[str]:
    """
    Download adjusted daily prices for all tickers.  Loads the cache and
    saves the panel exactly once.  Returns a list of tickers with errors.
    """
    total = len(tickers)
    _step(3, f"PRICE LOADER  ({start} → {end}) — {total} tickers")

    PRICES_DIR.mkdir(parents=True, exist_ok=True)
    cache  = _load_cache()
    fresh:  list[pd.DataFrame] = []
    errors: list[str] = []

    t0 = time.time()
    for i, ticker in enumerate(tickers, 1):
        elapsed = time.time() - t0
        eta     = _fmt_eta(elapsed, i - 1, total)
        log.info("[%d/%d]  %s  (ETA %s)", i, total, ticker, eta)

        if _is_covered(cache, ticker, start, end):
            log.info("  CACHED — skipping download")
            continue

        try:
            frame = _download_ticker(ticker, start, end)
            if frame is not None:
                fresh.append(frame)
        except Exception as exc:
            log.error("  Download failed for %s: %s", ticker, exc)
            errors.append(ticker)

    # Merge cache + fresh data, recompute returns, save once
    cached_rows = (
        cache.drop(columns=["daily_return"], errors="ignore")
        if not cache.empty else pd.DataFrame()
    )
    pieces = [p for p in [cached_rows, *fresh] if not p.empty]

    if pieces:
        panel = (
            pd.concat(pieces, ignore_index=True)
            .drop_duplicates(subset=["date", "ticker"])
            .dropna(subset=["adj_close"])
        )
        panel["date"] = pd.to_datetime(panel["date"])
        panel = _add_returns(panel).sort_values(["ticker", "date"]).reset_index(drop=True)
        save_prices(panel)
        log.info("Price panel saved — %d rows, %d tickers.", len(panel), panel["ticker"].nunique())
    else:
        log.warning("No price data to save.")

    elapsed_total = time.time() - t0
    log.info(
        "Prices complete in %s — %d error(s).",
        _fmt_elapsed(elapsed_total), len(errors),
    )
    if errors:
        log.warning("Price errors on: %s", ", ".join(errors))
    return errors


# ---------------------------------------------------------------------------
# Step 4 — coverage audit
# ---------------------------------------------------------------------------

def step4_audit(universe_set: set[str]) -> tuple[
    list[str], list[str], list[str],
    list[str], list[str],
]:
    _step(4, "COVERAGE AUDIT  (full universe)")

    print(f"\n  Price coverage ({len(universe_set)} tickers)")
    print("  " + "·" * 50)
    full, partial, price_missing, _ = _audit_prices(universe_set)

    print(f"\n  Filing coverage ({len(universe_set)} tickers)")
    print("  " + "·" * 50)
    with_filings, filing_missing = _audit_filings(universe_set)

    return full, partial, price_missing, with_filings, filing_missing


# ---------------------------------------------------------------------------
# Step 5 — final report
# ---------------------------------------------------------------------------

def step5_report(
    total:          int,
    edgar_errors:   list[str],
    price_errors:   list[str],
    full:           list[str],
    partial:        list[str],
    price_missing:  list[str],
    with_filings:   list[str],
    filing_missing: list[str],
) -> None:
    print(f"\n{'═' * _W}")
    print(f" FULL PULL — FINAL REPORT")
    print(f"{'═' * _W}")
    print(f"\n  Universe tickers : {total}")

    # Filings
    print(f"\n  FILINGS")
    if filing_missing:
        print(f"  ✗  {len(with_filings)} / {total} tickers have ≥1 filing")
        print(f"     Zero-filing tickers ({len(filing_missing)}):")
        print(_fmt_tickers(filing_missing))
    else:
        print(f"  ✓  {len(with_filings)} / {total} tickers have ≥1 filing")
    if edgar_errors:
        print(f"     Download errors on : {', '.join(edgar_errors)}")
    try:
        print(f"     Total filings in index : {len(load_filings_index()):,}")
    except FileNotFoundError:
        print("     (filings index not found)")

    # Prices
    print(f"\n  PRICES")
    n_prices = len(full) + len(partial)
    if price_missing:
        print(f"  ✗  {n_prices} / {total} tickers have price data")
        print(f"     Missing tickers ({len(price_missing)}):")
        print(_fmt_tickers(price_missing))
    else:
        print(f"  ✓  {n_prices} / {total} tickers have price data")
    if partial:
        print(f"     Partial-history tickers ({len(partial)}) — started after 2015-01-31:")
        print(_fmt_tickers(partial))
    if price_errors:
        print(f"     Download errors on : {', '.join(price_errors)}")
    try:
        prices_df = load_prices()
        print(f"     Total price rows : {len(prices_df):>10,}")
        print(
            f"     Date range       : "
            f"{prices_df['date'].min().date()}  →  {prices_df['date'].max().date()}"
        )
    except FileNotFoundError:
        print("     (price panel not found)")

    # Verdict
    passed = not filing_missing and not price_missing
    print()
    _rule("═")
    if passed:
        print(f"  PASS — all {total} tickers have filings and price data.")
    else:
        gaps = []
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
    tickers      = step1_load_universe()
    edgar_errors = step2_edgar(tickers)
    price_errors = step3_prices(tickers)

    full, partial, price_missing, with_filings, filing_missing = step4_audit(set(tickers))

    step5_report(
        total          = len(tickers),
        edgar_errors   = edgar_errors,
        price_errors   = price_errors,
        full           = full,
        partial        = partial,
        price_missing  = price_missing,
        with_filings   = with_filings,
        filing_missing = filing_missing,
    )


if __name__ == "__main__":
    main()
