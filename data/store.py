"""
Data store — single access layer for all on-disk data.

Every other module reads and writes through these functions.
No module outside this file should construct raw file paths.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Canonical paths  (everything derived from project root)
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent

_UNIVERSE_PATH    = _ROOT / "config" / "universe.parquet"
_PRICES_PATH      = _ROOT / "data"   / "prices"      / "prices.parquet"
_FILINGS_IDX_PATH = _ROOT / "data"   / "raw_filings" / "filings_index.parquet"
_RAW_FILINGS_DIR  = _ROOT / "data"   / "raw_filings"

# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _require(path: Path, script_hint: str) -> None:
    """Raise a friendly FileNotFoundError if a data file is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not built yet: {path.relative_to(_ROOT).as_posix()}\n"
            f"  Create it with:  {script_hint}"
        )

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------

def load_universe() -> pd.DataFrame:
    """
    Load the investment universe.

    Returns
    -------
    DataFrame with columns: ticker, wiki_symbol, company_name, sector.
    """
    _require(_UNIVERSE_PATH, "python config/universe.py")
    return pd.read_parquet(_UNIVERSE_PATH)

# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------

def load_prices(
    tickers: list[str] | None = None,
    start:   str       | None = None,
    end:     str       | None = None,
) -> pd.DataFrame:
    """
    Load the daily price panel.

    Columns: date, ticker, adj_close, daily_return.

    Parameters
    ----------
    tickers : ticker list to keep; None returns all tickers.
    start   : inclusive lower bound, e.g. ``"2018-01-01"``; None = no bound.
    end     : inclusive upper bound, e.g. ``"2023-12-31"``; None = no bound.
    """
    _require(_PRICES_PATH, "python data/price_loader.py")
    df = pd.read_parquet(_PRICES_PATH)
    df["date"] = pd.to_datetime(df["date"])

    if tickers is not None:
        df = df[df["ticker"].isin(tickers)]
    if start is not None:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["date"] <= pd.Timestamp(end)]

    return df.reset_index(drop=True)


def save_prices(df: pd.DataFrame) -> None:
    """Overwrite data/prices/prices.parquet with *df*."""
    _PRICES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_PRICES_PATH, index=False)

# ---------------------------------------------------------------------------
# Filings
# ---------------------------------------------------------------------------

def load_filings_index(
    tickers:   list[str] | None = None,
    form_type: str       | None = None,
) -> pd.DataFrame:
    """
    Load the filings metadata index.

    Columns: ticker, cik, form_type, filing_date, report_date,
             accession_number, primary_document_url.

    Parameters
    ----------
    tickers   : ticker list to keep; None returns all.
    form_type : e.g. ``"10-K"`` or ``"10-Q"``; None returns all form types.
    """
    _require(_FILINGS_IDX_PATH, "python data/edgar_downloader.py")
    df = pd.read_parquet(_FILINGS_IDX_PATH)

    if tickers is not None:
        df = df[df["ticker"].isin(tickers)]
    if form_type is not None:
        df = df[df["form_type"] == form_type]

    return df.reset_index(drop=True)


def load_filing_text(ticker: str, accession_number: str) -> str:
    """
    Return the saved plain-text content of one filing.

    Parameters
    ----------
    ticker           : e.g. ``"AAPL"``
    accession_number : dashed SEC form, e.g. ``"0000320193-23-000077"``
    """
    path = _RAW_FILINGS_DIR / ticker / f"{accession_number}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"Filing text not found: {path.relative_to(_ROOT).as_posix()}\n"
            f"  Create it with:  python data/edgar_downloader.py"
        )
    return path.read_text(encoding="utf-8", errors="replace")


def save_filings_index(df: pd.DataFrame) -> None:
    """Overwrite data/raw_filings/filings_index.parquet with *df*."""
    _FILINGS_IDX_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_FILINGS_IDX_PATH, index=False)


def save_filing_text(ticker: str, accession_number: str, text: str) -> Path:
    """
    Write *text* to data/raw_filings/{ticker}/{accession_number}.txt.

    Returns the path it was written to.
    """
    out_dir = _RAW_FILINGS_DIR / ticker
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{accession_number}.txt"
    path.write_text(text, encoding="utf-8", errors="replace")
    return path

# ---------------------------------------------------------------------------
# CLI — smoke-test every loader against whatever data exists
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _W = 62

    def _banner(label: str) -> None:
        print(f"\n{'─' * _W}")
        print(f" {label}")
        print("─" * _W)

    def _show(result: pd.DataFrame | str) -> None:
        if isinstance(result, pd.DataFrame):
            print(f"  shape  : {result.shape}")
            print(result.head(3).to_string(index=False))
        else:
            preview = result[:300].replace("\n", " ")
            print(f"  {len(result):,} chars")
            print(f"  preview: {preview!r}")

    def _check(label: str, fn) -> object:
        _banner(label)
        try:
            result = fn()
            _show(result)
            return result
        except FileNotFoundError as exc:
            print(f"  {exc}")
            return None

    _check("load_universe()",      load_universe)
    _check("load_prices()",        load_prices)
    idx = _check("load_filings_index()", load_filings_index)

    # Filing text: peek at the first row of the index when it exists
    _banner("load_filing_text(first row of index)")
    if idx is not None and not idx.empty:
        row = idx.iloc[0]
        try:
            _show(load_filing_text(row["ticker"], row["accession_number"]))
        except FileNotFoundError as exc:
            print(f"  {exc}")
    elif idx is not None:
        print("  Filings index is empty — no text to preview.")

    print(f"\n{'─' * _W}")
