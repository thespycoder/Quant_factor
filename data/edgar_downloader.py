"""
SEC EDGAR filing downloader.

Official APIs used
------------------
  /files/company_tickers.json          ticker → CIK mapping
  /submissions/CIK{cik}.json           filing metadata + overflow pages
  /Archives/edgar/data/{cik}/...       actual filing documents
"""

from __future__ import annotations

import sys
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
import pandas as pd

# Allow `from config.settings import …` whether the file is run as
# `python data/edgar_downloader.py` or imported from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import SEC_EDGAR_USER_AGENT
from config.universe import get_universe

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TICKERS: list[str] = ["AAPL", "MSFT", "XOM", "JPM", "WMT"]

SEC_BASE      = "https://data.sec.gov"
ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"
TICKERS_URL   = "https://www.sec.gov/files/company_tickers.json"

RAW_DIR    = Path(__file__).resolve().parent / "raw_filings"
INDEX_PATH = RAW_DIR / "filings_index.parquet"

# 0.13 s ≈ 7.7 req/s — comfortably under SEC's published 10 req/s cap
REQUEST_DELAY = 0.13

_INDEX_COLS = [
    "ticker", "cik", "form_type", "filing_date",
    "report_date", "accession_number", "primary_document_url",
]

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
# HTTP session (one persistent TCP connection, headers set once)
# ---------------------------------------------------------------------------

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": SEC_EDGAR_USER_AGENT})


def _get(url: str) -> requests.Response:
    """Rate-limited GET.  Raises HTTPError on non-2xx."""
    time.sleep(REQUEST_DELAY)
    resp = _SESSION.get(url, timeout=30)
    resp.raise_for_status()
    return resp


# ---------------------------------------------------------------------------
# Ticker → CIK resolution
# ---------------------------------------------------------------------------

def build_ticker_cik_map() -> dict[str, str]:
    """
    Returns {TICKER: zero-padded-10-digit-CIK} for every company
    in SEC's master ticker file (~10 000 entries, ~300 KB).
    """
    log.info("Fetching ticker→CIK map from SEC …")
    data = _get(TICKERS_URL).json()
    return {
        entry["ticker"].upper(): str(entry["cik_str"]).zfill(10)
        for entry in data.values()
    }


# ---------------------------------------------------------------------------
# Filing metadata
# ---------------------------------------------------------------------------

def _parse_page(
    page: dict,
    cik: str,
    form_types: list[str],
    start_dt: datetime,
    end_dt: datetime,
) -> list[dict]:
    """Extract matching filing rows from one SEC submissions page dict."""
    rows: list[dict] = []

    for acc, form, fd, rd, pdoc in zip(
        page.get("accessionNumber", []),
        page.get("form", []),
        page.get("filingDate", []),
        page.get("reportDate", []),
        page.get("primaryDocument", []),
    ):
        if form not in form_types:
            continue
        try:
            filing_dt = datetime.strptime(fd, "%Y-%m-%d")
        except ValueError:
            continue
        if not (start_dt <= filing_dt <= end_dt):
            continue

        acc_no_dash = acc.replace("-", "")
        # Archives URLs use the bare integer CIK (no leading zeros)
        doc_url = f"{ARCHIVES_BASE}/{int(cik)}/{acc_no_dash}/{pdoc}"

        rows.append(
            {
                "cik":                  cik,
                "form_type":            form,
                "filing_date":          fd,
                "report_date":          rd,
                "accession_number":     acc,
                "primary_document_url": doc_url,
            }
        )
    return rows


def get_filing_records(
    cik: str,
    form_types: list[str],
    start_year: int,
    end_year: int,
) -> list[dict]:
    """
    Fetch ALL filing metadata for a CIK, walking overflow pages when present
    (large filers like AAPL have 40+ years of filings spread across multiple
    JSON files).
    """
    start_dt = datetime(start_year, 1, 1)
    end_dt   = datetime(end_year, 12, 31)

    main_json = _get(f"{SEC_BASE}/submissions/CIK{cik}.json").json()
    filings   = main_json["filings"]

    # The main file contains 'recent'; overflow files are raw page dicts
    pages = [filings["recent"]]
    for overflow in filings.get("files", []):
        pages.append(_get(f"{SEC_BASE}/submissions/{overflow['name']}").json())

    records: list[dict] = []
    for page in pages:
        records.extend(_parse_page(page, cik, form_types, start_dt, end_dt))
    return records


# ---------------------------------------------------------------------------
# Document download
# ---------------------------------------------------------------------------

def download_filing(ticker: str, record: dict) -> Optional[Path]:
    """
    Save the filing's primary document to
    data/raw_filings/{ticker}/{accession_number}.txt.

    Returns the Path on success, None on failure.
    Already-downloaded files are skipped (cache check by path existence).
    """
    out_dir  = RAW_DIR / ticker
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{record['accession_number']}.txt"

    if out_path.exists():
        log.info("    CACHED  %s", out_path.name)
        return out_path

    log.info(
        "    GET    %-5s  %-5s  %s",
        record["form_type"], record["filing_date"], out_path.name,
    )
    try:
        resp = _get(record["primary_document_url"])
        out_path.write_text(resp.text, encoding="utf-8", errors="replace")
        return out_path
    except requests.HTTPError as exc:
        log.warning(
            "    HTTP %s — skipping %s",
            exc.response.status_code, out_path.name,
        )
    except Exception as exc:
        log.warning("    Download failed for %s: %s", out_path.name, exc)
    return None


# ---------------------------------------------------------------------------
# Metadata index (Parquet)
# ---------------------------------------------------------------------------

def load_index() -> pd.DataFrame:
    if INDEX_PATH.exists():
        return pd.read_parquet(INDEX_PATH)
    return pd.DataFrame(columns=_INDEX_COLS)


def save_index(df: pd.DataFrame) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(INDEX_PATH, index=False)
    log.info("Index saved → %s  (%d rows)", INDEX_PATH.name, len(df))


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run(
    tickers: list[str] | None    = None,
    form_types: list[str] | None = None,
    start_year: int              = 2015,
    end_year: int                = 2024,
) -> pd.DataFrame:
    """
    Download all matching filings for every ticker and return the
    complete (updated) metadata index as a DataFrame.
    """
    if tickers is None:
        tickers = get_universe()
    if form_types is None:
        form_types = ["10-K"]

    ticker_cik       = build_ticker_cik_map()
    index_df         = load_index()
    known_accessions = set(index_df["accession_number"].tolist())
    new_rows: list[dict] = []

    for ticker in tickers:
        ticker = ticker.upper()
        cik = ticker_cik.get(ticker)
        if not cik:
            log.warning("No CIK found for %s — skipping", ticker)
            continue

        log.info("── %s  (CIK %s)", ticker, cik)
        try:
            records = get_filing_records(cik, form_types, start_year, end_year)
        except Exception as exc:
            log.error("Metadata fetch failed for %s: %s", ticker, exc)
            continue

        log.info("  %d filing(s) matched in %d–%d", len(records), start_year, end_year)

        for rec in records:
            download_filing(ticker, rec)
            if rec["accession_number"] not in known_accessions:
                new_rows.append({"ticker": ticker, **rec})
                known_accessions.add(rec["accession_number"])

    if new_rows:
        new_df   = pd.DataFrame(new_rows)[_INDEX_COLS]
        index_df = pd.concat([index_df, new_df], ignore_index=True)

    save_index(index_df)
    return index_df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = run(form_types=["10-K"], start_year=2015, end_year=2024)
    divider = "─" * 62
    print(f"\n{divider}")
    print(f"Done.  {len(df)} total filing(s) in index.\n")
    if not df.empty:
        print(
            df[["ticker", "form_type", "filing_date", "report_date"]]
            .sort_values(["ticker", "filing_date"])
            .to_string(index=False)
        )
    print(divider)
