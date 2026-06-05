"""
Universe definition for the quantitative factor engine.

Builds a stable ~150-ticker universe sampled proportionally from the
S&P 500 across GICS sectors.  Saved to config/universe.parquet so the
constituent list never changes between runs (unless force_refresh=True),
which keeps backtest results reproducible.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

UNIVERSE_PATH = Path(__file__).resolve().parent / "universe.parquet"
WIKI_URL      = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
TARGET_SIZE   = 150
_SEED         = 42    # fixed → same tickers every time the universe is rebuilt

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
# Wikipedia fetch
# ---------------------------------------------------------------------------

def _fetch_sp500() -> pd.DataFrame:
    """
    Download and parse the Wikipedia S&P 500 constituents table.
    Returns a clean DataFrame: wiki_symbol, company_name, sector.
    """
    log.info("Fetching S&P 500 constituent list from Wikipedia …")
    try:
        resp = requests.get(
            WIKI_URL,
            headers={"User-Agent": "Mozilla/5.0 (quant-factor-research-engine)"},
            timeout=20,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Failed to reach Wikipedia: {exc}\n"
            "Check your internet connection, or pass force_refresh=False to "
            "reuse the cached universe."
        ) from exc

    # Prefer the table with id="constituents"; fall back to the first table
    # in case Wikipedia's HTML ever changes the attribute.
    try:
        tables = pd.read_html(resp.text, attrs={"id": "constituents"})
    except ValueError:
        log.warning("  Table id='constituents' not found — trying first table")
        tables = pd.read_html(resp.text)

    if not tables:
        raise RuntimeError("No HTML tables found on the Wikipedia S&P 500 page.")

    raw = tables[0].copy()
    raw.columns = [str(c).strip() for c in raw.columns]
    log.info("  Raw table: %d rows, columns: %s", len(raw), list(raw.columns))

    # Fuzzy column resolution — Wikipedia occasionally tweaks header text
    def _find_col(keywords: list[str]) -> str | None:
        for kw in keywords:
            for col in raw.columns:
                if kw.lower() in col.lower():
                    return col
        return None

    symbol_col = _find_col(["symbol", "ticker"])
    name_col   = _find_col(["security", "company", "name"])
    sector_col = _find_col(["gics sector", "sector"])

    missing = [
        label
        for label, col in [("Symbol", symbol_col), ("Security", name_col), ("Sector", sector_col)]
        if col is None
    ]
    if missing:
        raise RuntimeError(
            f"Could not locate columns {missing} in the Wikipedia table.\n"
            f"Available columns: {list(raw.columns)}"
        )

    df = pd.DataFrame(
        {
            "wiki_symbol":  raw[symbol_col].astype(str).str.strip(),
            "company_name": raw[name_col].astype(str).str.strip(),
            "sector":       raw[sector_col].astype(str).str.strip(),
        }
    ).dropna(subset=["wiki_symbol", "sector"])

    # Drop placeholder / malformed rows
    df = df[df["wiki_symbol"].str.match(r"^[A-Z]")]

    log.info(
        "  Parsed %d companies across %d GICS sectors",
        len(df), df["sector"].nunique(),
    )
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Proportional sampling — Largest Remainder Method
# ---------------------------------------------------------------------------

def _proportional_sample(df: pd.DataFrame, target: int) -> pd.DataFrame:
    """
    Sample exactly `target` rows from df, preserving each sector's share.

    Algorithm (Largest Remainder Method):
      1. Compute each sector's exact float quota = sector_size / total * target.
      2. Floor each quota; guarantee a minimum of 1 per sector.
      3. Distribute the remaining integer slots to the sectors with the
         largest fractional parts until the total reaches exactly `target`.
    """
    sector_sizes = df.groupby("sector").size()
    total        = len(df)

    exact  = sector_sizes * target / total
    floors = exact.astype(int).clip(lower=1)

    remainder = target - int(floors.sum())

    if remainder > 0:
        # Give extra slots to sectors with the largest fractional remainders
        frac_order = (exact - floors).sort_values(ascending=False)
        for sector in frac_order.index[:remainder]:
            floors[sector] += 1
    elif remainder < 0:
        # Clipping to 1 pushed the total over target — trim smallest-excess sectors
        trimable = floors[floors > 1].sort_values()
        for sector in trimable.index[: abs(remainder)]:
            floors[sector] -= 1

    frames = []
    for sector, n in floors.items():
        subset = df[df["sector"] == sector]
        frames.append(subset.sample(n=min(n, len(subset)), random_state=_SEED))

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def _build() -> pd.DataFrame:
    sp500    = _fetch_sp500()
    universe = _proportional_sample(sp500, TARGET_SIZE)

    # yfinance expects "-" where SEC/Wikipedia uses "." (e.g. BRK.B → BRK-B)
    universe["ticker"] = universe["wiki_symbol"].str.replace(".", "-", regex=False)

    universe = (
        universe[["ticker", "wiki_symbol", "company_name", "sector"]]
        .sort_values("ticker")
        .reset_index(drop=True)
    )

    breakdown = universe.groupby("sector").size().sort_values(ascending=False)
    log.info("Universe built — %d tickers across %d sectors:", len(universe), len(breakdown))
    for sector, count in breakdown.items():
        log.info("  %-42s %3d", sector, count)

    return universe


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_universe_df(force_refresh: bool = False) -> pd.DataFrame:
    """
    Return the universe as a DataFrame with columns:
        ticker        — yfinance-compatible (dots replaced with dashes)
        wiki_symbol   — original Wikipedia/SEC symbol
        company_name  — company name from Wikipedia
        sector        — GICS sector

    Loads from config/universe.parquet unless force_refresh=True.
    The cached file is the canonical source — never changes between runs.
    """
    if not force_refresh and UNIVERSE_PATH.exists():
        log.info("Loading universe from cache (%s)", UNIVERSE_PATH.name)
        return pd.read_parquet(UNIVERSE_PATH)

    df = _build()
    df.to_parquet(UNIVERSE_PATH, index=False)
    log.info("Universe cached → %s", UNIVERSE_PATH)
    return df


def get_universe(force_refresh: bool = False) -> list[str]:
    """Return the list of yfinance-compatible tickers."""
    return get_universe_df(force_refresh)["ticker"].tolist()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = get_universe_df(force_refresh=True)

    breakdown = df.groupby("sector").size().sort_values(ascending=False)
    longest   = max(len(s) for s in breakdown.index)
    divider   = "─" * (longest + 20)

    print(f"\n{divider}")
    print(f"Total tickers : {len(df)}")
    print(f"Sectors       : {len(breakdown)}")
    print(f"\n{'Sector':<{longest}}   n   chart")
    print(divider)
    for sector, count in breakdown.items():
        bar = "█" * count
        print(f"{sector:<{longest}}  {count:>3}  {bar}")
    print(divider)
