"""
Feature-statistics extractor (no LLM).

Computes compact, cross-filing summary statistics about candidate text
features — Loughran-McDonald sentiment-category densities and a handful of
common phrase densities — over the FULL filing universe.  These statistics
are meant to be handed to the Signal Finder LLM as grounding context, so it
proposes hypotheses informed by real, measurable distributions rather than
guesswork.

Reads exclusively from the preprocessed filing-token cache
(agents/filing_tokens.parquet, via agents.signal_computation.load_filing_tokens)
and data/store.py — raw filing text is never re-read.
"""

from __future__ import annotations

import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.store import load_filings_index, load_universe
from agents.signal_computation import load_filing_tokens, load_lm_words, _MIN_TOKENS

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

_AGENTS_DIR        = Path(__file__).resolve().parent
FEATURE_STATS_PATH = _AGENTS_DIR / "feature_stats.json"

_LM_CATEGORIES = ["LM_negative", "LM_positive", "LM_uncertainty", "LM_litigious"]

# A small set of common candidate phrases plausibly relevant to factor
# research.  All are one or two words so their counts can be resolved purely
# from cached word/bigram frequencies (see _phrase_density).
_CANDIDATE_PHRASES = [
    "risk", "litigation", "supply chain", "going concern", "restructuring",
    "impairment", "competition", "regulation", "uncertainty", "decline", "growth",
]

_ROUND_NDIGITS = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-filing density helpers (cache-only — no raw text)
# ---------------------------------------------------------------------------

def _phrase_density(phrase: str, word_freq: dict, bigram_freq: dict, n_tokens: int) -> float:
    """Normalised count of *phrase* (1-2 words) using cached frequencies."""
    words = phrase.split()
    if len(words) == 1:
        count = word_freq.get(words[0], 0)
    else:
        count = bigram_freq.get(phrase, 0)
    return count / n_tokens if n_tokens else float("nan")


def _lm_density(cat_words: frozenset, word_freq: dict, n_tokens: int) -> float:
    """Normalised count of all words in an LM category, using cached word frequencies."""
    count = sum(word_freq.get(w, 0) for w in cat_words)
    return count / n_tokens if n_tokens else float("nan")


# ---------------------------------------------------------------------------
# Distribution summary
# ---------------------------------------------------------------------------

def _distribution_stats(values: np.ndarray) -> dict:
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return {k: None for k in ("n", "mean", "std", "min", "p25", "median", "p75", "max")}
    return {
        "n":      int(len(values)),
        "mean":   float(np.mean(values)),
        "std":    float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "min":    float(np.min(values)),
        "p25":    float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "p75":    float(np.percentile(values, 75)),
        "max":    float(np.max(values)),
    }


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_feature_stats(form_type: str = "10-K") -> dict:
    """
    Compute cross-filing summary statistics for LM-category and candidate-
    phrase densities (all normalised by per-filing word count), plus coverage
    info and a sector-level LM_uncertainty breakdown.

    Returns a JSON-serialisable dict; also written to agents/feature_stats.json
    by ``save_feature_stats`` / the CLI entry point.
    """
    filings      = load_filings_index(form_type=form_type)
    tokens_cache = load_filing_tokens()
    lm_words     = load_lm_words()

    # Join the index against the cache, keeping only filings with enough tokens
    records: list[dict] = []
    for _, filing in filings.iterrows():
        tokens = tokens_cache.get((filing["ticker"], filing["accession_number"]))
        if tokens is None or tokens["total_word_count"] < _MIN_TOKENS:
            continue
        records.append({
            "ticker":           filing["ticker"],
            "filing_date":      filing["filing_date"],
            "total_word_count": tokens["total_word_count"],
            "word_freq":        tokens["word_freq"],
            "bigram_freq":      tokens["bigram_freq"],
        })

    log.info("Computing feature statistics over %d / %d %s filings ...",
             len(records), len(filings), form_type)

    # --- 1. LM-category density distributions -----------------------------
    lm_stats = {}
    for cat in _LM_CATEGORIES:
        cat_words = lm_words.get(cat, frozenset())
        densities = np.array([
            _lm_density(cat_words, r["word_freq"], r["total_word_count"])
            for r in records
        ])
        lm_stats[cat] = _distribution_stats(densities)

    # --- 2. Candidate-phrase density distributions ------------------------
    phrase_stats = {}
    for phrase in _CANDIDATE_PHRASES:
        densities = np.array([
            _phrase_density(phrase, r["word_freq"], r["bigram_freq"], r["total_word_count"])
            for r in records
        ])
        phrase_stats[phrase] = _distribution_stats(densities)

    # --- 3. Coverage -------------------------------------------------------
    dates = pd.to_datetime([r["filing_date"] for r in records])
    coverage = {
        "form_type":  form_type,
        "n_filings":  len(records),
        "n_tickers":  len({r["ticker"] for r in records}),
        "date_min":   str(dates.min().date()) if len(dates) else None,
        "date_max":   str(dates.max().date()) if len(dates) else None,
    }

    # --- 4. Sector-level LM_uncertainty averages (optional, cheap) --------
    sector_stats: dict[str, dict] = {}
    try:
        universe       = load_universe()
        ticker_sectors = dict(zip(universe["ticker"], universe["sector"]))
        cat_words      = lm_words.get("LM_uncertainty", frozenset())

        by_sector: dict[str, list[float]] = {}
        for r in records:
            sector = ticker_sectors.get(r["ticker"])
            if sector is None:
                continue
            density = _lm_density(cat_words, r["word_freq"], r["total_word_count"])
            by_sector.setdefault(sector, []).append(density)

        sector_stats = {
            sector: {"n": len(vals), "mean": float(np.mean(vals))}
            for sector, vals in sorted(by_sector.items())
        }
    except FileNotFoundError as exc:
        log.warning("Universe not available — skipping sector breakdown (%s)", exc)

    return {
        "coverage":             coverage,
        "lm_categories":        lm_stats,
        "phrases":              phrase_stats,
        "sector_LM_uncertainty": sector_stats,
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _round_floats(obj, ndigits: int = _ROUND_NDIGITS):
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, ndigits) for v in obj]
    return obj


def save_feature_stats(stats: dict, path: Path = FEATURE_STATS_PATH) -> Path:
    """Write a compact, rounded JSON version of *stats* — small enough to drop into an LLM prompt."""
    path.write_text(json.dumps(_round_floats(stats), indent=2), encoding="utf-8")
    log.info("Feature stats written -> %s  (%.1f KB)", path.name, path.stat().st_size / 1024)
    return path


# ---------------------------------------------------------------------------
# Pretty-print summary
# ---------------------------------------------------------------------------

def _print_distribution_table(title: str, stats_by_key: dict[str, dict]) -> None:
    cols = ("mean", "std", "min", "p25", "median", "p75", "max")
    print(f"\n--- {title} (density = count / total_word_count) ---")
    print(f"  {'':<16}" + "".join(f"{c:>9}" for c in cols))
    for key, s in stats_by_key.items():
        if s["mean"] is None:
            print(f"  {key:<16}  (no data)")
            continue
        print(f"  {key:<16}" + "".join(f"{s[c]:>9.5f}" for c in cols))


def print_feature_stats_summary(stats: dict) -> None:
    W = 78
    cov = stats["coverage"]

    print(f"\n{'=' * W}")
    print("  FEATURE STATISTICS SUMMARY")
    print(f"{'=' * W}")
    print(f"\n  form_type   : {cov['form_type']}")
    print(f"  filings     : {cov['n_filings']}")
    print(f"  tickers     : {cov['n_tickers']}")
    print(f"  date range  : {cov['date_min']}  ->  {cov['date_max']}")

    _print_distribution_table("LM-category densities", stats["lm_categories"])
    _print_distribution_table("Candidate-phrase densities", stats["phrases"])

    sector_stats = stats["sector_LM_uncertainty"]
    if sector_stats:
        print(f"\n--- Sector-level LM_uncertainty density (mean) ---")
        for sector, s in sector_stats.items():
            print(f"  {sector:<30} mean={s['mean']:.5f}   (n={s['n']})")

    print(f"\n{'=' * W}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\nBuilding feature statistics over the full 10-K filing universe ...")
    stats = build_feature_stats(form_type="10-K")
    save_feature_stats(stats)
    print_feature_stats_summary(stats)
