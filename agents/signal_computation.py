"""
Signal computation layer.

Translates a FactorHypothesis into per-filing numeric signal values using:
  * Loughran-McDonald sentiment word lists (negative, positive, uncertainty,
    litigious) — loaded from a local CSV and cached to agents/lm_words.parquet.
  * Simple phrase-counting for custom term specs.

The output DataFrame [filing_date, ticker, signal_value] is directly compatible
with backtest.ic_engine.compute_ic and backtest.portfolio_engine.run_backtest.
"""

from __future__ import annotations

import re
import sys
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agents.hypothesis_schema import FactorHypothesis, SignalSpec, TermSpec
from data.store import load_filings_index, load_filing_text

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_AGENTS_DIR     = Path(__file__).resolve().parent
LM_CSV_PATH     = _AGENTS_DIR / "lm_dictionary.csv"
LM_CACHE_PATH   = _AGENTS_DIR / "lm_words.parquet"
TOKENS_CACHE_PATH = _AGENTS_DIR / "filing_tokens.parquet"

# Possible download URLs for the LM Master Dictionary (try newest first)
_LM_DOWNLOAD_URLS = [
    "https://sraf.nd.edu/loughranmcdonald-master-dictionary/"
    "Loughran-McDonald_MasterDictionary_1993-2024.csv",
    "https://sraf.nd.edu/loughranmcdonald-master-dictionary/"
    "Loughran-McDonald_MasterDictionary_1993-2023.csv",
]

# LM CSV column → our category name
_LM_COL_MAP = {
    "negative":    "LM_negative",
    "positive":    "LM_positive",
    "uncertainty": "LM_uncertainty",
    "litigious":   "LM_litigious",
}

# Regex: only alphabetic tokens (no numbers, no punctuation)
_TOKEN_RE = re.compile(r"\b[a-z]+\b")

# Minimum token count to attempt signal computation
_MIN_TOKENS = 50

# Bigrams occurring fewer than this many times in a filing are pruned from the
# filing-token cache (see preprocess_filings.py) to bound its size — a 200K-word
# 10-K can contain 100K+ distinct bigrams, the large majority occurring exactly
# once, and a singleton bigram contributes a negligible ~5e-6 to any
# length-normalised phrase signal.  word_freq is NEVER pruned.
_MIN_BIGRAM_COUNT = 2


def _prune_bigram_counts(counts: dict[str, int]) -> dict[str, int]:
    """Drop bigrams occurring fewer than _MIN_BIGRAM_COUNT times."""
    return {bg: c for bg, c in counts.items() if c >= _MIN_BIGRAM_COUNT}

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
# In-process cache (avoid re-parsing on each call within a run)
# ---------------------------------------------------------------------------

_lm_words_cache: dict[str, frozenset[str]] | None = None
_filing_tokens_cache: dict[tuple[str, str], dict] | None = None

# ---------------------------------------------------------------------------
# LM dictionary loading
# ---------------------------------------------------------------------------

def _try_download_lm() -> bool:
    """
    Attempt to download the LM Master Dictionary CSV from Notre Dame.
    Returns True on success, False if all URLs fail.
    """
    for url in _LM_DOWNLOAD_URLS:
        try:
            log.info("Attempting LM dictionary download: %s ...", url.rsplit("/", 1)[-1])
            resp = requests.get(
                url,
                headers={"User-Agent": "quant-factor-research/1.0 (academic)"},
                timeout=30,
            )
            resp.raise_for_status()
            LM_CSV_PATH.write_bytes(resp.content)
            log.info("LM dictionary saved to %s", LM_CSV_PATH)
            return True
        except Exception as exc:
            log.warning("  Download failed (%s): %s", url.rsplit("/", 1)[-1], exc)
    return False


def _parse_lm_csv(path: Path) -> dict[str, set[str]]:
    """
    Parse the Loughran-McDonald Master Dictionary CSV.

    The CSV has a 'Word' column (uppercase) and numeric category columns
    (Negative, Positive, Uncertainty, Litigious) where non-zero means the
    word belongs to that category.  Returns {category: set_of_lowercase_words}.
    """
    df = pd.read_csv(path, encoding="latin-1", low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]

    word_col = next((c for c in df.columns if c == "word"), None)
    if word_col is None:
        raise RuntimeError(
            f"No 'Word' column found in {path.name}. "
            f"Columns: {df.columns.tolist()}"
        )

    result: dict[str, set[str]] = {}
    for csv_col, cat_name in _LM_COL_MAP.items():
        if csv_col not in df.columns:
            log.warning("Column '%s' not found in %s — %s will be empty.",
                        csv_col, path.name, cat_name)
            result[cat_name] = set()
            continue
        mask  = pd.to_numeric(df[csv_col], errors="coerce").fillna(0) != 0
        words = df.loc[mask, word_col].dropna().str.lower().tolist()
        result[cat_name] = set(words)
        log.info("  %-20s  %d words", cat_name, len(words))

    return result


def load_lm_words(force_refresh: bool = False) -> dict[str, frozenset[str]]:
    """
    Return the Loughran-McDonald word sets.
    Loads from cache (lm_words.parquet) when available; otherwise parses
    lm_dictionary.csv and writes the cache.

    If lm_dictionary.csv is missing, a download is attempted automatically.
    If the download also fails, a FileNotFoundError is raised with clear
    instructions on where to obtain the file manually.
    """
    global _lm_words_cache
    if _lm_words_cache is not None and not force_refresh:
        return _lm_words_cache

    # Fast path: parquet cache exists
    if not force_refresh and LM_CACHE_PATH.exists():
        log.info("Loading LM word sets from cache (%s) ...", LM_CACHE_PATH.name)
        df  = pd.read_parquet(LM_CACHE_PATH)
        out = {
            cat: frozenset(df.loc[df["category"] == cat, "word"].tolist())
            for cat in _LM_COL_MAP.values()
        }
        _lm_words_cache = out
        for cat, words in out.items():
            log.info("  %-20s  %d words", cat, len(words))
        return out

    # Need the CSV; try to download it if missing
    if not LM_CSV_PATH.exists():
        log.info("LM dictionary CSV not found — attempting automatic download ...")
        ok = _try_download_lm()
        if not ok:
            raise FileNotFoundError(
                f"\nLM Master Dictionary not found at:\n"
                f"  {LM_CSV_PATH}\n\n"
                "Automatic download failed.  Please download it manually:\n"
                "  1. Visit:  https://sraf.nd.edu/loughranmcdonald-master-dictionary/\n"
                "  2. Download the most recent 'Master Dictionary' CSV file.\n"
                "  3. Save it as:  agents/lm_dictionary.csv\n"
                "  4. Re-run this script.\n"
            )

    log.info("Parsing LM dictionary from %s ...", LM_CSV_PATH.name)
    word_sets = _parse_lm_csv(LM_CSV_PATH)

    # Write parquet cache: long-format (word, category) pairs
    rows = [
        {"word": w, "category": cat}
        for cat, words in word_sets.items()
        for w in words
    ]
    cache_df = pd.DataFrame(rows)
    LM_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    cache_df.to_parquet(LM_CACHE_PATH, index=False)
    log.info("LM word cache written → %s  (%d rows)", LM_CACHE_PATH.name, len(cache_df))

    result = {cat: frozenset(words) for cat, words in word_sets.items()}
    _lm_words_cache = result
    return result


# ---------------------------------------------------------------------------
# Filing-token cache loading (built by agents/preprocess_filings.py)
# ---------------------------------------------------------------------------

def load_filing_tokens(force_refresh: bool = False) -> dict[tuple[str, str], dict]:
    """
    Load the preprocessed per-filing token cache.

    Returns ``{(ticker, accession_number): record}`` where each record is
    ``{"total_word_count": int, "word_freq": dict[str,int], "bigram_freq": dict[str,int]}``.

    Raises FileNotFoundError with build instructions if the cache is missing.
    """
    global _filing_tokens_cache
    if _filing_tokens_cache is not None and not force_refresh:
        return _filing_tokens_cache

    if not TOKENS_CACHE_PATH.exists():
        raise FileNotFoundError(
            f"\nFiling token cache not found at:\n  {TOKENS_CACHE_PATH}\n\n"
            "Build it once with:\n  python agents/preprocess_filings.py\n"
        )

    # The cache is a directory of batched part-files (see preprocess_filings.py);
    # pd.read_parquet transparently reads and concatenates all of them.
    log.info("Loading filing-token cache from %s ...", TOKENS_CACHE_PATH.name)
    df = pd.read_parquet(TOKENS_CACHE_PATH)
    cache = {
        (row.ticker, row.accession_number): {
            "total_word_count": int(row.total_word_count),
            "word_freq":        dict(row.word_freq),
            "bigram_freq":      dict(row.bigram_freq),
        }
        for row in df.itertuples(index=False)
    }
    _filing_tokens_cache = cache
    log.info("  %d filings cached", len(cache))
    return cache


# ---------------------------------------------------------------------------
# Per-filing signal computation
# ---------------------------------------------------------------------------

def _phrase_count_from_cache(
    phrase:           str,
    word_freq:        dict[str, int],
    bigram_freq:      dict[str, int],
    ticker:           str | None = None,
    accession_number: str | None = None,
) -> float:
    """
    Count occurrences of *phrase* (already lowercased) using cached
    token/bigram frequencies.

      * single word   → exact lookup in the word-frequency dict.
      * two words     → exact lookup in the adjacent-bigram dict
                        (covers phrases like "going concern").
      * three+ words  → can't be resolved from unigram/bigram frequencies;
                        fall back to a direct substring scan of the raw
                        filing text (rare — most phrase hypotheses are
                        one or two words).
    """
    words = phrase.split()
    if len(words) == 1:
        return float(word_freq.get(words[0], 0))
    if len(words) == 2:
        return float(bigram_freq.get(phrase, 0))

    if ticker is None or accession_number is None:
        return 0.0
    text_lower = load_filing_text(ticker, accession_number).lower()
    return float(text_lower.count(phrase))


def compute_signal_for_filing(
    hypothesis:       FactorHypothesis,
    tokens:           dict,
    lm_words:         dict[str, frozenset[str]] | None = None,
    *,
    ticker:           str | None = None,
    accession_number: str | None = None,
) -> float:
    """
    Compute the signal value for a single filing from its cached token
    representation (see ``load_filing_tokens`` / ``preprocess_filings.py``),
    avoiding any re-read or re-tokenisation of the raw filing text.

    Parameters
    ----------
    tokens : the cached record for this filing —
        ``{"total_word_count": int, "word_freq": dict, "bigram_freq": dict}``.
    ticker, accession_number : only used as a fallback for phrase terms with
        three or more words, which cannot be resolved from unigram/bigram
        frequencies alone (rare).

    Steps
    -----
    1. For each term in the spec:
         phrase      → look up the (bi)gram count in the cached frequencies.
         lm_category → sum cached word frequencies over the LM word set.
    2. Combine counts per the spec's 'combine' rule.
    3. Optionally normalise by total token count (normalize_by_length).
    4. Multiply by hypothesis.direction so that high signal_value always means
       "expected positive return" (consistent with IC and backtest conventions).

    Returns NaN for filings with fewer than 50 cached tokens.
    """
    n_tokens = tokens["total_word_count"]
    if n_tokens < _MIN_TOKENS:
        return float("nan")

    word_freq   = tokens["word_freq"]
    bigram_freq = tokens["bigram_freq"]

    # Lazy-load LM words only when actually needed
    if lm_words is None:
        needs_lm = any(
            t.type == "lm_category"
            for t in hypothesis.signal_computation.terms
        )
        if needs_lm:
            lm_words = load_lm_words()
        else:
            lm_words = {}

    spec         = hypothesis.signal_computation
    term_counts: list[float] = []

    for term in spec.terms:
        if term.type == "phrase":
            count = _phrase_count_from_cache(
                term.value.lower(), word_freq, bigram_freq, ticker, accession_number,
            )
        else:  # lm_category
            cat_words = lm_words.get(term.value, frozenset())
            count = float(sum(word_freq.get(w, 0) for w in cat_words))
        term_counts.append(count)

    if not term_counts:
        return float("nan")

    # Combine
    if spec.combine == "sum":
        raw = sum(term_counts)
    elif spec.combine == "mean":
        raw = sum(term_counts) / len(term_counts)
    else:  # ratio — schema already validated len == 2
        if term_counts[1] == 0.0:
            return float("nan")      # undefined ratio
        raw = term_counts[0] / term_counts[1]

    if spec.normalize_by_length and n_tokens > 0:
        raw /= n_tokens

    # Apply direction so downstream IC and backtest always see "high = bullish"
    return float(raw * hypothesis.direction)


# ---------------------------------------------------------------------------
# Batch function
# ---------------------------------------------------------------------------

def compute_signal_df(
    hypothesis:   FactorHypothesis,
    form_type:    str       = "10-K",
    max_filings:  int | None = None,
) -> pd.DataFrame:
    """
    Compute the signal value for every filing of the given form type.

    Uses the preprocessed token cache (agents/filing_tokens.parquet, built by
    agents/preprocess_filings.py) instead of re-reading and re-tokenising raw
    filing text — this is what makes repeated signal computation fast.
    Filings missing from the cache are silently skipped (logged).

    Parameters
    ----------
    hypothesis  : a validated FactorHypothesis.
    form_type   : SEC form type to restrict to (default "10-K").
    max_filings : cap on number of filings processed (for testing).

    Returns
    -------
    DataFrame with columns [filing_date, ticker, signal_value], sorted by
    (ticker, filing_date).  Shape matches what compute_ic and run_backtest expect.
    """
    filings = load_filings_index(form_type=form_type)
    if max_filings is not None:
        filings = filings.head(max_filings)

    total     = len(filings)
    log.info(
        "Computing signal '%s' over %d %s filings (cached path) ...",
        hypothesis.signal_name, total, form_type,
    )

    # Pre-load LM word sets and the token cache once (expensive) before the loop
    needs_lm = any(t.type == "lm_category" for t in hypothesis.signal_computation.terms)
    lm_words    = load_lm_words() if needs_lm else {}
    tokens_by_filing = load_filing_tokens()

    rows: list[dict] = []
    n_missing = n_short = n_ok = 0

    for _, filing in filings.iterrows():
        ticker, accession = filing["ticker"], filing["accession_number"]
        tokens = tokens_by_filing.get((ticker, accession))
        if tokens is None:
            n_missing += 1
            continue

        val = compute_signal_for_filing(
            hypothesis, tokens, lm_words, ticker=ticker, accession_number=accession,
        )
        if np.isnan(val):
            n_short += 1
            continue

        rows.append({
            "filing_date":  filing["filing_date"],
            "ticker":       ticker,
            "signal_value": val,
        })
        n_ok += 1

    log.info(
        "Done: %d valid  |  %d missing from cache  |  %d too short / error  "
        "(total attempted: %d)",
        n_ok, n_missing, n_short, total,
    )

    if not rows:
        return pd.DataFrame(columns=["filing_date", "ticker", "signal_value"])

    df = pd.DataFrame(rows)
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    return df.sort_values(["ticker", "filing_date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Cache-correctness check: cached lookup vs. fresh tokenisation of raw text
# ---------------------------------------------------------------------------

def _fresh_tokens_from_text(text: str) -> dict:
    """
    Tokenise raw text the slow way — used only to verify the cache.

    Applies the SAME bigram pruning as preprocess_filings._build_record so
    this is an apples-to-apples comparison against the cached representation
    (not against an unpruned ground truth).
    """
    toks = _TOKEN_RE.findall(text.lower())
    bigram_counts = Counter(f"{a} {b}" for a, b in zip(toks, toks[1:]))
    return {
        "total_word_count": len(toks),
        "word_freq":        dict(Counter(toks)),
        "bigram_freq":      _prune_bigram_counts(dict(bigram_counts)),
    }


def _verify_cache_matches_raw_text(
    hypothesis: FactorHypothesis,
    filings:    pd.DataFrame,
    lm_words:   dict[str, frozenset[str]],
    n_sample:   int = 5,
) -> int:
    """
    For a sample of filings, confirm that compute_signal_for_filing gives the
    SAME value whether fed the cached token record or a record freshly built
    by re-reading and re-tokenising the raw filing text.  Returns the number
    of mismatches found (0 == cache is faithful).
    """
    tokens_cache = load_filing_tokens()
    n_mismatch   = 0

    for _, filing in filings.head(n_sample).iterrows():
        ticker, accession = filing["ticker"], filing["accession_number"]
        cached = tokens_cache.get((ticker, accession))
        if cached is None:
            continue

        try:
            text = load_filing_text(ticker, accession)
        except FileNotFoundError:
            continue
        fresh = _fresh_tokens_from_text(text)

        v_cached = compute_signal_for_filing(hypothesis, cached, lm_words, ticker=ticker, accession_number=accession)
        v_fresh  = compute_signal_for_filing(hypothesis, fresh,  lm_words, ticker=ticker, accession_number=accession)

        same = (np.isnan(v_cached) and np.isnan(v_fresh)) or np.isclose(v_cached, v_fresh, equal_nan=True)
        if not same:
            n_mismatch += 1
        flag = "OK " if same else "MISMATCH"
        print(f"    [{flag}]  {ticker:<6} {accession:<22}  cached={v_cached:.8f}  fresh={v_fresh:.8f}")

    return n_mismatch


# ---------------------------------------------------------------------------
# CLI — self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    W = 62
    N_TEST_FILINGS = 30

    print(f"\n{'=' * W}")
    print("  SIGNAL COMPUTATION — SELF-TEST (cached path)")
    print(f"{'=' * W}")

    # --- 1. Load / check LM dictionary ---
    print("\n--- LM Dictionary ---")
    lm = load_lm_words()
    for cat, words in lm.items():
        print(f"  {cat:<22} {len(words):>5,} words")

    # --- 2. Ensure the filing-token cache covers the test filings ---
    print(f"\n--- Filing-token cache (first {N_TEST_FILINGS} filings) ---")
    try:
        load_filing_tokens()
    except FileNotFoundError:
        print("  Cache missing — building it now via preprocess_filings ...")
        from agents.preprocess_filings import preprocess_filings
        preprocess_filings(form_type="10-K", max_filings=N_TEST_FILINGS)
        load_filing_tokens(force_refresh=True)

    test_filings = load_filings_index(form_type="10-K").head(N_TEST_FILINGS)

    # --- 3. Build the two test hypotheses ---
    risk_hyp = FactorHypothesis(
        signal_name        = "risk_phrase",
        signal_description = "Frequency of the word 'risk' in the filing, length-normalised.",
        signal_computation = SignalSpec(
            terms                = [TermSpec(type="phrase", value="risk")],
            combine              = "sum",
            normalize_by_length  = True,
        ),
        direction          = -1,
        horizon            = "fwd_ret_21d",
        economic_rationale = "More 'risk' mentions signal heightened management concern, "
                             "predicting negative near-term returns.",
    )

    uncertainty_hyp = FactorHypothesis(
        signal_name        = "uncertainty_tone",
        signal_description = "Fraction of LM uncertainty-category words in the filing.",
        signal_computation = SignalSpec(
            terms                = [TermSpec(type="lm_category", value="LM_uncertainty")],
            combine              = "sum",
            normalize_by_length  = True,
        ),
        direction          = -1,
        horizon            = "fwd_ret_21d",
        economic_rationale = "More uncertainty language signals management concern about "
                             "future performance, predicting negative near-term returns.",
    )

    pd.set_option("display.float_format", "{:.6f}".format)

    for hyp in (risk_hyp, uncertainty_hyp):
        print(f"\n--- Hypothesis: {hyp.signal_name}  "
              f"(terms={[t.model_dump() for t in hyp.signal_computation.terms]}) ---")
        signal_df = compute_signal_df(hyp, form_type="10-K", max_filings=N_TEST_FILINGS)

        n_total   = len(signal_df)
        n_nonzero = int((signal_df["signal_value"] != 0).sum()) if n_total else 0
        print(f"\n  valid values : {n_total}")
        print(f"  nonzero      : {n_nonzero}")
        if n_total:
            print(f"  mean         : {signal_df['signal_value'].mean():.6f}")
            print(f"  min / max    : {signal_df['signal_value'].min():.6f} / "
                  f"{signal_df['signal_value'].max():.6f}")
            print(signal_df.head(5).to_string(index=False))

    # --- 4. Correctness check: cached lookup vs. fresh raw-text tokenisation ---
    print(f"\n--- Correctness check: cache vs. fresh tokenisation (sample) ---")
    for hyp in (risk_hyp, uncertainty_hyp):
        print(f"\n  {hyp.signal_name}:")
        n_bad = _verify_cache_matches_raw_text(hyp, test_filings, lm, n_sample=5)
        verdict = "ALL MATCH" if n_bad == 0 else f"{n_bad} MISMATCH(ES)"
        print(f"    -> {verdict}")

    print(f"\n{'=' * W}")
    print("  signal_computation.py — OK")
    print(f"{'=' * W}")
