"""
Filing preprocessing cache.

Tokenising every 10-K filing on every signal run is the dominant cost of
``compute_signal_df`` (~20 minutes over 1,435 filings).  This script reads
each filing's text exactly once, tokenises it, and stores a compact
per-filing representation — total word count, a word-frequency dict, and a
(pruned) bigram-frequency dict (for fast multi-word phrase lookups like
"going concern") — to ``agents/filing_tokens.parquet/``, keyed by
(ticker, accession_number).

Storage scheme
--------------
``agents/filing_tokens.parquet`` is a DIRECTORY of part-files
(``part-00000.parquet``, ``part-00001.parquet``, ...), each holding one batch
of ``_BATCH_SIZE`` filings.  This keeps every write bounded to one small
in-memory pa.Table — building one giant table for all 1,435 filings at once
exhausts memory because bigram dictionaries are huge (a 200K-word 10-K can
have 100K+ distinct bigrams).  Re-running only tokenises filings that are not
yet present in any part-file and simply appends new part-files — existing
parts are never re-read or rewritten, so resuming is cheap and safe.
"""

from __future__ import annotations

import sys
import logging
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.store import load_filings_index, load_filing_text
from agents.signal_computation import _TOKEN_RE, _prune_bigram_counts

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

_AGENTS_DIR       = Path(__file__).resolve().parent
TOKENS_CACHE_PATH = _AGENTS_DIR / "filing_tokens.parquet"   # a directory of part-files

# Filings per part-file / per in-memory pa.Table.  Small enough that one
# batch's word_freq + bigram_freq dicts comfortably fit in memory; large
# enough to keep the part-file count (and read overhead) manageable.
_BATCH_SIZE = 150

# word -> count and bigram -> count are both stored as parquet MAP columns
# (string -> int64): compact, and round-trip to plain python dicts.
_FREQ_MAP_TYPE = pa.map_(pa.string(), pa.int64())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lowercase, alphabetic word tokens — identical to the signal layer's tokeniser."""
    return _TOKEN_RE.findall(text.lower())


def _bigram_counts(tokens: list[str]) -> Counter:
    """Counts of adjacent-token pairs, joined by a single space (e.g. 'going concern')."""
    return Counter(f"{a} {b}" for a, b in zip(tokens, tokens[1:]))


def _build_record(ticker: str, accession_number: str, text: str) -> dict:
    tokens = _tokenize(text)
    return {
        "ticker":           ticker,
        "accession_number": accession_number,
        "total_word_count": len(tokens),
        "word_freq":        dict(Counter(tokens)),
        # Pruned: see signal_computation._MIN_BIGRAM_COUNT for the rationale.
        "bigram_freq":      _prune_bigram_counts(dict(_bigram_counts(tokens))),
    }


def _records_to_table(records: list[dict]) -> pa.Table:
    return pa.table({
        "ticker":           pa.array([r["ticker"] for r in records],           type=pa.string()),
        "accession_number": pa.array([r["accession_number"] for r in records], type=pa.string()),
        "total_word_count": pa.array([r["total_word_count"] for r in records], type=pa.int64()),
        "word_freq":        pa.array([r["word_freq"] for r in records],        type=_FREQ_MAP_TYPE),
        "bigram_freq":      pa.array([r["bigram_freq"] for r in records],      type=_FREQ_MAP_TYPE),
    })


# ---------------------------------------------------------------------------
# Part-file bookkeeping (append-only — existing parts are never rewritten)
# ---------------------------------------------------------------------------

def _part_files() -> list[Path]:
    if not TOKENS_CACHE_PATH.exists():
        return []
    return sorted(TOKENS_CACHE_PATH.glob("part-*.parquet"))


def _next_part_index() -> int:
    parts = _part_files()
    if not parts:
        return 0
    return max(int(p.stem.split("-")[1]) for p in parts) + 1


def _cached_keys() -> set[tuple[str, str]]:
    if not _part_files():
        return set()
    df = pd.read_parquet(TOKENS_CACHE_PATH, columns=["ticker", "accession_number"])
    return set(zip(df["ticker"], df["accession_number"]))


def _flush_batch(records: list[dict], part_index: int) -> Path:
    TOKENS_CACHE_PATH.mkdir(parents=True, exist_ok=True)
    path = TOKENS_CACHE_PATH / f"part-{part_index:05d}.parquet"
    pq.write_table(_records_to_table(records), path)
    return path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def preprocess_filings(form_type: str = "10-K", max_filings: int | None = None) -> None:
    """
    Tokenise every *form_type* filing exactly once and append the result to
    ``agents/filing_tokens.parquet/`` in batches of ``_BATCH_SIZE``.

    Filings already present in any part-file are skipped — only newly
    downloaded filings get (re)processed, and existing part-files are left
    untouched (append-only), so resuming after an interruption is cheap.
    """
    filings = load_filings_index(form_type=form_type)
    if max_filings is not None:
        filings = filings.head(max_filings)

    cached = _cached_keys()
    total  = len(filings)
    log.info("Preprocessing %d %s filings  (%d already cached) ...",
             total, form_type, len(cached))

    part_index   = _next_part_index()
    batch:        list[dict] = []
    n_new = n_skip = n_missing = 0

    def _flush():
        nonlocal part_index
        if not batch:
            return
        path = _flush_batch(batch, part_index)
        log.info("  flushed %s  (%d filings)", path.name, len(batch))
        part_index += 1
        batch.clear()

    for i, (_, filing) in enumerate(filings.iterrows(), start=1):
        key = (filing["ticker"], filing["accession_number"])
        if key in cached:
            n_skip += 1
            continue
        try:
            text = load_filing_text(*key)
        except FileNotFoundError:
            n_missing += 1
            continue

        batch.append(_build_record(key[0], key[1], text))
        n_new += 1

        if len(batch) >= _BATCH_SIZE:
            _flush()

        if i % 50 == 0 or i == total:
            log.info("  %4d / %4d   new=%-4d  cached=%-4d  missing=%-4d",
                     i, total, n_new, n_skip, n_missing)

    _flush()

    if n_new == 0:
        log.info("Nothing new to process — cache already covers all %d filings.", n_skip)
    else:
        log.info("Done: %d newly tokenised, %d already cached, %d missing text  "
                 "-> %s (%d part-files)",
                 n_new, n_skip, n_missing, TOKENS_CACHE_PATH.name, len(_part_files()))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    W = 62
    print(f"\n{'=' * W}")
    print("  FILING PREPROCESSOR — tokenise & cache 10-K filings")
    print(f"{'=' * W}\n")

    preprocess_filings(form_type="10-K")

    if TOKENS_CACHE_PATH.exists():
        df = pd.read_parquet(TOKENS_CACHE_PATH, columns=["ticker", "accession_number", "total_word_count"])
        size_mb = sum(p.stat().st_size for p in _part_files()) / 1e6
        print(f"\n  Cache: {TOKENS_CACHE_PATH.name}/  ({len(df)} filings, "
              f"{len(_part_files())} part-files, {size_mb:.1f} MB)")
        print(f"  total_word_count  min={df['total_word_count'].min():,}  "
              f"median={int(df['total_word_count'].median()):,}  "
              f"max={df['total_word_count'].max():,}")

    print(f"\n{'=' * W}")
    print("  preprocess_filings.py — OK")
    print(f"{'=' * W}")
