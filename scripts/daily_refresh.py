#!/usr/bin/env python3
# Production entry point for the nightly incremental data refresh.
#
# In a real deployment this script would be triggered on a schedule via cron
# (Linux/macOS), Windows Task Scheduler, or a GitHub Actions `schedule:` workflow.
# The existing caching logic inside the loaders makes daily re-runs cheap:
#   - price_loader skips tickers whose cache already covers the requested range.
#   - edgar_downloader skips filings already present in the local index/raw dir.
# Only genuinely new data is fetched, so same-day reruns are near-instant.
#
# Scope: 10-K only.
#   The engine — signal computation, IC backtest, factor decay, Fama-French
#   orthogonalization, and the filing-token preprocessing cache — was built and
#   validated end-to-end on annual 10-K filings.  10-Qs are intentionally out of
#   scope for the current version.  Extending to 10-Qs would require:
#     1. Rebuilding the forward-return panel (the current panel anchors on 10-K
#        filing dates; adding quarterly filings changes the event cadence and the
#        point-in-time join logic).
#     2. Re-running preprocess_filings.py so the token cache includes 10-Q text.
#     3. Re-validating the IC significance and factor-decay thresholds, which may
#        behave differently on quarterly vs annual disclosures.
#   Until that work is done, passing --forms 10-Q (or 10-K,10-Q) is safe but the
#   extra filings will not be used by any downstream step.

import argparse
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402 — after sys.path setup

from config.universe import get_universe
from data.price_loader import PRICES_PATH, run as _price_run
from data.edgar_downloader import load_index as _filing_index, run as _filing_run
from agents.preprocess_filings import preprocess_filings


# ── helpers ────────────────────────────────────────────────────────────────

def _row_count(path: Path) -> int:
    """Return parquet row count; 0 if the file/directory does not yet exist."""
    if not path.exists():
        return 0
    try:
        return len(pd.read_parquet(path))
    except Exception:
        return 0


def _today() -> str:
    return date.today().isoformat()


# ── refresh steps ──────────────────────────────────────────────────────────

def refresh_prices(tickers: list[str]) -> int:
    """Pull any missing price rows up to today.  Returns number of new rows."""
    before = _row_count(PRICES_PATH)
    try:
        _price_run(tickers=tickers, end=_today())
    except Exception as exc:
        print(f"  [ERROR] Price refresh failed: {exc}")
        return 0
    return max(0, _row_count(PRICES_PATH) - before)


def refresh_filings(tickers: list[str], forms: list[str]) -> int:
    """Download any missing filings up to the current year.  Returns new filing count."""
    before = len(_filing_index())
    try:
        _filing_run(tickers=tickers, form_types=forms, end_year=date.today().year)
    except Exception as exc:
        print(f"  [ERROR] Filing refresh failed: {exc}")
        return 0
    return max(0, len(_filing_index()) - before)


def run_preprocessing(forms: list[str]) -> None:
    """Tokenize filings not yet in the cache, one form type at a time."""
    for form_type in forms:
        print(f"  Tokenizing {form_type} filings ...")
        try:
            preprocess_filings(form_type=form_type)
        except Exception as exc:
            print(f"  [ERROR] Preprocessing {form_type} failed: {exc}")


def cleanup_orphan_filings() -> int:
    """Delete raw filing .txt files whose accession number is not in the index.

    These are filings that were downloaded (e.g. 10-Qs from an earlier run) but
    were never added to filings_index.parquet — so no downstream step uses them.
    Returns the number of files deleted.
    """
    from data.edgar_downloader import RAW_DIR, load_index as _filing_index

    indexed = set(_filing_index()["accession_number"].tolist())
    deleted = 0
    for txt_file in RAW_DIR.rglob("*.txt"):
        if txt_file.stem not in indexed:
            try:
                txt_file.unlink()
                deleted += 1
            except Exception as exc:
                print(f"  [WARN] Could not delete {txt_file.name}: {exc}")
    return deleted


def run_research_cycle() -> None:
    """Run one factor research cycle as a subprocess."""
    script = ROOT / "orchestration" / "research_graph.py"
    result = subprocess.run([sys.executable, str(script)], check=False)
    if result.returncode != 0:
        print(f"  [ERROR] Research cycle exited with code {result.returncode}")


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Incrementally refresh price and filing data, then optionally run a research cycle.",
    )
    parser.add_argument(
        "--run-research", action="store_true", default=False,
        help="Trigger one factor research cycle after the data refresh.",
    )
    parser.add_argument(
        "--universe", default="all",
        help="Universe key (default: all — uses config/universe.py).",
    )
    parser.add_argument(
        "--forms", default="10-K",
        help="Comma-separated SEC form types to refresh (default: 10-K).",
    )
    args = parser.parse_args()

    forms = [f.strip() for f in args.forms.split(",") if f.strip()]

    # ── banner ─────────────────────────────────────────────────────────────
    print("=" * 64)
    print(f"  Quant Factor — Daily Refresh   {datetime.now().isoformat(timespec='seconds')}")
    print("=" * 64)

    # ── universe ───────────────────────────────────────────────────────────
    print("\n[1/5] Loading universe ...")
    tickers = get_universe()
    print(f"      {len(tickers)} tickers.")

    # ── prices ─────────────────────────────────────────────────────────────
    print(f"\n[2/5] Refreshing prices (end={_today()}) ...")
    new_prices = refresh_prices(tickers)
    print(f"      {new_prices} new rows added to price cache.")

    # ── orphan cleanup ─────────────────────────────────────────────────────
    print("\n[3/5] Removing orphan filing files (not in index) ...")
    deleted = cleanup_orphan_filings()
    print(f"      {deleted} file(s) deleted.")

    # ── filings ────────────────────────────────────────────────────────────
    print(f"\n[4/5] Refreshing filings ({', '.join(forms)}) ...")
    new_filings = refresh_filings(tickers, forms)
    print(f"      {new_filings} new filings downloaded.")

    # ── preprocessing ──────────────────────────────────────────────────────
    if new_filings > 0:
        print(f"\n[5/5] Tokenizing {new_filings} new filing(s) ...")
        run_preprocessing(forms)
        preprocessed = True
    else:
        print("\n[5/5] No new filings — skipping preprocessing.")
        preprocessed = False

    # ── summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  Refresh summary")
    print(f"    New price rows    : {new_prices}")
    print(f"    Orphan files deleted : {deleted}")
    print(f"    New filings       : {new_filings}")
    print(f"    Preprocessing ran : {'yes' if preprocessed else 'no (nothing new)'}")
    print("=" * 64)

    # ── optional research cycle ────────────────────────────────────────────
    if args.run_research:
        print("\n[--run-research] Starting research cycle ...\n")
        run_research_cycle()
    else:
        print("\nPass --run-research to also run a factor research cycle.")


if __name__ == "__main__":
    main()
