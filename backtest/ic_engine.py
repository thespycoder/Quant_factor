"""
Information Coefficient (IC) engine.

What is IC?
-----------
IC is the Spearman rank correlation between a signal and a forward return,
computed cross-sectionally within each rebalance period.

    IC_t = Spearman( signal_i,t , fwd_ret_i,t )   for all stocks i in period t

A signal with IC_mean near +1 perfectly predicts returns; near 0 means noise.
The IC Information Ratio (ICIR = IC_mean / IC_std) is the risk-adjusted score
— a high mean IC that is erratic is worth less than a steady, moderate IC.

Why Spearman and not Pearson?
------------------------------
Forward returns are fat-tailed and outlier-prone.  Pearson correlation is
distorted by a single crash or spike.  Spearman works on ranks, so outliers
count as just the extreme rank — no distortion.

Why group by filing_date (not t0)?
------------------------------------
We group by when we *learn* about the filing (filing_date), because that is
the moment a live strategy would rerank stocks.  t0 differs only for weekend/
holiday filings, but filing_date is the correct causal anchor.
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_FWD_PATH = Path(__file__).resolve().parent / "forward_returns.parquet"

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
# Internal loader
# ---------------------------------------------------------------------------

def _load_forward_returns() -> pd.DataFrame:
    """Load backtest/forward_returns.parquet with a friendly error if absent."""
    if not _FWD_PATH.exists():
        raise FileNotFoundError(
            f"Forward-returns table not found: {_FWD_PATH}\n"
            "Run `python backtest/point_in_time.py` first."
        )
    df = pd.read_parquet(_FWD_PATH)
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    return df


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def compute_ic(
    signal_df:  pd.DataFrame,
    return_col: str = "fwd_ret_21d",
    period:     str = "Q",
    min_stocks: int = 5,
) -> dict:
    """
    Compute the Information Coefficient for a signal against forward returns.

    Parameters
    ----------
    signal_df :
        DataFrame with columns ``[filing_date, ticker, signal_value]``.
        One row per (filing, ticker) pair — the same grain as the forward-
        return table.  ``signal_value`` must be numeric; NaN rows are dropped.
    return_col :
        Column in the forward-return table to predict.
        One of ``fwd_ret_5d``, ``fwd_ret_21d`` (default), ``fwd_ret_63d``.
    period :
        Rebalance-period frequency for grouping ``filing_date``.
        ``"Q"`` (calendar quarter, default), ``"M"`` (month), ``"Y"`` (year).
        Periods with fewer than ``min_stocks`` cross-sectional observations
        are skipped.
    min_stocks :
        Minimum number of stocks required in a period to compute IC.
        Default 5 — ranking fewer stocks is not statistically meaningful.

    Returns
    -------
    dict with keys:
        ic_mean      – mean IC across valid periods
        ic_std       – std of per-period ICs
        ic_ir        – IC Information Ratio = ic_mean / ic_std
        n_periods    – number of valid periods used
        ic_by_period – pd.Series (period_end_date -> IC value)

    Algorithm
    ---------
    1. Inner-join signal_df onto the forward-return table on (filing_date, ticker).
    2. Drop any row where signal_value or return_col is NaN.
    3. Group surviving rows by calendar period (pd.Grouper on filing_date).
    4. For each group with >= min_stocks rows:
           IC_t = scipy.stats.spearmanr(signal_value, return_col).statistic
    5. Aggregate: mean, std, IR = mean/std.
    """
    fwd = _load_forward_returns()

    # Normalise legacy pandas frequency aliases (deprecated in pandas >= 2.2)
    _FREQ_MAP = {"Q": "QE", "M": "ME", "Y": "YE"}
    period = _FREQ_MAP.get(period.upper(), period)

    if return_col not in fwd.columns:
        raise ValueError(
            f"return_col={return_col!r} not in forward-return table. "
            f"Available: {[c for c in fwd.columns if c.startswith('fwd_')]}"
        )

    # Prepare signal
    sig = signal_df[["filing_date", "ticker", "signal_value"]].copy()
    sig["filing_date"] = pd.to_datetime(sig["filing_date"])

    # Join — inner so only rows with both signal and return survive
    merged = fwd.merge(sig, on=["filing_date", "ticker"], how="inner")

    n_before = len(merged)
    merged = merged.dropna(subset=["signal_value", return_col])
    n_dropped = n_before - len(merged)
    if n_dropped:
        log.info("Dropped %d row(s) with NaN in signal or %s", n_dropped, return_col)

    if merged.empty:
        log.warning("No valid rows after join — check that signal tickers match the universe.")
        return {
            "ic_mean":      np.nan,
            "ic_std":       np.nan,
            "ic_ir":        np.nan,
            "n_periods":    0,
            "ic_by_period": pd.Series(dtype=float),
        }

    # Per-period Spearman IC
    ic_records: dict[pd.Timestamp, float] = {}
    n_skipped = 0

    for period_end, group in merged.groupby(
        pd.Grouper(key="filing_date", freq=period)
    ):
        if len(group) < min_stocks:
            n_skipped += 1
            continue
        # spearmanr returns a SpearmanrResult; .statistic is the correlation
        result = spearmanr(
            group["signal_value"].values,
            group[return_col].values,
        )
        # scipy >= 1.9 uses .statistic; older versions return a named tuple
        ic = float(getattr(result, "statistic", result[0]))
        ic_records[period_end] = ic

    if n_skipped:
        log.info(
            "%d period(s) skipped — fewer than %d stocks (period=%s)",
            n_skipped, min_stocks, period,
        )

    if not ic_records:
        log.warning("No periods had >= %d stocks.  Try a coarser period.", min_stocks)
        return {
            "ic_mean":      np.nan,
            "ic_std":       np.nan,
            "ic_ir":        np.nan,
            "n_periods":    0,
            "ic_by_period": pd.Series(dtype=float),
        }

    ic_series = pd.Series(ic_records).sort_index()
    ic_mean   = float(ic_series.mean())
    ic_std    = float(ic_series.std(ddof=1))

    if ic_std > 1e-10:
        ic_ir = ic_mean / ic_std
    elif ic_mean > 0:
        ic_ir = np.inf      # perfect, zero-variance predictor
    elif ic_mean < 0:
        ic_ir = -np.inf
    else:
        ic_ir = np.nan

    log.info(
        "IC result: mean=%.4f  std=%.4f  IR=%.3f  periods=%d",
        ic_mean, ic_std, ic_ir if np.isfinite(ic_ir) else 0.0, len(ic_series),
    )

    return {
        "ic_mean":      ic_mean,
        "ic_std":       ic_std,
        "ic_ir":        ic_ir,
        "n_periods":    len(ic_series),
        "ic_by_period": ic_series,
    }


# ---------------------------------------------------------------------------
# Test-signal helpers
# ---------------------------------------------------------------------------

def make_random_signal(seed: int = 42) -> pd.DataFrame:
    """
    Return a signal_df whose ``signal_value`` is pure white noise.

    Expected IC: near 0 in every period.  Useful for confirming that the
    engine does not manufacture spurious correlation from noise.
    """
    fwd = _load_forward_returns()
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "filing_date":  fwd["filing_date"],
        "ticker":       fwd["ticker"],
        "signal_value": rng.standard_normal(len(fwd)),
    })


def make_cheating_signal(return_col: str = "fwd_ret_21d") -> pd.DataFrame:
    """
    Return a signal_df where ``signal_value`` IS the forward return.

    This is deliberate lookahead — never use in real backtests.
    Its only purpose is to verify that ``compute_ic`` correctly recovers
    IC = 1.0 for a perfect predictor, confirming the join and grouping
    logic are working.
    """
    fwd = _load_forward_returns()
    return pd.DataFrame({
        "filing_date":  fwd["filing_date"],
        "ticker":       fwd["ticker"],
        "signal_value": fwd[return_col],
    })


# ---------------------------------------------------------------------------
# CLI — validation test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    W = 60

    def _report(label: str, result: dict) -> None:
        print(f"\n  Signal : {label}")
        ic_ir_str = (
            f"{result['ic_ir']:.4f}"
            if np.isfinite(result["ic_ir"])
            else str(result["ic_ir"])
        )
        print(f"  ic_mean  : {result['ic_mean']:>9.4f}")
        print(f"  ic_std   : {result['ic_std']:>9.4f}")
        print(f"  ic_ir    : {ic_ir_str:>9}")
        print(f"  n_periods: {result['n_periods']:>9}")

    print(f"\n{'=' * W}")
    print("  IC ENGINE -- VALIDATION TEST")
    print(f"{'=' * W}")

    print("\n--- Test 1: random noise signal ---")
    rand_sig  = make_random_signal(seed=42)
    rand_ic   = compute_ic(rand_sig, return_col="fwd_ret_21d", period="Q")
    _report("RANDOM (seed=42)", rand_ic)

    print("\n--- Test 2: cheating signal (signal = fwd_ret_21d) ---")
    cheat_sig = make_cheating_signal(return_col="fwd_ret_21d")
    cheat_ic  = compute_ic(cheat_sig, return_col="fwd_ret_21d", period="Q")
    _report("CHEATING (signal_value == fwd_ret_21d)", cheat_ic)

    print(f"\n{'=' * W}")
    rand_ok  = abs(rand_ic["ic_mean"])  < 0.15
    cheat_ok = abs(cheat_ic["ic_mean"]) > 0.95
    print(f"  Random IC near 0   : {'PASS' if rand_ok  else 'FAIL'}  "
          f"(ic_mean={rand_ic['ic_mean']:.4f})")
    print(f"  Cheating IC near 1 : {'PASS' if cheat_ok else 'FAIL'}  "
          f"(ic_mean={cheat_ic['ic_mean']:.4f})")
    print(f"{'=' * W}")
