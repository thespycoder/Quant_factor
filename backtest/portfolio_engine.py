"""
Portfolio backtest engine.

Turns a per-filing signal into a long-short quantile portfolio and computes
the equity curve, Sharpe ratio, and related performance metrics.

Portfolio construction (per rebalance period)
----------------------------------------------
1. Rank all stocks in the period by signal_value.
   Ties broken by rank(method="first") so pd.qcut always gets clean edges.
2. Split into n_quantiles equal-count buckets (default: quintiles 0–4).
   Bucket 0  = lowest signal  →  SHORT leg.
   Bucket n-1 = highest signal →  LONG  leg.
3. Period spread return = mean(LONG returns) − mean(SHORT returns).
   Equal-weighting within each leg.

Equity curve
------------
Compounds the per-period spread returns:  E_t = ∏(1 + spread_s,  s ≤ t)

Sharpe ratio
------------
Sharpe = mean(spread_ret) / std(spread_ret,ddof=1) × √(periods_per_year)
Risk-free rate = 0.  Annualisation:  Q→4, M→12, Y→1.
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Reuse the loader and test-signal helpers from the IC engine —
# no duplication of file-path logic or signal construction.
from backtest.ic_engine import (
    _load_forward_returns,
    make_random_signal,
    make_cheating_signal,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Map deprecated pandas freq aliases (≥ 2.2) to their modern equivalents
_FREQ_MAP: dict[str, str] = {"Q": "QE", "M": "ME", "Y": "YE"}

# How many rebalance periods fit in a calendar year (for Sharpe annualisation)
_PERIODS_PER_YEAR: dict[str, int] = {
    "QE": 4,  "Q": 4,
    "ME": 12, "M": 12,
    "YE": 1,  "Y": 1,
}

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
# Helpers
# ---------------------------------------------------------------------------

def _empty_result() -> dict:
    """Sentinel return value when the backtest cannot produce any periods."""
    return {
        "total_return":    np.nan,
        "mean_period_ret": np.nan,
        "sharpe":          np.nan,
        "n_periods":       0,
        "equity_curve":    pd.Series(dtype=float),
        "period_returns":  pd.Series(dtype=float),
    }


def _quantile_spread(
    group:       pd.DataFrame,
    return_col:  str,
    n_quantiles: int,
) -> float | None:
    """
    Assign quantile labels within one period and return the long-short spread.

    Returns None when the spread cannot be computed (NaN in either leg).

    rank(method='first') ensures every stock gets a unique integer rank so
    pd.qcut never encounters duplicate bin edges.
    """
    g = group.copy()
    g["_rank"] = g["signal_value"].rank(method="first")
    g["_q"]    = pd.qcut(g["_rank"], q=n_quantiles, labels=False).astype(int)

    long_ret  = g.loc[g["_q"] == n_quantiles - 1, return_col].mean()
    short_ret = g.loc[g["_q"] == 0,               return_col].mean()
    spread    = long_ret - short_ret

    return None if pd.isna(spread) else float(spread)


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def run_backtest(
    signal_df:   pd.DataFrame,
    return_col:  str = "fwd_ret_21d",
    period:      str = "Q",
    n_quantiles: int = 5,
) -> dict:
    """
    Run a long-short quantile backtest for a given signal.

    Parameters
    ----------
    signal_df :
        DataFrame with columns ``[filing_date, ticker, signal_value]``.
        One row per filing — same grain as the forward-return table.
    return_col :
        Forward-return column to use as each stock's holding-period return.
        One of ``fwd_ret_5d``, ``fwd_ret_21d`` (default), ``fwd_ret_63d``.
    period :
        Rebalance frequency for grouping filing dates.
        ``"Q"`` quarterly (default), ``"M"`` monthly, ``"Y"`` annual.
    n_quantiles :
        Number of signal buckets.  Default 5 (quintiles).
        Periods with fewer than ``n_quantiles * 2`` stocks are skipped.

    Returns
    -------
    dict
        total_return    – ``equity_curve[-1] − 1``
        mean_period_ret – arithmetic mean of per-period spread returns
        sharpe          – annualised Sharpe (risk-free = 0)
        n_periods       – number of valid periods used
        equity_curve    – ``pd.Series``: cumulative (1+spread) product by period
        period_returns  – ``pd.Series``: per-period spread return by period
    """
    # Normalise deprecated freq aliases first so _PERIODS_PER_YEAR lookup works
    period     = _FREQ_MAP.get(period.upper(), period)
    ppy        = _PERIODS_PER_YEAR.get(period, 4)
    min_stocks = n_quantiles * 2

    # ------------------------------------------------------------------
    # 1. Load forward returns and join signal
    # ------------------------------------------------------------------
    fwd = _load_forward_returns()

    if return_col not in fwd.columns:
        raise ValueError(
            f"return_col={return_col!r} not in forward-return table. "
            f"Available: {[c for c in fwd.columns if c.startswith('fwd_')]}"
        )

    sig = signal_df[["filing_date", "ticker", "signal_value"]].copy()
    sig["filing_date"] = pd.to_datetime(sig["filing_date"])

    merged = fwd.merge(sig, on=["filing_date", "ticker"], how="inner")

    n_before = len(merged)
    merged   = merged.dropna(subset=["signal_value", return_col])
    if (n_dropped := n_before - len(merged)):
        log.info("Dropped %d NaN row(s) in signal or %s", n_dropped, return_col)

    if merged.empty:
        log.warning("No rows after join — verify that signal tickers match the universe.")
        return _empty_result()

    log.info(
        "Joined %d rows across %d tickers for backtest",
        len(merged), merged["ticker"].nunique(),
    )

    # ------------------------------------------------------------------
    # 2. Per-period quantile construction and spread return
    # ------------------------------------------------------------------
    period_rets: dict[pd.Timestamp, float] = {}
    n_skipped = 0

    for period_end, group in merged.groupby(
        pd.Grouper(key="filing_date", freq=period)
    ):
        if len(group) < min_stocks:
            n_skipped += 1
            continue

        spread = _quantile_spread(group, return_col, n_quantiles)
        if spread is None:
            n_skipped += 1
            continue

        period_rets[period_end] = spread

    if n_skipped:
        log.info(
            "%d period(s) skipped — fewer than %d stocks or NaN spread "
            "(period=%s, n_quantiles=%d)",
            n_skipped, min_stocks, period, n_quantiles,
        )

    if not period_rets:
        log.warning("No valid periods produced — try a coarser period or larger dataset.")
        return _empty_result()

    # ------------------------------------------------------------------
    # 3. Equity curve and metrics
    # ------------------------------------------------------------------
    period_returns = pd.Series(period_rets).sort_index()
    equity_curve   = (1 + period_returns).cumprod()

    total_return    = float(equity_curve.iloc[-1] - 1)
    mean_period_ret = float(period_returns.mean())
    std_period_ret  = float(period_returns.std(ddof=1))

    if std_period_ret > 1e-10:
        sharpe = mean_period_ret / std_period_ret * np.sqrt(ppy)
    else:
        sharpe = np.nan

    log.info(
        "Backtest complete — total=%.2f%%  mean_period=%.4f  "
        "sharpe=%.3f  periods=%d",
        total_return * 100,
        mean_period_ret,
        sharpe if np.isfinite(sharpe) else 0.0,
        len(period_returns),
    )

    return {
        "total_return":    total_return,
        "mean_period_ret": mean_period_ret,
        "sharpe":          sharpe,
        "n_periods":       len(period_returns),
        "equity_curve":    equity_curve,
        "period_returns":  period_returns,
    }


# ---------------------------------------------------------------------------
# CLI — validation test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    W = 60

    def _report(label: str, r: dict) -> None:
        sharpe_str = (
            f"{r['sharpe']:.3f}" if np.isfinite(r["sharpe"]) else str(r["sharpe"])
        )
        print(f"\n  Signal          : {label}")
        print(f"  total_return    : {r['total_return']:>10.2%}")
        print(f"  mean_period_ret : {r['mean_period_ret']:>10.4f}")
        print(f"  sharpe          : {sharpe_str:>10}")
        print(f"  n_periods       : {r['n_periods']:>10}")

    print(f"\n{'=' * W}")
    print("  PORTFOLIO ENGINE -- VALIDATION TEST")
    print(f"{'=' * W}")

    print("\n--- Test 1: random noise signal ---")
    rand_r  = run_backtest(make_random_signal(),  return_col="fwd_ret_21d", period="Q")
    _report("RANDOM (seed=42)", rand_r)

    print("\n--- Test 2: cheating signal (signal == fwd_ret_21d) ---")
    cheat_r = run_backtest(make_cheating_signal(), return_col="fwd_ret_21d", period="Q")
    _report("CHEATING (signal_value == fwd_ret_21d)", cheat_r)

    print(f"\n{'=' * W}")

    rand_ok  = abs(rand_r["mean_period_ret"]) < 0.05
    cheat_ok = (
        cheat_r["total_return"] > 0.5
        and np.isfinite(cheat_r["sharpe"])
        and cheat_r["sharpe"] > 2.0
    )

    sharpe_str = (
        f"{cheat_r['sharpe']:.3f}"
        if np.isfinite(cheat_r["sharpe"])
        else str(cheat_r["sharpe"])
    )

    print(f"  Random ~0 return and low Sharpe   : {'PASS' if rand_ok  else 'FAIL'}  "
          f"(mean={rand_r['mean_period_ret']:.4f})")
    print(f"  Cheating high return + high Sharpe: {'PASS' if cheat_ok else 'FAIL'}  "
          f"(total={cheat_r['total_return']:.2%}, sharpe={sharpe_str})")
    print(f"\n  Note: random noise compiles no edge — long/short legs cancel out.")
    print(f"        cheating signal has perfect foresight — Sharpe >> 0 is expected.")
    print(f"{'=' * W}")
