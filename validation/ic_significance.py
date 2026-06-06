"""
IC significance validator.

Answers: is a factor's Information Coefficient reliably different from
zero, or could it be explained by sampling noise?

Statistical test
----------------
One-sample t-test (H₀: E[IC] = 0, two-sided).

  t = IC_mean / (IC_std / sqrt(n_periods))

A factor is declared SIGNIFICANT at the 5 % level when p < 0.05.
This is the standard test used in quantitative research (e.g. Grinold &
Kahn "Active Portfolio Management").

Why a t-test on the IC series?
---------------------------------
The per-period IC values are approximately i.i.d. samples from the
factor's true predictive distribution.  Testing their mean against zero
directly measures whether the signal has a reliable edge across the
sample of periods, accounting for both the magnitude and variability
of that edge.

The IC Information Ratio (ICIR = IC_mean / IC_std) is a complementary
rule-of-thumb: ICIR > 0.5 is often cited as practically significant in
industry, independent of the statistical test.
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backtest.ic_engine import (
    compute_ic,
    make_random_signal,
    make_cheating_signal,
    _load_forward_returns,
)

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
# Core significance test
# ---------------------------------------------------------------------------

def test_ic_significance(ic_by_period: pd.Series) -> dict:
    """
    One-sample t-test on a per-period IC series against H₀: IC_mean = 0.

    Parameters
    ----------
    ic_by_period :
        pd.Series of period-label → IC value, typically the
        ``ic_by_period`` field returned by ``backtest.ic_engine.compute_ic``.

    Returns
    -------
    dict with keys:

        ic_mean    – sample mean of the IC series
        ic_std     – sample std (ddof=1)
        n_periods  – number of valid (non-NaN) periods
        t_stat     – t-statistic  (np.inf when IC_std ≈ 0 and IC_mean ≠ 0)
        p_value    – two-sided p-value  (0.0 in the degenerate IC_std≈0 case)
        ic_ir      – IC_mean / IC_std (information ratio; NaN when std≈0)
        significant – True when p_value < 0.05
        verdict    – human-readable "SIGNIFICANT (p<0.05)" or "NOT SIGNIFICANT"
    """
    series = ic_by_period.dropna()
    n      = len(series)

    if n < 2:
        return dict(
            ic_mean=np.nan, ic_std=np.nan, n_periods=n,
            t_stat=np.nan, p_value=np.nan, ic_ir=np.nan,
            significant=False,
            verdict="NOT SIGNIFICANT (insufficient data — need >= 2 periods)",
        )

    ic_mean = float(series.mean())
    ic_std  = float(series.std(ddof=1))
    ic_ir   = ic_mean / ic_std if ic_std > 1e-10 else np.nan

    if ic_std < 1e-10:
        # Degenerate: all ICs are identical (e.g. perfect cheating signal).
        # t → ±∞, p → 0 when mean ≠ 0.
        if abs(ic_mean) > 1e-10:
            t_stat, p_value = float(np.inf * np.sign(ic_mean)), 0.0
            significant, verdict = True, "SIGNIFICANT (p<0.05)"
        else:
            t_stat, p_value = 0.0, 1.0
            significant, verdict = False, "NOT SIGNIFICANT"
    else:
        result  = ttest_1samp(series.values, popmean=0)
        t_stat  = float(result.statistic)
        p_value = float(result.pvalue)
        significant = p_value < 0.05
        verdict = "SIGNIFICANT (p<0.05)" if significant else "NOT SIGNIFICANT"

    return dict(
        ic_mean    = ic_mean,
        ic_std     = ic_std,
        n_periods  = n,
        t_stat     = t_stat,
        p_value    = p_value,
        ic_ir      = ic_ir,
        significant = significant,
        verdict    = verdict,
    )


# ---------------------------------------------------------------------------
# Yearly IC breakdown
# ---------------------------------------------------------------------------

def ic_by_year(
    signal_df:           pd.DataFrame,
    return_col:          str,
    forward_returns_df:  pd.DataFrame,
    min_obs:             int = 5,
) -> pd.DataFrame:
    """
    Compute cross-sectional Spearman IC separately for each calendar year.

    A factor with a genuinely positive IC should show consistent positive
    values across years.  A single lucky year producing a large IC while the
    rest are near zero is a red flag, even if the t-test looks good.

    Parameters
    ----------
    signal_df :
        DataFrame with columns ``[filing_date, ticker, signal_value]``.
    return_col :
        Forward-return column to predict (e.g. ``"fwd_ret_21d"``).
    forward_returns_df :
        The full forward-return table, typically from
        ``backtest.ic_engine._load_forward_returns()``.
    min_obs :
        Minimum number of valid observations in a year; years below this
        threshold are skipped.

    Returns
    -------
    DataFrame with columns: ``year`` (int), ``ic`` (float), ``n_obs`` (int).
    Each row is one IC value computed across all filings in that year.
    """
    sig = signal_df[["filing_date", "ticker", "signal_value"]].copy()
    sig["filing_date"] = pd.to_datetime(sig["filing_date"])

    fwd = forward_returns_df.copy()
    fwd["filing_date"] = pd.to_datetime(fwd["filing_date"])

    merged = (
        fwd.merge(sig, on=["filing_date", "ticker"], how="inner")
        .dropna(subset=["signal_value", return_col])
    )

    if merged.empty:
        return pd.DataFrame(columns=["year", "ic", "n_obs"])

    merged["year"] = merged["filing_date"].dt.year
    records: list[dict] = []

    for year, grp in merged.groupby("year"):
        if len(grp) < min_obs:
            log.info("Year %d skipped — only %d observations", year, len(grp))
            continue
        result = spearmanr(grp["signal_value"].values, grp[return_col].values)
        ic_val = float(getattr(result, "statistic", result[0]))
        records.append({"year": int(year), "ic": ic_val, "n_obs": len(grp)})

    return (
        pd.DataFrame(records)
        .sort_values("year")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def _print_report(
    label:      str,
    return_col: str,
    period:     str,
    sig:        dict,
    yearly:     pd.DataFrame,
) -> None:
    W = 58

    p_str  = f"{sig['p_value']:.4f}"  if np.isfinite(sig["p_value"])  else "inf/NaN"
    t_str  = f"{sig['t_stat']:.4f}"   if np.isfinite(sig["t_stat"])   else str(sig["t_stat"])
    ir_str = f"{sig['ic_ir']:.4f}"    if np.isfinite(sig["ic_ir"])    else "N/A"

    print(f"\n{'-' * W}")
    print(f"  IC Significance Report")
    print(f"  Signal   : {label}")
    print(f"  Return   : {return_col}   Period: {period}")
    print(f"{'-' * W}")
    print(f"  {'ic_mean':<18} {sig['ic_mean']:>+10.4f}")
    print(f"  {'ic_std':<18} {sig['ic_std']:>10.4f}")
    print(f"  {'ic_ir':<18} {ir_str:>10}")
    print(f"  {'n_periods':<18} {sig['n_periods']:>10}")
    print(f"  {'t_stat':<18} {t_str:>10}")
    print(f"  {'p_value':<18} {p_str:>10}")
    print(f"  {'verdict':<18} {sig['verdict']}")

    if not yearly.empty:
        bar_max   = yearly["ic"].abs().max()
        bar_scale = 18 / max(bar_max, 1e-6)
        print(f"\n  Yearly IC breakdown (cross-sectional per year):")
        print(f"  {'Year':>6}  {'IC':>8}  {'n_obs':>5}  Chart")
        print(f"  {'--':->6}  {'--':->8}  {'--':->5}  {'--':->18}")
        for _, row in yearly.iterrows():
            bar  = int(abs(row["ic"]) * bar_scale)
            sign = "+" if row["ic"] >= 0 else "-"
            print(f"  {int(row['year']):>6}  {row['ic']:>+8.4f}  {int(row['n_obs']):>5}  {sign}{'|' * bar}")

    print(f"{'-' * W}")


# ---------------------------------------------------------------------------
# Convenience validator
# ---------------------------------------------------------------------------

def validate_factor(
    signal_df:  pd.DataFrame,
    return_col: str = "fwd_ret_21d",
    period:     str = "Q",
    label:      str = "factor",
) -> dict:
    """
    Full factor validation in one call: IC significance + yearly stability.

    1. Calls ``compute_ic`` to get the per-period IC series.
    2. Runs ``test_ic_significance`` on that series.
    3. Computes ``ic_by_year`` for the stability breakdown.
    4. Prints a clean report.
    5. Returns a combined dict.

    Parameters
    ----------
    signal_df  : DataFrame ``[filing_date, ticker, signal_value]``.
    return_col : Forward-return column to predict.
    period     : Rebalance period for IC grouping (``"Q"``, ``"M"``, ``"Y"``).
    label      : Human-readable name printed in the report header.

    Returns
    -------
    dict merging the significance-test keys with:
        yearly_ic    – DataFrame from ``ic_by_year``
        ic_by_period – raw per-period IC Series
    """
    log.info("Validating '%s' against %s (period=%s) ...", label, return_col, period)

    ic_result  = compute_ic(signal_df, return_col=return_col, period=period)
    sig_result = test_ic_significance(ic_result["ic_by_period"])

    fwd        = _load_forward_returns()
    yearly     = ic_by_year(signal_df, return_col, fwd)

    _print_report(label, return_col, period, sig_result, yearly)

    return {
        **sig_result,
        "yearly_ic":    yearly,
        "ic_by_period": ic_result["ic_by_period"],
    }


# ---------------------------------------------------------------------------
# CLI — validate two test signals
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    W = 60

    print(f"\n{'=' * W}")
    print("  IC SIGNIFICANCE VALIDATOR")
    print(f"{'=' * W}")

    # --- Signal 1: random noise ---
    print("\n--- Signal 1: RANDOM NOISE ---")
    print("  EXPECTED: NOT SIGNIFICANT  |  p well above 0.05  |  |t| small")

    rand_result = validate_factor(
        make_random_signal(),
        return_col="fwd_ret_21d",
        period="Q",
        label="Random noise (seed=42)",
    )

    rand_ok = (not rand_result["significant"]) and (rand_result["p_value"] > 0.05)
    print(f"\n  EXPECTED vs ACTUAL")
    print(f"  Verdict   : NOT SIGNIFICANT  |  {rand_result['verdict']}")
    print(f"  p > 0.05  : {'PASS' if rand_ok else 'FAIL'}  "
          f"(p = {rand_result['p_value']:.4f})")

    # --- Signal 2: cheating signal ---
    print(f"\n{'=' * W}")
    print("\n--- Signal 2: CHEATING (signal = fwd_ret_21d) ---")
    print("  EXPECTED: SIGNIFICANT  |  p near 0  |  |t_stat| very large")

    cheat_result = validate_factor(
        make_cheating_signal(),
        return_col="fwd_ret_21d",
        period="Q",
        label="Cheating (signal == fwd_ret_21d)",
    )

    cheat_sig_ok = cheat_result["significant"] and cheat_result["p_value"] < 0.05
    cheat_t_ok   = (
        np.isfinite(cheat_result["t_stat"]) and abs(cheat_result["t_stat"]) > 10
    ) or not np.isfinite(cheat_result["t_stat"])   # inf also counts
    t_display = (
        f"{cheat_result['t_stat']:.2f}"
        if np.isfinite(cheat_result["t_stat"])
        else str(cheat_result["t_stat"])
    )

    print(f"\n  EXPECTED vs ACTUAL")
    print(f"  Verdict   : SIGNIFICANT    |  {cheat_result['verdict']}")
    print(f"  Sig (p<0.05): {'PASS' if cheat_sig_ok else 'FAIL'}  "
          f"(p = {cheat_result['p_value']:.2e})")
    print(f"  Large |t| : {'PASS' if cheat_t_ok else 'FAIL'}  "
          f"(t = {t_display})")

    # Summary
    print(f"\n{'=' * W}")
    all_ok = rand_ok and cheat_sig_ok and cheat_t_ok
    print(f"  OVERALL: {'PASS' if all_ok else 'FAIL'}")
    print(f"  Random correctly NOT significant, cheating correctly SIGNIFICANT.")
    print(f"{'=' * W}")
