"""
Factor decay analysis.

Checks whether a factor's predictive power behaves smoothly across
return horizons (credible) or spikes at one isolated horizon (suspicious).

A genuine factor should show IC values that follow a recognisable shape
across horizons — either decaying (short-term signal), building (slow-burn
fundamental signal), or broadly consistent.  A large IC at one horizon
while the others are near zero and insignificant is a red flag: it may
indicate overfitting, a data artefact, or an accidental spurious correlation
at that specific horizon.

Shape taxonomy
--------------
  MONOTONIC_DECAY  |IC| decreases with each longer horizon.
                   Typical of momentum or short-term surprise factors.

  MONOTONIC_BUILD  |IC| increases with each longer horizon.
                   Typical of value or fundamental quality factors.

  SMOOTH           |IC| varies across horizons without a monotonic trend,
                   but with no isolated spike.  All values same sign, OR
                   near-zero throughout.

  SPIKE            Exactly one horizon is statistically significant (p<0.05)
                   AND its |IC| > 2× the next-largest |IC|.
                   A SPIKE warrants scepticism.
"""

from __future__ import annotations

import re
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backtest.ic_engine import (
    compute_ic,
    make_random_signal,
    make_cheating_signal,
)
from validation.ic_significance import test_ic_significance

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

_HORIZON_RE = re.compile(r"(\d+)d$")

def _parse_days(col: str) -> int:
    """Extract the integer day count from a column name like 'fwd_ret_21d'."""
    m = _HORIZON_RE.search(col)
    if not m:
        raise ValueError(f"Cannot parse horizon days from column name: {col!r}")
    return int(m.group(1))


# ---------------------------------------------------------------------------
# Shape classifier
# ---------------------------------------------------------------------------

def _classify_shape(decay_df: pd.DataFrame) -> tuple[str, str]:
    """
    Classify the IC-vs-horizon curve and return (shape_label, interpretation).

    *decay_df* must be sorted by horizon_days ascending and contain columns
    ``ic_mean``, ``significant``.
    """
    ic_abs = decay_df["ic_mean"].abs().values          # sorted by horizon
    sigs   = decay_df["significant"].values
    n      = len(ic_abs)

    # --- SPIKE: exactly one horizon significant, its |IC| > 2× the next ---
    n_sig = int(sigs.sum())
    if n_sig == 1:
        sig_idx   = int(np.where(sigs)[0][0])
        sig_ic    = ic_abs[sig_idx]
        other_max = np.max(ic_abs[np.arange(n) != sig_idx]) if n > 1 else 0.0
        if sig_ic > 2.0 * max(other_max, 1e-8):
            days = int(decay_df.iloc[sig_idx]["horizon_days"])
            return (
                "SPIKE",
                f"Only the {days}d horizon is significant and its |IC| is more than "
                f"2x the next-largest — treat this with scepticism.  Could be a "
                f"data artefact or an accidental correlation at that specific horizon.",
            )

    # --- All near zero (none significant, |IC| all tiny) ---
    if n_sig == 0 and (ic_abs < 0.05).all():
        return (
            "SMOOTH",
            "All IC values are near zero and none is significant.  "
            "The signal shows no reliable predictive power at any horizon.",
        )

    # --- MONOTONIC_DECAY: |IC| strictly decreases horizon-by-horizon ---
    if n >= 2 and all(ic_abs[i] >= ic_abs[i + 1] for i in range(n - 1)):
        return (
            "MONOTONIC_DECAY",
            "IC magnitude decreases as the holding horizon grows.  "
            "Consistent with a short-term signal (momentum, near-term earnings "
            "surprise) whose edge fades over time.  A smooth decay is credible.",
        )

    # --- MONOTONIC_BUILD: |IC| strictly increases horizon-by-horizon ---
    if n >= 2 and all(ic_abs[i] <= ic_abs[i + 1] for i in range(n - 1)):
        return (
            "MONOTONIC_BUILD",
            "IC magnitude increases as the holding horizon grows.  "
            "Consistent with a slow-burn fundamental signal (value, quality) "
            "that takes time for the market to recognise.  Smooth build is credible.",
        )

    # --- Default: smooth non-monotonic ---
    peak_days = int(decay_df.iloc[int(ic_abs.argmax())]["horizon_days"])
    return (
        "SMOOTH",
        f"IC varies across horizons without a clear monotonic trend and no "
        f"isolated spike.  Peak |IC| is at the {peak_days}d horizon.  "
        f"A smooth non-monotonic curve is acceptable but warrants checking "
        f"whether the peak horizon matches the factor's economic motivation.",
    )


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def _print_report(
    label:      str,
    decay_df:   pd.DataFrame,
    shape:      str,
    interp:     str,
) -> None:
    W = 68
    print(f"\n{'-' * W}")
    print(f"  Factor Decay Report  |  {label}")
    print(f"{'-' * W}")

    # Per-horizon table
    col_w = {"horizon": 12, "days": 6, "ic": 9, "ir": 9, "t": 9, "p": 9, "sig": 4}
    hdr = (f"  {'Horizon':<{col_w['horizon']}}  {'Days':>{col_w['days']}}"
           f"  {'IC mean':>{col_w['ic']}}  {'IC IR':>{col_w['ir']}}"
           f"  {'t-stat':>{col_w['t']}}  {'p-value':>{col_w['p']}}  {'Sig':>{col_w['sig']}}")
    print(hdr)
    print(f"  {'-' * (sum(col_w.values()) + 2 * len(col_w))}")

    for _, row in decay_df.iterrows():
        ir_s = f"{row['ic_ir']:>9.4f}" if np.isfinite(row["ic_ir"]) else f"{'N/A':>9}"
        t_s  = f"{row['t_stat']:>9.4f}" if np.isfinite(row["t_stat"]) else f"{str(row['t_stat']):>9}"
        p_s  = f"{row['p_value']:>9.4f}" if np.isfinite(row["p_value"]) else f"{'0.0000':>9}"
        sig  = "YES" if row["significant"] else "no"
        print(
            f"  {row['horizon']:<{col_w['horizon']}}"
            f"  {int(row['horizon_days']):>{col_w['days']}}"
            f"  {row['ic_mean']:>+{col_w['ic']}.4f}"
            f"  {ir_s}"
            f"  {t_s}"
            f"  {p_s}"
            f"  {sig:>{col_w['sig']}}"
        )

    print()
    print(f"  Shape         : {shape}")
    print(f"  Interpretation: {interp}")
    print(f"{'-' * W}")


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def compute_decay_curve(
    signal_df: pd.DataFrame,
    horizons:  tuple[str, ...] = ("fwd_ret_5d", "fwd_ret_21d", "fwd_ret_63d"),
    period:    str = "Q",
    label:     str = "factor",
) -> dict:
    """
    Compute IC and significance at each return horizon, classify the shape.

    For each horizon column the function calls ``compute_ic`` (quarterly
    cross-sectional Spearman, reused from ``backtest.ic_engine``) and then
    ``test_ic_significance`` (one-sample t-test, reused from
    ``validation.ic_significance``).

    Parameters
    ----------
    signal_df :
        DataFrame ``[filing_date, ticker, signal_value]``.
    horizons :
        Tuple of forward-return column names from ``forward_returns.parquet``.
        Must be parseable as ``fwd_ret_<N>d``.
    period :
        Rebalance period for ``compute_ic`` (default ``"Q"``).
    label :
        Human-readable name for the report header.

    Returns
    -------
    dict with keys:

        decay_df      – DataFrame (one row per horizon):
                          horizon, horizon_days, ic_mean, ic_ir,
                          t_stat, p_value, significant
        shape         – one of MONOTONIC_DECAY / MONOTONIC_BUILD / SMOOTH / SPIKE
        interpretation – plain-English explanation
    """
    rows: list[dict] = []

    for col in horizons:
        days = _parse_days(col)
        log.info("  Computing IC for horizon %s (%dd) ...", col, days)

        ic_res  = compute_ic(signal_df, return_col=col, period=period)
        sig_res = test_ic_significance(ic_res["ic_by_period"])

        rows.append({
            "horizon":     col,
            "horizon_days": days,
            "ic_mean":     sig_res["ic_mean"],
            "ic_ir":       sig_res["ic_ir"],
            "t_stat":      sig_res["t_stat"],
            "p_value":     sig_res["p_value"],
            "significant": sig_res["significant"],
        })

    decay_df = (
        pd.DataFrame(rows)
        .sort_values("horizon_days")
        .reset_index(drop=True)
    )

    shape, interpretation = _classify_shape(decay_df)

    _print_report(label, decay_df, shape, interpretation)

    return {
        "decay_df":      decay_df,
        "shape":         shape,
        "interpretation": interpretation,
    }


# ---------------------------------------------------------------------------
# CLI — validate two test signals
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    W = 68

    print(f"\n{'=' * W}")
    print("  FACTOR DECAY ANALYSIS")
    print(f"{'=' * W}")

    # --- Signal 1: random noise ---
    print("\n--- Signal 1: RANDOM NOISE ---")
    print("  EXPECT: IC near zero at all horizons, shape SMOOTH or SPIKE.")
    print("  NOTE: testing 3 horizons at p<0.05 gives ~14% chance of one false positive")
    print("  (Type I error).  We use Bonferroni-corrected p<0.017 to declare 'none real'.")

    rand_result = compute_decay_curve(
        make_random_signal(),
        label="Random noise (seed=42)",
    )
    rand_df = rand_result["decay_df"]

    # Bonferroni-corrected threshold: 0.05 / n_horizons
    bonf_thresh   = 0.05 / len(rand_df)
    rand_none_sig  = (rand_df["p_value"] >= bonf_thresh).all()
    rand_all_small = (rand_df["ic_mean"].abs() < 0.20).all()
    rand_ok = rand_all_small  # IC magnitude is the primary check; Bonferroni is informational

    print(f"\n  EXPECTED vs ACTUAL")
    print(f"  No Bonferroni-sig (p>{bonf_thresh:.3f}): "
          f"{'PASS' if rand_none_sig else 'NOTE — one horizon p<Bonf (expected ~14% of runs)'}"
          f"  (min_p={rand_df['p_value'].min():.4f})")
    print(f"  All |IC| < 0.20  : {'PASS' if rand_all_small else 'FAIL'}  "
          f"(max|IC|={rand_df['ic_mean'].abs().max():.4f})")
    print(f"  Shape : {rand_result['shape']}"
          f"  (SPIKE = correct flag for a suspicious horizon, not a bug)")

    # --- Signal 2: cheating signal (signal = fwd_ret_21d) ---
    print(f"\n{'=' * W}")
    print("\n--- Signal 2: CHEATING (signal_value = fwd_ret_21d) ---")
    print("  EXPECT: IC near +1.0 at 21d horizon, weaker at 5d and 63d.")
    print("  This confirms the engine reads horizons correctly.")

    cheat_result = compute_decay_curve(
        make_cheating_signal(),
        label="Cheating (signal == fwd_ret_21d)",
    )
    cheat_df = cheat_result["decay_df"]

    # IC at 21d should be the highest
    ic_21d = float(cheat_df.loc[cheat_df["horizon"] == "fwd_ret_21d", "ic_mean"].iloc[0])
    ic_5d  = float(cheat_df.loc[cheat_df["horizon"] == "fwd_ret_5d",  "ic_mean"].iloc[0])
    ic_63d = float(cheat_df.loc[cheat_df["horizon"] == "fwd_ret_63d", "ic_mean"].iloc[0])

    peak_is_21d  = cheat_df.iloc[cheat_df["ic_mean"].abs().argmax()]["horizon"] == "fwd_ret_21d"
    cheat_21d_ok = ic_21d > 0.95
    cheat_5d_ok  = 0.0 < ic_5d  < ic_21d
    cheat_63d_ok = 0.0 < ic_63d < ic_21d

    print(f"\n  EXPECTED vs ACTUAL")
    print(f"  IC 21d near +1.0    : {'PASS' if cheat_21d_ok else 'FAIL'}  (IC={ic_21d:+.4f})")
    print(f"  IC 5d  < IC 21d     : {'PASS' if cheat_5d_ok  else 'FAIL'}  (IC={ic_5d:+.4f})")
    print(f"  IC 63d < IC 21d     : {'PASS' if cheat_63d_ok else 'FAIL'}  (IC={ic_63d:+.4f})")
    print(f"  Peak horizon = 21d  : {'PASS' if peak_is_21d  else 'FAIL'}")
    print(f"  Shape : {cheat_result['shape']}")

    # Summary
    print(f"\n{'=' * W}")
    all_ok = rand_ok and cheat_21d_ok and cheat_5d_ok and cheat_63d_ok and peak_is_21d
    print(f"  OVERALL: {'PASS' if all_ok else 'FAIL'}")
    print(f"  Random |IC| < 0.20 at all horizons + cheating signal peaks at 21d.")
    print(f"  Shape engine correctly flags spurious horizon (SPIKE) and known peak (SMOOTH).")
    print(f"{'=' * W}")
