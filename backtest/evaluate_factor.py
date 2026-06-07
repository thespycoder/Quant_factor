"""
Factor evaluation pipeline.

Connects the Signal Finder LLM's structured output (FactorHypothesis, see
agents/hypothesis_schema.py) to the existing backtest + validation engine,
producing one combined PASS/FAIL verdict per hypothesis.

Pipeline (each stage reuses an existing, already-validated module — none of
their logic is reimplemented here):

    1. agents.signal_computation.compute_signal_df   hypothesis -> signal_df
    2. validation.ic_significance.validate_factor    IC + significance + yearly IC
    3. backtest.portfolio_engine.run_backtest        long-short quantile backtest
    4. validation.factor_decay.compute_decay_curve   IC across 5d/21d/63d, shape
    5. validation.fama_french.orthogonalize          alpha vs. FF3 + Momentum

A factor only earns the "CANDIDATE" verdict if it clears all three gates:
significant IC (p < 0.05), a non-suspicious decay shape (no SPIKE), and a
genuinely novel alpha (significant after controlling for known factors).
Anything else is labelled by its first failing gate, with every failing gate
listed in ``failed_gates`` so a researcher can see the whole picture at once.
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agents.hypothesis_schema import FactorHypothesis
from agents.signal_computation import compute_signal_df
from backtest.portfolio_engine import run_backtest
from validation.ic_significance import validate_factor
from validation.factor_decay import compute_decay_curve
from validation.fama_french import orthogonalize

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Below this many filings with a usable (non-NaN) signal value, IC/backtest
# results are too noisy to be meaningful — short-circuit instead of running
# the full (expensive) suite on a near-empty signal.
_MIN_FILINGS = 100

# Gate-failure labels, in priority order — when a factor fails more than one
# gate, `verdict` reports the first one (in this order) while `failed_gates`
# lists all of them.
_GATE_NOT_SIGNIFICANT  = "NOT_SIGNIFICANT"
_GATE_SUSPICIOUS_DECAY = "SUSPICIOUS_DECAY"
_GATE_NOT_NOVEL        = "NOT_NOVEL"
_GATE_PRIORITY = (_GATE_NOT_SIGNIFICANT, _GATE_SUSPICIOUS_DECAY, _GATE_NOT_NOVEL)

_CANDIDATE = "CANDIDATE"
_INSUFFICIENT_DATA = "INSUFFICIENT_DATA"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def evaluate_factor(hypothesis: FactorHypothesis) -> dict:
    """
    Run a FactorHypothesis through the full backtest + validation suite and
    return a single combined verdict dict.

    Returns
    -------
    dict with keys:
        hypothesis       – hypothesis.model_dump()
        n_filings_used   – count of filings with a usable (non-NaN) signal value
        ic_mean, ic_t_stat, ic_p_value, ic_by_year  – IC + significance (primary horizon)
        total_return, sharpe                        – long-short quantile backtest
        decay_shape                                  – MONOTONIC_DECAY / MONOTONIC_BUILD / SMOOTH / SPIKE
        ff_alpha, ff_alpha_pvalue, ff_is_novel       – Fama-French orthogonalisation
        verdict       – "CANDIDATE" | "INSUFFICIENT_DATA" | first failed-gate label
        failed_gates  – list of every gate this factor failed (empty for CANDIDATE)

    If fewer than ``_MIN_FILINGS`` filings produce a usable signal value, the
    expensive suite is skipped entirely and verdict="INSUFFICIENT_DATA" is
    returned with whatever count was achieved.
    """
    horizon = hypothesis.horizon
    label   = hypothesis.signal_name

    # --- 1. Signal --------------------------------------------------------
    signal_df      = compute_signal_df(hypothesis)
    n_filings_used = int(signal_df["signal_value"].notna().sum())

    if n_filings_used < _MIN_FILINGS:
        log.warning(
            "'%s': only %d filings with a usable signal (< %d) — INSUFFICIENT_DATA",
            label, n_filings_used, _MIN_FILINGS,
        )
        return {
            "hypothesis":     hypothesis.model_dump(),
            "n_filings_used": n_filings_used,
            "ic_mean": None, "ic_t_stat": None, "ic_p_value": None, "ic_by_year": None,
            "total_return": None, "sharpe": None,
            "decay_shape": None,
            "ff_alpha": None, "ff_alpha_pvalue": None, "ff_is_novel": None,
            "verdict":      _INSUFFICIENT_DATA,
            "failed_gates": [_INSUFFICIENT_DATA],
        }

    # --- 2. IC + significance + yearly breakdown (primary horizon) -------
    val_result = validate_factor(signal_df, return_col=horizon, label=label)
    ic_mean    = val_result["ic_mean"]
    ic_t_stat  = val_result["t_stat"]
    ic_p_value = val_result["p_value"]
    ic_by_year = val_result["yearly_ic"]

    # --- 3. Long-short quantile backtest (primary horizon) ---------------
    bt_result      = run_backtest(signal_df, return_col=horizon)
    total_return   = bt_result["total_return"]
    sharpe         = bt_result["sharpe"]
    period_returns = bt_result["period_returns"]

    # --- 4. Decay curve across all three horizons ------------------------
    decay_result = compute_decay_curve(signal_df, label=label)
    decay_shape  = decay_result["shape"]

    # --- 5. Fama-French orthogonalisation of the backtest's per-period returns
    try:
        ff_result       = orthogonalize(period_returns)
        ff_alpha        = ff_result["alpha"]
        ff_alpha_pvalue = ff_result["alpha_pvalue"]
        ff_is_novel     = ff_result["is_novel"]
    except ValueError as exc:
        # Too few aligned periods for a meaningful regression — cannot claim
        # novelty, so fail that gate conservatively rather than crash.
        log.warning("'%s': Fama-French orthogonalisation skipped (%s)", label, exc)
        ff_alpha = ff_alpha_pvalue = None
        ff_is_novel = False

    # --- 6. Combine into gates + verdict ----------------------------------
    failed_gates: list[str] = []
    if ic_p_value is None or not (ic_p_value < 0.05):
        failed_gates.append(_GATE_NOT_SIGNIFICANT)
    if decay_shape == "SPIKE":
        failed_gates.append(_GATE_SUSPICIOUS_DECAY)
    if not ff_is_novel:
        failed_gates.append(_GATE_NOT_NOVEL)

    if not failed_gates:
        verdict = _CANDIDATE
    else:
        verdict = next(g for g in _GATE_PRIORITY if g in failed_gates)

    return {
        "hypothesis":     hypothesis.model_dump(),
        "n_filings_used": n_filings_used,
        "ic_mean":        ic_mean,
        "ic_t_stat":      ic_t_stat,
        "ic_p_value":     ic_p_value,
        "ic_by_year":     ic_by_year,
        "total_return":   total_return,
        "sharpe":         sharpe,
        "decay_shape":    decay_shape,
        "ff_alpha":         ff_alpha,
        "ff_alpha_pvalue":  ff_alpha_pvalue,
        "ff_is_novel":      ff_is_novel,
        "verdict":      verdict,
        "failed_gates": failed_gates,
    }


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_many(hypotheses: list[FactorHypothesis]) -> pd.DataFrame:
    """
    Evaluate a list of hypotheses and return a one-row-per-hypothesis summary
    DataFrame, sorted so CANDIDATEs appear first, then by ic_p_value ascending.

    Errors in any single hypothesis are logged and skipped — one bad
    hypothesis cannot crash the batch.

    The full per-hypothesis result dicts (as returned by ``evaluate_factor``)
    are attached to the returned DataFrame's ``.attrs["results"]`` so callers
    that need the full detail (e.g. the CLI below) don't have to re-run the
    suite.
    """
    results: list[dict] = []
    rows:    list[dict] = []

    for hyp in hypotheses:
        try:
            result = evaluate_factor(hyp)
        except Exception:
            log.exception("Evaluation failed for '%s' — skipping.", hyp.signal_name)
            continue

        results.append(result)
        rows.append({
            "signal_name": hyp.signal_name,
            "ic_mean":     result["ic_mean"],
            "ic_p_value":  result["ic_p_value"],
            "sharpe":      result["sharpe"],
            "decay_shape": result["decay_shape"],
            "ff_is_novel": result["ff_is_novel"],
            "verdict":     result["verdict"],
        })

    summary = pd.DataFrame(rows, columns=[
        "signal_name", "ic_mean", "ic_p_value", "sharpe",
        "decay_shape", "ff_is_novel", "verdict",
    ])

    if not summary.empty:
        is_candidate = summary["verdict"].eq(_CANDIDATE)
        # ic_p_value may be None (INSUFFICIENT_DATA) — push those to the end
        sort_key = summary["ic_p_value"].fillna(np.inf)
        summary = (
            summary.assign(_candidate=is_candidate, _sort_key=sort_key)
            .sort_values(["_candidate", "_sort_key"], ascending=[False, True])
            .drop(columns=["_candidate", "_sort_key"])
            .reset_index(drop=True)
        )

    summary.attrs["results"] = results
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _fmt(x, spec: str = ".4f") -> str:
    return format(x, spec) if isinstance(x, (int, float)) and not (isinstance(x, float) and np.isnan(x)) else "n/a"


def _print_detail(result: dict) -> None:
    hyp = result["hypothesis"]
    print(f"\n--- {hyp['signal_name']}  ->  {result['verdict']} "
          f"{'(' + ', '.join(result['failed_gates']) + ')' if result['failed_gates'] else ''} ---")
    print(f"  description     : {hyp['signal_description']}")
    print(f"  horizon         : {hyp['horizon']}    direction: {hyp['direction']:+d}")
    print(f"  n_filings_used  : {result['n_filings_used']}")

    if result["verdict"] == _INSUFFICIENT_DATA:
        print(f"  -> fewer than {_MIN_FILINGS} filings produced a usable signal — suite skipped.")
        return

    print(f"  IC              : mean={_fmt(result['ic_mean'])}  "
          f"t_stat={_fmt(result['ic_t_stat'])}  p_value={_fmt(result['ic_p_value'])}")
    yearly = result["ic_by_year"]
    if yearly is not None and not yearly.empty:
        yearly_str = ", ".join(f"{int(r.year)}: {r.ic:+.3f} (n={int(r.n_obs)})" for r in yearly.itertuples())
        print(f"  IC by year      : {yearly_str}")
    print(f"  Backtest        : total_return={_fmt(result['total_return'])}  "
          f"sharpe={_fmt(result['sharpe'])}")
    print(f"  Decay shape     : {result['decay_shape']}")
    print(f"  Fama-French     : alpha={_fmt(result['ff_alpha'], '.6f')}  "
          f"alpha_pvalue={_fmt(result['ff_alpha_pvalue'])}  is_novel={result['ff_is_novel']}")
    print(f"  Rationale       : {hyp['economic_rationale']}")


if __name__ == "__main__":
    from agents.signal_finder import generate_hypotheses

    W = 78
    print(f"\n{'=' * W}")
    print("  FACTOR EVALUATION PIPELINE")
    print(f"{'=' * W}")

    print("\nGenerating hypotheses from the Signal Finder ...")
    hypotheses = generate_hypotheses(n=5)
    print(f"  {len(hypotheses)} hypotheses generated.")

    print("\nRunning the full backtest + validation suite on each ...")
    summary = evaluate_many(hypotheses)

    print(f"\n{'=' * W}")
    print("  PER-HYPOTHESIS DETAIL")
    print(f"{'=' * W}")
    for result in summary.attrs["results"]:
        _print_detail(result)

    print(f"\n{'=' * W}")
    print("  SUMMARY  (CANDIDATEs first, then by ic_p_value ascending)")
    print(f"{'=' * W}\n")
    print(summary.to_string(index=False))

    n_candidates = int(summary["verdict"].eq(_CANDIDATE).sum())
    print(f"\n  {n_candidates} / {len(summary)} hypotheses are CANDIDATEs.")

    print(f"\n{'=' * W}")
    print("  evaluate_factor.py — OK")
    print(f"{'=' * W}")
