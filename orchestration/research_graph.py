"""
LangGraph orchestration — wires the existing agents/backtest/validation/
persistence components into a single straight-line research cycle:

    feedback -> generate -> dedup -> evaluate -> report -> persist -> summary

Every node is a thin wrapper around an existing, already-tested function —
no new logic, only wiring. Per-hypothesis failures are caught and logged so
one bad hypothesis can't crash the run; whenever a node drops an item it
drops it from every parallel list (hypotheses/evaluated/memos) so they stay
aligned by index for the nodes downstream.

The feedback_node closes the loop across runs: it reads the factor library's
index for past CANDIDATEs (winners) and rejections (losers), and generate_node
passes them to generate_hypotheses so the LLM can lean toward themes that have
worked and away from patterns that haven't. On a cold start (empty/missing
index) both lists are empty and the prompt is unchanged.

(Conditional edges within a single cycle are a later step — this graph is
intentionally linear.)
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agents.hypothesis_schema import FactorHypothesis
from agents.signal_finder import generate_hypotheses
from agents.dedup_store import is_duplicate, add_hypothesis
from agents.report_writer import write_memo
from backtest.evaluate_factor import evaluate_factor
from factor_library.store import save_factor, load_index, load_record

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Below this many survivors, dedup has effectively wiped out the batch — worth
# one retry of generate_node with a stronger "be substantially different" nudge
# before we accept a (rare) zero-survivor cycle and move on.
_MIN_SURVIVORS      = 2
_MAX_DEDUP_RETRIES  = 1

_RETRY_INSTRUCTION = (
    "IMPORTANT — RETRY: your previous batch of hypotheses were ALL near-duplicates "
    "of ideas already in the research library; none of them survived deduplication. "
    "This is a one-shot retry — you MUST produce genuinely novel ideas this time. "
    "Do not resubmit variants of the same themes (supply chain, litigation, "
    "restructuring, regulation, uncertainty, or whatever else appeared in the prior "
    "failures listed below) with a different phrase or LM category bolted on — that "
    "is exactly what got rejected last time. Pick DIFFERENT linguistic dimensions "
    "entirely: e.g. forward-looking vs. backward-looking language, hedging/commitment "
    "language, growth or expansion language, workforce or liquidity language, or "
    "phrase pairings that have never been combined before."
)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class ResearchState(TypedDict):
    past_winners:      list[dict]
    past_losers:       list[dict]
    hypotheses:        list[FactorHypothesis]
    evaluated:         list[dict]
    memos:             list[str]
    factor_ids:        list[str]
    summary:           dict
    dedup_retry_count: int
    dedup_route:       str


# ---------------------------------------------------------------------------
# Nodes — each wraps one existing function; no new logic
# ---------------------------------------------------------------------------

def _prior_item_from_index_row(row) -> dict:
    """
    Build the {signal_name, signal_description, economic_rationale, ic_mean,
    verdict} dict generate_hypotheses expects, combining the cheap index row
    (signal_name/ic_mean/verdict) with the full record (description/rationale,
    only available via load_record).
    """
    item = {
        "signal_name":        row["signal_name"],
        "ic_mean":            row["ic_mean"],
        "verdict":            row["verdict"],
        "signal_description": "",
        "economic_rationale": "",
    }
    try:
        hyp = load_record(row["factor_id"]).get("hypothesis", {}) or {}
        item["signal_description"] = hyp.get("signal_description", "")
        item["economic_rationale"] = hyp.get("economic_rationale", "")
    except Exception:
        log.exception("feedback_node: failed to load record for '%s' (factor_id=%s) — using bare metadata",
                      row["signal_name"], row["factor_id"])
    return item


def feedback_node(state: ResearchState) -> dict:
    try:
        index = load_index()
    except Exception:
        log.exception("feedback_node: failed to load factor index — proceeding cold-start")
        index = None

    past_winners: list[dict] = []
    past_losers:  list[dict] = []

    if index is not None and not index.empty:
        winners_df = (
            index[index["is_candidate"] == True]                       # noqa: E712
            .assign(_abs_ic=lambda df: df["ic_mean"].abs())
            .sort_values("_abs_ic", ascending=False)
            .head(5)
        )
        losers_df = (
            index[index["is_candidate"] == False]                      # noqa: E712
            .sort_values("evaluated_at", ascending=False)
            .head(10)
        )
        past_winners = [_prior_item_from_index_row(row) for _, row in winners_df.iterrows()]
        past_losers  = [_prior_item_from_index_row(row) for _, row in losers_df.iterrows()]

    log.info("feedback_node: loaded %d winners, %d losers", len(past_winners), len(past_losers))
    return {"past_winners": past_winners, "past_losers": past_losers}


def generate_node(state: ResearchState) -> dict:
    past_winners = state.get("past_winners") or []
    past_losers  = state.get("past_losers")  or []
    retry_count  = state.get("dedup_retry_count", 0)

    extra_instruction = _RETRY_INSTRUCTION if retry_count > 0 else None
    if extra_instruction:
        log.warning("generate_node: dedup retry %d/%d — appending stronger "
                    "'be substantially different' instruction", retry_count, _MAX_DEDUP_RETRIES)

    hypotheses = generate_hypotheses(n=3, past_winners=past_winners, past_losers=past_losers,
                                     extra_instruction=extra_instruction)
    log.info("generate_node: received %d winners / %d losers as context (retry=%d), produced %d hypotheses",
             len(past_winners), len(past_losers), retry_count, len(hypotheses))

    summary = dict(state.get("summary") or {})
    summary["n_generated"] = len(hypotheses)
    return {"hypotheses": hypotheses, "summary": summary}


def dedup_node(state: ResearchState) -> dict:
    hypotheses = state["hypotheses"]
    survivors: list[FactorHypothesis] = []

    for hyp in hypotheses:
        try:
            dup, info = is_duplicate(hyp)
        except Exception:
            log.exception("dedup_node: dedup check failed for '%s' — keeping it", hyp.signal_name)
            survivors.append(hyp)
            continue

        if dup:
            log.info("dedup_node: '%s' ~ '%s' (similarity=%.4f) — dropping as duplicate",
                     hyp.signal_name, info["matched_name"], info["similarity"])
        else:
            survivors.append(hyp)

    log.info("dedup_node: received %d hypotheses, %d survived dedup",
             len(hypotheses), len(survivors))

    summary = dict(state.get("summary") or {})
    summary["n_after_dedup"] = len(survivors)

    retry_count = state.get("dedup_retry_count", 0)
    result = {"hypotheses": survivors, "summary": summary}

    if len(survivors) < _MIN_SURVIVORS and retry_count < _MAX_DEDUP_RETRIES:
        result["dedup_retry_count"] = retry_count + 1
        result["dedup_route"] = "retry"
        log.warning("dedup_node: only %d/%d survived dedup (< %d) — triggering retry %d/%d "
                    "of generate_node with a stronger instruction",
                    len(survivors), len(hypotheses), _MIN_SURVIVORS, retry_count + 1, _MAX_DEDUP_RETRIES)
    else:
        result["dedup_retry_count"] = retry_count
        result["dedup_route"] = "proceed"
        if len(survivors) < _MIN_SURVIVORS and retry_count >= _MAX_DEDUP_RETRIES:
            log.warning("dedup_node: only %d/%d survived dedup after %d retr%s — "
                        "proceeding anyway (a low/zero-survivor cycle is a valid, if rare, outcome)",
                        len(survivors), len(hypotheses), retry_count,
                        "y" if retry_count == 1 else "ies")

    return result


def _route_after_dedup(state: ResearchState) -> str:
    return "generate" if state.get("dedup_route") == "retry" else "evaluate"


def evaluate_node(state: ResearchState) -> dict:
    hypotheses = state["hypotheses"]
    survivors: list[FactorHypothesis] = []
    evaluated: list[dict] = []

    for hyp in hypotheses:
        try:
            verdict = evaluate_factor(hyp)
        except Exception:
            log.exception("evaluate_node: evaluation failed for '%s' — skipping", hyp.signal_name)
            continue
        survivors.append(hyp)
        evaluated.append(verdict)

    n_candidates = sum(1 for v in evaluated if v["verdict"] == "CANDIDATE")
    log.info("evaluate_node: received %d hypotheses, evaluated %d (%d CANDIDATE)",
             len(hypotheses), len(evaluated), n_candidates)

    summary = dict(state.get("summary") or {})
    summary["n_evaluated"]  = len(evaluated)
    summary["n_candidates"] = n_candidates
    return {"hypotheses": survivors, "evaluated": evaluated, "summary": summary}


def report_node(state: ResearchState) -> dict:
    hypotheses = state["hypotheses"]
    evaluated  = state["evaluated"]
    surv_hyps:     list[FactorHypothesis] = []
    surv_verdicts: list[dict] = []
    memos:         list[str] = []

    for hyp, verdict in zip(hypotheses, evaluated):
        try:
            memo = write_memo(verdict)
        except Exception:
            log.exception("report_node: memo writing failed for '%s' — skipping", hyp.signal_name)
            continue
        surv_hyps.append(hyp)
        surv_verdicts.append(verdict)
        memos.append(memo)

    log.info("report_node: received %d evaluated factors, wrote %d memos",
             len(evaluated), len(memos))
    return {"hypotheses": surv_hyps, "evaluated": surv_verdicts, "memos": memos}


def persist_node(state: ResearchState) -> dict:
    hypotheses = state["hypotheses"]
    evaluated  = state["evaluated"]
    memos      = state["memos"]
    factor_ids: list[str] = []

    for hyp, verdict, memo in zip(hypotheses, evaluated, memos):
        try:
            factor_id = save_factor(verdict, memo_md=memo)
            add_hypothesis(hyp, extra_metadata={
                "verdict":   verdict["verdict"],
                "factor_id": factor_id,
            })
        except Exception:
            log.exception("persist_node: persistence failed for '%s' — skipping", hyp.signal_name)
            continue
        factor_ids.append(factor_id)

    log.info("persist_node: received %d (verdict, memo) pairs, persisted %d factors",
             len(evaluated), len(factor_ids))
    return {"factor_ids": factor_ids}


def summary_node(state: ResearchState) -> dict:
    factor_ids = state["factor_ids"]
    summary = dict(state.get("summary") or {})
    summary["factor_ids"] = factor_ids

    W = 78
    print(f"\n{'=' * W}")
    print("  RESEARCH CYCLE SUMMARY")
    print(f"{'=' * W}")
    print(f"  Hypotheses generated   : {summary.get('n_generated', 0)}")
    print(f"  Survived dedup         : {summary.get('n_after_dedup', 0)}")
    print(f"  Evaluated              : {summary.get('n_evaluated', 0)}")
    print(f"  CANDIDATEs             : {summary.get('n_candidates', 0)}")
    print(f"  Factors saved          : {len(factor_ids)}")
    for fid in factor_ids:
        print(f"    - {fid}")
    print(f"{'=' * W}")

    log.info("summary_node: cycle complete — %s", summary)
    return {"summary": summary}


# ---------------------------------------------------------------------------
# Graph assembly
#
#   feedback -> generate -> dedup -+-> evaluate -> report -> persist -> summary
#                  ^                |
#                  +--- (retry, capped at _MAX_DEDUP_RETRIES) ---+
#
# dedup -> generate is a conditional edge: dedup_node sets dedup_route to
# "retry" (and bumps dedup_retry_count) only when fewer than _MIN_SURVIVORS
# hypotheses survived AND the retry budget isn't spent yet; otherwise it sets
# "proceed" and the cycle continues normally — including the (rare) case of
# proceeding with zero surviving hypotheses after the retry is exhausted.
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("feedback", feedback_node)
    graph.add_node("generate", generate_node)
    graph.add_node("dedup",    dedup_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("report",   report_node)
    graph.add_node("persist",  persist_node)
    graph.add_node("summary",  summary_node)

    graph.add_edge(START,      "feedback")
    graph.add_edge("feedback", "generate")
    graph.add_edge("generate", "dedup")
    graph.add_conditional_edges("dedup", _route_after_dedup, {
        "generate": "generate",
        "evaluate": "evaluate",
    })
    graph.add_edge("evaluate", "report")
    graph.add_edge("report",   "persist")
    graph.add_edge("persist",  "summary")
    graph.add_edge("summary",  END)

    return graph.compile()


# ---------------------------------------------------------------------------
# CLI — run one research cycle end-to-end
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    W = 78
    print(f"\n{'=' * W}")
    print("  RESEARCH GRAPH — one end-to-end cycle "
          "(feedback -> generate -> dedup -> evaluate -> report -> persist -> summary)")
    print(f"{'=' * W}")

    app = build_graph()

    initial_state: ResearchState = {
        "past_winners":      [],
        "past_losers":       [],
        "hypotheses":        [],
        "evaluated":         [],
        "memos":             [],
        "factor_ids":        [],
        "summary":           {},
        "dedup_retry_count": 0,
        "dedup_route":       "",
    }

    final_state = app.invoke(initial_state)

    print(f"\n{'=' * W}")
    print("  research_graph.py — OK")
    print(f"{'=' * W}")
