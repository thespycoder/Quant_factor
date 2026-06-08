"""
LangGraph orchestration — wires the existing agents/backtest/validation/
persistence components into a single straight-line research cycle:

    generate -> dedup -> evaluate -> report -> persist -> summary

Every node is a thin wrapper around an existing, already-tested function —
no new logic, only wiring. Per-hypothesis failures are caught and logged so
one bad hypothesis can't crash the run; whenever a node drops an item it
drops it from every parallel list (hypotheses/evaluated/memos) so they stay
aligned by index for the nodes downstream.

(Conditional edges and the feedback loop are a later step — this graph is
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
from factor_library.store import save_factor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class ResearchState(TypedDict):
    hypotheses: list[FactorHypothesis]
    evaluated:  list[dict]
    memos:      list[str]
    factor_ids: list[str]
    summary:    dict


# ---------------------------------------------------------------------------
# Nodes — each wraps one existing function; no new logic
# ---------------------------------------------------------------------------

def generate_node(state: ResearchState) -> dict:
    hypotheses = generate_hypotheses(n=5)
    log.info("generate_node: received nothing, produced %d hypotheses", len(hypotheses))

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
    return {"hypotheses": survivors, "summary": summary}


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
# Graph assembly — linear: generate -> dedup -> evaluate -> report -> persist -> summary
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("generate", generate_node)
    graph.add_node("dedup",    dedup_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("report",   report_node)
    graph.add_node("persist",  persist_node)
    graph.add_node("summary",  summary_node)

    graph.add_edge(START,      "generate")
    graph.add_edge("generate", "dedup")
    graph.add_edge("dedup",    "evaluate")
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
          "(generate -> dedup -> evaluate -> report -> persist -> summary)")
    print(f"{'=' * W}")

    app = build_graph()

    initial_state: ResearchState = {
        "hypotheses": [],
        "evaluated":  [],
        "memos":      [],
        "factor_ids": [],
        "summary":    {},
    }

    final_state = app.invoke(initial_state)

    print(f"\n{'=' * W}")
    print("  research_graph.py — OK")
    print(f"{'=' * W}")
