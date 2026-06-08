"""
Report Writer — LLM agent (Groq / langchain-groq).

Turns an evaluate_factor verdict dict into a clean markdown equity-research
memo.  Numbers are the load-bearing part of any factor memo, so they are
NEVER handed to the LLM to transcribe: the code builds the "Hypothesis &
Rationale", "Metrics", and "Verdict" sections directly from the verdict dict
with f-strings, and the LLM is asked only for the surrounding NARRATIVE prose
(Summary, Interpretation, Recommendation) — framing and explanation around
numbers it never gets to restate. The final memo is assembled by the code,
splicing the LLM's prose between the code-built sections, so a hallucinated
statistic is structurally impossible.
"""

from __future__ import annotations

import sys
import json
import logging
from pathlib import Path

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import GROQ_API_KEY, GROQ_MODEL_REPORT
from agents.hypothesis_schema import FactorHypothesis
from agents.signal_finder import generate_hypotheses
from backtest.evaluate_factor import evaluate_factor
from factor_library.store import save_factor

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
# Formatting helpers (code-built sections — numbers live ONLY here)
# ---------------------------------------------------------------------------

def _fmt(x, spec: str = ".4f") -> str:
    """Render a verdict-dict number for markdown; None/NaN -> 'n/a'."""
    if x is None:
        return "n/a"
    try:
        if isinstance(x, float) and x != x:   # NaN
            return "n/a"
        return format(x, spec)
    except (TypeError, ValueError):
        return str(x)


def _render_terms(hyp: dict) -> str:
    spec  = hyp["signal_computation"]
    terms = ", ".join(f"`{t['type']}:{t['value']}`" for t in spec["terms"])
    norm  = "normalized by filing length" if spec["normalize_by_length"] else "raw count"
    return f"{spec['combine']}({terms}), {norm}"


def _build_hypothesis_section(hyp: dict) -> str:
    return (
        "## Hypothesis & Rationale\n\n"
        f"- **Description:** {hyp['signal_description']}\n"
        f"- **Computation:** {_render_terms(hyp)}\n"
        f"- **Direction:** {hyp['direction']:+d}    **Horizon:** {hyp['horizon']}    "
        f"**Universe:** {hyp['universe_filter']}\n"
        f"- **Economic rationale:** {hyp['economic_rationale']}\n"
    )


def _build_metrics_section(v: dict) -> str:
    lines = [
        "## Metrics\n",
        "| Metric | Value |",
        "|---|---|",
        f"| Filings used | {v['n_filings_used']} |",
        f"| IC mean | {_fmt(v['ic_mean'])} |",
        f"| IC t-stat | {_fmt(v['ic_t_stat'])} |",
        f"| IC p-value | {_fmt(v['ic_p_value'])} |",
        f"| Sharpe | {_fmt(v['sharpe'])} |",
        f"| Total return | {_fmt(v['total_return'], '.2%')} |",
        f"| Decay shape | {v['decay_shape'] if v['decay_shape'] is not None else 'n/a'} |",
        f"| Fama-French alpha | {_fmt(v['ff_alpha'], '.6f')} |",
        f"| Fama-French alpha p-value | {_fmt(v['ff_alpha_pvalue'])} |",
        f"| Fama-French is_novel | {v['ff_is_novel'] if v['ff_is_novel'] is not None else 'n/a'} |",
    ]

    yearly = v.get("ic_by_year")
    if yearly is not None:
        # yearly is a DataFrame coming straight out of evaluate_factor, or a
        # list-of-records if the dict has been round-tripped through JSON.
        rows = yearly.to_dict(orient="records") if hasattr(yearly, "to_dict") else yearly
        if rows:
            lines.append("\n**IC by year**\n")
            lines.append("| Year | IC | n obs |")
            lines.append("|---|---|---|")
            for r in rows:
                lines.append(f"| {int(r['year'])} | {_fmt(r['ic'], '+.4f')} | {int(r['n_obs'])} |")

    return "\n".join(lines) + "\n"


def _build_verdict_section(v: dict) -> str:
    gates = v.get("failed_gates") or []
    lines = [
        "## Verdict\n",
        f"**{v['verdict']}**\n",
    ]
    if gates:
        lines.append("Failed gate(s):")
        for g in gates:
            lines.append(f"- `{g}`")
    else:
        lines.append("No gates failed — this factor cleared every check in the suite.")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# LLM prompt construction — narrative-only, no numbers handed over
# ---------------------------------------------------------------------------

def _build_system_prompt() -> str:
    return """You are an equity-research analyst writing the NARRATIVE portions of a \
factor research memo. You will be given a JSON summary describing a quantitative \
factor's hypothesis and the high-level VERDICT it received from a validation suite \
(but NOT its detailed statistics — those are written directly into the memo by other \
code, not by you).

Your job is to write ONLY plain-English framing and interpretation around that verdict \
— never to state, estimate, or imply any specific number, statistic, or metric value. \
If you find yourself wanting to write a number, don't: describe it qualitatively instead \
(e.g. "the factor's predictive power was not statistically distinguishable from noise" \
rather than "p = 0.4").

Respond with ONLY a JSON object with exactly these three string fields — no markdown \
fences, no commentary outside the object:
{
  "summary": "2-3 sentences: what this factor measures and the basic idea behind it",
  "interpretation": "what the verdict means in plain English. If the factor was \
REJECTED, explain WHY in terms a researcher would understand (e.g. a SPIKE decay shape \
suggests the apparent edge is concentrated suspiciously in one horizon and is likely an \
artefact rather than a real, persistent effect; failing the novelty check means the \
apparent return is just exposure to already-known risk factors in disguise). If the \
factor was a CANDIDATE, explain that it cleared every check in the suite, but caveat \
that this is one validation pass on a fixed historical sample and out-of-sample \
performance is never guaranteed.",
  "recommendation": "a measured recommendation. For a CANDIDATE: suggest cautious next \
steps (e.g. paper-trading, out-of-sample testing, combining with other factors) rather \
than wholesale adoption. For a REJECTED factor: state plainly that it should not be used \
as-is and briefly why, but you may suggest whether the underlying idea seems worth \
revisiting in modified form or abandoning entirely."
}"""


def _build_user_prompt(hyp: dict, v: dict) -> str:
    """
    Compact context for the LLM — verdict + hypothesis framing only.
    Deliberately OMITS every numeric statistic (ic_mean, p_value, sharpe,
    alpha, ...): the LLM cannot misstate a number it never receives.
    """
    context = {
        "signal_name":        hyp["signal_name"],
        "signal_description": hyp["signal_description"],
        "economic_rationale": hyp["economic_rationale"],
        "direction":          hyp["direction"],
        "horizon":            hyp["horizon"],
        "verdict":            v["verdict"],
        "failed_gates":       v.get("failed_gates") or [],
        "decay_shape":        v.get("decay_shape"),
        "ff_is_novel":        v.get("ff_is_novel"),
        "n_filings_used":     v.get("n_filings_used"),
    }
    return (
        "Write the narrative sections for this factor's research memo.\n\n"
        f"Context (no numeric statistics included on purpose):\n{json.dumps(context, indent=2)}\n\n"
        "Output ONLY the JSON object described in the system prompt."
    )


# ---------------------------------------------------------------------------
# Robust LLM call
# ---------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    before_sleep=before_sleep_log(log, logging.WARNING),
    reraise=True,
)
def _invoke(llm: ChatGroq, messages: list) -> str:
    return llm.invoke(messages).content


def _extract_json_object(text: str) -> dict:
    """Parse the LLM's {summary, interpretation, recommendation} object, tolerating fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        cleaned = cleaned[len("json"):].strip() if cleaned.lower().startswith("json") else cleaned.strip()
    return json.loads(cleaned)


_FALLBACK_PROSE = {
    "summary":        "(Narrative summary unavailable — the LLM response could not be parsed.)",
    "interpretation": "(Narrative interpretation unavailable — the LLM response could not be parsed. "
                      "See the Verdict and Metrics sections above for the factual record.)",
    "recommendation": "(Narrative recommendation unavailable — the LLM response could not be parsed. "
                      "Treat any unverified factor as not actionable until reviewed manually.)",
}


def _generate_prose(hyp: dict, v: dict) -> dict:
    """Call Groq for the narrative sections; fall back to placeholder text on failure."""
    llm = ChatGroq(model=GROQ_MODEL_REPORT, api_key=GROQ_API_KEY, temperature=0.4)
    messages = [
        SystemMessage(content=_build_system_prompt()),
        HumanMessage(content=_build_user_prompt(hyp, v)),
    ]
    try:
        raw = _invoke(llm, messages)
        prose = _extract_json_object(raw)
        missing = [k for k in ("summary", "interpretation", "recommendation") if not prose.get(k)]
        if missing:
            raise ValueError(f"missing field(s): {missing}")
        return prose
    except Exception as exc:
        log.warning("Report Writer: could not parse narrative prose (%s) — using fallback text. "
                    "Raw response (truncated): %r", exc, locals().get("raw", "")[:300])
        return dict(_FALLBACK_PROSE)


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def write_memo(verdict_dict: dict) -> str:
    """
    Build a full markdown research memo for one evaluate_factor verdict dict.

    Numbers come ONLY from *verdict_dict*, written verbatim by f-strings in
    ``_build_metrics_section`` / ``_build_verdict_section`` /
    ``_build_hypothesis_section``. The LLM (Groq, GROQ_MODEL_REPORT) supplies
    only the surrounding narrative prose (Summary / Interpretation /
    Recommendation), and is never shown a single statistic — so it cannot
    misstate one.
    """
    hyp = verdict_dict["hypothesis"]
    log.info("Writing memo for '%s' (verdict=%s) ...", hyp["signal_name"], verdict_dict["verdict"])

    prose = _generate_prose(hyp, verdict_dict)

    sections = [
        f"# Factor Research Memo: {hyp['signal_name']}",
        "## Summary\n\n" + prose["summary"],
        _build_hypothesis_section(hyp),
        _build_metrics_section(verdict_dict),
        _build_verdict_section(verdict_dict),
        "## Interpretation\n\n" + prose["interpretation"],
        "## Recommendation\n\n" + prose["recommendation"],
    ]
    return "\n\n".join(sections) + "\n"


# ---------------------------------------------------------------------------
# Integration: evaluate -> write memo -> persist
# ---------------------------------------------------------------------------

def evaluate_and_report(hypothesis: FactorHypothesis) -> tuple[str, dict, str]:
    """
    Run a hypothesis through the full evaluation suite, write its memo, and
    persist both (verdict + memo) to the factor library.

    Returns (factor_id, verdict_dict, memo_markdown).
    """
    verdict = evaluate_factor(hypothesis)
    memo    = write_memo(verdict)
    factor_id = save_factor(verdict, memo_md=memo)
    log.info("evaluate_and_report: '%s' -> factor_id=%s  verdict=%s",
             hypothesis.signal_name, factor_id, verdict["verdict"])
    return factor_id, verdict, memo


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    W = 78
    print(f"\n{'=' * W}")
    print(f"  REPORT WRITER — LLM memo generation  (Groq / {GROQ_MODEL_REPORT})")
    print(f"{'=' * W}")

    print("\nGenerating 2 hypotheses from the Signal Finder ...")
    hypotheses = generate_hypotheses(n=2)
    print(f"  {len(hypotheses)} hypotheses generated.")

    for hyp in hypotheses:
        print(f"\n{'=' * W}")
        print(f"  {hyp.signal_name}")
        print(f"{'=' * W}")

        factor_id, verdict, memo = evaluate_and_report(hyp)

        print(f"\n  factor_id : {factor_id}")
        print(f"  verdict   : {verdict['verdict']}")
        print(f"\n{memo}")

    print(f"\n{'=' * W}")
    print("  report_writer.py — OK")
    print(f"{'=' * W}")
