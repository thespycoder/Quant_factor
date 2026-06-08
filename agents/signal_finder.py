"""
Signal Finder — LLM agent (Groq / langchain-groq).

Proposes structured, machine-executable factor hypotheses (FactorHypothesis,
see agents/hypothesis_schema.py) for the signal-computation layer.  The model
is grounded in measured cross-filing feature statistics
(agents/feature_stats.json, built without any LLM by agents/feature_stats.py)
so it reasons about real, varying distributions rather than guessing — and is
constrained to a small "guided palette" of building blocks that the
computation layer can actually execute mechanically.

Every proposal is validated against the FactorHypothesis schema; invalid
proposals trigger a retry loop that feeds the model its own validation
errors so it can self-correct.
"""

from __future__ import annotations

import sys
import re
import json
import logging
from pathlib import Path

from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import GROQ_API_KEY, GROQ_MODEL_SIGNAL
from agents.hypothesis_schema import FactorHypothesis, parse_hypothesis
from agents.feature_stats import FEATURE_STATS_PATH

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
# Grounding context (feature_stats.json, built by agents/feature_stats.py)
# ---------------------------------------------------------------------------

def _load_feature_stats() -> dict:
    if not FEATURE_STATS_PATH.exists():
        raise FileNotFoundError(
            f"\nFeature stats not found at:\n  {FEATURE_STATS_PATH}\n\n"
            "Build them once with:\n  python agents/feature_stats.py\n"
        )
    return json.loads(FEATURE_STATS_PATH.read_text(encoding="utf-8"))


def _grounding_highlights(stats: dict, top_n: int = 4) -> str:
    """
    Identify the features with the widest cross-sectional spread (highest
    coefficient of variation = std / mean) — these vary most across companies
    and are therefore the most likely to carry cross-sectional signal.
    """
    rows: list[tuple[float, str, str, dict]] = []
    for group, label in (("lm_categories", "LM category"), ("phrases", "phrase")):
        for key, s in stats.get(group, {}).items():
            mean = s.get("mean")
            if mean:
                rows.append((s["std"] / mean, label, key, s))
    rows.sort(key=lambda r: -r[0])

    return "\n".join(
        f"  - {label} '{key}':  mean={s['mean']:.5f}  std={s['std']:.5f}  "
        f"(coefficient of variation = {cv:.2f} — wide spread across filings)"
        for cv, label, key, s in rows[:top_n]
    )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_ALLOWED_BUILDING_BLOCKS = """\
ALLOWED BUILDING BLOCKS — every hypothesis MUST stay strictly within these
(nothing else can be computed by the engine):
  * term {"type": "lm_category", "value": <one of LM_negative, LM_positive,
    LM_uncertainty, LM_litigious>}  — counts words in that Loughran-McDonald list
  * term {"type": "phrase", "value": "<short phrase, 1-3 words>"}  — counts
    occurrences of that exact phrase, e.g. "going concern", "supply chain"
  * combine: "sum" (add term counts) | "mean" (average term counts) |
    "ratio" (terms[0] / terms[1] — REQUIRES exactly 2 terms)
  * normalize_by_length: true | false  — divide by the filing's total word count
  * direction: +1 (higher signal_value predicts POSITIVE forward returns) or
    -1 (higher RAW count predicts NEGATIVE forward returns; the engine flips the sign)
  * horizon: "fwd_ret_5d" | "fwd_ret_21d" | "fwd_ret_63d"
  * universe_filter: "all"  (the only supported value right now)
"""

_OUTPUT_SCHEMA = """\
{
  "signal_name": "snake_case_identifier",
  "signal_description": "one sentence describing what is measured",
  "signal_computation": {
    "terms": [{"type": "lm_category"|"phrase", "value": "..."}, ...],
    "combine": "sum"|"mean"|"ratio",
    "normalize_by_length": true|false
  },
  "direction": 1 | -1,
  "horizon": "fwd_ret_5d"|"fwd_ret_21d"|"fwd_ret_63d",
  "universe_filter": "all",
  "economic_rationale": "concrete causal explanation of WHY this predicts returns"
}"""


def _render_prior_item(item: dict) -> str:
    name      = item.get("signal_name", "?")
    desc      = item.get("signal_description", "")
    rationale = item.get("economic_rationale", "")
    ic_mean   = item.get("ic_mean")
    verdict   = item.get("verdict")
    ic_is_number = isinstance(ic_mean, (int, float)) and not (isinstance(ic_mean, float) and ic_mean != ic_mean)
    ic_str       = f"{ic_mean:+.4f}" if ic_is_number else "n/a"
    return f"  - '{name}' (IC={ic_str}, verdict={verdict}): {desc} Rationale: {rationale}"


def _build_prior_results_section(past_winners: list[dict] | None,
                                 past_losers:  list[dict] | None) -> str:
    """
    Compact "PRIOR RESULTS" block summarising what worked and what didn't in
    earlier research cycles, so the model can build on winners and steer away
    from losers. Returns "" if there is nothing to report (cold start).
    """
    winners = (past_winners or [])[:5]
    losers  = (past_losers  or [])[:10]
    if not winners and not losers:
        return ""

    parts = ["\n=== PRIOR RESULTS — learn from past research cycles ==="]
    if winners:
        parts.append(
            "\nThese past hypotheses were validated as CANDIDATEs with strong IC — "
            "favor SIMILAR underlying themes and economic mechanisms (but propose "
            "genuinely new specs, not copies):"
        )
        parts += [_render_prior_item(w) for w in winners]
    if losers:
        loser_names = ", ".join(f"'{l.get('signal_name', '?')}'" for l in losers)
        parts.append(
            "\nThese past hypotheses FAILED validation (insignificant IC, suspicious "
            "decay, or no novel alpha after controlling for known factors). Each line "
            "is [name] (stats): [description] Rationale: [rationale] — read the names "
            "AND descriptions carefully:"
        )
        parts += [_render_prior_item(l) for l in losers]
        parts.append(
            "\n*** Your new proposals MUST be SUBSTANTIALLY DIFFERENT from every failure "
            f"listed above ({loser_names}) — NOT minor rewordings, synonym swaps, or "
            "variations on the same underlying theme. Swapping the LM category or phrase "
            "on an otherwise-identical spec to one of these is NOT substantially "
            "different and will be rejected as a near-duplicate before it ever reaches "
            "the backtest. Concretely: do not propose new variants of supply chain, "
            "litigation, restructuring, regulation, or uncertainty signals if those "
            "names (or close synonyms) appear above — propose hypotheses on DIFFERENT "
            "linguistic dimensions instead. Explore genuinely new angles: LM-category "
            "combinations not yet tried together, ratios/contrasts between phrase "
            "families that have never been paired (e.g. forward-looking vs. "
            "backward-looking language, hedging/commitment language, growth/expansion "
            "language, liquidity or workforce language), or structurally different "
            "combine/normalize specs built on entirely new phrase sets. ***"
        )
    parts.append("")
    return "\n".join(parts)


def _build_system_prompt(stats: dict,
                         past_winners: list[dict] | None = None,
                         past_losers:  list[dict] | None = None) -> str:
    cov           = stats["coverage"]
    highlights    = _grounding_highlights(stats)
    stats_json    = json.dumps(stats, separators=(",", ":"))
    prior_results = _build_prior_results_section(past_winners, past_losers)

    return f"""You are an expert quantitative researcher whose job is to propose \
TESTABLE, text-based alpha factors derived purely from the language of 10-K annual \
report filings.

=== GROUNDING: measured statistics over the real filing universe ===
Coverage: {cov['n_filings']} filings, {cov['n_tickers']} tickers, \
{cov['date_min']} to {cov['date_max']}.
Full per-feature distribution statistics (density = count / total_word_count, \
computed across every filing) as compact JSON:
{stats_json}

The features below vary the MOST across filings (highest std/mean) and are therefore \
the most likely to actually distinguish companies — and to carry cross-sectional \
return signal. A feature that barely varies across filings cannot predict relative \
returns, so prefer building around features like these (or deliberate combinations \
of them):
{highlights}

=== GUIDED PALETTE ===
{_ALLOWED_BUILDING_BLOCKS}
You are encouraged to COMBINE these blocks in non-obvious ways rather than proposing \
only single-term signals — for example a "ratio" of litigious-language density to \
positive-language density, or the "mean" of two related phrase counts that jointly \
indicate a theme (e.g. "supply chain" + "shortage").

=== ECONOMIC RATIONALE — REQUIRED AND STRICT ===
Every hypothesis MUST include a concrete, specific economic_rationale: one or two \
sentences naming WHO changes their language and WHY (e.g. management hedging, \
disclosure obligations, sentiment leakage), and WHAT that implies about forward \
returns over the chosen horizon. Generic statements such as "this might affect investor \
sentiment" or "language reflects company health" are VAGUE and will be REJECTED — be \
precise about the causal mechanism.

=== HARD CONSTRAINTS ===
  - signal_computation may reference ONLY the allowed building blocks above.
  - NEVER reference financial-statement figures, prices, valuation ratios, analyst \
estimates, macro data, or anything outside of phrase/LM word counts within the filing \
text. If a signal cannot be computed purely from word/phrase counts, it is INVALID — \
do not propose it.
{prior_results}
=== OUTPUT FORMAT — FOLLOW EXACTLY ===
Respond with ONLY a JSON array of hypothesis objects. No prose, no markdown code \
fences, no commentary before or after the array. Each object must have EXACTLY these \
fields, matching this shape:
{_OUTPUT_SCHEMA}
"""


def _build_initial_user_prompt(n: int, extra_instruction: str | None = None) -> str:
    base = (
        f"Propose exactly {n} DISTINCT factor hypotheses. Each must explore a "
        f"genuinely different economic mechanism — do not just swap the LM category "
        f"on an otherwise-identical spec. Output ONLY the JSON array, nothing else."
    )
    if extra_instruction:
        return f"{extra_instruction}\n\n{base}"
    return base


def _build_retry_user_prompt(n: int, n_valid: int, n_total: int, errors: list[str]) -> str:
    feedback = "\n".join(errors)
    return (
        f"{n_valid} of your {n_total} hypotheses passed schema validation; "
        f"the rest failed with these errors:\n{feedback}\n\n"
        f"Return a CORRECTED JSON array of exactly {n} hypotheses that ALL pass "
        f"validation (fix every issue listed above — e.g. stay within the allowed "
        f"building blocks, use a valid horizon, make rationales concrete and specific). "
        f"Output ONLY the JSON array — no prose, no markdown fences."
    )


def _build_json_error_prompt(n: int, parse_err: str) -> str:
    return (
        f"Your response could not be parsed as a JSON array ({parse_err}). "
        f"Reply again with ONLY a valid JSON array of exactly {n} hypothesis objects — "
        f"no prose, no markdown code fences, no trailing commentary."
    )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def _extract_json_array(text: str) -> tuple[list | None, str | None]:
    """Best-effort extraction of a top-level JSON array from an LLM response."""
    cleaned = _FENCE_RE.sub("", text.strip()).strip()

    try:
        obj = json.loads(cleaned)
        if isinstance(obj, list):
            return obj, None
        return None, f"expected a JSON array, got {type(obj).__name__}"
    except json.JSONDecodeError:
        pass

    match = _ARRAY_RE.search(cleaned)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, list):
                return obj, None
            return None, f"expected a JSON array, got {type(obj).__name__}"
        except json.JSONDecodeError as exc:
            return None, f"could not parse JSON array ({exc})"

    return None, "no JSON array found in the response"


# ---------------------------------------------------------------------------
# Robust LLM call (handles Groq rate limits / transient errors)
# ---------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    before_sleep=before_sleep_log(log, logging.WARNING),
    reraise=True,
)
def _invoke(llm: ChatGroq, messages: list) -> str:
    return llm.invoke(messages).content


# ---------------------------------------------------------------------------
# Core entry point
# ---------------------------------------------------------------------------

def generate_hypotheses(n: int = 3, max_retries: int = 3,
                        past_winners: list[dict] | None = None,
                        past_losers:  list[dict] | None = None,
                        extra_instruction: str | None = None) -> list[FactorHypothesis]:
    """
    Ask the Groq-hosted LLM for *n* FactorHypothesis-shaped proposals,
    grounded in agents/feature_stats.json, and validate every one against the
    FactorHypothesis schema.

    *past_winners* / *past_losers* (each a list of dicts with signal_name,
    signal_description, economic_rationale, ic_mean, verdict) let the model
    learn from prior research cycles: it's instructed to favor themes similar
    to past CANDIDATEs and avoid patterns similar to past failures. Pass
    None/[] (the default) for a cold start — the prompt is then identical to
    the no-feedback version.

    *extra_instruction*, if given, is prepended to the initial user prompt —
    e.g. a one-off "your last batch was all duplicates, try harder" nudge from
    a calling orchestration graph's retry path. Pass None for the normal prompt.

    On parse/validation failure, re-prompts (up to *max_retries* additional
    times) including the specific errors so the model can self-correct.
    Returns the most valid hypotheses seen across all attempts (length <= n);
    logs how many succeeded vs. failed.
    """
    stats = _load_feature_stats()
    llm   = ChatGroq(model=GROQ_MODEL_SIGNAL, api_key=GROQ_API_KEY, temperature=0.7)

    log.info("Signal Finder: generating with %d past winner(s), %d past loser(s) as context%s",
             len(past_winners or []), len(past_losers or []),
             "  (+ extra retry instruction)" if extra_instruction else "")

    messages: list = [
        SystemMessage(content=_build_system_prompt(stats, past_winners, past_losers)),
        HumanMessage(content=_build_initial_user_prompt(n, extra_instruction)),
    ]

    best_valid: list[FactorHypothesis] = []

    for attempt in range(1, max_retries + 2):  # initial attempt + max_retries
        log.info("Signal Finder — attempt %d/%d: requesting %d hypotheses ...",
                 attempt, max_retries + 1, n)
        try:
            raw = _invoke(llm, messages)
        except Exception as exc:
            log.error("LLM call failed after retries: %s", exc)
            break

        objs, parse_err = _extract_json_array(raw)
        if parse_err is not None:
            log.warning("Attempt %d: %s", attempt, parse_err)
            log.debug("Raw response (truncated to 500 chars): %r", raw[:500])
            if attempt > max_retries:
                break
            messages += [AIMessage(content=raw), HumanMessage(content=_build_json_error_prompt(n, parse_err))]
            continue

        valid:  list[FactorHypothesis] = []
        errors: list[str] = []
        for i, obj in enumerate(objs, start=1):
            hyp, err = parse_hypothesis(obj if isinstance(obj, dict) else {})
            if hyp is not None:
                valid.append(hyp)
            else:
                name = obj.get("signal_name", "?") if isinstance(obj, dict) else "?"
                errors.append(f"  #{i} ({name}):\n" + "\n".join(f"    {ln}" for ln in err.splitlines()))

        log.info("Attempt %d: %d / %d hypotheses passed validation", attempt, len(valid), len(objs))

        if len(valid) > len(best_valid):
            best_valid = valid

        if len(valid) >= n or not errors or attempt > max_retries:
            break

        messages += [
            AIMessage(content=raw),
            HumanMessage(content=_build_retry_user_prompt(n, len(valid), len(objs), errors)),
        ]

    n_failed = max(n - len(best_valid), 0)
    log.info("Signal Finder done: %d / %d hypotheses valid (%d unrecoverable after retries).",
             len(best_valid), n, n_failed)
    return best_valid[:n]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_hypothesis(idx: int, hyp: FactorHypothesis) -> None:
    spec = hyp.signal_computation
    print(f"\n  [{idx}] {hyp.signal_name}")
    print(f"      description : {hyp.signal_description}")
    print(f"      terms       : {[t.model_dump() for t in spec.terms]}")
    print(f"      combine     : {spec.combine}    normalize_by_length: {spec.normalize_by_length}")
    print(f"      direction   : {hyp.direction:+d}    horizon: {hyp.horizon}")
    print(f"      rationale   : {hyp.economic_rationale}")


if __name__ == "__main__":
    W = 70
    N = 5

    print(f"\n{'=' * W}")
    print(f"  SIGNAL FINDER — LLM hypothesis generation  (Groq / {GROQ_MODEL_SIGNAL})")
    print(f"{'=' * W}")

    hypotheses = generate_hypotheses(n=N)

    print(f"\n  {len(hypotheses)} / {N} requested hypotheses passed validation:")
    for idx, hyp in enumerate(hypotheses, start=1):
        _print_hypothesis(idx, hyp)

    print(f"\n{'=' * W}")
    print("  signal_finder.py — OK")
    print(f"{'=' * W}")
