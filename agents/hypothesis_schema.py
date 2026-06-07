"""
FactorHypothesis schema — the structured contract between the LLM Signal Finder
agent and the signal-computation layer.

Every hypothesis the LLM proposes must validate against this schema before
any computation is attempted.  Invalid proposals are rejected with a clear
error so the LLM can retry.

Design notes
------------
* signal_computation is a STRUCTURED spec (not free text) so the
  computation layer can execute it mechanically without re-parsing prose.
* direction (+1 / -1) encodes the expected sign of the relationship so
  downstream IC and backtest code always sees "high signal_value = bullish".
* Strict validators on direction, horizon, and universe_filter prevent
  silent logic errors that would be hard to debug later in the pipeline.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError

# ---------------------------------------------------------------------------
# Term spec — one item in the signal_computation.terms list
# ---------------------------------------------------------------------------

_LM_CATEGORIES = frozenset(
    {"LM_negative", "LM_positive", "LM_uncertainty", "LM_litigious"}
)


class TermSpec(BaseModel):
    """One term to measure in the filing text."""

    type:  Literal["phrase", "lm_category"]
    value: str = Field(..., description=(
        "For type='phrase'   : the text to count (e.g. 'going concern').\n"
        "For type='lm_category': one of LM_negative, LM_positive, "
        "LM_uncertainty, LM_litigious."
    ))

    @model_validator(mode="after")
    def _check_value(self) -> "TermSpec":
        if self.type == "lm_category":
            if self.value not in _LM_CATEGORIES:
                raise ValueError(
                    f"lm_category value must be one of {sorted(_LM_CATEGORIES)}, "
                    f"got {self.value!r}."
                )
        else:  # phrase
            if not self.value.strip():
                raise ValueError("phrase value must be a non-empty string.")
        return self


# ---------------------------------------------------------------------------
# Signal spec — how to combine the terms into one number
# ---------------------------------------------------------------------------

class SignalSpec(BaseModel):
    """Structured computation specification for a factor signal."""

    terms: list[TermSpec] = Field(..., min_length=1)

    combine: Literal["sum", "mean", "ratio"] = Field(
        default="sum",
        description=(
            "How to merge multiple term counts into a single value.\n"
            "  'sum'   : add all counts.\n"
            "  'mean'  : average all counts.\n"
            "  'ratio' : terms[0] / terms[1]  (exactly 2 terms required)."
        ),
    )

    normalize_by_length: bool = Field(
        default=True,
        description=(
            "Divide the combined count by the filing's total word count. "
            "Prevents long filings from dominating purely because of length."
        ),
    )

    @model_validator(mode="after")
    def _ratio_needs_two_terms(self) -> "SignalSpec":
        if self.combine == "ratio" and len(self.terms) != 2:
            raise ValueError(
                f"combine='ratio' requires exactly 2 terms; "
                f"got {len(self.terms)}."
            )
        return self


# ---------------------------------------------------------------------------
# Top-level hypothesis
# ---------------------------------------------------------------------------

_VALID_HORIZONS = frozenset({"fwd_ret_5d", "fwd_ret_21d", "fwd_ret_63d"})


class FactorHypothesis(BaseModel):
    """
    A fully-specified, machine-executable factor hypothesis.

    All fields are validated on construction; a ValidationError is raised
    for any invalid input so the LLM can receive a clear retry prompt.
    """

    signal_name: str = Field(
        ...,
        description="Short snake_case identifier, e.g. 'uncertainty_tone'.",
    )
    signal_description: str = Field(
        ...,
        description="One sentence describing what the signal measures.",
    )
    signal_computation: SignalSpec
    direction: int = Field(
        ...,
        description=(
            "+1 if higher signal_value predicts POSITIVE returns; "
            "-1 if higher raw count predicts NEGATIVE returns "
            "(the computation layer applies this sign flip)."
        ),
    )
    horizon: str = Field(
        ...,
        description="Forward-return column to target; one of fwd_ret_5d/21d/63d.",
    )
    universe_filter: str = Field(
        default="all",
        description="Stock universe filter. Only 'all' is currently supported.",
    )
    economic_rationale: str = Field(
        ...,
        description=(
            "Non-empty rationale explaining WHY this signal should predict returns. "
            "A hypothesis with no rationale is automatically rejected."
        ),
    )

    # ------------------------------------------------------------------
    # Field validators
    # ------------------------------------------------------------------

    @field_validator("signal_name")
    @classmethod
    def _name_snake_case(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("signal_name must not be empty.")
        if " " in v:
            raise ValueError(
                f"signal_name must not contain spaces (use snake_case); "
                f"got {v!r}."
            )
        return v

    @field_validator("direction")
    @classmethod
    def _direction_binary(cls, v: int) -> int:
        if v not in (1, -1):
            raise ValueError(
                f"direction must be exactly +1 or -1; got {v}."
            )
        return v

    @field_validator("horizon")
    @classmethod
    def _horizon_valid(cls, v: str) -> str:
        if v not in _VALID_HORIZONS:
            raise ValueError(
                f"horizon must be one of {sorted(_VALID_HORIZONS)}; "
                f"got {v!r}."
            )
        return v

    @field_validator("universe_filter")
    @classmethod
    def _universe_all_only(cls, v: str) -> str:
        if v != "all":
            raise ValueError(
                f"universe_filter must be 'all' (the only supported value); "
                f"got {v!r}."
            )
        return v

    @field_validator("economic_rationale")
    @classmethod
    def _rationale_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError(
                "economic_rationale must not be empty — a hypothesis "
                "without a rationale is rejected."
            )
        return v


# ---------------------------------------------------------------------------
# Parse helper (for the LLM retry loop)
# ---------------------------------------------------------------------------

def parse_hypothesis(data: dict) -> tuple[FactorHypothesis | None, str | None]:
    """
    Attempt to construct a FactorHypothesis from a raw dict.

    Returns
    -------
    (hypothesis, None)      on success
    (None, error_message)   on ValidationError — the message is formatted
                            for feeding back to the LLM as a retry prompt.
    """
    try:
        return FactorHypothesis(**data), None
    except ValidationError as exc:
        lines = ["Validation failed — fix the following fields:"]
        for err in exc.errors():
            loc = " -> ".join(str(x) for x in err["loc"]) or "(root)"
            lines.append(f"  [{loc}]  {err['msg']}")
        return None, "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI — demonstrate valid and invalid construction
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    W = 60

    # ---- Valid hypothesis ----
    valid_dict = {
        "signal_name": "uncertainty_tone",
        "signal_description": (
            "Measures the fraction of uncertainty-category words "
            "in 10-K filings."
        ),
        "signal_computation": {
            "terms": [{"type": "lm_category", "value": "LM_uncertainty"}],
            "combine": "sum",
            "normalize_by_length": True,
        },
        "direction": -1,
        "horizon": "fwd_ret_21d",
        "economic_rationale": (
            "Filings with more uncertainty language signal management's "
            "concern about future performance, predicting negative near-term returns."
        ),
    }

    print(f"\n{'=' * W}")
    print("  TEST 1: Valid hypothesis")
    print(f"{'=' * W}")
    hyp, err = parse_hypothesis(valid_dict)
    if hyp:
        print(f"  RESULT  : Parsed successfully")
        print(f"  name    : {hyp.signal_name}")
        print(f"  horizon : {hyp.horizon}")
        print(f"  direction: {hyp.direction:+d}")
        print(f"  terms   : {[t.model_dump() for t in hyp.signal_computation.terms]}")
        print(f"  combine : {hyp.signal_computation.combine}")
        print(f"  norm    : {hyp.signal_computation.normalize_by_length}")
    else:
        print(f"  ERROR (unexpected): {err}")

    # ---- Invalid hypothesis: direction=2, name has spaces ----
    invalid_dict = {
        "signal_name": "bad factor name",        # spaces not allowed
        "signal_description": "a test signal",
        "signal_computation": {
            "terms": [{"type": "phrase", "value": "going concern"}],
        },
        "direction": 2,                           # must be +1 or -1
        "horizon": "fwd_ret_99d",                 # not a valid horizon
        "economic_rationale": "test rationale",
    }

    print(f"\n{'=' * W}")
    print("  TEST 2: Invalid hypothesis (direction=2, bad name, bad horizon)")
    print(f"{'=' * W}")
    _, err = parse_hypothesis(invalid_dict)
    if err:
        print("  RESULT  : ValidationError raised (expected)")
        for line in err.splitlines():
            print(f"  {line}")
    else:
        print("  ERROR: unexpectedly parsed as valid — this is a bug.")

    print(f"\n{'=' * W}")
    print("  hypothesis_schema.py — OK")
    print(f"{'=' * W}")
