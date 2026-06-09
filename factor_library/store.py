"""
Factor library persistence layer (no LLM).

Stores the full output of backtest.evaluate_factor.evaluate_factor — one
verdict dict per evaluated hypothesis — plus an optional markdown memo, so
that evaluated factors persist across runs and can later be browsed by a
dashboard without re-running the (expensive) backtest + validation suite.

Storage layout
--------------
factor_library/
    factors_index.parquet   one row per factor: cheap, queryable summary columns
    records/{factor_id}.json   the full verdict dict, JSON-serialised
    memos/{factor_id}.md       optional markdown memo (analyst notes, write-up)
    chroma/                    the hypothesis-dedup vector store — a SEPARATE
                               concern (agents/dedup_store.py); never touched here

factor_id is a fresh uuid4 per saved factor — it is the join key between the
index, the JSON records, and the memo files.
"""

from __future__ import annotations

import sys
import json
import uuid
import logging
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_FACTOR_LIBRARY_DIR = Path(__file__).resolve().parent
INDEX_PATH          = _FACTOR_LIBRARY_DIR / "factors_index.parquet"
RECORDS_DIR         = _FACTOR_LIBRARY_DIR / "records"
MEMOS_DIR           = _FACTOR_LIBRARY_DIR / "memos"

# Snapshot layout (tracked by Git; used by Streamlit Cloud where the live
# factor_library/ root is gitignored and therefore absent).
_SNAPSHOT_DIR = _FACTOR_LIBRARY_DIR / "snapshot"


def _snapshot_or(live: Path) -> Path:
    """Return the snapshot-copy of *live* if the snapshot exists, else *live*.

    Reads prefer snapshot/ so the dashboard works on Streamlit Cloud without
    a local data pull.  Writes are never redirected — they always target the
    live paths so local development is unaffected.
    """
    if not _SNAPSHOT_DIR.exists():
        return live
    snap = _SNAPSHOT_DIR / live.relative_to(_FACTOR_LIBRARY_DIR)
    return snap if snap.exists() else live

_INDEX_COLUMNS = [
    "factor_id", "signal_name", "verdict", "ic_mean", "ic_p_value",
    "sharpe", "decay_shape", "ff_is_novel", "evaluated_at", "is_candidate",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JSON serialisation — verdict dicts contain DataFrames / numpy scalars / NaN
# ---------------------------------------------------------------------------

def _to_jsonable(obj):
    """Recursively convert a verdict dict into something json.dumps can handle."""
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        val = float(obj)
        return None if np.isnan(val) else val
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------

def load_index() -> pd.DataFrame:
    """Load the factor index; returns an empty (correctly-columned) DataFrame if missing."""
    path = _snapshot_or(INDEX_PATH)
    if not path.exists():
        return pd.DataFrame(columns=_INDEX_COLUMNS)
    return pd.read_parquet(path)


def _save_index(df: pd.DataFrame) -> None:
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(INDEX_PATH, index=False)


def _index_row(factor_id: str, verdict_dict: dict) -> dict:
    hyp = verdict_dict.get("hypothesis", {}) or {}
    verdict = verdict_dict.get("verdict")
    return {
        "factor_id":    factor_id,
        "signal_name":  hyp.get("signal_name"),
        "verdict":      verdict,
        "ic_mean":      verdict_dict.get("ic_mean"),
        "ic_p_value":   verdict_dict.get("ic_p_value"),
        "sharpe":       verdict_dict.get("sharpe"),
        "decay_shape":  verdict_dict.get("decay_shape"),
        "ff_is_novel":  verdict_dict.get("ff_is_novel"),
        "evaluated_at": pd.Timestamp.now(tz="UTC"),
        "is_candidate": verdict == "CANDIDATE",
    }


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def save_factor(verdict_dict: dict, memo_md: str | None = None) -> str:
    """
    Persist one evaluated factor (the full ``evaluate_factor`` verdict dict)
    plus an optional markdown memo.

    Generates a fresh factor_id, appends a summary row to the index, writes
    the full verdict dict as JSON to ``records/{factor_id}.json``, and — if
    *memo_md* is given — writes it to ``memos/{factor_id}.md``.

    Creates the index and the records/memos subfolders if they don't exist
    yet.  Returns the new factor_id.
    """
    RECORDS_DIR.mkdir(parents=True, exist_ok=True)
    MEMOS_DIR.mkdir(parents=True, exist_ok=True)

    factor_id = str(uuid.uuid4())

    # 1. Append to index
    index   = load_index()
    new_row = pd.DataFrame([_index_row(factor_id, verdict_dict)])
    index   = new_row if index.empty else pd.concat([index, new_row], ignore_index=True)
    _save_index(index)

    # 2. Full record as JSON
    record_path = RECORDS_DIR / f"{factor_id}.json"
    record_path.write_text(json.dumps(_to_jsonable(verdict_dict), indent=2), encoding="utf-8")

    # 3. Optional memo
    if memo_md is not None:
        (MEMOS_DIR / f"{factor_id}.md").write_text(memo_md, encoding="utf-8")

    log.info("Saved factor '%s'  id=%s  verdict=%s%s",
             verdict_dict.get("hypothesis", {}).get("signal_name", "?"),
             factor_id, verdict_dict.get("verdict"),
             "  (+memo)" if memo_md is not None else "")
    return factor_id


def load_record(factor_id: str) -> dict:
    """Load the full verdict dict for *factor_id* from its JSON record."""
    path = _snapshot_or(RECORDS_DIR / f"{factor_id}.json")
    if not path.exists():
        raise FileNotFoundError(f"No record found for factor_id={factor_id!r} at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_memo(factor_id: str) -> str | None:
    """Load the markdown memo for *factor_id*, or None if it has none."""
    path = _snapshot_or(MEMOS_DIR / f"{factor_id}.md")
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def update_memo(factor_id: str, memo_md: str) -> None:
    """Attach (or replace) the markdown memo for an already-saved factor."""
    MEMOS_DIR.mkdir(parents=True, exist_ok=True)
    (MEMOS_DIR / f"{factor_id}.md").write_text(memo_md, encoding="utf-8")
    log.info("Memo updated for factor_id=%s", factor_id)


def list_candidates() -> pd.DataFrame:
    """Return index rows where is_candidate == True (empty DataFrame if none / no index)."""
    index = load_index()
    if index.empty:
        return index
    return index[index["is_candidate"] == True].reset_index(drop=True)  # noqa: E712


# ---------------------------------------------------------------------------
# CLI — round-trip demo with fake verdict dicts
# ---------------------------------------------------------------------------

def _fake_hypothesis(name: str, description: str, rationale: str, direction: int, horizon: str) -> dict:
    return {
        "signal_name": name,
        "signal_description": description,
        "signal_computation": {
            "terms": [{"type": "phrase", "value": "going concern"}],
            "combine": "sum",
            "normalize_by_length": True,
        },
        "direction": direction,
        "horizon": horizon,
        "universe_filter": "all",
        "economic_rationale": rationale,
    }


def _fake_verdict(name: str, verdict: str, ic_mean: float, ic_p_value: float,
                  sharpe: float, decay_shape: str, ff_is_novel: bool,
                  failed_gates: list[str]) -> dict:
    return {
        "hypothesis": _fake_hypothesis(
            name,
            f"Fake demo signal for '{name}'.",
            "Fabricated rationale used only to demonstrate store.py round-tripping.",
            direction=-1, horizon="fwd_ret_21d",
        ),
        "n_filings_used": 1435,
        "ic_mean":    ic_mean,
        "ic_t_stat":  ic_mean / 0.05,
        "ic_p_value": ic_p_value,
        "ic_by_year": pd.DataFrame([
            {"year": 2022, "ic": ic_mean * 0.9, "n_obs": 146},
            {"year": 2023, "ic": ic_mean * 1.1, "n_obs": 146},
        ]),
        "total_return": sharpe * 0.2,
        "sharpe":       sharpe,
        "decay_shape":  decay_shape,
        "ff_alpha":         0.02 if ff_is_novel else -0.01,
        "ff_alpha_pvalue":  0.01 if ff_is_novel else 0.40,
        "ff_is_novel":      ff_is_novel,
        "verdict":      verdict,
        "failed_gates": failed_gates,
    }


if __name__ == "__main__":
    W = 74
    print(f"\n{'=' * W}")
    print("  FACTOR LIBRARY — persistence layer round-trip demo")
    print(f"{'=' * W}")
    print(f"\n  Index path  : {INDEX_PATH}")
    print(f"  Records dir : {RECORDS_DIR}")
    print(f"  Memos dir   : {MEMOS_DIR}")
    print(f"  (chroma/    : untouched — separate dedup concern)")

    fake_candidate = _fake_verdict(
        "demo_candidate_signal", verdict="CANDIDATE",
        ic_mean=0.12, ic_p_value=0.018, sharpe=1.05,
        decay_shape="MONOTONIC_DECAY", ff_is_novel=True, failed_gates=[],
    )
    fake_rejected = _fake_verdict(
        "demo_rejected_signal", verdict="NOT_SIGNIFICANT",
        ic_mean=-0.02, ic_p_value=0.61, sharpe=-0.30,
        decay_shape="SMOOTH", ff_is_novel=False,
        failed_gates=["NOT_SIGNIFICANT", "NOT_NOVEL"],
    )

    print(f"\n--- Saving fake CANDIDATE verdict ---")
    id_candidate = save_factor(
        fake_candidate,
        memo_md="### demo_candidate_signal\n\nLooks promising — significant IC, smooth decay, novel alpha. Worth a closer look.\n",
    )
    print(f"  factor_id = {id_candidate}")

    print(f"\n--- Saving fake NOT_SIGNIFICANT verdict ---")
    id_rejected = save_factor(
        fake_rejected,
        memo_md="### demo_rejected_signal\n\nNo edge — IC indistinguishable from zero, alpha fully explained by FF3+Mom.\n",
    )
    print(f"  factor_id = {id_rejected}")

    print(f"\n--- load_index() ---")
    index = load_index()
    print(index[["signal_name", "verdict", "ic_mean", "ic_p_value", "sharpe", "is_candidate"]].to_string(index=False))

    print(f"\n--- list_candidates() ---")
    candidates = list_candidates()
    print(candidates[["signal_name", "verdict", "ic_mean", "ic_p_value", "is_candidate"]].to_string(index=False))

    print(f"\n--- load_record({id_candidate[:8]}...) ---")
    record = load_record(id_candidate)
    print(f"  signal_name : {record['hypothesis']['signal_name']}")
    print(f"  verdict     : {record['verdict']}")
    print(f"  ic_by_year  : {record['ic_by_year']}")

    print(f"\n--- load_memo({id_candidate[:8]}...) ---")
    print(f"  {load_memo(id_candidate)!r}")

    print(f"\n--- update_memo({id_rejected[:8]}...) then reload ---")
    update_memo(id_rejected, "### demo_rejected_signal (revised)\n\nConfirmed dead end — archiving.\n")
    print(f"  {load_memo(id_rejected)!r}")

    print(f"\n  Total factors in index : {len(load_index())}")
    print(f"  Candidates             : {len(list_candidates())}")

    print(f"\n{'=' * W}")
    print("  store.py — OK")
    print(f"{'=' * W}")
