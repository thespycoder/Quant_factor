"""
Hypothesis deduplication layer (ChromaDB, local embeddings only).

Purpose: the Signal Finder LLM will happily propose the same underlying idea
worded a dozen different ways ("supply chain disruption density" vs.
"frequency of logistics-disruption language" vs. ...).  Running the full
backtest + validation suite (see backtest/evaluate_factor.py) on each reworded
copy wastes time and pollutes the factor library with near-duplicates.

This module embeds each hypothesis's MEANING (description + economic
rationale + a readable rendering of its computation spec) with ChromaDB's
built-in local sentence-transformer embedding function — no external API, no
API key — and checks new proposals against everything already stored via
cosine similarity.

Storage is a persistent ChromaDB client on disk at factor_library/chroma/, so
the dedup store survives across runs/sessions.
"""

from __future__ import annotations

import sys
import uuid
import logging
from pathlib import Path

import chromadb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agents.hypothesis_schema import FactorHypothesis

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

_FACTOR_LIBRARY_DIR = Path(__file__).resolve().parent.parent / "factor_library"
CHROMA_PATH         = _FACTOR_LIBRARY_DIR / "chroma"

_COLLECTION_NAME = "hypotheses"

# Cosine similarity threshold above which a new hypothesis is considered a
# reworded duplicate of an existing one.
#
# Empirically derived (see __main__ demo) from ChromaDB's local MiniLM
# embedding model: a genuine reworded-duplicate pair scores ~0.76 cosine
# similarity, while a genuinely different hypothesis scores ~0.49. 0.72 sits
# safely between the two — catching paraphrases without rejecting distinct
# ideas. Tunable: re-run the demo if the embedding model or phrasing style
# changes and adjust if the gap shifts.
_DEFAULT_THRESHOLD = 0.72

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Persistent client / collection (module-level singletons)
# ---------------------------------------------------------------------------

_client:     chromadb.ClientAPI | None = None
_collection                            = None


def _get_collection():
    """
    Lazily create the persistent client + collection.

    The collection is explicitly configured for cosine distance
    (``hnsw:space: "cosine"``) so that ``1 - distance`` is a valid cosine
    similarity — Chroma's default space is squared-L2, which would make that
    conversion meaningless.  No embedding_function is passed, so Chroma uses
    its built-in local default (a small sentence-transformer that downloads
    on first use — no API key required).
    """
    global _client, _collection
    if _collection is not None:
        return _collection

    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    _client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    _collection = _client.get_or_create_collection(
        name=_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    log.info("Chroma collection '%s' ready at %s  (%d entries)",
             _COLLECTION_NAME, CHROMA_PATH, _collection.count())
    return _collection


# ---------------------------------------------------------------------------
# Embedding text — what captures a hypothesis's MEANING
# ---------------------------------------------------------------------------

def _render_computation(hypothesis: FactorHypothesis) -> str:
    """Readable one-line rendering of the signal_computation spec."""
    spec  = hypothesis.signal_computation
    terms = ", ".join(f"{t.type}:{t.value}" for t in spec.terms)
    norm  = "normalized by filing length" if spec.normalize_by_length else "raw count"
    return f"{spec.combine} of [{terms}], {norm}"


def _embedding_text(hypothesis: FactorHypothesis) -> str:
    """
    Combine the meaning-bearing fields of a hypothesis into one string.

    Deliberately excludes signal_name (an arbitrary identifier) and
    direction/horizon/universe_filter (mechanical parameters, not "meaning")
    — two hypotheses that measure the same thing but target different
    horizons are still the same underlying idea.
    """
    return (
        f"{hypothesis.signal_description.strip()} "
        f"{hypothesis.economic_rationale.strip()} "
        f"Computation: {_render_computation(hypothesis)}."
    )


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def _nearest_match(hypothesis: FactorHypothesis) -> tuple[str, float] | None:
    """
    Query the collection for the single nearest stored entry and convert its
    distance to a cosine similarity.  Returns ``(matched_name, similarity)``,
    or ``None`` if the collection is empty.

    Exposed (semi-)publicly so callers — e.g. the CLI demo below — can
    inspect the raw similarity number even when it falls below the
    duplicate threshold (handy for picking a sensible threshold).
    """
    collection = _get_collection()
    if collection.count() == 0:
        return None

    text   = _embedding_text(hypothesis)
    result = collection.query(query_texts=[text], n_results=1, include=["metadatas", "distances"])

    distances = result["distances"][0]
    metadatas = result["metadatas"][0]
    if not distances:
        return None

    # Cosine space: similarity = 1 - distance
    similarity   = 1.0 - float(distances[0])
    matched_name = metadatas[0].get("signal_name", "?")
    return matched_name, similarity


def is_duplicate(hypothesis: FactorHypothesis, threshold: float = _DEFAULT_THRESHOLD) -> tuple[bool, dict | None]:
    """
    Check whether *hypothesis* means roughly the same thing as something
    already stored.

    Returns
    -------
    (True,  {"matched_name": str, "similarity": float})  if the nearest
        existing entry's cosine similarity >= threshold
    (False, None)                                         otherwise, including
        when the collection is empty
    """
    nearest = _nearest_match(hypothesis)
    if nearest is None:
        return False, None
    matched_name, similarity = nearest

    if similarity >= threshold:
        log.info("Duplicate match: '%s' ~ '%s'  (similarity=%.4f >= %.2f)",
                 hypothesis.signal_name, matched_name, similarity, threshold)
        return True, {"matched_name": matched_name, "similarity": similarity}

    log.info("No duplicate for '%s'  (nearest='%s', similarity=%.4f < %.2f)",
             hypothesis.signal_name, matched_name, similarity, threshold)
    return False, None


def add_hypothesis(hypothesis: FactorHypothesis, extra_metadata: dict | None = None) -> str:
    """
    Store *hypothesis* in the collection under a fresh uuid id.

    Metadata always includes signal_name / direction / horizon (handy for
    ``list_all`` and for filtering later); ``extra_metadata`` (e.g. an
    evaluation verdict) is merged in on top.
    """
    collection = _get_collection()

    hyp_id   = str(uuid.uuid4())
    metadata = {
        "signal_name": hypothesis.signal_name,
        "direction":   hypothesis.direction,
        "horizon":     hypothesis.horizon,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    collection.add(
        ids=[hyp_id],
        documents=[_embedding_text(hypothesis)],
        metadatas=[metadata],
    )
    log.info("Stored '%s' as id=%s  (%d entries total)",
             hypothesis.signal_name, hyp_id, collection.count())
    return hyp_id


def add_if_new(hypothesis: FactorHypothesis, threshold: float = _DEFAULT_THRESHOLD) -> tuple[bool, dict]:
    """
    Add *hypothesis* only if it isn't a near-duplicate of something stored.

    Returns
    -------
    (False, {"reason": "duplicate", "matched_name": str, "similarity": float})
        if a near-duplicate (similarity >= threshold) already exists
    (True,  {"id": str})
        once the hypothesis has been stored
    """
    dup, info = is_duplicate(hypothesis, threshold=threshold)
    if dup:
        return False, {"reason": "duplicate", **info}

    hyp_id = add_hypothesis(hypothesis)
    return True, {"id": hyp_id}


def count() -> int:
    """How many hypotheses are currently stored."""
    return _get_collection().count()


def list_all() -> list[dict]:
    """
    Return every stored hypothesis as ``{"id", "signal_name", "metadata"}`` —
    useful for quick inspection of the dedup store's contents.
    """
    collection = _get_collection()
    if collection.count() == 0:
        return []

    got = collection.get(include=["metadatas"])
    return [
        {"id": hyp_id, "signal_name": meta.get("signal_name", "?"), "metadata": meta}
        for hyp_id, meta in zip(got["ids"], got["metadatas"])
    ]


# ---------------------------------------------------------------------------
# CLI — demonstrate dedup behaviour
# ---------------------------------------------------------------------------

def _make_hypothesis(name: str, description: str, rationale: str,
                     terms: list[dict], combine: str = "sum",
                     direction: int = -1, horizon: str = "fwd_ret_21d") -> FactorHypothesis:
    return FactorHypothesis(
        signal_name=name,
        signal_description=description,
        signal_computation={"terms": terms, "combine": combine, "normalize_by_length": True},
        direction=direction,
        horizon=horizon,
        economic_rationale=rationale,
    )


if __name__ == "__main__":
    W = 74
    print(f"\n{'=' * W}")
    print("  HYPOTHESIS DEDUPLICATION STORE  (ChromaDB, local embeddings)")
    print(f"{'=' * W}")

    print(f"\n  Collection path : {CHROMA_PATH}")
    print(f"  Current count   : {count()}")

    # --- 1. Base hypothesis -------------------------------------------------
    base = _make_hypothesis(
        name="supply_chain_density",
        description="Density of supply chain phrases in 10-K filings.",
        rationale=(
            "Companies that mention supply chain issues more frequently are "
            "signalling operational disruption to investors, which predicts "
            "lower forward returns as the market prices in the risk."
        ),
        terms=[{"type": "phrase", "value": "supply chain"}],
    )
    print(f"\n--- Adding base hypothesis: '{base.signal_name}' ---")
    print(f"  embedding text: {_embedding_text(base)!r}")
    added, info = add_if_new(base)
    print(f"  added={added}  info={info}")

    # --- 2. Reworded duplicate ----------------------------------------------
    reworded = _make_hypothesis(
        name="logistics_disruption_language",
        description="Frequency of logistics disruption language in annual filings.",
        rationale=(
            "When management talks more about logistics disruptions, it signals "
            "operational trouble to the market, which the market then prices in, "
            "leading to lower future stock returns."
        ),
        terms=[{"type": "phrase", "value": "supply chain"}],
    )
    print(f"\n--- Testing REWORDED duplicate: '{reworded.signal_name}' ---")
    print(f"  embedding text: {_embedding_text(reworded)!r}")
    nearest_reworded = _nearest_match(reworded)
    dup, dup_info = is_duplicate(reworded)
    print(f"  nearest match  -> {nearest_reworded}")
    print(f"  is_duplicate   -> ({dup}, {dup_info})")
    print(f"  EXPECTATION: high similarity, flagged as duplicate of '{base.signal_name}'")

    # --- 3. Genuinely different hypothesis -----------------------------------
    different = _make_hypothesis(
        name="sentiment_ratio",
        description="Ratio of positive to negative sentiment language in 10-K filings.",
        rationale=(
            "A higher ratio of positive to negative language reflects management's "
            "genuine optimism about the company's prospects, which predicts higher "
            "forward returns as that optimism is gradually validated by results."
        ),
        terms=[{"type": "lm_category", "value": "LM_positive"},
               {"type": "lm_category", "value": "LM_negative"}],
        combine="ratio",
        direction=1,
    )
    print(f"\n--- Testing DIFFERENT hypothesis: '{different.signal_name}' ---")
    print(f"  embedding text: {_embedding_text(different)!r}")
    nearest_different = _nearest_match(different)
    dup2, dup2_info = is_duplicate(different)
    print(f"  nearest match  -> {nearest_different}")
    print(f"  is_duplicate   -> ({dup2}, {dup2_info})")
    print(f"  EXPECTATION: low similarity, NOT a duplicate")

    # --- Summary -------------------------------------------------------------
    print(f"\n{'=' * W}")
    print("  SIMILARITY SUMMARY  (threshold = %.2f)" % _DEFAULT_THRESHOLD)
    print(f"{'=' * W}")
    sim_reworded  = nearest_reworded[1]  if nearest_reworded  else float("nan")
    sim_different = nearest_different[1] if nearest_different else float("nan")
    print(f"  reworded  vs. base : similarity = {sim_reworded:.4f}   "
          f"-> {'DUPLICATE' if dup else 'not duplicate'}")
    print(f"  different vs. base : similarity = {sim_different:.4f}   "
          f"-> {'DUPLICATE' if dup2 else 'not duplicate'}")

    print(f"\n  Final count        : {count()}")
    print(f"  Stored hypotheses  : {[h['signal_name'] for h in list_all()]}")

    print(f"\n{'=' * W}")
    print("  dedup_store.py — OK")
    print(f"{'=' * W}")
