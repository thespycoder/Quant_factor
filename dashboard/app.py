"""
Streamlit dashboard — read-only window into the factor research engine.

Shows what the system has discovered so far by reading directly from the
existing storage layers:

    factor_library.store    the evaluated-factor index, full records, memos
    agents.dedup_store      the hypothesis dedup vector store (counts/listing only)

This dashboard NEVER writes to the library or to chroma — it only loads and
displays. New factors appear here once orchestration/research_graph.py (run
via the CLI) persists them; there is no "trigger a run" control here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from factor_library.store import load_index, load_record, load_memo
try:
    from agents.dedup_store import count as dedup_count, list_all as dedup_list_all
    DEDUP_AVAILABLE = True
    _DEDUP_ERR = ""
except Exception as e:
    DEDUP_AVAILABLE = False
    _DEDUP_ERR = str(e)

_CACHE_TTL = 60  # seconds — short enough that a fresh CLI run shows up promptly

_VERDICT_OPTIONS = [
    "CANDIDATE", "NOT_SIGNIFICANT", "SUSPICIOUS_DECAY", "NOT_NOVEL", "INSUFFICIENT_DATA",
]
_REJECTION_REASONS = ["NOT_SIGNIFICANT", "SUSPICIOUS_DECAY", "NOT_NOVEL"]

_TABLE_COLUMNS = [
    "signal_name", "verdict", "ic_mean", "ic_p_value",
    "sharpe", "decay_shape", "ff_is_novel", "evaluated_at",
]


# ---------------------------------------------------------------------------
# Cached loaders — read-only, short TTL so new CLI runs surface quickly
# ---------------------------------------------------------------------------

@st.cache_data(ttl=_CACHE_TTL)
def _load_index_cached() -> pd.DataFrame:
    return load_index()


@st.cache_data(ttl=_CACHE_TTL)
def _load_record_cached(factor_id: str) -> dict:
    return load_record(factor_id)


@st.cache_data(ttl=_CACHE_TTL)
def _load_memo_cached(factor_id: str) -> str | None:
    return load_memo(factor_id)


@st.cache_data(ttl=_CACHE_TTL)
def _dedup_snapshot() -> tuple[int, list[dict]]:
    if not DEDUP_AVAILABLE:
        return 0, []
    return dedup_count(), dedup_list_all()


# ---------------------------------------------------------------------------
# Header + top-level metrics
# ---------------------------------------------------------------------------

def _render_header(index: pd.DataFrame, dedup_total: int) -> None:
    st.title("Quant Factor Research Engine")

    n_total      = len(index)
    n_candidates = int(index["is_candidate"].sum()) if not index.empty else 0
    reason_counts = (
        index.loc[index["verdict"].isin(_REJECTION_REASONS), "verdict"].value_counts()
        if not index.empty else pd.Series(dtype=int)
    )

    cols = st.columns(2 + len(_REJECTION_REASONS) + 1)
    cols[0].metric("Factors evaluated", n_total)
    cols[1].metric("CANDIDATEs", n_candidates)
    for i, reason in enumerate(_REJECTION_REASONS, start=2):
        cols[i].metric(reason, int(reason_counts.get(reason, 0)))
    cols[-1].metric("Hypotheses in dedup store", dedup_total)


# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------

def _render_filters(index: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")

    if index.empty:
        st.sidebar.caption("Library is empty — nothing to filter yet.")
        return index

    verdicts = st.sidebar.multiselect(
        "Verdict", options=_VERDICT_OPTIONS, default=[],
        help="Leave empty to show all verdicts.",
    )
    decay_shapes = sorted(index["decay_shape"].dropna().unique().tolist())
    shapes = st.sidebar.multiselect(
        "Decay shape", options=decay_shapes, default=[],
        help="Leave empty to show all decay shapes.",
    )

    min_date = pd.Timestamp(index["evaluated_at"].min()).date()
    max_date = pd.Timestamp(index["evaluated_at"].max()).date()
    date_range = st.sidebar.date_input(
        "Evaluated between", value=(min_date, max_date),
        min_value=min_date, max_value=max_date,
    )

    name_filter = st.sidebar.text_input("Signal name contains", value="")

    filtered = index.copy()
    if verdicts:
        filtered = filtered[filtered["verdict"].isin(verdicts)]
    if shapes:
        filtered = filtered[filtered["decay_shape"].isin(shapes)]
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
        evaluated_date = filtered["evaluated_at"].dt.tz_localize(None).dt.date
        filtered = filtered[(evaluated_date >= start) & (evaluated_date <= end)]
    if name_filter.strip():
        filtered = filtered[filtered["signal_name"].str.contains(name_filter.strip(), case=False, na=False)]

    return filtered


# ---------------------------------------------------------------------------
# Factor library table
# ---------------------------------------------------------------------------

def _render_library_table(filtered: pd.DataFrame) -> None:
    st.header("Factor library")

    if filtered.empty:
        st.info("No factors match the current filters (or the library is still empty — "
                "run `python orchestration/research_graph.py` to populate it).")
        return

    ordered = filtered.sort_values(
        by=["is_candidate", "ic_p_value"],
        ascending=[False, True],
        na_position="last",
    )
    st.caption(f"{len(ordered)} factor(s) — sorted CANDIDATEs first, then by ic_p_value ascending.")
    st.dataframe(ordered[_TABLE_COLUMNS], width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Factor detail view
# ---------------------------------------------------------------------------

def _ic_by_year_chart(ic_by_year: list[dict]) -> go.Figure:
    df = pd.DataFrame(ic_by_year)
    colors = ["seagreen" if v >= 0 else "indianred" for v in df["ic"]]
    fig = go.Figure(go.Bar(x=df["year"], y=df["ic"], marker_color=colors))
    fig.update_layout(
        title="IC by year", xaxis_title="Year", yaxis_title="Information coefficient",
        height=320, margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def _render_detail(filtered: pd.DataFrame) -> None:
    st.header("Factor detail")

    if filtered.empty:
        st.caption("Pick a factor once the library has results matching your filters.")
        return

    options = filtered["factor_id"].tolist()
    labels  = {
        row["factor_id"]: f"{row['signal_name']}  —  {row['verdict']}  (ic={row['ic_mean']:.4f})"
        for _, row in filtered.iterrows()
    }
    factor_id = st.selectbox(
        "Choose a factor", options=options, format_func=lambda fid: labels.get(fid, fid),
    )
    if not factor_id:
        return

    memo = _load_memo_cached(factor_id)
    if memo:
        st.markdown(memo)
    else:
        st.info("This factor has no memo on file.")

    record = _load_record_cached(factor_id)
    metrics_keys = [
        "signal_name", "n_filings_used", "ic_mean", "ic_t_stat", "ic_p_value",
        "total_return", "sharpe", "decay_shape", "ff_alpha", "ff_alpha_pvalue",
        "ff_is_novel", "verdict", "failed_gates",
    ]
    hyp = record.get("hypothesis", {}) or {}
    metrics_block = {"signal_name": hyp.get("signal_name")}
    metrics_block.update({k: record.get(k) for k in metrics_keys[1:]})

    with st.expander("Raw metrics (JSON)"):
        st.json(metrics_block)

    ic_by_year = record.get("ic_by_year")
    if ic_by_year:
        st.plotly_chart(_ic_by_year_chart(ic_by_year), width="stretch")


# ---------------------------------------------------------------------------
# Run stats — dedup store snapshot
# ---------------------------------------------------------------------------

def _render_run_stats(dedup_total: int, dedup_entries: list[dict], n: int = 15) -> None:
    st.header("Run stats")

    if not DEDUP_AVAILABLE:
        st.info(
            f"Dedup store unavailable in this environment (chromadb import failed): {_DEDUP_ERR}"
        )
        return

    st.caption(f"{dedup_total} hypothesis/hypotheses stored in the dedup vector store "
               "(see the metric in the header above for the running total).")

    if not dedup_entries:
        st.caption("Dedup store is empty — no hypotheses have been proposed yet.")
        return

    st.caption(f"Most recent {min(n, len(dedup_entries))} hypothesis name(s) in the dedup store:")
    names = [entry["signal_name"] for entry in dedup_entries[-n:]][::-1]
    st.write("\n".join(f"- {name}" for name in names))


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Quant Factor Research Engine", layout="wide")

    index = _load_index_cached()
    dedup_total, dedup_entries = _dedup_snapshot()

    _render_header(index, dedup_total)
    st.divider()

    filtered = _render_filters(index)
    _render_library_table(filtered)
    st.divider()
    _render_detail(filtered)
    st.divider()
    _render_run_stats(dedup_total, dedup_entries)


if __name__ == "__main__":
    main()
