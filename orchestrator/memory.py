"""
Session memory for the investment committee.

Stores summaries of prior analyses so the Portfolio Manager can reference
previous runs of the same ticker within a session. This enables the PM
to identify how the investment case has evolved over multiple analyses.

Implementation: simple module-level dict. Not a database, not MemorySaver —
just enough state to remember what happened earlier in the session.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Session store: {ticker: [summary_dict, ...]}
# ---------------------------------------------------------------------------

_session_store: dict[str, list[dict[str, Any]]] = {}


def store_analysis(ticker: str, result: Any) -> None:
    """
    Store a completed analysis summary for PM reference.

    Extracts a small summary dict from a CommitteeResult — just enough
    for the PM to understand what happened previously, not the full result.

    Args:
        ticker: Stock ticker (uppercased)
        result: CommitteeResult or dict with the analysis output
    """
    ticker = ticker.upper()

    summary: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "ticker": ticker,
    }

    # Extract from CommitteeResult object
    if hasattr(result, "committee_memo") and result.committee_memo:
        memo = result.committee_memo
        summary["recommendation"] = memo.recommendation
        summary["conviction"] = memo.conviction
        summary["thesis_summary"] = memo.thesis_summary
        summary["position_size"] = memo.position_size
        summary["time_horizon"] = memo.time_horizon

    if hasattr(result, "bull_case") and result.bull_case:
        summary["bull_conviction"] = result.bull_case.conviction_score

    if hasattr(result, "bear_case") and result.bear_case:
        summary["bear_bearish_conviction"] = result.bear_case.bearish_conviction

    if hasattr(result, "macro_view") and result.macro_view:
        summary["macro_favorability"] = result.macro_view.macro_favorability

    # Also accept plain dicts (for testing or manual injection)
    if isinstance(result, dict):
        for key in ("recommendation", "conviction", "thesis_summary",
                     "bull_conviction", "bear_risk", "macro_favorability",
                     "position_size", "time_horizon"):
            if key in result and key not in summary:
                summary[key] = result[key]

    if ticker not in _session_store:
        _session_store[ticker] = []

    _session_store[ticker].append(summary)
    logger.info(
        f"Stored analysis #{len(_session_store[ticker])} for {ticker}: "
        f"{summary.get('recommendation', 'N/A')} @ {summary.get('conviction', 'N/A')}/10"
    )


def get_prior_analyses(ticker: str) -> list[dict[str, Any]]:
    """
    Retrieve prior analyses of this ticker from the current session.

    Returns:
        List of summary dicts, ordered chronologically (oldest first).
        Empty list if no prior analyses exist.
    """
    ticker = ticker.upper()
    return list(_session_store.get(ticker, []))


def clear_session() -> None:
    """Clear all session memory."""
    _session_store.clear()
    logger.info("Session memory cleared")


def get_session_tickers() -> list[str]:
    """Return list of tickers that have been analyzed in this session."""
    return list(_session_store.keys())


def get_session_summary() -> dict[str, int]:
    """Return a summary of how many analyses per ticker."""
    return {ticker: len(analyses) for ticker, analyses in _session_store.items()}


# ---------------------------------------------------------------------------
# Agent memory: BM25-based retrieval of past reflections
# ---------------------------------------------------------------------------

def build_agent_memory(
    agent_role: str,
    ticker: str,
    context: str = "",
    top_n: int = 3,
    db: Any = None,
) -> list[dict[str, Any]]:
    """
    Retrieve relevant past reflections for an agent using BM25 similarity.

    Loads reflections for *agent_role* from SQLite, builds a BM25 index
    from each reflection's lesson field, and queries with a combination
    of the ticker and any user context.

    Args:
        db: Optional SignalDatabase instance.  When *None*, opens and
            closes its own connection to the default database path.

    Returns:
        List of dicts with keys: ticker, lesson, what_worked, what_failed,
        was_correct, conviction_calibration.  Empty list if rank_bm25 is
        not installed or no reflections exist.
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        logger.debug("rank_bm25 not installed — agent memory disabled")
        return []

    try:
        own_db = False
        if db is None:
            from backtest.database import SignalDatabase
            db = SignalDatabase()
            own_db = True
        reflections = db.get_reflections(agent_role=agent_role, limit=500)
        if own_db:
            db.close()
    except Exception:
        logger.debug("Could not load reflections for %s", agent_role, exc_info=True)
        return []

    if not reflections:
        return []

    # Small corpus: skip BM25 ranking and return most recent reflections
    if len(reflections) <= top_n:
        results = []
        for r in reflections:
            results.append({
                "ticker": r.ticker,
                "lesson": r.lesson,
                "what_worked": r.what_worked,
                "what_failed": r.what_failed,
                "was_correct": bool(r.was_correct),
                "confidence_calibration": r.confidence_calibration,
            })
        logger.debug("Agent memory for %s/%s: %d results (small corpus, no ranking)", agent_role, ticker, len(results))
        return results

    # Build BM25 index from lessons
    corpus = []
    for r in reflections:
        doc = f"{r.ticker} {r.lesson} {r.what_worked} {r.what_failed}"
        corpus.append(doc.lower().split())

    bm25 = BM25Okapi(corpus)

    # Query with ticker + context
    query = f"{ticker} {context}".lower().split()
    scores = bm25.get_scores(query)

    # Rank and return top-N
    scored = sorted(zip(scores, reflections), key=lambda x: x[0], reverse=True)
    results = []
    for score, r in scored[:top_n]:
        results.append({
            "ticker": r.ticker,
            "lesson": r.lesson,
            "what_worked": r.what_worked,
            "what_failed": r.what_failed,
            "was_correct": bool(r.was_correct),
            "confidence_calibration": r.confidence_calibration,
        })

    logger.debug("Agent memory for %s/%s: %d results (from %d reflections)", agent_role, ticker, len(results), len(reflections))
    return results
