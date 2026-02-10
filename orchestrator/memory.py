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
from datetime import datetime, timezone
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
        "timestamp": datetime.now(timezone.utc).isoformat(),
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
