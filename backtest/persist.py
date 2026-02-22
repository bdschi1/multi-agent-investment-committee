"""Shared signal persistence utility.

Used by app.py (Gradio UI) and scripts/accumulate_signals.py (batch CLI).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from backtest.database import SignalDatabase
from backtest.models import SignalRecord
from orchestrator.committee import CommitteeResult

logger = logging.getLogger(__name__)


def persist_signal(
    result: CommitteeResult,
    provider_name: str,
    model_name: str,
    db: SignalDatabase | None = None,
) -> int | None:
    """Persist a CommitteeResult as a signal in the backtest database.

    Args:
        result: The committee analysis result.
        provider_name: LLM provider name (e.g., "anthropic", "ollama").
        model_name: Model identifier (e.g., "claude-sonnet-4-20250514").
        db: Optional pre-existing database connection.  Creates default if None.

    Returns:
        Signal ID if persisted successfully, None if no memo available.
    """
    if not result.committee_memo:
        return None

    own_db = db is None
    if own_db:
        db = SignalDatabase()

    memo = result.committee_memo

    signal = SignalRecord(
        ticker=result.ticker,
        signal_date=datetime.now(UTC),
        provider=provider_name,
        model_name=model_name,
        recommendation=memo.recommendation,
        t_signal=memo.t_signal,
        conviction=memo.conviction,
        position_direction=memo.position_direction,
        raw_confidence=memo.raw_confidence,
        bull_conviction=result.bull_case.conviction_score if result.bull_case else 5.0,
        bear_conviction=result.bear_case.bearish_conviction if result.bear_case else 5.0,
        macro_favorability=result.macro_view.macro_favorability if result.macro_view else 5.0,
        duration_s=result.total_duration_ms / 1000,
        total_tokens=result.total_tokens,
    )

    # Add BL optimizer results if available
    if result.optimization_result and hasattr(result.optimization_result, "success"):
        opt = result.optimization_result
        if opt.success:
            signal.bl_optimal_weight = opt.optimal_weight
            signal.bl_sharpe = opt.computed_sharpe
            signal.bl_sortino = opt.computed_sortino

    # Add XAI results if available
    xai = result.xai_result
    if xai and hasattr(xai, "distress"):
        signal.xai_pfd = xai.distress.pfd
        signal.xai_z_score = xai.distress.z_score
        signal.xai_distress_zone = xai.distress.distress_zone
        signal.xai_expected_return = xai.returns.expected_return
        signal.xai_model_used = xai.distress.model_used
        if xai.distress.top_risk_factors:
            first_factor = xai.distress.top_risk_factors[0]
            if isinstance(first_factor, dict):
                signal.xai_top_risk_factor = next(iter(first_factor), "")
            else:
                signal.xai_top_risk_factor = str(first_factor)

    signal_id = db.store_signal(signal)
    logger.info(f"Signal persisted: {result.ticker} → id={signal_id}")

    if own_db:
        db.close()

    return signal_id
