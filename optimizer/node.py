"""LangGraph node for the Black-Litterman optimizer."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    from langchain_core.runnables import RunnableConfig
except ImportError:
    RunnableConfig = Any


def run_optimizer(state: dict, config: RunnableConfig | None = None) -> dict:
    """
    LangGraph node: run Black-Litterman optimization on PM output.

    Extracts memo, bull_case, bear_case, macro_view, and sector from state.
    Returns {"optimization_result": OptimizationResult | OptimizerFallback}.

    Graceful: if anything fails, returns OptimizerFallback without
    disrupting the pipeline.
    """
    from optimizer.bl_optimizer import run_black_litterman
    from optimizer.models import OptimizerFallback

    ticker = state.get("ticker", "")

    # Status callback
    on_status = None
    if config and hasattr(config, "get"):
        configurable = config.get("configurable", {})
        on_status = configurable.get("on_status")
    if on_status is None:
        on_status = state.get("on_status")

    if on_status:
        try:
            on_status("Running Black-Litterman portfolio optimizer...")
        except Exception:
            pass

    memo = state.get("committee_memo")
    bull_case = state.get("bull_case")
    bear_case = state.get("bear_case")
    macro_view = state.get("macro_view")

    if not memo:
        logger.warning("No committee memo in state — skipping optimizer")
        return {
            "optimization_result": OptimizerFallback(
                ticker=ticker,
                error_message="No committee memo available for optimization",
            )
        }

    # Detect sector from context or bull case
    sector = "Technology"  # default
    context = state.get("context", {})
    market_data = context.get("market_data", {})
    if isinstance(market_data, dict):
        sector = market_data.get("sector", sector)
    # Try bull case info
    if hasattr(bull_case, "key_metrics") and bull_case.key_metrics:
        sector = bull_case.key_metrics.get("sector", sector)

    result = run_black_litterman(
        ticker=ticker,
        sector=sector,
        bull_case=bull_case,
        bear_case=bear_case,
        macro_view=macro_view,
        committee_memo=memo,
    )

    if on_status:
        try:
            if result.success:
                on_status(
                    f"  BL optimizer: {ticker} weight={result.optimal_weight_pct}, "
                    f"Sharpe={result.computed_sharpe:.2f}"
                )
            else:
                on_status(f"  BL optimizer: {result.error_message}")
        except Exception:
            pass

    return {"optimization_result": result}
