"""LangGraph node for the portfolio optimizer."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    from langchain_core.runnables import RunnableConfig
except ImportError:
    RunnableConfig = Any


def run_optimizer(state: dict, config: RunnableConfig) -> dict:
    """
    LangGraph node: run portfolio optimization on PM output.

    Reads optimizer_method from config["configurable"] (default: black_litterman).
    Returns {"optimization_result": OptimizationResult | OptimizerFallback}.

    Graceful: if anything fails, returns OptimizerFallback without
    disrupting the pipeline.
    """
    from optimizer.bl_optimizer import run_optimization
    from optimizer.ensemble import run_ensemble
    from optimizer.models import OptimizerFallback
    from optimizer.strategies import STRATEGY_DISPLAY_NAMES

    ticker = state.get("ticker", "")

    # Read config
    configurable = config.get("configurable", {}) if config else {}
    on_status = configurable.get("on_status")
    if on_status is None:
        on_status = state.get("on_status")

    optimizer_method = configurable.get("optimizer_method", "black_litterman")
    display_name = STRATEGY_DISPLAY_NAMES.get(optimizer_method, optimizer_method)

    if on_status:
        try:
            on_status(f"Running {display_name} portfolio optimizer...")
        except Exception:
            pass

    memo = state.get("committee_memo")
    bull_case = state.get("bull_case")
    bear_case = state.get("bear_case")
    macro_view = state.get("macro_view")

    # Detect sector from context or bull case
    sector = "Technology"  # default
    context = state.get("context", {})
    market_data = context.get("market_data", {})
    if isinstance(market_data, dict):
        sector = market_data.get("sector", sector)
    if hasattr(bull_case, "key_metrics") and bull_case.key_metrics:
        sector = bull_case.key_metrics.get("sector", sector)

    # ── Ensemble mode ──
    if optimizer_method == "ensemble":
        result = run_ensemble(
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
                    n = len(result.strategy_results)
                    on_status(
                        f"  Ensemble: {n} strategies completed, "
                        f"blended weight={result.blended_target_weight * 100:.1f}%"
                    )
                else:
                    on_status(f"  Ensemble: {result.error_message}")
            except Exception:
                pass

        return {"optimization_result": result}

    # ── Single-strategy mode ──

    # BL requires a memo for view extraction; non-BL methods can proceed without one
    if not memo and optimizer_method == "black_litterman":
        logger.warning("No committee memo in state — skipping BL optimizer")
        return {
            "optimization_result": OptimizerFallback(
                ticker=ticker,
                error_message="No committee memo available for BL optimization",
            )
        }

    result = run_optimization(
        ticker=ticker,
        sector=sector,
        optimizer_method=optimizer_method,
        bull_case=bull_case,
        bear_case=bear_case,
        macro_view=macro_view,
        committee_memo=memo,
    )

    if on_status:
        try:
            if result.success:
                on_status(
                    f"  {display_name}: {ticker} weight={result.optimal_weight_pct}, "
                    f"Sharpe={result.computed_sharpe:.2f}"
                )
            else:
                on_status(f"  {display_name}: {result.error_message}")
        except Exception:
            pass

    return {"optimization_result": result}
