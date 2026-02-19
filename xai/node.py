"""LangGraph node for the XAI analysis step."""

from __future__ import annotations

import contextlib
import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    from langchain_core.runnables import RunnableConfig
except ImportError:
    RunnableConfig = Any


def run_xai_analysis(state: dict, config: RunnableConfig) -> dict:
    """LangGraph node: run XAI pre-screen analysis on gathered fundamentals.

    Extracts financial_metrics from state context, runs the 5-step XAI
    pipeline, and injects results into context["xai_analysis"] for
    downstream agents to reference.

    Graceful: if XAI is disabled, data is missing, or anything fails,
    returns {} (no state change) so the pipeline continues normally.
    """
    from xai.pipeline import XAIPipeline

    ticker = state.get("ticker", "")

    # Status callback
    on_status = None
    configurable = config.get("configurable", {}) if config else {}
    on_status = configurable.get("on_status")
    if on_status is None:
        on_status = state.get("on_status")

    # Check if XAI is enabled
    try:
        from config.settings import settings
        if not getattr(settings, "enable_xai", True):
            logger.info("XAI analysis disabled in settings")
            return {}
    except Exception:
        pass  # If settings unavailable, proceed with XAI enabled

    if on_status:
        with contextlib.suppress(Exception):
            on_status("Running XAI pre-screen analysis...")

    # Extract fundamentals from context
    context = state.get("context", {})
    fundamentals = context.get("financial_metrics", {})

    if not fundamentals:
        logger.warning("No financial_metrics in context — skipping XAI analysis")
        return {}

    try:
        # Get config options
        threshold = 0.5
        artifact_path = None
        try:
            from config.settings import settings
            threshold = getattr(settings, "xai_distress_threshold", 0.5)
            artifact_path = getattr(settings, "xai_model_path", None)
        except Exception:
            pass

        pipeline = XAIPipeline(
            distress_threshold=threshold,
            artifact_path=artifact_path,
        )
        result = pipeline.analyze(ticker, fundamentals)

        # Inject into context — copy to avoid mutating shared state
        ctx = dict(context)
        ctx["xai_analysis"] = result.model_dump()

        if on_status:
            try:
                zone = result.distress.distress_zone
                pfd_pct = f"{result.distress.pfd:.1%}"
                er_pct = result.returns.expected_return_pct
                on_status(
                    f"  XAI: {ticker} zone={zone}, PFD={pfd_pct}, "
                    f"ER={er_pct} ({result.computation_time_ms:.0f}ms)"
                )
            except Exception:
                pass

        return {"context": ctx}

    except Exception:
        logger.warning("XAI analysis failed for %s — continuing without it", ticker, exc_info=True)
        return {}
