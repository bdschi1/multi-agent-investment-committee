"""
Node functions for the LangGraph investment committee graph.

Each node wraps an existing agent's .run() or .rebut() method.
Agents are NOT modified — they receive the same inputs as in v1.

Node contract: (state: dict, config: dict) -> dict (partial state update)

Phase C: All nodes accept an optional `config` parameter for
RunnableConfig injection. Non-serializable objects (model, on_status,
tool_registry) are read from config["configurable"] first, falling
back to state for backward compatibility with Phase A/B tests.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional

from langgraph.types import RunnableConfig

from agents.sector_analyst import SectorAnalystAgent
from agents.risk_manager import RiskManagerAgent
from agents.macro_analyst import MacroAnalystAgent
from agents.portfolio_manager import PortfolioManagerAgent
from orchestrator.committee import ConvictionSnapshot

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config helpers — read from config first, fall back to state
# ---------------------------------------------------------------------------

def _get_model(state: dict, config: Optional[RunnableConfig] = None) -> Any:
    """Get LLM model callable from config (preferred) or state (fallback)."""
    if config:
        cfg = config.get("configurable", {})
        model = cfg.get("model")
        if model is not None:
            return model
    return state.get("model")


def _get_on_status(state: dict, config: Optional[RunnableConfig] = None) -> Optional[Callable]:
    """Get on_status callback from config (preferred) or state (fallback)."""
    if config:
        cfg = config.get("configurable", {})
        cb = cfg.get("on_status")
        if cb is not None:
            return cb
    return state.get("on_status")


def _get_tool_registry(state: dict, config: Optional[RunnableConfig] = None) -> Any:
    """Get tool registry from config (preferred) or state (fallback)."""
    if config:
        cfg = config.get("configurable", {})
        registry = cfg.get("tool_registry")
        if registry is not None:
            return registry
    return state.get("tool_registry")


# ---------------------------------------------------------------------------
# Status helper
# ---------------------------------------------------------------------------

def _status(state: dict, msg: str, config: Optional[RunnableConfig] = None) -> None:
    """Fire the on_status callback if present, and log."""
    cb = _get_on_status(state, config)
    if cb:
        cb(msg)
    logger.info(msg)


# ---------------------------------------------------------------------------
# Phase 0: Data Gathering
# ---------------------------------------------------------------------------

def gather_data(state: dict, config: Optional[RunnableConfig] = None) -> dict:
    """
    Gather market data for the ticker.

    If context is already populated (pre-gathered by app.py), this
    is a passthrough. Emits the Phase 1 status message so the
    progress bar fires in app.py.
    """
    _status(
        state,
        "Phase 1/3: Sector Analyst, Risk Manager, and Macro Analyst "
        "analyzing in parallel...",
        config,
    )

    if state.get("context"):
        return {}

    from tools.data_aggregator import DataAggregator

    context = DataAggregator.gather_context(
        state["ticker"],
        state.get("user_context", ""),
    )
    return {"context": context}


# ---------------------------------------------------------------------------
# Phase 1: Parallel Analysis Nodes
# ---------------------------------------------------------------------------

def run_sector_analyst(state: dict, config: Optional[RunnableConfig] = None) -> dict:
    """Run the Sector Analyst agent (bull case)."""
    model = _get_model(state, config)
    tool_registry = _get_tool_registry(state, config)

    agent = SectorAnalystAgent(model=model, tool_registry=tool_registry)
    result = agent.run(state["ticker"], state["context"])

    bull_case = result["output"]
    trace = result["trace"]

    return {
        "bull_case": bull_case,
        "traces": {"sector_analyst": trace},
        "conviction_timeline": [
            ConvictionSnapshot(
                phase="Initial Analysis",
                agent="Sector Analyst",
                score=bull_case.conviction_score,
                score_type="conviction",
            )
        ],
    }


def run_risk_manager(state: dict, config: Optional[RunnableConfig] = None) -> dict:
    """Run the Risk Manager agent (bear case)."""
    model = _get_model(state, config)
    tool_registry = _get_tool_registry(state, config)

    agent = RiskManagerAgent(model=model, tool_registry=tool_registry)
    result = agent.run(state["ticker"], state["context"])

    bear_case = result["output"]
    trace = result["trace"]

    return {
        "bear_case": bear_case,
        "traces": {"risk_manager": trace},
        "conviction_timeline": [
            ConvictionSnapshot(
                phase="Initial Analysis",
                agent="Risk Manager",
                score=bear_case.risk_score,
                score_type="risk",
            )
        ],
    }


def run_macro_analyst(state: dict, config: Optional[RunnableConfig] = None) -> dict:
    """Run the Macro Analyst agent (top-down context)."""
    model = _get_model(state, config)
    tool_registry = _get_tool_registry(state, config)

    agent = MacroAnalystAgent(model=model, tool_registry=tool_registry)
    result = agent.run(state["ticker"], state["context"])

    macro_view = result["output"]
    trace = result["trace"]

    return {
        "macro_view": macro_view,
        "traces": {"macro_analyst": trace},
        "conviction_timeline": [
            ConvictionSnapshot(
                phase="Initial Analysis",
                agent="Macro Analyst",
                score=macro_view.macro_favorability,
                score_type="favorability",
            )
        ],
    }


# ---------------------------------------------------------------------------
# Phase 1 Reporter (fan-in convergence point)
# ---------------------------------------------------------------------------

def report_phase1(state: dict, config: Optional[RunnableConfig] = None) -> dict:
    """
    Log Phase 1 scores after all three analysts complete.
    Initializes debate_round counter.
    """
    bc = state.get("bull_case")
    bear = state.get("bear_case")
    mv = state.get("macro_view")

    bull_score = bc.conviction_score if bc else 0
    bear_score = bear.risk_score if bear else 0
    macro_score = mv.macro_favorability if mv else 0

    _status(state, (
        f"  Bull case: conviction {bull_score}/10 | "
        f"Bear case: risk {bear_score}/10 | "
        f"Macro favorability: {macro_score}/10"
    ), config)

    return {"debate_round": 0, "debate_skipped": False}


# ---------------------------------------------------------------------------
# Debate skip marker
# ---------------------------------------------------------------------------

def mark_debate_skipped(state: dict, config: Optional[RunnableConfig] = None) -> dict:
    """Set flag when debate is skipped due to convergence."""
    return {"debate_skipped": True}


# ---------------------------------------------------------------------------
# Phase 2: Debate
# ---------------------------------------------------------------------------

def run_debate_round(state: dict, config: Optional[RunnableConfig] = None) -> dict:
    """
    Execute one debate round: analyst.rebut() and risk_mgr.rebut()
    in parallel via ThreadPoolExecutor, exactly as v1 does.
    """
    model = _get_model(state, config)

    current_round = state.get("debate_round", 0) + 1
    max_rounds = state.get("max_debate_rounds", 2)

    # Emit Phase 2 header on first round
    if current_round == 1:
        _status(
            state,
            "Phase 2/3: Adversarial debate — agents challenging each other...",
            config,
        )

    _status(state, f"  Debate round {current_round}/{max_rounds}...", config)

    analyst = SectorAnalystAgent(model=model)
    risk_mgr = RiskManagerAgent(model=model)

    bull_case = state["bull_case"]
    bear_case = state["bear_case"]

    analyst_rebuttal = None
    risk_rebuttal = None

    with ThreadPoolExecutor(max_workers=2) as executor:
        analyst_future = executor.submit(
            analyst.rebut, state["ticker"], bear_case, bull_case
        )
        risk_future = executor.submit(
            risk_mgr.rebut, state["ticker"], bull_case, bear_case
        )

        try:
            analyst_rebuttal = analyst_future.result()
        except Exception as e:
            logger.warning(f"Analyst rebuttal failed: {e}")
        try:
            risk_rebuttal = risk_future.result()
        except Exception as e:
            logger.warning(f"Risk rebuttal failed: {e}")

    # Build conviction timeline entries for post-debate
    timeline_entries: list[ConvictionSnapshot] = []
    if analyst_rebuttal and analyst_rebuttal.revised_conviction is not None:
        timeline_entries.append(ConvictionSnapshot(
            phase="Post-Debate",
            agent="Sector Analyst",
            score=analyst_rebuttal.revised_conviction,
            score_type="conviction",
        ))
    if risk_rebuttal and risk_rebuttal.revised_conviction is not None:
        timeline_entries.append(ConvictionSnapshot(
            phase="Post-Debate",
            agent="Risk Manager",
            score=risk_rebuttal.revised_conviction,
            score_type="risk",
        ))

    update: dict[str, Any] = {
        "debate_round": current_round,
        "analyst_rebuttal": analyst_rebuttal,
        "risk_rebuttal": risk_rebuttal,
    }

    if timeline_entries:
        update["conviction_timeline"] = timeline_entries

    return update


# ---------------------------------------------------------------------------
# Phase 2 Reporter
# ---------------------------------------------------------------------------

def report_debate_complete(state: dict, config: Optional[RunnableConfig] = None) -> dict:
    """Log debate completion or skip status."""
    ar = state.get("analyst_rebuttal")
    rr = state.get("risk_rebuttal")

    if state.get("debate_skipped"):
        _status(state, (
            "  Debate SKIPPED — bull/bear scores within convergence threshold"
        ), config)
    else:
        _status(state, (
            f"  Debate complete | "
            f"Analyst revised conviction: "
            f"{ar.revised_conviction if ar else 'N/A'} | "
            f"Risk Mgr revised risk: "
            f"{rr.revised_conviction if rr else 'N/A'}"
        ), config)

    return {}


# ---------------------------------------------------------------------------
# Phase 3: Portfolio Manager
# ---------------------------------------------------------------------------

def run_portfolio_manager(state: dict, config: Optional[RunnableConfig] = None) -> dict:
    """
    Run the PM synthesis — identical context assembly to v1
    (orchestrator/committee.py lines 238-247).

    Phase C additions:
    - pm_guidance: user-provided guidance from HITL review step
    - prior_analyses: session memory of prior runs for this ticker
    """
    model = _get_model(state, config)
    tool_registry = _get_tool_registry(state, config)

    _status(
        state,
        "Phase 3/3: Portfolio Manager synthesizing committee decision...",
        config,
    )

    pm = PortfolioManagerAgent(model=model, tool_registry=tool_registry)

    # Retrieve prior analyses from session memory
    prior_analyses: list[dict] = []
    try:
        from orchestrator.memory import get_prior_analyses
        prior_analyses = get_prior_analyses(state["ticker"])
    except Exception as e:
        logger.debug(f"Session memory unavailable: {e}")

    # Assemble PM context exactly as v1 does, plus Phase C additions
    pm_context: dict[str, Any] = {
        "bull_case": state["bull_case"],
        "bear_case": state["bear_case"],
        "macro_view": state.get("macro_view"),
        "debate": {
            "analyst_rebuttal": (
                state["analyst_rebuttal"].model_dump()
                if state.get("analyst_rebuttal") else {}
            ),
            "risk_rebuttal": (
                state["risk_rebuttal"].model_dump()
                if state.get("risk_rebuttal") else {}
            ),
        },
        "user_context": state.get("context", {}).get("user_context", ""),
        "pm_guidance": state.get("pm_guidance", ""),
        "prior_analyses": prior_analyses,
    }

    pm_result = pm.run(state["ticker"], pm_context)
    memo = pm_result["output"]
    trace = pm_result["trace"]

    timeline_entries: list[ConvictionSnapshot] = []
    if memo:
        timeline_entries.append(ConvictionSnapshot(
            phase="PM Decision",
            agent="Portfolio Manager",
            score=memo.conviction,
            score_type="conviction",
        ))

    update: dict[str, Any] = {
        "committee_memo": memo,
        "traces": {"portfolio_manager": trace},
    }

    if timeline_entries:
        update["conviction_timeline"] = timeline_entries

    return update


# ---------------------------------------------------------------------------
# Finalize
# ---------------------------------------------------------------------------

def finalize(state: dict, config: Optional[RunnableConfig] = None) -> dict:
    """Compute final timing and token totals, log completion."""
    duration_ms = (time.time() - state["start_time"]) * 1000
    total_tokens = sum(t.total_tokens for t in state.get("traces", {}).values())

    memo = state.get("committee_memo")
    if memo:
        _status(state, (
            f"  Decision: {memo.recommendation} | "
            f"Conviction: {memo.conviction}/10 | "
            f"Size: {memo.position_size}"
        ), config)
    _status(state, f"\nCommittee complete in {duration_ms / 1000:.1f}s", config)

    return {
        "total_duration_ms": duration_ms,
        "total_tokens": total_tokens,
    }
