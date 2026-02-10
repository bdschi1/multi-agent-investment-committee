"""
LangGraph StateGraph for the investment committee workflow.

This replaces v1's ThreadPoolExecutor orchestration in committee.py
with a declarative state-machine graph. Same inputs, same outputs,
but now with conditional edges (debate loop) and parallel fan-out
via Send.  Debate always runs (convergence is noted, not skipped).

Graph topology (full pipeline):
    START → gather_data → [Send 3 analysts] → report_phase1
                                                    │
                                            run_debate_round ◄──┐
                                                │                │
                                         [debate_done?]          │
                                         /           \\          │
                                    done          continue ──────┘
                                       │
                                report_debate_complete
                                       │
                                run_portfolio_manager
                                       │
                                   finalize → END

Phase C: Two-phase execution (HITL)
    Phase 1 graph: START → ... → report_debate_complete → END
    Phase 2 graph: START → run_portfolio_manager → finalize → END
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

from langgraph.graph import StateGraph, END, START
from langgraph.types import Send

from orchestrator.state import CommitteeState
from orchestrator.committee import CommitteeResult, ConvictionSnapshot
from orchestrator import nodes


# ---------------------------------------------------------------------------
# Routing functions (conditional edges)
# ---------------------------------------------------------------------------

def _fan_out_analysts(state: CommitteeState) -> list[Send]:
    """Route from gather_data to 3 parallel analyst nodes."""
    return [
        Send("run_sector_analyst", state),
        Send("run_risk_manager", state),
        Send("run_macro_analyst", state),
    ]


def _should_debate(state: CommitteeState) -> str:
    """
    After report_phase1: always enter the debate loop.

    Previously this would skip debate when bull/bear scores converged
    (spread < 2.0).  We now always debate so the user can observe the
    convergence first-hand in the Debate tab.  The convergence spread is
    still reported in report_debate_complete for informational purposes.
    """
    return "enter_debate"


def _debate_or_exit(state: CommitteeState) -> str:
    """After a debate round: continue debating or proceed to PM."""
    current = state.get("debate_round", 0)
    max_rounds = state.get("max_debate_rounds", 2)

    if current >= max_rounds:
        return "debate_done"
    return "debate_continue"


# ---------------------------------------------------------------------------
# Graph construction — Full pipeline (backward compatible)
# ---------------------------------------------------------------------------

def build_graph() -> Any:
    """
    Construct and compile the investment committee StateGraph.

    Returns a compiled graph ready for .invoke().
    """
    graph = StateGraph(CommitteeState)

    # ── Add all nodes ──
    graph.add_node("gather_data", nodes.gather_data)
    graph.add_node("run_sector_analyst", nodes.run_sector_analyst)
    graph.add_node("run_risk_manager", nodes.run_risk_manager)
    graph.add_node("run_macro_analyst", nodes.run_macro_analyst)
    graph.add_node("report_phase1", nodes.report_phase1)
    graph.add_node("mark_debate_skipped", nodes.mark_debate_skipped)
    graph.add_node("run_debate_round", nodes.run_debate_round)
    graph.add_node("report_debate_complete", nodes.report_debate_complete)
    graph.add_node("run_portfolio_manager", nodes.run_portfolio_manager)
    graph.add_node("finalize", nodes.finalize)

    # ── Entry ──
    graph.add_edge(START, "gather_data")

    # ── Fan-out to 3 parallel analysts ──
    graph.add_conditional_edges(
        "gather_data",
        _fan_out_analysts,
        ["run_sector_analyst", "run_risk_manager", "run_macro_analyst"],
    )

    # ── Fan-in: all three analyst nodes converge to report_phase1 ──
    graph.add_edge("run_sector_analyst", "report_phase1")
    graph.add_edge("run_risk_manager", "report_phase1")
    graph.add_edge("run_macro_analyst", "report_phase1")

    # ── Convergence check: debate or skip ──
    graph.add_conditional_edges(
        "report_phase1",
        _should_debate,
        {
            "convergence_met": "mark_debate_skipped",
            "enter_debate": "run_debate_round",
        },
    )

    # ── Skipped debate path ──
    graph.add_edge("mark_debate_skipped", "report_debate_complete")

    # ── Debate loop ──
    graph.add_conditional_edges(
        "run_debate_round",
        _debate_or_exit,
        {
            "debate_done": "report_debate_complete",
            "debate_continue": "run_debate_round",
        },
    )

    # ── Post-debate → PM ──
    graph.add_edge("report_debate_complete", "run_portfolio_manager")

    # ── PM → Finalize → END ──
    graph.add_edge("run_portfolio_manager", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Graph construction — Phase 1 only (HITL: stops after debate)
# ---------------------------------------------------------------------------

def build_graph_phase1() -> Any:
    """
    Build graph that runs analysts + debate, stopping after report_debate_complete.

    This is the first half of the HITL two-phase execution.
    Returns intermediate state with bull_case, bear_case, macro_view,
    and debate results — but NO committee_memo.
    """
    graph = StateGraph(CommitteeState)

    # Same nodes as full graph minus PM and finalize
    graph.add_node("gather_data", nodes.gather_data)
    graph.add_node("run_sector_analyst", nodes.run_sector_analyst)
    graph.add_node("run_risk_manager", nodes.run_risk_manager)
    graph.add_node("run_macro_analyst", nodes.run_macro_analyst)
    graph.add_node("report_phase1", nodes.report_phase1)
    graph.add_node("mark_debate_skipped", nodes.mark_debate_skipped)
    graph.add_node("run_debate_round", nodes.run_debate_round)
    graph.add_node("report_debate_complete", nodes.report_debate_complete)

    # ── Entry ──
    graph.add_edge(START, "gather_data")

    # ── Fan-out to 3 parallel analysts ──
    graph.add_conditional_edges(
        "gather_data",
        _fan_out_analysts,
        ["run_sector_analyst", "run_risk_manager", "run_macro_analyst"],
    )

    # ── Fan-in ──
    graph.add_edge("run_sector_analyst", "report_phase1")
    graph.add_edge("run_risk_manager", "report_phase1")
    graph.add_edge("run_macro_analyst", "report_phase1")

    # ── Convergence check ──
    graph.add_conditional_edges(
        "report_phase1",
        _should_debate,
        {
            "convergence_met": "mark_debate_skipped",
            "enter_debate": "run_debate_round",
        },
    )

    graph.add_edge("mark_debate_skipped", "report_debate_complete")

    graph.add_conditional_edges(
        "run_debate_round",
        _debate_or_exit,
        {
            "debate_done": "report_debate_complete",
            "debate_continue": "run_debate_round",
        },
    )

    # ── STOP after debate report (no PM, no finalize) ──
    graph.add_edge("report_debate_complete", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Graph construction — Phase 2 only (HITL: PM synthesis)
# ---------------------------------------------------------------------------

def build_graph_phase2() -> Any:
    """
    Build graph for PM synthesis only.

    This is the second half of the HITL two-phase execution.
    Expects intermediate state from Phase 1 as input.
    """
    graph = StateGraph(CommitteeState)

    graph.add_node("run_portfolio_manager", nodes.run_portfolio_manager)
    graph.add_node("finalize", nodes.finalize)

    graph.add_edge(START, "run_portfolio_manager")
    graph.add_edge("run_portfolio_manager", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# State → CommitteeResult conversion
# ---------------------------------------------------------------------------

def _state_to_result(state: dict) -> CommitteeResult:
    """Convert final graph state to CommitteeResult for backward compat."""
    return CommitteeResult(
        ticker=state.get("ticker", ""),
        bull_case=state.get("bull_case"),
        bear_case=state.get("bear_case"),
        macro_view=state.get("macro_view"),
        analyst_rebuttal=state.get("analyst_rebuttal"),
        risk_rebuttal=state.get("risk_rebuttal"),
        committee_memo=state.get("committee_memo"),
        traces=state.get("traces", {}),
        conviction_timeline=state.get("conviction_timeline", []),
        parsing_failures=state.get("parsing_failures", []),
        total_duration_ms=state.get("total_duration_ms", 0.0),
        total_tokens=state.get("total_tokens", 0),
    )


# ---------------------------------------------------------------------------
# Config builder helper
# ---------------------------------------------------------------------------

def _build_config(
    model: Any = None,
    on_status: Optional[Callable[[str], None]] = None,
    tool_registry: Any = None,
) -> dict:
    """Build a LangGraph config dict with configurable non-serializable objects."""
    configurable: dict[str, Any] = {}
    if model is not None:
        configurable["model"] = model
    if on_status is not None:
        configurable["on_status"] = on_status
    if tool_registry is not None:
        configurable["tool_registry"] = tool_registry
    return {"configurable": configurable} if configurable else {}


# ---------------------------------------------------------------------------
# Tool registry builder helper
# ---------------------------------------------------------------------------

def _maybe_build_tool_registry(tool_registry: Any = None) -> Any:
    """Build default tool registry if none provided and tool calling is enabled."""
    if tool_registry is not None:
        return tool_registry
    try:
        from tools.registry import build_default_registry
        from config.settings import settings
        if settings.max_tool_calls_per_agent > 0:
            return build_default_registry(
                max_calls=settings.max_tool_calls_per_agent
            )
    except Exception:
        pass  # Graceful degradation: no tools if registry fails to build
    return None


# ---------------------------------------------------------------------------
# Public entry point — Full pipeline (backward compatible)
# ---------------------------------------------------------------------------

def run_graph(
    ticker: str,
    context: dict[str, Any],
    model: Any,
    max_debate_rounds: int = 2,
    on_status: Optional[Callable[[str], None]] = None,
    tool_registry: Any = None,
) -> CommitteeResult:
    """
    High-level entry point: invoke the compiled graph and return CommitteeResult.

    This is the function that committee.py's InvestmentCommittee.run() calls.
    Same interface, same return type — drop-in replacement for v1.

    Args:
        ticker: Stock ticker to analyze
        context: Pre-gathered data from DataAggregator
        model: LLM callable(str) -> str
        max_debate_rounds: Maximum debate rounds (1-20)
        on_status: Optional callback for UI progress updates
        tool_registry: Optional ToolRegistry for dynamic tool calling (Phase B)

    Returns:
        CommitteeResult with all analyses, debate, and final memo
    """
    compiled = build_graph()

    # Build default tool registry if none provided
    tool_registry = _maybe_build_tool_registry(tool_registry)

    # Build config for non-serializable objects
    config = _build_config(model=model, on_status=on_status, tool_registry=tool_registry)

    initial_state: dict[str, Any] = {
        "ticker": ticker,
        "context": context,
        # Keep in state for backward compat with Phase A/B tests
        "model": model,
        "max_debate_rounds": max_debate_rounds,
        "on_status": on_status,
        "tool_registry": tool_registry,
        "start_time": time.time(),
        # Initialize reducer base values
        "traces": {},
        "conviction_timeline": [],
        "parsing_failures": [],
        # Initialize debate state
        "debate_round": 0,
        "debate_skipped": False,
    }

    final_state = compiled.invoke(initial_state, config=config)

    return _state_to_result(final_state)


# ---------------------------------------------------------------------------
# Public entry point — Phase 1 (HITL: analysts + debate)
# ---------------------------------------------------------------------------

def run_graph_phase1(
    ticker: str,
    context: dict[str, Any],
    model: Any,
    max_debate_rounds: int = 2,
    on_status: Optional[Callable[[str], None]] = None,
    tool_registry: Any = None,
) -> dict[str, Any]:
    """
    Run Phase 1 of the HITL two-phase execution.

    Runs analysts in parallel, then debate. Stops BEFORE the PM synthesis.
    Returns the intermediate graph state dict (not a CommitteeResult).

    The returned dict can be passed directly to run_graph_phase2() after
    the user reviews and optionally adds PM guidance.

    Args:
        ticker: Stock ticker to analyze
        context: Pre-gathered data from DataAggregator
        model: LLM callable(str) -> str
        max_debate_rounds: Maximum debate rounds (1-20)
        on_status: Optional callback for UI progress updates
        tool_registry: Optional ToolRegistry for dynamic tool calling

    Returns:
        Intermediate state dict with bull_case, bear_case, macro_view,
        debate results, traces, and conviction_timeline.
    """
    compiled = build_graph_phase1()

    tool_registry = _maybe_build_tool_registry(tool_registry)
    config = _build_config(model=model, on_status=on_status, tool_registry=tool_registry)

    initial_state: dict[str, Any] = {
        "ticker": ticker,
        "context": context,
        "model": model,
        "max_debate_rounds": max_debate_rounds,
        "on_status": on_status,
        "tool_registry": tool_registry,
        "start_time": time.time(),
        "traces": {},
        "conviction_timeline": [],
        "parsing_failures": [],
        "debate_round": 0,
        "debate_skipped": False,
    }

    final_state = compiled.invoke(initial_state, config=config)

    # Return as plain dict so it's easy to manipulate before Phase 2
    return dict(final_state)


# ---------------------------------------------------------------------------
# Public entry point — Phase 2 (HITL: PM synthesis)
# ---------------------------------------------------------------------------

def run_graph_phase2(
    intermediate_state: dict[str, Any],
    model: Any,
    pm_guidance: str = "",
    on_status: Optional[Callable[[str], None]] = None,
    tool_registry: Any = None,
) -> CommitteeResult:
    """
    Run Phase 2 of the HITL two-phase execution.

    Takes the intermediate state from Phase 1 and runs the PM synthesis.
    Optionally injects PM guidance from the user's review.

    Args:
        intermediate_state: State dict from run_graph_phase1()
        model: LLM callable(str) -> str
        pm_guidance: Optional user guidance for the PM (from HITL review)
        on_status: Optional callback for UI progress updates
        tool_registry: Optional ToolRegistry for dynamic tool calling

    Returns:
        CommitteeResult with the complete analysis including PM memo
    """
    compiled = build_graph_phase2()

    tool_registry = _maybe_build_tool_registry(tool_registry)
    config = _build_config(model=model, on_status=on_status, tool_registry=tool_registry)

    # Inject PM guidance and update non-serializable refs
    state = dict(intermediate_state)
    state["model"] = model
    state["on_status"] = on_status
    state["tool_registry"] = tool_registry

    if pm_guidance:
        state["pm_guidance"] = pm_guidance

    # Ensure start_time is set (needed by finalize)
    if "start_time" not in state:
        state["start_time"] = time.time()

    final_state = compiled.invoke(state, config=config)

    return _state_to_result(final_state)
