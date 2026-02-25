"""
CommitteeState — typed state for the LangGraph investment committee graph.

This TypedDict flows through every node in the graph. Keys with
Annotated reducers support automatic fan-in merging from parallel
Send nodes (e.g., three analyst nodes writing to traces simultaneously).
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import Annotated, Any

from typing_extensions import TypedDict

from agents.base import (
    BearCase,
    BullCase,
    CommitteeMemo,
    MacroView,
    ReasoningTrace,
    Rebuttal,
    ShortCase,
)
from orchestrator.committee import ConvictionSnapshot

# ---------------------------------------------------------------------------
# Custom reducers for fan-in merging
# ---------------------------------------------------------------------------

def _merge_dicts(left: dict, right: dict) -> dict:
    """
    Reducer that merges two dicts.

    Used for traces: each parallel analyst node returns
    {"traces": {"sector_analyst": trace}}, and the reducer
    combines them into a single dict with all agents.
    """
    merged = left.copy()
    merged.update(right)
    return merged


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class CommitteeState(TypedDict, total=False):
    """
    State flowing through the investment committee StateGraph.

    Fields with Annotated reducers handle fan-in from parallel Send nodes.
    Fields without reducers are simple overwrites (last-writer-wins).

    Note: `on_status` and `model` are non-serializable callables.
    This is fine because Phase A uses no checkpointer (no serialization).
    """

    # -- Inputs (set once at graph invocation) --
    ticker: str
    context: dict[str, Any]                   # DataAggregator output (includes user_kb if uploaded)
    max_debate_rounds: int
    on_status: Callable[[str], None] | None
    model: Any                                 # LLM callable(str) -> str
    tool_registry: Any | None               # ToolRegistry instance (Phase B)

    # -- Phase 1: Parallel analyst outputs --
    bull_case: BullCase | None
    bear_case: BearCase | None
    short_case: ShortCase | None
    macro_view: MacroView | None

    # -- Aggregated across parallel branches (need reducers) --
    traces: Annotated[dict[str, ReasoningTrace], _merge_dicts]
    conviction_timeline: Annotated[list[ConvictionSnapshot], operator.add]

    # -- Phase 2: Debate --
    long_rebuttal: Rebuttal | None      # Long Analyst's rebuttal (was analyst_rebuttal)
    short_rebuttal: Rebuttal | None     # Short Analyst's rebuttal (new)
    risk_rebuttal: Rebuttal | None
    debate_round: int

    # -- HITL: PM guidance from user review (Phase C) --
    pm_guidance: str | None

    # -- Phase 3: PM synthesis --
    committee_memo: CommitteeMemo | None

    # -- Phase 3b: Black-Litterman optimizer output --
    optimization_result: Any | None

    # -- Parsing quality (tracks which agents had JSON parsing failures) --
    parsing_failures: Annotated[list[str], operator.add]

    # -- Timing / meta --
    start_time: float
    total_duration_ms: float
    total_tokens: int
