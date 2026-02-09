"""
CommitteeState — typed state for the LangGraph investment committee graph.

This TypedDict flows through every node in the graph. Keys with
Annotated reducers support automatic fan-in merging from parallel
Send nodes (e.g., three analyst nodes writing to traces simultaneously).
"""

from __future__ import annotations

import operator
from typing import Any, Annotated, Optional, Callable

from typing_extensions import TypedDict

from agents.base import (
    BullCase,
    BearCase,
    MacroView,
    CommitteeMemo,
    Rebuttal,
    ReasoningTrace,
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
    context: dict[str, Any]                   # DataAggregator output
    max_debate_rounds: int
    on_status: Optional[Callable[[str], None]]
    model: Any                                 # LLM callable(str) -> str
    tool_registry: Optional[Any]               # ToolRegistry instance (Phase B)

    # -- Phase 1: Parallel analyst outputs --
    bull_case: Optional[BullCase]
    bear_case: Optional[BearCase]
    macro_view: Optional[MacroView]

    # -- Aggregated across parallel branches (need reducers) --
    traces: Annotated[dict[str, ReasoningTrace], _merge_dicts]
    conviction_timeline: Annotated[list[ConvictionSnapshot], operator.add]

    # -- Phase 2: Debate --
    analyst_rebuttal: Optional[Rebuttal]
    risk_rebuttal: Optional[Rebuttal]
    debate_round: int
    debate_skipped: bool

    # -- HITL: PM guidance from user review (Phase C) --
    pm_guidance: Optional[str]

    # -- Phase 3: PM synthesis --
    committee_memo: Optional[CommitteeMemo]

    # -- Timing / meta --
    start_time: float
    total_duration_ms: float
    total_tokens: int
