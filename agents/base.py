"""
Base agent protocol for the Investment Committee.

Every agent follows a structured reasoning loop:
    think → plan → [execute_tools] → act → reflect

v2 adds dynamic tool calling: agents can request data tools
during plan() via a TOOL_CALLS: block. The base class parses
the block, executes tools via ToolRegistry, and passes results
to act(). If no tools are requested (or no registry is provided),
the loop behaves identically to v1.
"""

from __future__ import annotations

import ast
import logging
import re
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AgentRole(str, Enum):
    SECTOR_ANALYST = "sector_analyst"
    RISK_MANAGER = "risk_manager"
    PORTFOLIO_MANAGER = "portfolio_manager"
    MACRO_ANALYST = "macro_analyst"


class StepType(str, Enum):
    THINK = "think"
    PLAN = "plan"
    ACT = "act"
    REFLECT = "reflect"
    TOOL_CALL = "tool_call"
    REBUTTAL = "rebuttal"


# ---------------------------------------------------------------------------
# Reasoning trace models
# ---------------------------------------------------------------------------

class ReasoningStep(BaseModel):
    """A single step in an agent's reasoning chain."""

    step_type: StepType
    agent_role: AgentRole
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    duration_ms: Optional[float] = None
    tokens_used: Optional[int] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReasoningTrace(BaseModel):
    """Complete reasoning trace for an agent's execution."""

    agent_role: AgentRole
    ticker: str
    steps: list[ReasoningStep] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    total_tokens: int = 0
    total_duration_ms: float = 0.0

    def add_step(self, step: ReasoningStep) -> None:
        self.steps.append(step)
        if step.tokens_used:
            self.total_tokens += step.tokens_used
        if step.duration_ms:
            self.total_duration_ms += step.duration_ms

    def finalize(self) -> None:
        self.completed_at = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Shared output schemas
# ---------------------------------------------------------------------------

class BullCase(BaseModel):
    """Structured output from the Sector Analyst."""

    ticker: str
    thesis: str
    supporting_evidence: list[str] = Field(default_factory=list)
    catalysts: list[str] = Field(default_factory=list)
    catalyst_calendar: list[dict[str, str]] = Field(
        default_factory=list,
        description="12-month catalyst timeline: [{timeframe, event, impact}, ...]",
    )
    conviction_score: float = Field(ge=0.0, le=10.0, description="0 = no conviction, 10 = max")
    time_horizon: str = Field(description="e.g. '6-12 months', '2-3 years'")
    key_metrics: dict[str, Any] = Field(default_factory=dict)
    technical_outlook: str = Field(default="", description="Technical analysis summary")


class BearCase(BaseModel):
    """Structured output from the Risk Manager."""

    ticker: str
    risks: list[str] = Field(default_factory=list)
    second_order_effects: list[str] = Field(default_factory=list)
    third_order_effects: list[str] = Field(default_factory=list)
    worst_case_scenario: str = ""
    risk_score: float = Field(ge=0.0, le=10.0, description="0 = no risk, 10 = extreme risk")
    key_vulnerabilities: dict[str, Any] = Field(default_factory=dict)
    # Active short positioning fields
    short_thesis: str = Field(default="", description="Active short pitch if warranted")
    actionable_recommendation: str = Field(
        default="AVOID",
        description="AVOID / UNDERWEIGHT / ACTIVE SHORT / HEDGE — not just defensive",
    )
    technical_levels: dict[str, Any] = Field(
        default_factory=dict,
        description="Key technical levels: support, resistance, breakdown triggers",
    )


class MacroView(BaseModel):
    """Structured output from the Macro Analyst — top-down economic context."""

    ticker: str
    economic_cycle_phase: str = Field(
        default="",
        description="Current cycle phase: early expansion, mid-cycle, late cycle, recession",
    )
    cycle_evidence: list[str] = Field(
        default_factory=list,
        description="Evidence supporting the cycle assessment",
    )
    rate_environment: str = Field(
        default="",
        description="Rate regime: tightening, pausing, easing, QE",
    )
    central_bank_outlook: str = Field(
        default="",
        description="Expected central bank trajectory over 6-12 months",
    )
    sector_positioning: str = Field(
        default="",
        description="Where this stock's sector sits in the rotation cycle",
    )
    rotation_implications: str = Field(
        default="",
        description="What sector rotation means for this stock",
    )
    geopolitical_risks: list[str] = Field(
        default_factory=list,
        description="Key geopolitical risks affecting this stock",
    )
    cross_asset_signals: dict[str, str] = Field(
        default_factory=dict,
        description="Signals from other asset classes: bonds, commodities, FX, credit",
    )
    macro_impact_on_stock: str = Field(
        default="",
        description="Net macro impact narrative: how the macro backdrop affects this specific stock",
    )
    macro_favorability: float = Field(
        ge=0.0, le=10.0, default=5.0,
        description="0 = extremely hostile macro, 10 = perfect macro tailwind",
    )
    tailwinds: list[str] = Field(
        default_factory=list,
        description="Macro factors working in the stock's favor",
    )
    headwinds: list[str] = Field(
        default_factory=list,
        description="Macro factors working against the stock",
    )


class CommitteeMemo(BaseModel):
    """Final synthesized output from the Portfolio Manager."""

    ticker: str
    recommendation: str = Field(description="STRONG BUY / BUY / HOLD / UNDERWEIGHT / SELL / ACTIVE SHORT")
    position_size: str = Field(description="e.g. 'Full position', 'Half position', 'No position'")
    conviction: float = Field(ge=0.0, le=10.0)
    thesis_summary: str = ""
    key_factors: list[str] = Field(default_factory=list)
    bull_points_accepted: list[str] = Field(default_factory=list)
    bear_points_accepted: list[str] = Field(default_factory=list)
    dissenting_points: list[str] = Field(default_factory=list)
    risk_mitigants: list[str] = Field(default_factory=list)
    time_horizon: str = ""


class Rebuttal(BaseModel):
    """A structured rebuttal from one agent to another's analysis."""

    agent_role: AgentRole
    responding_to: AgentRole
    points: list[str] = Field(default_factory=list)
    concessions: list[str] = Field(default_factory=list)
    revised_conviction: Optional[float] = None


# ---------------------------------------------------------------------------
# Tool call parsing
# ---------------------------------------------------------------------------

# Matches lines like: - get_earnings_history(ticker="NVDA")
_TOOL_CALL_PATTERN = re.compile(r'^-\s+(\w+)\((.+)\)\s*$', re.MULTILINE)


def parse_tool_calls(plan_output: str) -> list[dict[str, Any]]:
    """
    Parse TOOL_CALLS: block from an agent's plan output.

    Expected format at end of plan text:
        TOOL_CALLS:
        - get_earnings_history(ticker="NVDA")
        - compare_peers(ticker="NVDA")

    Returns:
        List of {"tool_name": str, "kwargs": dict} dicts.
        Empty list if no TOOL_CALLS block found.
    """
    # Find the TOOL_CALLS: block
    marker = "TOOL_CALLS:"
    idx = plan_output.find(marker)
    if idx == -1:
        return []

    tool_section = plan_output[idx + len(marker):]
    calls = []

    for match in _TOOL_CALL_PATTERN.finditer(tool_section):
        tool_name = match.group(1)
        args_str = match.group(2).strip()

        # Parse kwargs: convert "key=value, key2=value2" to dict
        kwargs = _parse_kwargs(args_str)
        calls.append({"tool_name": tool_name, "kwargs": kwargs})

    return calls


def _parse_kwargs(args_str: str) -> dict[str, Any]:
    """
    Parse key=value kwargs string into a dict.

    Handles: ticker="NVDA", max_peers=3, flag=True
    Uses ast.literal_eval for each value to preserve types (int, bool, etc).
    """
    kwargs: dict[str, Any] = {}
    # Split on commas, but be careful with quoted strings containing commas
    parts = _smart_split(args_str)
    for part in parts:
        part = part.strip()
        if "=" not in part:
            continue
        key, value_str = part.split("=", 1)
        key = key.strip()
        value_str = value_str.strip()
        try:
            # ast.literal_eval handles strings, ints, floats, bools, None, lists, dicts
            kwargs[key] = ast.literal_eval(value_str)
        except (ValueError, SyntaxError):
            # Last resort: treat as plain string
            kwargs[key] = value_str.strip('"').strip("'")
    return kwargs


def _smart_split(s: str) -> list[str]:
    """Split on commas that aren't inside quotes."""
    parts = []
    current = []
    in_quote = False
    quote_char = None

    for ch in s:
        if ch in ('"', "'") and not in_quote:
            in_quote = True
            quote_char = ch
            current.append(ch)
        elif ch == quote_char and in_quote:
            in_quote = False
            quote_char = None
            current.append(ch)
        elif ch == ',' and not in_quote:
            parts.append(''.join(current))
            current = []
        else:
            current.append(ch)

    if current:
        parts.append(''.join(current))
    return parts


# ---------------------------------------------------------------------------
# Base Agent ABC
# ---------------------------------------------------------------------------

class BaseInvestmentAgent(ABC):
    """
    Abstract base class for all investment committee agents.

    Implements the think → plan → [execute_tools] → act → reflect reasoning loop.
    Subclasses implement the domain-specific logic for each step.

    v2: Optional tool_registry enables dynamic tool calling between plan and act.
    """

    def __init__(self, model: Any, role: AgentRole, tool_registry: Any = None):
        self.model = model
        self.role = role
        self.tool_registry = tool_registry
        self.trace: Optional[ReasoningTrace] = None

    def _start_trace(self, ticker: str) -> ReasoningTrace:
        self.trace = ReasoningTrace(agent_role=self.role, ticker=ticker)
        return self.trace

    def _record_step(
        self,
        step_type: StepType,
        content: str,
        duration_ms: float = 0.0,
        tokens_used: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> ReasoningStep:
        step = ReasoningStep(
            step_type=step_type,
            agent_role=self.role,
            content=content,
            duration_ms=duration_ms,
            tokens_used=tokens_used,
            metadata=metadata or {},
        )
        if self.trace:
            self.trace.add_step(step)
        return step

    def execute_tools(self, plan_output: str) -> dict[str, Any]:
        """
        Parse TOOL_CALLS block from plan output and execute requested tools.

        Uses the tool_registry for lookup and budget enforcement.

        Args:
            plan_output: The raw plan text from the agent's plan() step.

        Returns:
            {"tool_results": {tool_name: result, ...}, "tools_called": [...]}
            Empty dict if no tools requested or no registry available.
        """
        if not self.tool_registry:
            return {"tool_results": {}, "tools_called": []}

        calls = parse_tool_calls(plan_output)
        if not calls:
            return {"tool_results": {}, "tools_called": []}

        agent_id = self.role.value
        tool_results: dict[str, Any] = {}
        tools_called: list[str] = []

        for call in calls:
            tool_name = call["tool_name"]
            kwargs = call["kwargs"]

            logger.info(f"[{agent_id}] Executing tool: {tool_name}({kwargs})")
            result = self.tool_registry.execute(agent_id, tool_name, kwargs)
            tool_results[tool_name] = result
            tools_called.append(tool_name)

            # Stop if budget exceeded (the registry returns error dicts)
            if isinstance(result, dict) and result.get("skipped"):
                logger.info(f"[{agent_id}] Tool budget reached, stopping further calls.")
                break

        return {"tool_results": tool_results, "tools_called": tools_called}

    def get_tool_catalog(self) -> str:
        """Get formatted tool catalog for prompt injection. Empty string if no registry."""
        if not self.tool_registry:
            return ""
        return self.tool_registry.get_catalog()

    def run(self, ticker: str, context: dict[str, Any]) -> dict[str, Any]:
        """
        Execute the full reasoning loop.

        v2: Inserts execute_tools() between plan and act when tool_registry
        is available. Tool results are passed to act() as additional context.

        Returns a dict containing the agent's output and reasoning trace.
        """
        trace = self._start_trace(ticker)

        # 1. THINK — assess the situation
        t0 = time.time()
        thinking = self.think(ticker, context)
        self._record_step(StepType.THINK, thinking, duration_ms=(time.time() - t0) * 1000)

        # 2. PLAN — decide what tools/data to gather
        t0 = time.time()
        plan = self.plan(ticker, context, thinking)
        self._record_step(StepType.PLAN, plan, duration_ms=(time.time() - t0) * 1000)

        # 2.5 EXECUTE TOOLS — dynamic tool calling (new in v2)
        tool_results: dict[str, Any] = {}
        if self.tool_registry:
            t0 = time.time()
            tool_data = self.execute_tools(plan)
            tool_results = tool_data.get("tool_results", {})
            tools_called = tool_data.get("tools_called", [])
            if tools_called:
                self._record_step(
                    StepType.TOOL_CALL,
                    f"Called {len(tools_called)} tools: {', '.join(tools_called)}",
                    duration_ms=(time.time() - t0) * 1000,
                    metadata={"tools_called": tools_called, "results_summary": list(tool_results.keys())},
                )

        # 3. ACT — execute the plan (now with optional tool_results)
        t0 = time.time()
        result = self.act(ticker, context, plan, tool_results=tool_results)
        self._record_step(StepType.ACT, str(result), duration_ms=(time.time() - t0) * 1000)

        # 4. REFLECT — evaluate output quality
        t0 = time.time()
        reflection = self.reflect(ticker, result)
        self._record_step(StepType.REFLECT, reflection, duration_ms=(time.time() - t0) * 1000)

        trace.finalize()

        return {
            "output": result,
            "trace": trace,
        }

    @abstractmethod
    def think(self, ticker: str, context: dict[str, Any]) -> str:
        """Assess the situation and form initial hypotheses."""
        ...

    @abstractmethod
    def plan(self, ticker: str, context: dict[str, Any], thinking: str) -> str:
        """Decide what data to gather and how to analyze it."""
        ...

    @abstractmethod
    def act(
        self,
        ticker: str,
        context: dict[str, Any],
        plan: str,
        tool_results: dict[str, Any] | None = None,
    ) -> Any:
        """Execute analysis — call tools, compute metrics, build thesis."""
        ...

    @abstractmethod
    def reflect(self, ticker: str, result: Any) -> str:
        """Evaluate output quality and identify gaps or biases."""
        ...

    @abstractmethod
    def rebut(self, ticker: str, opposing_view: Any, own_result: Any) -> Rebuttal:
        """Produce a rebuttal to the opposing agent's analysis."""
        ...
