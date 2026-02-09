"""
Base agent protocol for the Investment Committee.

Every agent follows a structured reasoning loop:
    think → plan → act → reflect

This ensures transparent, auditable reasoning chains — not just
prompt-in / answer-out black boxes.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


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
# Base Agent ABC
# ---------------------------------------------------------------------------

class BaseInvestmentAgent(ABC):
    """
    Abstract base class for all investment committee agents.

    Implements the think → plan → act → reflect reasoning loop.
    Subclasses implement the domain-specific logic for each step.
    """

    def __init__(self, model: Any, role: AgentRole):
        self.model = model
        self.role = role
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

    def run(self, ticker: str, context: dict[str, Any]) -> dict[str, Any]:
        """
        Execute the full reasoning loop.

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

        # 3. ACT — execute the plan (tool calls, analysis)
        t0 = time.time()
        result = self.act(ticker, context, plan)
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
    def act(self, ticker: str, context: dict[str, Any], plan: str) -> Any:
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
