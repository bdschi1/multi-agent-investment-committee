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
import json
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
    # Sentiment extraction fields
    sentiment_factors: list[dict[str, str]] = Field(
        default_factory=list,
        description="Extracted sentiment factors from news: [{headline, sentiment, signal_strength, catalyst_type}, ...]",
    )
    aggregate_news_sentiment: str = Field(
        default="neutral",
        description="Overall news sentiment: strongly_bullish / bullish / neutral / bearish / strongly_bearish",
    )
    sentiment_divergence: str = Field(
        default="",
        description="Where news sentiment diverges from price action or fundamentals",
    )
    # Quantitative heuristic fields (LLM-estimated, not computed)
    # NOTE: These are heuristic estimates from LLM reasoning, not precise calculations.
    # The agents use available market data to approximate these values as part of their
    # qualitative-quantitative reasoning process.
    price_target: str = Field(
        default="",
        description="Price target + reasoning (e.g. '$185 in 12m — DCF with 12% WACC, 25x terminal multiple on normalized FCF')",
    )
    forecasted_total_return: str = Field(
        default="",
        description="Total return estimate + how derived (e.g. '22% — (185/152)-1, includes 1.2% div yield')",
    )
    estimated_industry_return: str = Field(
        default="",
        description="Sector return estimate + reasoning (e.g. '8% — semis historically track GDP+4%, current cycle mid-expansion')",
    )
    idiosyncratic_return: str = Field(
        default="",
        description="Idio return (alpha) + reasoning (e.g. '14% — 22% total minus 8% sector; driven by share gains + margin expansion')",
    )
    estimated_sharpe: str = Field(
        default="",
        description="Heuristic Sharpe + reasoning (e.g. '0.47 — 14% idio / 30% vol; moderate, reflects binary FDA outcome')",
    )
    estimated_sortino: str = Field(
        default="",
        description="Heuristic Sortino + reasoning (e.g. '0.64 — 14% idio / 22% downside vol; downside limited by asset floor')",
    )
    # Risk-unit framework (ported from llm-long-short-arena)
    conviction_levers: list[ConvictionLever] = Field(
        default_factory=list,
        description="Key levers that change sizing/conviction (e.g., margins, FDA outcome, guidance revision)",
    )
    risk_sizing: RiskSizing | None = Field(
        default=None,
        description="Position sizing in risk units with role-in-book and drawdown tolerance",
    )


class BearCase(BaseModel):
    """Structured output from the Risk Manager."""

    ticker: str
    risks: list[str] = Field(default_factory=list)
    second_order_effects: list[str] = Field(default_factory=list)
    third_order_effects: list[str] = Field(default_factory=list)
    worst_case_scenario: str = ""
    bearish_conviction: float = Field(ge=0.0, le=10.0, description="0 = minimal concern, 10 = maximum bearish conviction")
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
    # Risk-unit framework (ported from llm-long-short-arena)
    risk_sizing: RiskSizing | None = Field(
        default=None,
        description="Short/hedge sizing in risk units with drawdown triggers",
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
    # Portfolio strategist fields
    annualized_vol_regime: str = Field(
        default="",
        description="Current vol regime: low (<12), normal (12-18), elevated (18-25), crisis (>25)",
    )
    vol_budget_guidance: str = Field(
        default="",
        description="Guidance on position sizing within annualized vol boundaries",
    )
    portfolio_directionality: str = Field(
        default="",
        description="Recommended net exposure direction: net long / market neutral / net short",
    )
    sector_style_assessment: str = Field(
        default="",
        description="Sector and style agnostic assessment: growth vs value rotation, large vs small cap, defensive vs cyclical",
    )
    correlation_regime: str = Field(
        default="",
        description="Current correlation regime and diversification implications",
    )
    # Quantitative portfolio construction heuristics (LLM-estimated, not computed)
    # NOTE: These are heuristic reasoning outputs, not precise calculations.
    sector_avg_volatility: str = Field(
        default="",
        description="Estimated annualized vol for this stock's sector (e.g. '22% annualized')",
    )
    recommended_sizing_method: str = Field(
        default="",
        description=(
            "Recommended NMV sizing method given the regime: "
            "proportional (NMV=k*alpha) / risk_parity (NMV=k*alpha/sigma) / "
            "mean_variance (NMV=k*alpha/sigma^2) / shrunk_mean_variance "
            "(NMV=k*alpha/[p*sigma^2+(1-p)*sigma_sector^2])"
        ),
    )
    portfolio_vol_target: str = Field(
        default="",
        description=(
            "Recommended portfolio annualized vol target and rationale. "
            "Vol targeting controls risk better than GMV targeting because "
            "volatility is persistent and partially predictable."
        ),
    )


# ---------------------------------------------------------------------------
# Risk-unit framework (ported from llm-long-short-arena)
# ---------------------------------------------------------------------------

class ConvictionLever(BaseModel):
    """A feature that changes position sizing or conviction.

    Institutional PMs think in terms of levers — specific developments
    that move a position from Tactical to Core, or trigger a cut.
    Each lever has a sensitivity score indicating how thesis-dependent
    the position is on that particular factor.
    """

    lever_name: str = Field(
        ..., description="Name of the feature (e.g., 'Margins', 'FDA Approval')"
    )
    impact_description: str = Field(
        ..., description="How this lever changes sizing/conviction"
    )
    sensitivity_score: int = Field(
        ge=1, le=5, description="1-5: How sensitive is the thesis to this lever?"
    )


class RiskSizing(BaseModel):
    """Position sizing in risk units with drawdown tolerance.

    Risk units (1-10 scale) abstract away dollar amounts, focusing
    on relative portfolio risk budget. ``role_in_book`` classifies the
    position's strategic purpose; ``max_drawdown_tolerance`` defines
    the loss threshold that triggers a cut.
    """

    risk_units: float = Field(
        ge=0.0, le=10.0, description="Proposed size in risk units (1-10 scale)"
    )
    role_in_book: str = Field(
        default="Tactical",
        description="Core / Tactical / Hedge / Pair Leg",
    )
    max_drawdown_tolerance: str = Field(
        default="",
        description="At what drawdown do you cut? (e.g., '-15% from entry')",
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
    # Trading-fluent PM fields (PM consults closely with his head trader)
    implied_vol_assessment: str = Field(
        default="",
        description="Assessment of implied vs historical vol and what it signals",
    )
    event_path: list[str] = Field(
        default_factory=list,
        description="Ordered sequence of near-term events that define the trade path",
    )
    conviction_change_triggers: dict[str, str] = Field(
        default_factory=dict,
        description="Specific triggers that would change conviction: {trigger: impact}",
    )
    factor_exposures: dict[str, str] = Field(
        default_factory=dict,
        description="Key factor exposures: momentum, value, quality, size, volatility",
    )
    # Quantitative heuristic synthesis (LLM-estimated, not computed)
    # NOTE: These are heuristic reasoning outputs — the PM reasons through quant
    # frameworks qualitatively using available market data, not running optimizers.
    idio_return_estimate: str = Field(
        default="",
        description="PM's validated idio return estimate after weighing bull/bear (e.g. '+8% over 12m')",
    )
    sharpe_estimate: str = Field(
        default="",
        description=(
            "Heuristic Sharpe ratio: idio_return / vol. For shorts, uses -1*expected_return. "
            "Reasoning, not precise math."
        ),
    )
    sortino_estimate: str = Field(
        default="",
        description=(
            "Heuristic Sortino ratio: idio_return / downside_vol. For shorts, uses -1*expected_return. "
            "Reasoning, not precise math."
        ),
    )
    sizing_method_used: str = Field(
        default="",
        description=(
            "Which NMV sizing method the PM applied: proportional / risk_parity / "
            "mean_variance / shrunk_mean_variance, and why"
        ),
    )
    target_nmv_rationale: str = Field(
        default="",
        description=(
            "Rationale for target net market value: NMV proportional to forecasted alpha, "
            "adjusted for idio vol and sector vol per the chosen sizing method"
        ),
    )
    vol_target_rationale: str = Field(
        default="",
        description=(
            "Why the PM chose a specific vol target for this position. "
            "Vol targeting > GMV targeting because vol is persistent and predictable."
        ),
    )
    # T signal: direction * entropy-adjusted confidence
    position_direction: int = Field(
        default=0,
        description="Position direction: +1 (long), -1 (short), 0 (no position)",
    )
    raw_confidence: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Scaled confidence score [0,1]",
    )
    t_signal: float = Field(
        default=0.0,
        description="T = direction * entropy-adjusted confidence. Range [-1, 1].",
    )


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

    Handles: ticker="NVDA", max_peers=3, flag=True, fundamentals={"roe": "25%"}
    Uses ast.literal_eval for each value to preserve types (int, bool, etc).
    Falls back to json.loads for JSON-formatted values, then plain string.
    """
    kwargs: dict[str, Any] = {}
    # Split on commas, but respect quotes and braces/brackets
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
            # Try json.loads for JSON-formatted values (e.g., dicts with double-quoted keys)
            try:
                parsed = json.loads(value_str)
                kwargs[key] = parsed
            except (json.JSONDecodeError, ValueError):
                # Last resort: treat as plain string
                kwargs[key] = value_str.strip('"').strip("'")
    return kwargs


def _smart_split(s: str) -> list[str]:
    """Split on commas that aren't inside quotes, braces, or brackets."""
    parts = []
    current = []
    in_quote = False
    quote_char = None
    depth = 0  # tracks {}, [] nesting

    for ch in s:
        if ch in ('"', "'") and not in_quote:
            in_quote = True
            quote_char = ch
            current.append(ch)
        elif ch == quote_char and in_quote:
            in_quote = False
            quote_char = None
            current.append(ch)
        elif not in_quote and ch in ('{', '['):
            depth += 1
            current.append(ch)
        elif not in_quote and ch in ('}', ']'):
            depth = max(0, depth - 1)
            current.append(ch)
        elif ch == ',' and not in_quote and depth == 0:
            parts.append(''.join(current))
            current = []
        else:
            current.append(ch)

    if current:
        parts.append(''.join(current))
    return parts


# ---------------------------------------------------------------------------
# Shared JSON extraction — replaces per-agent _extract_json() duplicates
# ---------------------------------------------------------------------------

def _find_brace_bounded(text: str) -> str:
    """Find the outermost { ... } substring in text."""
    start = text.index("{")
    end = text.rindex("}") + 1
    return text[start:end]


def _fix_trailing_commas(text: str) -> str:
    """Remove trailing commas before } and ]."""
    return re.sub(r',\s*([}\]])', r'\1', text)


def _fix_single_quotes(text: str) -> str:
    """Replace single quotes with double quotes for JSON keys/values."""
    # Only operate on the brace-bounded portion to avoid breaking prose
    return text.replace("'", '"')


def _fix_unbalanced_braces(text: str) -> str:
    """Append missing closing braces/brackets."""
    opens = text.count("{") - text.count("}")
    brackets = text.count("[") - text.count("]")
    if opens > 0:
        text += "}" * opens
    if brackets > 0:
        text += "]" * brackets
    return text


def extract_json(text: str) -> tuple[dict, bool]:
    """
    Extract a JSON dict from LLM output with progressive repair strategies.

    Tries increasingly aggressive repair in order:
      1. Direct parse
      2. Markdown ```json block
      3. Markdown ``` block
      4. Brace boundary ({...})
      5. Fix trailing commas
      6. Single-quote → double-quote
      7. Unbalanced brace repair

    Returns:
        (parsed_dict, was_repaired) — was_repaired is True if anything
        beyond direct parse was needed.

    Raises:
        ValueError if all strategies fail.
    """
    text = text.strip()

    # Strategy 1: direct parse
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result, False
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: ```json ... ``` markdown block
    if "```json" in text:
        try:
            start = text.index("```json") + 7
            end = text.index("```", start)
            candidate = text[start:end].strip()
            result = json.loads(candidate)
            if isinstance(result, dict):
                return result, True
        except (json.JSONDecodeError, ValueError, IndexError):
            pass

    # Strategy 3: ``` ... ``` generic code block
    if "```" in text:
        try:
            start = text.index("```") + 3
            # Skip optional language tag on same line
            newline = text.find("\n", start)
            if newline != -1 and newline - start < 20:
                start = newline + 1
            end = text.index("```", start)
            candidate = text[start:end].strip()
            result = json.loads(candidate)
            if isinstance(result, dict):
                return result, True
        except (json.JSONDecodeError, ValueError, IndexError):
            pass

    # Strategy 4: brace boundary — first { to last }
    try:
        candidate = _find_brace_bounded(text)
        result = json.loads(candidate)
        if isinstance(result, dict):
            return result, True
    except (json.JSONDecodeError, ValueError, IndexError):
        pass

    # From here, all strategies operate on the best candidate we can find
    try:
        best = _find_brace_bounded(text)
    except (ValueError, IndexError):
        best = text  # no braces at all — try repairs on raw text

    # Strategy 5: fix trailing commas
    try:
        repaired = _fix_trailing_commas(best)
        result = json.loads(repaired)
        if isinstance(result, dict):
            return result, True
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 6: single-quote → double-quote
    try:
        repaired = _fix_single_quotes(best)
        repaired = _fix_trailing_commas(repaired)  # also fix commas
        result = json.loads(repaired)
        if isinstance(result, dict):
            return result, True
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 7: unbalanced brace repair
    try:
        repaired = _fix_trailing_commas(best)
        repaired = _fix_unbalanced_braces(repaired)
        result = json.loads(repaired)
        if isinstance(result, dict):
            return result, True
    except (json.JSONDecodeError, ValueError):
        pass

    raise ValueError(
        f"All JSON extraction strategies failed for text of length {len(text)}"
    )


# ---------------------------------------------------------------------------
# JSON artifact cleaner — for fallback text fields
# ---------------------------------------------------------------------------

def clean_json_artifacts(text: str, max_length: int = 500) -> str:
    """
    Clean JSON syntax from raw LLM output for display in fallback text fields.

    When JSON parsing fails completely, agents stuff response_text[:500] into
    text fields like thesis, worst_case_scenario, etc. This function strips
    JSON artifacts so users see readable prose instead of raw JSON.

    Args:
        text: Raw LLM response text (may contain JSON)
        max_length: Maximum output length

    Returns:
        Cleaned text suitable for display, truncated to max_length.
    """
    if not text or not text.strip():
        return ""

    text = text.strip()

    # If it looks like JSON, try to extract the longest string value
    if text.startswith("{") or text.startswith("[") or '"' in text[:50]:
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                # Find the longest string value — likely the most useful content
                best_val = ""
                for v in parsed.values():
                    if isinstance(v, str) and len(v) > len(best_val):
                        best_val = v
                if best_val:
                    return best_val[:max_length]
        except (json.JSONDecodeError, ValueError):
            pass

        # Try brace-bounded extraction first
        try:
            bounded = _find_brace_bounded(text)
            parsed = json.loads(_fix_trailing_commas(bounded))
            if isinstance(parsed, dict):
                best_val = ""
                for v in parsed.values():
                    if isinstance(v, str) and len(v) > len(best_val):
                        best_val = v
                if best_val:
                    return best_val[:max_length]
        except (json.JSONDecodeError, ValueError, IndexError):
            pass

    # Regex cleanup: remove JSON syntax artifacts
    cleaned = text
    # Remove "key": patterns
    cleaned = re.sub(r'"[a-zA-Z_][a-zA-Z0-9_]*"\s*:', ' ', cleaned)
    # Remove braces, brackets, and stray quotes
    cleaned = re.sub(r'[{}\[\]]', ' ', cleaned)
    # Remove standalone double-quotes (not part of contractions)
    cleaned = re.sub(r'(?<!\w)"(?!\w)', ' ', cleaned)
    # Collapse whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    # Remove leading/trailing commas
    cleaned = cleaned.strip(',').strip()

    return cleaned[:max_length] if cleaned else text[:max_length]


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
