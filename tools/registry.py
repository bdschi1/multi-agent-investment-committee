"""
Tool Registry — central registration and execution engine for agent tools.

Agents request tools during plan() via a TOOL_CALLS: block. The registry
validates the tool name, enforces per-agent budgets, executes the call,
and returns structured results.

Usage:
    registry = build_default_registry(max_calls=5)
    catalog = registry.get_catalog()        # For agent prompt injection
    result = registry.execute("sector_analyst", "get_earnings_history", {"ticker": "NVDA"})
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable

from tools.market_data import MarketDataTool
from tools.news_retrieval import NewsRetrievalTool
from tools.financial_metrics import FinancialMetricsTool
from tools.peer_comparison import PeerComparisonTool
from tools.insider_data import InsiderDataTool
from tools.earnings_data import EarningsDataTool

logger = logging.getLogger(__name__)


@dataclass
class ToolSpec:
    """Specification for a registered tool."""

    name: str
    description: str
    func: Callable[..., Any]
    parameters: dict[str, str] = field(default_factory=dict)


class ToolRegistry:
    """
    Central tool registry with budget enforcement.

    Each agent can call up to max_calls_per_agent tools per run.
    The registry tracks calls per agent and refuses further calls
    once the budget is exhausted (returns an error dict, not exception).
    """

    def __init__(self, max_calls_per_agent: int = 5):
        self._tools: dict[str, ToolSpec] = {}
        self._call_counts: dict[str, int] = {}
        self.max_calls = max_calls_per_agent

    def register(self, spec: ToolSpec) -> None:
        """Register a tool. Overwrites if name already exists."""
        self._tools[spec.name] = spec

    def list_tools(self) -> list[str]:
        """Return registered tool names."""
        return list(self._tools.keys())

    def get_spec(self, name: str) -> ToolSpec | None:
        """Look up a tool spec by name."""
        return self._tools.get(name)

    def get_catalog(self) -> str:
        """
        Format all registered tools as a text catalog for prompt injection.

        Returns a readable string agents can use to decide which tools to call.
        """
        lines = []
        for spec in self._tools.values():
            params_str = ", ".join(f'{k}: {v}' for k, v in spec.parameters.items())
            lines.append(f"  - {spec.name}({params_str})")
            lines.append(f"    {spec.description}")
        return "\n".join(lines)

    def execute(
        self,
        agent_id: str,
        tool_name: str,
        kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Execute a registered tool with budget enforcement.

        Args:
            agent_id: Identifier for the calling agent (for budget tracking)
            tool_name: Name of the tool to execute
            kwargs: Keyword arguments to pass to the tool function

        Returns:
            Tool result dict, or error dict if tool not found or budget exceeded.
        """
        kwargs = kwargs or {}

        # Budget check
        count = self._call_counts.get(agent_id, 0)
        if count >= self.max_calls:
            logger.warning(
                f"Agent {agent_id} exceeded tool budget ({self.max_calls}). "
                f"Skipping {tool_name}."
            )
            return {
                "error": f"Tool budget exceeded ({count}/{self.max_calls})",
                "tool": tool_name,
                "skipped": True,
            }

        # Lookup tool
        spec = self._tools.get(tool_name)
        if spec is None:
            logger.warning(f"Unknown tool requested: {tool_name}")
            return {
                "error": f"Unknown tool: {tool_name}",
                "available_tools": self.list_tools(),
            }

        # Execute — auto-coerce string kwargs that look like JSON dicts/lists
        # (safety net for edge cases where the parser returns a stringified dict)
        try:
            coerced = {}
            for k, v in kwargs.items():
                if isinstance(v, str) and v.strip()[:1] in ('{', '['):
                    try:
                        coerced[k] = json.loads(v)
                    except (json.JSONDecodeError, ValueError):
                        coerced[k] = v
                else:
                    coerced[k] = v
            result = spec.func(**coerced)
            self._call_counts[agent_id] = count + 1
            logger.info(f"Agent {agent_id} called {tool_name} ({count + 1}/{self.max_calls})")
            return result
        except Exception as e:
            logger.error(f"Tool {tool_name} failed for agent {agent_id}: {e}")
            return {"error": f"Tool execution failed: {e}", "tool": tool_name}

    def reset_budget(self, agent_id: str) -> None:
        """Reset the call counter for an agent (e.g., between runs)."""
        self._call_counts.pop(agent_id, None)

    def get_usage(self, agent_id: str) -> dict[str, int]:
        """Return current usage stats for an agent."""
        return {
            "calls_used": self._call_counts.get(agent_id, 0),
            "calls_remaining": max(0, self.max_calls - self._call_counts.get(agent_id, 0)),
            "max_calls": self.max_calls,
        }


def build_default_registry(max_calls: int = 5) -> ToolRegistry:
    """
    Create a ToolRegistry pre-loaded with all available tools.

    Registers 10 tools wrapping existing and new data sources.
    """
    registry = ToolRegistry(max_calls_per_agent=max_calls)

    # --- Existing tools (wrapped from v1 static methods) ---

    registry.register(ToolSpec(
        name="get_company_overview",
        description="Get company info: sector, industry, market cap, description",
        func=MarketDataTool.get_company_overview,
        parameters={"ticker": "Stock ticker symbol (e.g. 'NVDA')"},
    ))

    registry.register(ToolSpec(
        name="get_price_data",
        description="Get 6-month price data: current price, returns, 52w high/low, volume",
        func=MarketDataTool.get_price_data,
        parameters={"ticker": "Stock ticker symbol"},
    ))

    registry.register(ToolSpec(
        name="get_price_data_extended",
        description="Get 1-year price data for longer-term trend analysis",
        func=partial(MarketDataTool.get_price_data, period="1y"),
        parameters={"ticker": "Stock ticker symbol"},
    ))

    registry.register(ToolSpec(
        name="get_fundamentals",
        description="Get fundamental data: P/E, margins, growth, balance sheet, analyst targets",
        func=MarketDataTool.get_fundamentals,
        parameters={"ticker": "Stock ticker symbol"},
    ))

    registry.register(ToolSpec(
        name="get_news",
        description="Get recent news headlines for a stock (up to 10 articles)",
        func=lambda ticker, max_articles=10: NewsRetrievalTool.format_for_agent(
            NewsRetrievalTool.get_news(ticker, max_articles=max_articles)
        ),
        parameters={"ticker": "Stock ticker symbol"},
    ))

    registry.register(ToolSpec(
        name="compute_valuation",
        description="Compute valuation assessment from fundamentals (P/E, PEG, EV/EBITDA flags)",
        func=FinancialMetricsTool.compute_valuation_assessment,
        parameters={"fundamentals": "Dict of fundamental data from get_fundamentals()"},
    ))

    registry.register(ToolSpec(
        name="compute_quality_score",
        description="Compute quality score from fundamentals (profitability, growth, balance sheet)",
        func=FinancialMetricsTool.compute_quality_score,
        parameters={"fundamentals": "Dict of fundamental data from get_fundamentals()"},
    ))

    # --- New tools (Phase B) ---

    registry.register(ToolSpec(
        name="compare_peers",
        description="Compare stock vs sector peers on P/E, growth, margins, ROE (auto-detects peers)",
        func=PeerComparisonTool.compare_peers,
        parameters={"ticker": "Stock ticker symbol"},
    ))

    registry.register(ToolSpec(
        name="get_insider_activity",
        description="Get insider transactions (buys/sells) and net sentiment signal",
        func=InsiderDataTool.get_insider_activity,
        parameters={"ticker": "Stock ticker symbol"},
    ))

    registry.register(ToolSpec(
        name="get_earnings_history",
        description="Get earnings history with beat/miss data and trend assessment",
        func=EarningsDataTool.get_earnings_history,
        parameters={"ticker": "Stock ticker symbol"},
    ))

    # --- Knowledge Base tools (optional — graceful fallback if KB not available) ---
    try:
        from tools.knowledge_base import KnowledgeBaseTool

        registry.register(ToolSpec(
            name="search_kb",
            description="Search curated knowledge base for relevant document chunks (quant, biotech, AI/ML, research)",
            func=KnowledgeBaseTool.search_kb,
            parameters={"query": "Natural language search query", "top_k": "Number of results (default 5)", "category": "Optional category filter"},
        ))

        registry.register(ToolSpec(
            name="ask_kb",
            description="Get formatted KB context string for LLM prompt injection",
            func=KnowledgeBaseTool.ask_kb,
            parameters={"query": "Natural language query", "top_k": "Number of chunks (default 5)", "category": "Optional category filter"},
        ))

        registry.register(ToolSpec(
            name="answer_kb",
            description="End-to-end KB answer with LLM synthesis and automatic fallback",
            func=KnowledgeBaseTool.answer_kb,
            parameters={"query": "Natural language question", "category": "Optional category filter"},
        ))

        logger.info("Knowledge base tools registered successfully")
    except ImportError:
        logger.info("Knowledge base not available — KB tools not registered")

    return registry
