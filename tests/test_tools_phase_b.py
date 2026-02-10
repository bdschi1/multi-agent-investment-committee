"""
Tests for Phase B: Dynamic Tool Calling.

Validates:
    - ToolRegistry registration, lookup, execution, budget enforcement
    - TOOL_CALLS parsing from plan output
    - New tools (PeerComparison, InsiderData, EarningsData) structure
    - Agent run with tool_registry (tool_results reach act())
    - Backward compatibility: agents work without tool_registry
    - Full graph integration with tool registry
"""

from __future__ import annotations

import json
import pytest

from agents.base import parse_tool_calls, AgentRole
from tools.registry import ToolRegistry, ToolSpec, build_default_registry


# ---------------------------------------------------------------------------
# ToolRegistry Tests
# ---------------------------------------------------------------------------

class TestToolRegistry:
    """Tests for the ToolRegistry class."""

    def test_register_and_lookup(self):
        """Registry should store and retrieve tool specs."""
        registry = ToolRegistry(max_calls_per_agent=5)
        spec = ToolSpec(
            name="test_tool",
            description="A test tool",
            func=lambda ticker: {"ticker": ticker, "data": "test"},
            parameters={"ticker": "Stock ticker"},
        )
        registry.register(spec)

        assert "test_tool" in registry.list_tools()
        assert registry.get_spec("test_tool") is spec
        assert registry.get_spec("nonexistent") is None

    def test_get_catalog(self):
        """Catalog should format tool info for prompt injection."""
        registry = ToolRegistry(max_calls_per_agent=5)
        registry.register(ToolSpec(
            name="get_data",
            description="Get some data",
            func=lambda: {},
            parameters={"ticker": "Symbol"},
        ))
        catalog = registry.get_catalog()
        assert "get_data" in catalog
        assert "Get some data" in catalog
        assert "ticker" in catalog

    def test_execute_success(self):
        """Execute should call the function and return result."""
        registry = ToolRegistry(max_calls_per_agent=5)
        registry.register(ToolSpec(
            name="add_numbers",
            description="Add two numbers",
            func=lambda a, b: {"sum": a + b},
            parameters={"a": "First number", "b": "Second number"},
        ))

        result = registry.execute("test_agent", "add_numbers", {"a": 3, "b": 4})
        assert result == {"sum": 7}

    def test_execute_unknown_tool(self):
        """Unknown tool should return error dict, not raise."""
        registry = ToolRegistry(max_calls_per_agent=5)
        result = registry.execute("test_agent", "nonexistent", {})
        assert "error" in result
        assert "Unknown tool" in result["error"]

    def test_budget_enforcement(self):
        """Registry should stop after max_calls_per_agent."""
        registry = ToolRegistry(max_calls_per_agent=2)
        registry.register(ToolSpec(
            name="cheap_tool",
            description="A cheap tool",
            func=lambda: {"ok": True},
            parameters={},
        ))

        # First two calls should succeed
        r1 = registry.execute("agent_a", "cheap_tool", {})
        assert r1 == {"ok": True}
        r2 = registry.execute("agent_a", "cheap_tool", {})
        assert r2 == {"ok": True}

        # Third call should be rejected
        r3 = registry.execute("agent_a", "cheap_tool", {})
        assert "error" in r3
        assert r3.get("skipped") is True

        # Different agent should still work
        r4 = registry.execute("agent_b", "cheap_tool", {})
        assert r4 == {"ok": True}

    def test_reset_budget(self):
        """Reset should allow agent to call tools again."""
        registry = ToolRegistry(max_calls_per_agent=1)
        registry.register(ToolSpec(
            name="tool",
            description="Tool",
            func=lambda: {"ok": True},
        ))

        registry.execute("agent_a", "tool", {})
        r2 = registry.execute("agent_a", "tool", {})
        assert r2.get("skipped") is True

        registry.reset_budget("agent_a")
        r3 = registry.execute("agent_a", "tool", {})
        assert r3 == {"ok": True}

    def test_get_usage(self):
        """Usage stats should track calls correctly."""
        registry = ToolRegistry(max_calls_per_agent=5)
        registry.register(ToolSpec(name="t", description="", func=lambda: {}))

        usage = registry.get_usage("agent_x")
        assert usage["calls_used"] == 0
        assert usage["calls_remaining"] == 5

        registry.execute("agent_x", "t", {})
        registry.execute("agent_x", "t", {})

        usage = registry.get_usage("agent_x")
        assert usage["calls_used"] == 2
        assert usage["calls_remaining"] == 3

    def test_execute_handles_exception(self):
        """Tool execution failure should return error dict, not raise."""
        registry = ToolRegistry(max_calls_per_agent=5)
        registry.register(ToolSpec(
            name="broken",
            description="Broken tool",
            func=lambda: 1 / 0,
        ))

        result = registry.execute("agent", "broken", {})
        assert "error" in result
        assert "division by zero" in result["error"]


class TestBuildDefaultRegistry:
    """Tests for the default registry builder."""

    def test_builds_with_10_tools(self):
        """Default registry should have all 10 tools."""
        registry = build_default_registry(max_calls=5)
        tools = registry.list_tools()
        assert len(tools) == 10

        expected_tools = {
            "get_company_overview",
            "get_price_data",
            "get_price_data_extended",
            "get_fundamentals",
            "get_news",
            "compute_valuation",
            "compute_quality_score",
            "compare_peers",
            "get_insider_activity",
            "get_earnings_history",
        }
        assert set(tools) == expected_tools

    def test_catalog_renders(self):
        """Catalog should render without error."""
        registry = build_default_registry()
        catalog = registry.get_catalog()
        assert len(catalog) > 100  # Should be substantial
        assert "get_earnings_history" in catalog
        assert "compare_peers" in catalog


# ---------------------------------------------------------------------------
# Tool Call Parsing Tests
# ---------------------------------------------------------------------------

class TestParseToolCalls:
    """Tests for the TOOL_CALLS: block parser."""

    def test_parse_single_call(self):
        """Should parse a single tool call."""
        plan = """I will analyze the earnings.

TOOL_CALLS:
- get_earnings_history(ticker="NVDA")
"""
        calls = parse_tool_calls(plan)
        assert len(calls) == 1
        assert calls[0]["tool_name"] == "get_earnings_history"
        assert calls[0]["kwargs"] == {"ticker": "NVDA"}

    def test_parse_multiple_calls(self):
        """Should parse multiple tool calls."""
        plan = """My plan is to compare peers and check insiders.

TOOL_CALLS:
- compare_peers(ticker="AAPL")
- get_insider_activity(ticker="AAPL")
- get_earnings_history(ticker="AAPL")
"""
        calls = parse_tool_calls(plan)
        assert len(calls) == 3
        assert calls[0]["tool_name"] == "compare_peers"
        assert calls[1]["tool_name"] == "get_insider_activity"
        assert calls[2]["tool_name"] == "get_earnings_history"

    def test_parse_no_tool_calls(self):
        """Should return empty list when no TOOL_CALLS block."""
        plan = "I will analyze the fundamentals manually."
        calls = parse_tool_calls(plan)
        assert calls == []

    def test_parse_empty_tool_calls(self):
        """Should return empty list when TOOL_CALLS has no entries."""
        plan = """My plan.

TOOL_CALLS:
Nothing here.
"""
        calls = parse_tool_calls(plan)
        assert calls == []

    def test_parse_multiple_kwargs(self):
        """Should parse multiple keyword arguments."""
        plan = """Plan.

TOOL_CALLS:
- compare_peers(ticker="NVDA", max_peers=3)
"""
        calls = parse_tool_calls(plan)
        assert len(calls) == 1
        assert calls[0]["kwargs"]["ticker"] == "NVDA"
        assert calls[0]["kwargs"]["max_peers"] == 3

    def test_parse_dict_kwarg(self):
        """Should parse a dict-typed kwarg without splitting on internal commas."""
        plan = """Plan.

TOOL_CALLS:
- compute_quality_score(fundamentals={"roe": "25.0%", "profit_margin": "20.0%", "debt_to_equity": "0.5"})
"""
        calls = parse_tool_calls(plan)
        assert len(calls) == 1
        assert calls[0]["tool_name"] == "compute_quality_score"
        f = calls[0]["kwargs"]["fundamentals"]
        assert isinstance(f, dict), f"Expected dict, got {type(f).__name__}: {f!r}"
        assert f["roe"] == "25.0%"
        assert f["profit_margin"] == "20.0%"
        assert f["debt_to_equity"] == "0.5"

    def test_parse_dict_kwarg_with_other_args(self):
        """Should handle a mix of simple args and dict args."""
        plan = """Plan.

TOOL_CALLS:
- some_tool(ticker="NVDA", data={"a": 1, "b": 2}, limit=5)
"""
        calls = parse_tool_calls(plan)
        assert len(calls) == 1
        assert calls[0]["kwargs"]["ticker"] == "NVDA"
        assert calls[0]["kwargs"]["data"] == {"a": 1, "b": 2}
        assert calls[0]["kwargs"]["limit"] == 5


class TestSmartSplitBraces:
    """Tests for _smart_split with brace/bracket-containing values."""

    def test_split_respects_braces(self):
        """Should not split on commas inside curly braces."""
        from agents.base import _smart_split
        result = _smart_split('fundamentals={"roe": "25%", "margin": "20%"}')
        assert len(result) == 1
        assert result[0] == 'fundamentals={"roe": "25%", "margin": "20%"}'

    def test_split_respects_brackets(self):
        """Should not split on commas inside square brackets."""
        from agents.base import _smart_split
        result = _smart_split('items=["a", "b", "c"], limit=3')
        assert len(result) == 2
        assert result[0] == 'items=["a", "b", "c"]'
        assert result[1].strip() == 'limit=3'

    def test_split_nested_structures(self):
        """Should handle nested braces/brackets."""
        from agents.base import _smart_split
        result = _smart_split('data={"a": [1, 2], "b": {"x": 3}}, flag=True')
        assert len(result) == 2
        assert "flag=True" in result[1]


class TestRegistryCoercion:
    """Tests for the registry auto-coercion safety net."""

    def test_registry_coerces_string_dict_to_dict(self):
        """Registry should auto-coerce a stringified JSON dict before calling the tool."""
        received_args = {}

        def capture_tool(fundamentals):
            received_args["fundamentals"] = fundamentals
            return {"score": 5}

        registry = ToolRegistry(max_calls_per_agent=5)
        registry.register(ToolSpec(
            name="test_tool",
            description="test",
            func=capture_tool,
            parameters={"fundamentals": "dict"},
        ))

        # Simulate what happens when the parser returns a string instead of a dict
        result = registry.execute("test_agent", "test_tool", {
            "fundamentals": '{"roe": "25%", "margin": "20%"}'
        })
        assert result == {"score": 5}
        assert isinstance(received_args["fundamentals"], dict)
        assert received_args["fundamentals"]["roe"] == "25%"

    def test_registry_leaves_real_dicts_alone(self):
        """Registry should not modify kwargs that are already dicts."""
        received_args = {}

        def capture_tool(fundamentals):
            received_args["fundamentals"] = fundamentals
            return {"score": 5}

        registry = ToolRegistry(max_calls_per_agent=5)
        registry.register(ToolSpec(
            name="test_tool",
            description="test",
            func=capture_tool,
            parameters={"fundamentals": "dict"},
        ))

        original_dict = {"roe": "25%"}
        result = registry.execute("test_agent", "test_tool", {
            "fundamentals": original_dict
        })
        assert result == {"score": 5}
        assert received_args["fundamentals"] is original_dict


# ---------------------------------------------------------------------------
# Agent Integration Tests
# ---------------------------------------------------------------------------

class MockLLMWithTools:
    """Mock LLM that returns TOOL_CALLS in plan output."""

    def __init__(self):
        self.call_count = 0

    def __call__(self, prompt: str) -> str:
        self.call_count += 1
        prompt_lower = prompt.lower()

        # PLAN step — return TOOL_CALLS
        if "plan your analysis" in prompt_lower or "plan your" in prompt_lower:
            return """I will analyze earnings and compare peers.

TOOL_CALLS:
- get_earnings_history(ticker="TEST")
- compare_peers(ticker="TEST")
"""

        # Bull case (act step)
        if "build the bull case" in prompt_lower or "bull case" in prompt_lower:
            if "json" in prompt_lower:
                return json.dumps({
                    "ticker": "TEST",
                    "thesis": "Strong AI-driven growth with tool data",
                    "supporting_evidence": ["Revenue +40%", "Earnings beat trend"],
                    "catalysts": ["Product launch Q2"],
                    "conviction_score": 7.5,
                    "time_horizon": "6-12 months",
                    "key_metrics": {"pe": 30},
                })
            return "Thinking about bull case..."

        # Bear case
        if "bear case" in prompt_lower or "risk" in prompt_lower:
            if "json" in prompt_lower:
                return json.dumps({
                    "ticker": "TEST",
                    "risks": ["Valuation stretch"],
                    "second_order_effects": ["Margin pressure"],
                    "third_order_effects": ["Talent loss"],
                    "worst_case_scenario": "30% downside",
                    "bearish_conviction": 5.5,
                    "key_vulnerabilities": {"valuation": "Above mean"},
                })
            return "Thinking about risks..."

        # Macro
        if "macro" in prompt_lower or "top-down" in prompt_lower:
            if "json" in prompt_lower:
                return json.dumps({
                    "ticker": "TEST",
                    "economic_cycle_phase": "mid-cycle",
                    "cycle_evidence": ["GDP growing"],
                    "rate_environment": "pausing",
                    "central_bank_outlook": "Hold steady",
                    "sector_positioning": "Favorable",
                    "rotation_implications": "Tech leading",
                    "geopolitical_risks": [],
                    "cross_asset_signals": {},
                    "macro_impact_on_stock": "Positive",
                    "macro_favorability": 6.5,
                    "tailwinds": ["AI spending"],
                    "headwinds": [],
                })
            return "Thinking about macro..."

        # PM decision
        if "final decision" in prompt_lower or "committee memo" in prompt_lower:
            if "json" in prompt_lower:
                return json.dumps({
                    "ticker": "TEST",
                    "recommendation": "BUY",
                    "position_size": "Half position",
                    "conviction": 7.0,
                    "thesis_summary": "Growth outweighs risks",
                    "key_factors": ["Strong momentum"],
                    "bull_points_accepted": ["Revenue growth"],
                    "bear_points_accepted": ["Valuation elevated"],
                    "dissenting_points": [],
                    "risk_mitigants": ["Position sizing"],
                    "time_horizon": "6-12 months",
                })
            return "Synthesizing..."

        # Rebuttal
        if "rebut" in prompt_lower or "challenge" in prompt_lower:
            return json.dumps({
                "points": ["Counter-argument"],
                "concessions": ["Fair point"],
                "revised_conviction": 7.0,
            })

        return "Analysis step completed."


class TestAgentWithTools:
    """Test agents with tool registry integration."""

    def test_agent_run_with_tools(self):
        """Agent should execute tools from plan and pass results to act."""
        from agents.sector_analyst import SectorAnalystAgent

        # Create a simple mock registry
        mock_results = {"test_data": True}
        registry = ToolRegistry(max_calls_per_agent=5)
        registry.register(ToolSpec(
            name="get_earnings_history",
            description="Test",
            func=lambda ticker: {"ticker": ticker, "quarters": [], "trend": "test"},
            parameters={"ticker": "ticker"},
        ))
        registry.register(ToolSpec(
            name="compare_peers",
            description="Test",
            func=lambda ticker: {"ticker": ticker, "peers": []},
            parameters={"ticker": "ticker"},
        ))

        mock = MockLLMWithTools()
        agent = SectorAnalystAgent(model=mock, tool_registry=registry)
        result = agent.run("TEST", {"market_data": {}, "news": [], "financial_metrics": {}})

        assert result["output"].ticker == "TEST"
        assert result["trace"] is not None

        # Check that TOOL_CALL step was recorded
        step_types = [s.step_type.value for s in result["trace"].steps]
        assert "tool_call" in step_types

        # Check tool budget was consumed
        usage = registry.get_usage("sector_analyst")
        assert usage["calls_used"] == 2

    def test_backward_compat_no_registry(self):
        """Agent should work without tool_registry (v1 behavior)."""
        from agents.sector_analyst import SectorAnalystAgent

        mock = MockLLMWithTools()
        agent = SectorAnalystAgent(model=mock)
        result = agent.run("TEST", {"market_data": {}, "news": [], "financial_metrics": {}})

        assert result["output"].ticker == "TEST"
        assert result["trace"] is not None

        # No tool_call step should be present
        step_types = [s.step_type.value for s in result["trace"].steps]
        assert "tool_call" not in step_types

    def test_all_agents_accept_tool_registry(self):
        """All 4 agent constructors should accept tool_registry kwarg."""
        from agents.sector_analyst import SectorAnalystAgent
        from agents.risk_manager import RiskManagerAgent
        from agents.macro_analyst import MacroAnalystAgent
        from agents.portfolio_manager import PortfolioManagerAgent

        mock = MockLLMWithTools()
        registry = ToolRegistry(max_calls_per_agent=5)

        # Should not raise
        SectorAnalystAgent(model=mock, tool_registry=registry)
        RiskManagerAgent(model=mock, tool_registry=registry)
        MacroAnalystAgent(model=mock, tool_registry=registry)
        PortfolioManagerAgent(model=mock, tool_registry=registry)


# ---------------------------------------------------------------------------
# Graph Integration Test
# ---------------------------------------------------------------------------

class TestGraphWithTools:
    """Test full graph execution with tool registry."""

    def test_graph_run_with_tools(self):
        """Graph should run with tool registry and produce valid result."""
        from orchestrator.graph import run_graph
        from orchestrator.committee import CommitteeResult

        mock = MockLLMWithTools()

        # Create a mock registry with simple tools
        registry = ToolRegistry(max_calls_per_agent=5)
        registry.register(ToolSpec(
            name="get_earnings_history",
            description="Test",
            func=lambda ticker: {"ticker": ticker, "trend": "test"},
            parameters={"ticker": "ticker"},
        ))
        registry.register(ToolSpec(
            name="compare_peers",
            description="Test",
            func=lambda ticker: {"ticker": ticker, "peers": []},
            parameters={"ticker": "ticker"},
        ))

        result = run_graph(
            ticker="TEST",
            context={"market_data": {}, "news": [], "financial_metrics": {}},
            model=mock,
            max_debate_rounds=1,
            tool_registry=registry,
        )

        assert isinstance(result, CommitteeResult)
        assert result.ticker == "TEST"
        assert result.bull_case is not None
        assert result.bear_case is not None
        assert result.committee_memo is not None

    def test_graph_run_without_tools(self):
        """Graph should still work when tool_registry is explicitly None."""
        from orchestrator.graph import run_graph
        from orchestrator.committee import CommitteeResult

        mock = MockLLMWithTools()
        result = run_graph(
            ticker="TEST",
            context={"market_data": {}, "news": [], "financial_metrics": {}},
            model=mock,
            max_debate_rounds=1,
            tool_registry=None,
        )

        assert isinstance(result, CommitteeResult)
        assert result.ticker == "TEST"
