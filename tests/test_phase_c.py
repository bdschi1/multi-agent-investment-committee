"""
Tests for Phase C: Human-in-the-Loop + Session Memory.

Tests cover:
    1. Two-phase graph execution (Phase 1 → intermediate state → Phase 2)
    2. Session memory (store, retrieve, clear, isolation)
    3. PM guidance injection
    4. Config pattern (nodes accept config, fall back to state)
    5. Backward compatibility (existing full-run still works)
"""

import json
from typing import Any

# ---------------------------------------------------------------------------
# Mock LLM — same pattern as test_graph.py / test_tools_phase_b.py
# ---------------------------------------------------------------------------

class MockLLMPhaseC:
    """Mock LLM that tracks calls and returns valid JSON for all agent stages."""

    def __init__(self):
        self.call_count = 0
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.call_count += 1
        self.prompts.append(prompt)
        prompt_lower = prompt.lower()

        # PLAN step — no tool calls needed for Phase C tests
        if "plan your analysis" in prompt_lower or "plan your" in prompt_lower or "plan the" in prompt_lower:
            return "I will analyze the data systematically."

        # PM decision — check BEFORE bull/bear/macro because PM prompts
        # quote analyst results and would otherwise match "bull case" first
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

        # Bull case
        if "build the bull case" in prompt_lower or "bull case" in prompt_lower:
            if "json" in prompt_lower:
                return json.dumps({
                    "ticker": "TEST",
                    "thesis": "Strong growth thesis",
                    "supporting_evidence": ["Revenue +40%"],
                    "catalysts": ["Product launch"],
                    "conviction_score": 7.5,
                    "time_horizon": "6-12 months",
                    "key_metrics": {"pe": 30},
                })
            return "Thinking about bull case..."

        # Short case
        if "short case" in prompt_lower or "short thesis" in prompt_lower or "short opportunity" in prompt_lower:
            if "json" in prompt_lower:
                return json.dumps({
                    "ticker": "TEST",
                    "short_thesis": "Overvalued vs peers",
                    "thesis_type": "alpha_short",
                    "event_path": ["Earnings miss"],
                    "supporting_evidence": ["Insider selling"],
                    "conviction_score": 4.5,
                    "key_vulnerabilities": {"valuation": "Premium"},
                })
            return "Thinking about short case..."

        # Bear case
        if "bear case" in prompt_lower or "risk" in prompt_lower:
            if "json" in prompt_lower:
                return json.dumps({
                    "ticker": "TEST",
                    "risks": ["Valuation risk"],
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
                    "macro_impact_on_stock": "Positive backdrop",
                    "macro_favorability": 6.5,
                    "tailwinds": ["AI spending"],
                    "headwinds": [],
                })
            return "Thinking about macro..."

        # Rebuttal
        if "rebut" in prompt_lower or "challenge" in prompt_lower:
            return json.dumps({
                "points": ["Counter-argument"],
                "concessions": ["Fair point"],
                "revised_conviction": 7.0,
            })

        # Default
        return "Analysis step completed."


def _make_context() -> dict[str, Any]:
    """Create a minimal valid context for testing."""
    return {
        "market_data": {"price": 100, "volume": 1_000_000},
        "news": [],
        "financial_metrics": {},
        "user_context": "",
    }


# ===========================================================================
# Test: Session Memory
# ===========================================================================

class TestSessionMemory:
    """Tests for orchestrator/memory.py."""

    def setup_method(self):
        """Clear session memory before each test."""
        from orchestrator.memory import clear_session
        clear_session()

    def test_store_and_retrieve(self):
        from orchestrator.memory import get_prior_analyses, store_analysis

        store_analysis("NVDA", {
            "recommendation": "BUY",
            "conviction": 7.5,
            "thesis_summary": "AI growth story",
        })

        prior = get_prior_analyses("NVDA")
        assert len(prior) == 1
        assert prior[0]["ticker"] == "NVDA"
        assert prior[0]["recommendation"] == "BUY"
        assert prior[0]["conviction"] == 7.5

    def test_case_insensitive_ticker(self):
        from orchestrator.memory import get_prior_analyses, store_analysis

        store_analysis("nvda", {"recommendation": "BUY"})
        prior = get_prior_analyses("NVDA")
        assert len(prior) == 1

    def test_multiple_analyses_same_ticker(self):
        from orchestrator.memory import get_prior_analyses, store_analysis

        store_analysis("NVDA", {"recommendation": "BUY", "conviction": 7.0})
        store_analysis("NVDA", {"recommendation": "HOLD", "conviction": 5.0})

        prior = get_prior_analyses("NVDA")
        assert len(prior) == 2
        assert prior[0]["recommendation"] == "BUY"
        assert prior[1]["recommendation"] == "HOLD"

    def test_different_tickers_isolated(self):
        from orchestrator.memory import get_prior_analyses, store_analysis

        store_analysis("NVDA", {"recommendation": "BUY"})
        store_analysis("AAPL", {"recommendation": "HOLD"})

        nvda = get_prior_analyses("NVDA")
        aapl = get_prior_analyses("AAPL")
        assert len(nvda) == 1
        assert len(aapl) == 1
        assert nvda[0]["recommendation"] == "BUY"
        assert aapl[0]["recommendation"] == "HOLD"

    def test_clear_session(self):
        from orchestrator.memory import clear_session, get_prior_analyses, store_analysis

        store_analysis("NVDA", {"recommendation": "BUY"})
        assert len(get_prior_analyses("NVDA")) == 1

        clear_session()
        assert len(get_prior_analyses("NVDA")) == 0

    def test_empty_retrieval(self):
        from orchestrator.memory import get_prior_analyses
        assert get_prior_analyses("UNKNOWN") == []

    def test_session_tickers(self):
        from orchestrator.memory import get_session_tickers, store_analysis

        store_analysis("NVDA", {"recommendation": "BUY"})
        store_analysis("AAPL", {"recommendation": "HOLD"})

        tickers = get_session_tickers()
        assert "NVDA" in tickers
        assert "AAPL" in tickers

    def test_session_summary(self):
        from orchestrator.memory import get_session_summary, store_analysis

        store_analysis("NVDA", {"recommendation": "BUY"})
        store_analysis("NVDA", {"recommendation": "HOLD"})
        store_analysis("AAPL", {"recommendation": "SELL"})

        summary = get_session_summary()
        assert summary["NVDA"] == 2
        assert summary["AAPL"] == 1


# ===========================================================================
# Test: Config Pattern
# ===========================================================================

class TestConfigPattern:
    """Tests for the config-based node parameter passing."""

    def test_get_model_from_config(self):
        from orchestrator.nodes import _get_model

        mock = MockLLMPhaseC()
        config = {"configurable": {"model": mock}}
        state = {"model": "should_not_be_used"}

        result = _get_model(state, config)
        assert result is mock

    def test_get_model_fallback_to_state(self):
        from orchestrator.nodes import _get_model

        mock = MockLLMPhaseC()
        state = {"model": mock}

        # No config
        assert _get_model(state, None) is mock
        # Empty config
        assert _get_model(state, {}) is mock
        # Config without model
        assert _get_model(state, {"configurable": {}}) is mock

    def test_get_on_status_from_config(self):
        from orchestrator.nodes import _get_on_status

        cb = lambda msg: None
        config = {"configurable": {"on_status": cb}}
        state = {}

        assert _get_on_status(state, config) is cb

    def test_get_tool_registry_from_config(self):
        from orchestrator.nodes import _get_tool_registry

        registry = object()
        config = {"configurable": {"tool_registry": registry}}
        state = {}

        assert _get_tool_registry(state, config) is registry

    def test_nodes_accept_config_param(self):
        """All node functions should accept an optional config parameter."""
        import inspect

        from orchestrator import nodes

        node_funcs = [
            nodes.gather_data,
            nodes.run_sector_analyst,
            nodes.run_short_analyst,
            nodes.run_risk_manager,
            nodes.run_macro_analyst,
            nodes.report_phase1,
            nodes.mark_debate_skipped,
            nodes.run_debate_round,
            nodes.report_debate_complete,
            nodes.run_portfolio_manager,
            nodes.finalize,
        ]

        for func in node_funcs:
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            assert "config" in params, f"{func.__name__} missing 'config' parameter"


# ===========================================================================
# Test: Two-Phase Execution
# ===========================================================================

class TestTwoPhaseExecution:
    """Tests for Phase 1 → Phase 2 split execution."""

    def setup_method(self):
        from orchestrator.memory import clear_session
        clear_session()

    def test_phase1_returns_intermediate_state(self):
        from orchestrator.graph import run_graph_phase1

        mock = MockLLMPhaseC()
        context = _make_context()

        state = run_graph_phase1(
            ticker="TEST",
            context=context,
            model=mock,
            max_debate_rounds=1,
            tool_registry=None,
        )

        # Phase 1 should produce bull, short, bear, macro, and debate results
        assert state["ticker"] == "TEST"
        assert state["bull_case"] is not None
        assert state["short_case"] is not None
        assert state["bear_case"] is not None
        assert state["macro_view"] is not None
        # Should NOT have committee_memo (that's Phase 2)
        assert state.get("committee_memo") is None
        # Should have traces for 4 analysts
        assert "sector_analyst" in state.get("traces", {})
        assert "short_analyst" in state.get("traces", {})
        assert "risk_manager" in state.get("traces", {})
        assert "macro_analyst" in state.get("traces", {})

    def test_phase2_produces_committee_result(self):
        from orchestrator.graph import run_graph_phase1, run_graph_phase2

        mock = MockLLMPhaseC()
        context = _make_context()

        # Phase 1
        intermediate = run_graph_phase1(
            ticker="TEST",
            context=context,
            model=mock,
            max_debate_rounds=1,
            tool_registry=None,
        )

        # Phase 2
        result = run_graph_phase2(
            intermediate_state=intermediate,
            model=mock,
            pm_guidance="",
            tool_registry=None,
        )

        # Phase 2 should produce the committee memo
        assert result.committee_memo is not None
        assert result.committee_memo.ticker == "TEST"
        assert result.committee_memo.recommendation == "BUY"
        assert result.bull_case is not None
        assert result.bear_case is not None

    def test_pm_guidance_injected(self):
        from orchestrator.graph import run_graph_phase1, run_graph_phase2

        mock = MockLLMPhaseC()
        context = _make_context()

        intermediate = run_graph_phase1(
            ticker="TEST",
            context=context,
            model=mock,
            max_debate_rounds=1,
            tool_registry=None,
        )

        # Run Phase 2 with PM guidance
        run_graph_phase2(
            intermediate_state=intermediate,
            model=mock,
            pm_guidance="Weight the bear case more heavily",
            tool_registry=None,
        )

        # The PM guidance should have been in the prompts
        pm_prompts = [p for p in mock.prompts if "committee chair guidance" in p.lower()]
        assert len(pm_prompts) > 0, "PM guidance should appear in at least one PM prompt"

    def test_backward_compat_full_run(self):
        """Existing run_graph() still works unchanged."""
        from orchestrator.graph import run_graph

        mock = MockLLMPhaseC()
        context = _make_context()

        result = run_graph(
            ticker="TEST",
            context=context,
            model=mock,
            max_debate_rounds=1,
            tool_registry=None,
        )

        assert result.committee_memo is not None
        assert result.bull_case is not None
        assert result.bear_case is not None
        assert result.macro_view is not None

    def test_two_phase_preserves_analyst_results(self):
        """Phase 2 should carry over the same bull/bear/macro from Phase 1."""
        from orchestrator.graph import run_graph_phase1, run_graph_phase2

        mock = MockLLMPhaseC()
        context = _make_context()

        intermediate = run_graph_phase1(
            ticker="TEST",
            context=context,
            model=mock,
            max_debate_rounds=1,
            tool_registry=None,
        )

        bull_thesis = intermediate["bull_case"].thesis
        bear_risk = intermediate["bear_case"].bearish_conviction

        result = run_graph_phase2(
            intermediate_state=intermediate,
            model=mock,
            tool_registry=None,
        )

        # Bull and bear should be preserved exactly
        assert result.bull_case.thesis == bull_thesis
        assert result.bear_case.bearish_conviction == bear_risk


# ===========================================================================
# Test: PM with Prior Analyses (Session Memory Integration)
# ===========================================================================

class TestPMSessionMemory:
    """Tests for PM receiving prior analyses from session memory."""

    def setup_method(self):
        from orchestrator.memory import clear_session
        clear_session()

    def test_prior_analyses_in_pm_context(self):
        """After storing an analysis, the PM should receive prior analyses."""
        from orchestrator.graph import run_graph_phase1, run_graph_phase2
        from orchestrator.memory import store_analysis

        # Store a prior analysis
        store_analysis("TEST", {
            "recommendation": "HOLD",
            "conviction": 5.0,
            "thesis_summary": "Previous analysis was cautious",
        })

        mock = MockLLMPhaseC()
        context = _make_context()

        intermediate = run_graph_phase1(
            ticker="TEST",
            context=context,
            model=mock,
            max_debate_rounds=1,
            tool_registry=None,
        )

        run_graph_phase2(
            intermediate_state=intermediate,
            model=mock,
            tool_registry=None,
        )

        # The PM should have received prior analyses in its prompts
        pm_prompts = [p for p in mock.prompts if "prior analyses" in p.lower()]
        assert len(pm_prompts) > 0, "PM should receive prior analyses in prompts"


# ===========================================================================
# Test: Graph Builders Compile
# ===========================================================================

class TestGraphBuilders:
    """Tests that the new graph builders compile successfully."""

    def test_build_graph_phase1_compiles(self):
        from orchestrator.graph import build_graph_phase1
        graph = build_graph_phase1()
        assert graph is not None

    def test_build_graph_phase2_compiles(self):
        from orchestrator.graph import build_graph_phase2
        graph = build_graph_phase2()
        assert graph is not None

    def test_build_graph_full_still_compiles(self):
        from orchestrator.graph import build_graph
        graph = build_graph()
        assert graph is not None
