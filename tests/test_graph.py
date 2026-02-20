"""
Tests for the LangGraph investment committee graph (v2).

Validates:
    - Graph compiles without errors
    - Full workflow produces valid CommitteeResult
    - Status callbacks fire correctly
    - Conviction timeline is populated
    - Backward compatibility via InvestmentCommittee.run()
"""

from __future__ import annotations

import json

from orchestrator.committee import CommitteeResult, InvestmentCommittee
from orchestrator.graph import build_graph, run_graph


# Reuse the mock LLM from test_orchestrator
class GraphMockLLM:
    """Mock that returns valid JSON for each stage of the committee workflow."""

    def __init__(self):
        self.call_count = 0

    def __call__(self, prompt: str) -> str:
        self.call_count += 1
        prompt_lower = prompt.lower()

        # Sector Analyst (Long) responses
        if "build the bull case" in prompt_lower or "bull case" in prompt_lower:
            if "json" in prompt_lower:
                return json.dumps({
                    "ticker": "TEST",
                    "thesis": "Strong AI-driven growth",
                    "supporting_evidence": ["Revenue +40%", "Margin expansion"],
                    "catalysts": ["Product launch Q2"],
                    "conviction_score": 7.5,
                    "time_horizon": "6-12 months",
                    "key_metrics": {"pe": 30},
                })
            return "Thinking about bull case..."

        # Short Analyst responses
        if "short case" in prompt_lower or "short thesis" in prompt_lower or "short opportunity" in prompt_lower:
            if "json" in prompt_lower:
                return json.dumps({
                    "ticker": "TEST",
                    "short_thesis": "Overvalued relative to growth",
                    "thesis_type": "alpha_short",
                    "event_path": ["Earnings miss", "Guidance cut"],
                    "supporting_evidence": ["Insider selling"],
                    "conviction_score": 4.5,
                    "key_vulnerabilities": {"valuation": "Premium to peers"},
                })
            return "Thinking about short case..."

        # Risk Manager responses
        if "bear case" in prompt_lower or "risk" in prompt_lower:
            if "json" in prompt_lower:
                return json.dumps({
                    "ticker": "TEST",
                    "risks": ["Valuation stretch", "Competition"],
                    "second_order_effects": ["Margin pressure from competition"],
                    "third_order_effects": ["Talent loss if growth slows"],
                    "worst_case_scenario": "30% downside in recession",
                    "bearish_conviction": 5.5,
                    "key_vulnerabilities": {"valuation": "Above historical mean"},
                })
            return "Thinking about risks..."

        # PM responses
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
                    "bear_points_accepted": ["Valuation is elevated"],
                    "dissenting_points": [],
                    "risk_mitigants": ["Position sizing"],
                    "time_horizon": "6-12 months",
                })
            return "Synthesizing..."

        # Rebuttal responses
        if "rebut" in prompt_lower or "challenge" in prompt_lower:
            return json.dumps({
                "points": ["Counter-argument"],
                "concessions": ["Fair point"],
                "revised_conviction": 7.0,
            })

        # Default for think/plan/reflect steps
        return "Analysis step completed."


class TestGraphBuild:
    """Test graph construction."""

    def test_graph_compiles(self):
        """Graph should compile without errors."""
        compiled = build_graph()
        assert compiled is not None

    def test_graph_has_expected_nodes(self):
        """Graph should contain all expected node names."""
        compiled = build_graph()
        graph_repr = compiled.get_graph()
        node_names = set(graph_repr.nodes.keys())
        expected = {
            "gather_data",
            "run_sector_analyst",
            "run_short_analyst",
            "run_risk_manager",
            "run_macro_analyst",
            "report_phase1",
            "mark_debate_skipped",
            "run_debate_round",
            "report_debate_complete",
            "run_portfolio_manager",
            "finalize",
        }
        # START and END are also nodes in the graph representation
        assert expected.issubset(node_names)


class TestRunGraph:
    """Test full graph execution with mock LLM."""

    def test_full_workflow(self):
        """End-to-end test: graph produces valid CommitteeResult."""
        mock = GraphMockLLM()
        result = run_graph(
            ticker="TEST",
            context={"market_data": {}, "news": [], "financial_metrics": {}},
            model=mock,
            max_debate_rounds=1,
        )

        assert isinstance(result, CommitteeResult)
        assert result.ticker == "TEST"
        assert result.bull_case is not None
        assert result.bear_case is not None
        assert result.macro_view is not None
        assert result.committee_memo is not None
        assert result.total_duration_ms > 0

    def test_traces_populated(self):
        """All 5 agent traces should be present."""
        mock = GraphMockLLM()
        result = run_graph(
            ticker="TEST",
            context={"market_data": {}, "news": [], "financial_metrics": {}},
            model=mock,
            max_debate_rounds=1,
        )

        assert "sector_analyst" in result.traces
        assert "short_analyst" in result.traces
        assert "risk_manager" in result.traces
        assert "macro_analyst" in result.traces
        assert "portfolio_manager" in result.traces

    def test_status_callbacks(self):
        """on_status should fire with Phase 1 and Phase 3 messages."""
        mock = GraphMockLLM()
        statuses = []
        run_graph(
            ticker="TEST",
            context={"market_data": {}, "news": [], "financial_metrics": {}},
            model=mock,
            max_debate_rounds=1,
            on_status=lambda msg: statuses.append(msg),
        )

        assert len(statuses) > 0
        assert any("Phase 1" in s for s in statuses)
        assert any("Phase 3" in s for s in statuses)
        assert any("Committee complete" in s for s in statuses)

    def test_conviction_timeline_populated(self):
        """Should have: 4 initial + debate entries + PM entry."""
        mock = GraphMockLLM()
        result = run_graph(
            ticker="TEST",
            context={"market_data": {}, "news": [], "financial_metrics": {}},
            model=mock,
            max_debate_rounds=1,
        )

        # At minimum: 4 initial (bull, short, bear, macro) + 1 PM = 5
        assert len(result.conviction_timeline) >= 5

    def test_result_serialization(self):
        """CommitteeResult.to_dict() should work with graph output."""
        mock = GraphMockLLM()
        result = run_graph(
            ticker="TEST",
            context={"market_data": {}, "news": [], "financial_metrics": {}},
            model=mock,
            max_debate_rounds=1,
        )

        result_dict = result.to_dict()
        assert result_dict["ticker"] == "TEST"
        assert result_dict["bull_case"] is not None
        assert result_dict["committee_memo"] is not None

    def test_debate_runs_with_mock(self):
        """
        Mock LLM returns conviction=7.5 and risk=5.5 (spread=2.0).
        With strict < 2.0 convergence threshold, debate should NOT
        be skipped — it should run normally.
        """
        mock = GraphMockLLM()
        statuses = []
        run_graph(
            ticker="TEST",
            context={"market_data": {}, "news": [], "financial_metrics": {}},
            model=mock,
            max_debate_rounds=2,
            on_status=lambda msg: statuses.append(msg),
        )

        # Debate should have run (not skipped)
        assert any("Phase 2" in s for s in statuses)
        assert any("Debate round" in s for s in statuses)
        assert not any("SKIPPED" in s for s in statuses)

    def test_backward_compat_via_committee(self):
        """InvestmentCommittee.run() should delegate to graph and still work."""
        mock = GraphMockLLM()
        committee = InvestmentCommittee(model=mock)
        result = committee.run(
            ticker="TEST",
            context={"market_data": {}, "news": [], "financial_metrics": {}},
        )

        assert isinstance(result, CommitteeResult)
        assert result.ticker == "TEST"
        assert result.bull_case is not None
        assert result.committee_memo is not None
