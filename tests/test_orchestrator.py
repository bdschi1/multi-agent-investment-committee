"""
Tests for the Investment Committee orchestrator.

Tests the full multi-agent workflow with mock LLMs.
"""

from __future__ import annotations

import json

from agents.base import AgentRole, ReasoningStep, ReasoningTrace, StepType
from orchestrator.committee import CommitteeResult, InvestmentCommittee
from orchestrator.reasoning_trace import TraceRenderer


# Mock LLM that returns appropriate JSON for each agent phase
class OrchestratorMockLLM:
    """Mock that returns valid JSON for each stage of the committee workflow."""

    def __init__(self):
        self.call_count = 0

    def __call__(self, prompt: str) -> str:
        self.call_count += 1
        prompt_lower = prompt.lower()

        # Sector Analyst responses
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


class TestInvestmentCommittee:
    """Test the full committee orchestration."""

    def test_full_workflow(self):
        mock = OrchestratorMockLLM()
        committee = InvestmentCommittee(model=mock)

        result = committee.run(
            ticker="TEST",
            context={"market_data": {}, "news": [], "financial_metrics": {}},
        )

        assert isinstance(result, CommitteeResult)
        assert result.ticker == "TEST"
        assert result.bull_case is not None
        assert result.bear_case is not None
        assert result.committee_memo is not None
        assert result.total_duration_ms > 0

    def test_result_serialization(self):
        mock = OrchestratorMockLLM()
        committee = InvestmentCommittee(model=mock)

        result = committee.run(
            ticker="TEST",
            context={"market_data": {}, "news": [], "financial_metrics": {}},
        )

        result_dict = result.to_dict()
        assert result_dict["ticker"] == "TEST"
        assert result_dict["bull_case"] is not None
        assert result_dict["bear_case"] is not None
        assert result_dict["committee_memo"] is not None

    def test_status_callback(self):
        mock = OrchestratorMockLLM()
        committee = InvestmentCommittee(model=mock)

        statuses = []
        committee.run(
            ticker="TEST",
            context={"market_data": {}, "news": [], "financial_metrics": {}},
            on_status=lambda msg: statuses.append(msg),
        )

        assert len(statuses) > 0
        assert any("Phase 1" in s for s in statuses)
        assert any("Phase 2" in s for s in statuses)
        assert any("Phase 3" in s for s in statuses)


class TestTraceRenderer:
    """Test reasoning trace rendering."""

    def _make_trace(self) -> dict[str, ReasoningTrace]:
        trace = ReasoningTrace(
            agent_role=AgentRole.SECTOR_ANALYST,
            ticker="TEST",
        )
        trace.add_step(ReasoningStep(
            step_type=StepType.THINK,
            agent_role=AgentRole.SECTOR_ANALYST,
            content="Thinking about the opportunity...",
            duration_ms=1500.0,
        ))
        trace.add_step(ReasoningStep(
            step_type=StepType.ACT,
            agent_role=AgentRole.SECTOR_ANALYST,
            content="Building the thesis...",
            duration_ms=3000.0,
        ))
        return {"sector_analyst": trace}

    def test_markdown_rendering(self):
        traces = self._make_trace()
        md = TraceRenderer.to_markdown(traces)

        assert "Long Analyst (Bull)" in md
        assert "THINK" in md
        assert "ACT" in md

    def test_gradio_accordion_rendering(self):
        traces = self._make_trace()
        html = TraceRenderer.to_gradio_accordion(traces)

        assert "<details>" in html
        assert "Long Analyst (Bull)" in html

    def test_summary_stats(self):
        traces = self._make_trace()
        stats = TraceRenderer.summary_stats(traces)

        assert stats["total_agents"] == 1
        assert stats["total_steps"] == 2
        assert stats["total_duration_s"] > 0
