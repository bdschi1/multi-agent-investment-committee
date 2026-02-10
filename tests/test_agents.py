"""
Tests for investment committee agents.

Uses mock LLM responses to test agent logic without API calls.
This ensures CI doesn't burn credits and tests run fast.
"""

from __future__ import annotations

import json
import pytest

from agents.base import AgentRole, BullCase, BearCase, CommitteeMemo, Rebuttal
from agents.sector_analyst import SectorAnalystAgent
from agents.risk_manager import RiskManagerAgent
from agents.portfolio_manager import PortfolioManagerAgent


# ---------------------------------------------------------------------------
# Mock LLM
# ---------------------------------------------------------------------------

class MockLLM:
    """Mock LLM that returns pre-defined responses based on prompt content."""

    def __init__(self, responses: dict[str, str] | None = None):
        self.responses = responses or {}
        self.calls: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.calls.append(prompt)

        # Return matching response if keyword found
        for keyword, response in self.responses.items():
            if keyword.lower() in prompt.lower():
                return response

        # Default: return valid JSON for the most common case
        return json.dumps({
            "ticker": "TEST",
            "thesis": "Test thesis",
            "supporting_evidence": ["evidence 1"],
            "catalysts": ["catalyst 1"],
            "conviction_score": 7.0,
            "time_horizon": "6-12 months",
            "key_metrics": {"pe": 25},
        })


def _mock_bull_json(ticker: str = "TEST") -> str:
    return json.dumps({
        "ticker": ticker,
        "thesis": "Strong growth driven by AI demand",
        "supporting_evidence": [
            "Revenue growth of 40% YoY",
            "Expanding margins",
        ],
        "catalysts": [
            "New product launch in Q2",
            "Analyst day in March",
        ],
        "conviction_score": 8.0,
        "time_horizon": "6-12 months",
        "key_metrics": {"pe_forward": 30, "revenue_growth": "40%"},
    })


def _mock_bear_json(ticker: str = "TEST") -> str:
    return json.dumps({
        "ticker": ticker,
        "risks": [
            "Customer concentration risk",
            "Regulatory headwinds",
        ],
        "second_order_effects": [
            "If regulation passes, compliance costs could compress margins by 300bps",
        ],
        "third_order_effects": [
            "Margin compression forces R&D cuts, weakening competitive moat over 2-3 years",
        ],
        "worst_case_scenario": "Regulatory action + recession = 40% downside",
        "bearish_conviction": 6.5,
        "key_vulnerabilities": {"regulation": "pending legislation"},
    })


def _mock_memo_json(ticker: str = "TEST") -> str:
    return json.dumps({
        "ticker": ticker,
        "recommendation": "BUY",
        "position_size": "Half position",
        "conviction": 7.0,
        "thesis_summary": "Growth thesis intact despite regulatory risk",
        "key_factors": ["Strong revenue momentum", "Manageable risk profile"],
        "bull_points_accepted": ["40% revenue growth is compelling"],
        "bear_points_accepted": ["Regulatory risk is real but priced in"],
        "dissenting_points": ["Risk manager's worst case assumes multiple negatives"],
        "risk_mitigants": ["Position size limited to half", "Stop-loss at 15%"],
        "time_horizon": "6-12 months",
    })


def _mock_rebuttal_json() -> str:
    return json.dumps({
        "points": ["Risk is overstated", "Company has pricing power"],
        "concessions": ["Regulatory timeline is uncertain"],
        "revised_conviction": 7.5,
    })


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestSchemas:
    """Test that output schemas validate correctly."""

    def test_bull_case_schema(self):
        bc = BullCase(
            ticker="AAPL",
            thesis="Strong ecosystem",
            supporting_evidence=["Services growth"],
            catalysts=["iPhone launch"],
            conviction_score=7.5,
            time_horizon="12 months",
        )
        assert bc.ticker == "AAPL"
        assert 0 <= bc.conviction_score <= 10

    def test_bull_case_conviction_bounds(self):
        with pytest.raises(ValueError):
            BullCase(
                ticker="X",
                thesis="",
                conviction_score=11.0,  # Out of bounds
                time_horizon="1y",
            )

    def test_bear_case_schema(self):
        bc = BearCase(
            ticker="TSLA",
            risks=["Competition"],
            second_order_effects=["Market share loss"],
            third_order_effects=["Margin compression"],
            worst_case_scenario="50% decline",
            bearish_conviction=8.0,
        )
        assert bc.bearish_conviction == 8.0
        assert len(bc.risks) == 1

    def test_committee_memo_schema(self):
        memo = CommitteeMemo(
            ticker="NVDA",
            recommendation="BUY",
            position_size="Full position",
            conviction=9.0,
            thesis_summary="AI leader",
            time_horizon="2-3 years",
        )
        assert memo.recommendation == "BUY"

    def test_rebuttal_schema(self):
        r = Rebuttal(
            agent_role=AgentRole.SECTOR_ANALYST,
            responding_to=AgentRole.RISK_MANAGER,
            points=["Growth offsets risk"],
            concessions=["Regulation is a concern"],
            revised_conviction=7.0,
        )
        assert r.agent_role == AgentRole.SECTOR_ANALYST


# ---------------------------------------------------------------------------
# Agent tests
# ---------------------------------------------------------------------------

class TestSectorAnalyst:
    """Test the Sector Analyst agent."""

    def test_full_run(self):
        mock = MockLLM({"build the bull case": _mock_bull_json()})
        agent = SectorAnalystAgent(model=mock)
        result = agent.run("TEST", {"market_data": {}, "news": []})

        assert "output" in result
        assert "trace" in result
        assert result["trace"].agent_role == AgentRole.SECTOR_ANALYST
        assert len(result["trace"].steps) == 4  # think, plan, act, reflect

    def test_rebuttal(self):
        mock = MockLLM({"rebut": _mock_rebuttal_json()})
        agent = SectorAnalystAgent(model=mock)

        bear = BearCase(
            ticker="TEST", risks=["risk1"], bearish_conviction=6.0,
            worst_case_scenario="bad", second_order_effects=[], third_order_effects=[],
        )
        bull = BullCase(
            ticker="TEST", thesis="good", conviction_score=8.0,
            time_horizon="1y", supporting_evidence=[], catalysts=[],
        )

        rebuttal = agent.rebut("TEST", bear, bull)
        assert isinstance(rebuttal, Rebuttal)
        assert rebuttal.agent_role == AgentRole.SECTOR_ANALYST


class TestRiskManager:
    """Test the Risk Manager agent."""

    def test_full_run(self):
        mock = MockLLM({"build the bear case": _mock_bear_json()})
        agent = RiskManagerAgent(model=mock)
        result = agent.run("TEST", {"market_data": {}, "news": []})

        assert "output" in result
        assert result["trace"].agent_role == AgentRole.RISK_MANAGER


class TestPortfolioManager:
    """Test the Portfolio Manager agent."""

    def test_full_run(self):
        mock = MockLLM({"final decision": _mock_memo_json()})
        agent = PortfolioManagerAgent(model=mock)

        context = {
            "bull_case": BullCase(
                ticker="TEST", thesis="good", conviction_score=8.0,
                time_horizon="1y", supporting_evidence=[], catalysts=[],
            ),
            "bear_case": BearCase(
                ticker="TEST", risks=["risk1"], bearish_conviction=6.0,
                worst_case_scenario="bad", second_order_effects=[], third_order_effects=[],
            ),
            "debate": {},
        }

        result = agent.run("TEST", context)
        assert "output" in result
        assert result["trace"].agent_role == AgentRole.PORTFOLIO_MANAGER
