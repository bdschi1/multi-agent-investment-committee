"""
Tests for data tools.

Tests market data, news retrieval, and financial metrics computation.
Note: Market data tests require network access and may be skipped in CI.
"""

from __future__ import annotations

import pytest

from tools.market_data import MarketDataTool, _format_large_number, _pct
from tools.financial_metrics import FinancialMetricsTool
from tools.data_aggregator import DataAggregator


class TestMarketDataFormatters:
    """Test utility formatting functions."""

    def test_format_trillion(self):
        assert _format_large_number(1.5e12) == "$1.50T"

    def test_format_billion(self):
        assert _format_large_number(230e9) == "$230.00B"

    def test_format_million(self):
        assert _format_large_number(45e6) == "$45.0M"

    def test_format_small(self):
        assert _format_large_number(5000) == "$5,000"

    def test_format_none(self):
        assert _format_large_number(None) is None

    def test_pct(self):
        assert _pct(0.15) == "15.0%"

    def test_pct_none(self):
        assert _pct(None) is None


class TestFinancialMetrics:
    """Test derived financial metrics computation."""

    def test_valuation_assessment_bullish(self):
        fundamentals = {
            "ticker": "TEST",
            "pe_trailing": 15,
            "pe_forward": 12,
            "peg_ratio": 0.8,
            "profit_margin": "20.5%",
            "debt_to_equity": 30,
            "revenue_growth": "25.5%",
        }
        result = FinancialMetricsTool.compute_valuation_assessment(fundamentals)

        assert result["ticker"] == "TEST"
        assert len(result["flags"]) > 0
        assert result["overall_valuation"] in ["attractive", "mixed", "expensive_or_risky"]

    def test_valuation_assessment_bearish(self):
        fundamentals = {
            "ticker": "TEST",
            "pe_trailing": 50,
            "pe_forward": 55,
            "peg_ratio": 3.0,
            "ev_to_ebitda": 30,
            "profit_margin": "-5.0%",
            "debt_to_equity": 250,
            "current_ratio": 0.8,
            "revenue_growth": "-10.0%",
        }
        result = FinancialMetricsTool.compute_valuation_assessment(fundamentals)

        assert result["overall_valuation"] in ["expensive_or_risky", "mixed"]

    def test_quality_score_high(self):
        fundamentals = {
            "roe": "25.0%",
            "profit_margin": "20.0%",
            "revenue_growth": "30.0%",
            "debt_to_equity": 30,
        }
        result = FinancialMetricsTool.compute_quality_score(fundamentals)

        assert result["quality_score"] > 0
        assert result["max_score"] > 0
        assert result["quality_label"] in ["High Quality", "Medium Quality", "Low Quality"]

    def test_quality_score_empty(self):
        result = FinancialMetricsTool.compute_quality_score({})
        assert result["quality_score"] == 0
        assert result["quality_pct"] == 0


class TestMarketData:
    """Tests that require network access. Mark with network marker for CI."""

    @pytest.mark.skipif(
        not pytest.importorskip("yfinance"),
        reason="yfinance not available",
    )
    def test_company_overview_valid_ticker(self):
        """Test with a real ticker — requires network."""
        result = MarketDataTool.get_company_overview("AAPL")
        # Should at minimum return the ticker
        assert result.get("ticker") == "AAPL"

    def test_company_overview_invalid_ticker(self):
        """Invalid ticker should return gracefully."""
        result = MarketDataTool.get_company_overview("ZZZZZZNOTREAL")
        assert "ticker" in result


class TestDataAggregator:
    """Test the data aggregation pipeline."""

    @pytest.mark.skipif(
        not pytest.importorskip("yfinance"),
        reason="yfinance not available",
    )
    def test_gather_context(self):
        """Full integration test — requires network."""
        context = DataAggregator.gather_context("AAPL", "Test context")
        assert "market_data" in context
        assert "financial_metrics" in context
        assert "valuation_assessment" in context
        assert "quality_score" in context
        assert context["user_context"] == "Test context"
