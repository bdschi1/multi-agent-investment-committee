"""
Data aggregator — single entry point for gathering all tool data.

This pre-fetches all data before agents run, so agents receive
structured context rather than making ad-hoc tool calls.

This is an intentional design choice: for v1 (smolagents), we
pre-gather data. For v2 (LangGraph), agents will invoke tools
dynamically via the graph's tool nodes.
"""

from __future__ import annotations

import logging
from typing import Any

from tools.financial_metrics import FinancialMetricsTool
from tools.market_data import MarketDataTool
from tools.news_retrieval import NewsRetrievalTool

logger = logging.getLogger(__name__)


class DataAggregator:
    """Aggregates all data sources into a unified context for agents."""

    @staticmethod
    def gather_context(ticker: str, user_context: str = "") -> dict[str, Any]:
        """
        Gather all available data for a ticker.

        Returns a structured context dict that agents consume directly.

        Args:
            ticker: Stock ticker symbol (e.g., "NVDA", "AAPL")
            user_context: Optional user-provided context or thesis to evaluate

        Returns:
            Dict with market_data, financial_metrics, news, valuation, quality, etc.
        """
        logger.info(f"Gathering data for {ticker}...")

        # Fetch raw data
        overview = MarketDataTool.get_company_overview(ticker)
        price_data = MarketDataTool.get_price_data(ticker, period="6mo")
        fundamentals = MarketDataTool.get_fundamentals(ticker)
        news_articles = NewsRetrievalTool.get_news(ticker, max_articles=10)
        news_formatted = NewsRetrievalTool.format_for_agent(news_articles)

        # Compute derived metrics
        valuation = FinancialMetricsTool.compute_valuation_assessment(fundamentals)
        quality = FinancialMetricsTool.compute_quality_score(fundamentals)

        context = {
            # Raw data
            "market_data": {
                "overview": overview,
                "price": price_data,
            },
            "financial_metrics": fundamentals,
            "news": news_formatted,
            "news_raw": news_articles,

            # Derived analytics
            "valuation_assessment": valuation,
            "quality_score": quality,

            # User input
            "user_context": user_context,
            "ticker": ticker,
        }

        logger.info(
            f"Context gathered for {ticker}: "
            f"{len(news_formatted)} news articles, "
            f"quality={quality.get('quality_label', 'N/A')}, "
            f"valuation={valuation.get('overall_valuation', 'N/A')}"
        )

        return context
