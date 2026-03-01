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


def _fetch_daily_prices(ticker: str, period: str = "1y") -> list[float] | None:
    """Fetch raw daily close prices via yfinance for vol computation.

    Returns a list of daily close prices (oldest first), or None on failure.
    """
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period=period)
        if hist.empty or "Close" not in hist.columns:
            return None
        return hist["Close"].dropna().tolist()
    except Exception as e:
        logger.warning(f"Failed to fetch daily prices for {ticker}: {e}")
        return None


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
            Dict with market_data, financial_metrics, news, valuation, quality,
            vol_intelligence, etc.
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

        # Compute vol intelligence (realized vol + implied vol + signals)
        vol_intel = _compute_vol_context(ticker, price_data)

        # Fetch 13F fund conviction signals (from fund-tracker-13f bridge)
        fund_conviction = _fetch_fund_conviction(ticker)

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

            # Vol intelligence — quantitative vol analysis for agents
            "vol_intelligence": vol_intel,

            # 13F fund conviction signals — hedge fund positioning from SEC filings
            "fund_conviction": fund_conviction,

            # User input
            "user_context": user_context,
            "ticker": ticker,
        }

        vol_regime = "N/A"
        if vol_intel and not vol_intel.get("error"):
            regime_data = vol_intel.get("vol_regime_sizing", {})
            vol_regime = regime_data.get("regime", "N/A")

        fund_avail = fund_conviction.get("available", False)
        fund_sentiment = fund_conviction.get("net_sentiment", 0)
        fund_count = fund_conviction.get("total_funds_holding", 0)

        logger.info(
            f"Context gathered for {ticker}: "
            f"{len(news_formatted)} news articles, "
            f"quality={quality.get('quality_label', 'N/A')}, "
            f"valuation={valuation.get('overall_valuation', 'N/A')}, "
            f"vol_regime={vol_regime}, "
            f"13f_signals={'available' if fund_avail else 'N/A'}"
            f"{f' ({fund_count} funds, sentiment {fund_sentiment:+d})' if fund_avail else ''}"
        )

        return context


def _fetch_fund_conviction(ticker: str) -> dict[str, Any]:
    """Fetch 13F fund conviction signals via the fund-tracker-13f bridge.

    Graceful fallback: returns an unavailable dict if fund-tracker is
    not installed or the query fails. Never raises.
    """
    try:
        from tools.fund_tracker_signals import get_fund_conviction_signals
        return get_fund_conviction_signals(ticker)
    except Exception as e:
        logger.warning(f"Fund conviction signals unavailable for {ticker}: {e}")
        return {
            "available": False,
            "summary": f"Fund conviction data unavailable: {e}",
        }


def _compute_vol_context(ticker: str, price_data: dict[str, Any]) -> dict[str, Any] | None:
    """Compute vol intelligence from daily price history.

    Graceful fallback: returns None if price data is unavailable.
    """
    try:
        daily_prices = _fetch_daily_prices(ticker, period="1y")
        if daily_prices is None or len(daily_prices) < 30:
            logger.info(f"Insufficient price history for vol intelligence on {ticker}")
            return {"error": "Insufficient price history for vol computation"}

        spot = price_data.get("current_price") or daily_prices[-1]

        from tools.vol_intelligence import compute_vol_intelligence
        return compute_vol_intelligence(
            ticker=ticker,
            prices=daily_prices,
            spot=float(spot),
        )
    except Exception as e:
        logger.warning(f"Vol intelligence computation failed for {ticker}: {e}")
        return {"error": f"Vol intelligence failed: {e}"}
