"""Yahoo Finance data provider (default).

Wraps yfinance to implement the MarketDataProvider interface.
No special requirements beyond ``pip install yfinance``.
"""

from __future__ import annotations

import logging
from typing import Any

import yfinance as yf

from tools.data_providers.base import MarketDataProvider

logger = logging.getLogger(__name__)


def is_available() -> bool:
    """Yahoo Finance is always available if yfinance is installed."""
    return True


class YahooProvider(MarketDataProvider):
    """Fetch market data from Yahoo Finance via yfinance."""

    @property
    def name(self) -> str:
        return "Yahoo Finance"

    def get_ticker_object(self, ticker: str) -> Any:
        return yf.Ticker(ticker)

    def get_company_overview(self, ticker: str) -> dict[str, Any]:
        try:
            info = self.get_info(ticker)
            return {
                "ticker": ticker,
                "name": info.get("longName", info.get("shortName", ticker)),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "market_cap": info.get("marketCap"),
                "market_cap_formatted": _format_large_number(info.get("marketCap")),
                "currency": info.get("currency", "USD"),
                "exchange": info.get("exchange", "Unknown"),
                "description": (
                    info.get("longBusinessSummary", "")[:500]
                    if info.get("longBusinessSummary")
                    else ""
                ),
                "website": info.get("website", ""),
                "employees": info.get("fullTimeEmployees"),
                "country": info.get("country", "Unknown"),
            }
        except Exception as e:
            logger.error("Failed to get company overview for %s: %s", ticker, e)
            return {"ticker": ticker, "error": str(e)}

    def get_price_data(self, ticker: str, period: str = "6mo") -> dict[str, Any]:
        try:
            hist = self.get_history(ticker, period)
            if hist.empty:
                return {"ticker": ticker, "error": "No price data available"}

            current_price = float(hist["Close"].iloc[-1])
            start_price = float(hist["Close"].iloc[0])
            high_52w = float(hist["Close"].max())
            low_52w = float(hist["Close"].min())
            avg_volume = int(hist["Volume"].mean())

            return {
                "ticker": ticker,
                "current_price": round(current_price, 2),
                "period_return_pct": round((current_price / start_price - 1) * 100, 2),
                "high_52w": round(high_52w, 2),
                "low_52w": round(low_52w, 2),
                "pct_from_high": round((current_price / high_52w - 1) * 100, 2),
                "pct_from_low": round((current_price / low_52w - 1) * 100, 2),
                "avg_daily_volume": avg_volume,
                "avg_volume_formatted": _format_large_number(avg_volume),
                "period": period,
                "data_points": len(hist),
            }
        except Exception as e:
            logger.error("Failed to get price data for %s: %s", ticker, e)
            return {"ticker": ticker, "error": str(e)}

    def get_fundamentals(self, ticker: str) -> dict[str, Any]:
        try:
            info = self.get_info(ticker)
            return {
                "ticker": ticker,
                # Valuation
                "pe_trailing": info.get("trailingPE"),
                "pe_forward": info.get("forwardPE"),
                "peg_ratio": info.get("pegRatio"),
                "price_to_book": info.get("priceToBook"),
                "price_to_sales": info.get("priceToSalesTrailing12Months"),
                "ev_to_ebitda": info.get("enterpriseToEbitda"),
                "ev_to_revenue": info.get("enterpriseToRevenue"),
                # Profitability
                "profit_margin": _pct(info.get("profitMargins")),
                "operating_margin": _pct(info.get("operatingMargins")),
                "gross_margin": _pct(info.get("grossMargins")),
                "roe": _pct(info.get("returnOnEquity")),
                "roa": _pct(info.get("returnOnAssets")),
                # Growth
                "revenue_growth": _pct(info.get("revenueGrowth")),
                "earnings_growth": _pct(info.get("earningsGrowth")),
                # Income
                "revenue": info.get("totalRevenue"),
                "revenue_formatted": _format_large_number(info.get("totalRevenue")),
                "ebitda": info.get("ebitda"),
                "ebitda_formatted": _format_large_number(info.get("ebitda")),
                "net_income": info.get("netIncomeToCommon"),
                # Balance sheet
                "total_debt": info.get("totalDebt"),
                "total_debt_formatted": _format_large_number(info.get("totalDebt")),
                "total_cash": info.get("totalCash"),
                "total_cash_formatted": _format_large_number(info.get("totalCash")),
                "debt_to_equity": info.get("debtToEquity"),
                "current_ratio": info.get("currentRatio"),
                # Dividends
                "dividend_yield": _pct(info.get("dividendYield")),
                "payout_ratio": _pct(info.get("payoutRatio")),
                # Analyst
                "target_mean_price": info.get("targetMeanPrice"),
                "target_high_price": info.get("targetHighPrice"),
                "target_low_price": info.get("targetLowPrice"),
                "recommendation": info.get("recommendationKey"),
                "num_analysts": info.get("numberOfAnalystOpinions"),
            }
        except Exception as e:
            logger.error("Failed to get fundamentals for %s: %s", ticker, e)
            return {"ticker": ticker, "error": str(e)}

    def get_info(self, ticker: str) -> dict[str, Any]:
        return yf.Ticker(ticker).info

    def get_insider_transactions(self, ticker: str) -> Any:
        return yf.Ticker(ticker).insider_transactions

    def get_earnings_history(self, ticker: str) -> Any:
        return yf.Ticker(ticker).earnings_history

    def get_quarterly_earnings(self, ticker: str) -> Any:
        return yf.Ticker(ticker).quarterly_earnings

    def get_history(self, ticker: str, period: str = "6mo") -> Any:
        return yf.Ticker(ticker).history(period=period)


# ---------------------------------------------------------------------------
# Shared formatting helpers
# ---------------------------------------------------------------------------


def _format_large_number(n: int | float | None) -> str | None:
    """Format large numbers into readable strings (e.g., 1.5T, 230B, 45M)."""
    if n is None:
        return None
    n = float(n)
    if abs(n) >= 1e12:
        return f"${n / 1e12:.2f}T"
    if abs(n) >= 1e9:
        return f"${n / 1e9:.2f}B"
    if abs(n) >= 1e6:
        return f"${n / 1e6:.1f}M"
    return f"${n:,.0f}"


def _pct(value: float | None) -> str | None:
    """Convert decimal to percentage string."""
    if value is None:
        return None
    return f"{value * 100:.1f}%"
