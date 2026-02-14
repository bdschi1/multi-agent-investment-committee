"""Abstract market data provider interface.

Defines the contract that all data providers must implement.
Each method returns the same dict schema regardless of the underlying
data source, so tools can switch providers without code changes.

Concrete implementations:
    - yahoo_provider.py   (default, no special requirements)
    - bloomberg_provider.py (requires blpapi + Bloomberg Terminal)
    - ib_provider.py       (requires ib_insync + TWS/IB Gateway)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class MarketDataProvider(ABC):
    """Abstract interface for market data providers.

    Every provider must implement five methods that return plain dicts
    matching the schemas consumed by the tool layer (MarketDataTool,
    PeerComparisonTool, InsiderDataTool, EarningsDataTool).
    """

    @property
    def name(self) -> str:
        """Human-readable provider name (override in subclasses)."""
        return type(self).__name__

    # ------------------------------------------------------------------
    # Core data methods
    # ------------------------------------------------------------------

    @abstractmethod
    def get_ticker_object(self, ticker: str) -> Any:
        """Return the underlying ticker/session object for a symbol.

        For Yahoo this is a ``yf.Ticker``; for Bloomberg or IB this is
        a wrapper that exposes ``.info`` and ``.history()`` with the
        same key structure so downstream tools can consume it uniformly.
        """
        ...

    @abstractmethod
    def get_company_overview(self, ticker: str) -> dict[str, Any]:
        """Get structured company overview.

        Returns dict with keys:
            ticker, name, sector, industry, market_cap, market_cap_formatted,
            currency, exchange, description, website, employees, country
        """
        ...

    @abstractmethod
    def get_price_data(
        self, ticker: str, period: str = "6mo"
    ) -> dict[str, Any]:
        """Get price data and performance metrics.

        Returns dict with keys:
            ticker, current_price, period_return_pct, high_52w, low_52w,
            pct_from_high, pct_from_low, avg_daily_volume,
            avg_volume_formatted, period, data_points
        """
        ...

    @abstractmethod
    def get_fundamentals(self, ticker: str) -> dict[str, Any]:
        """Get fundamental financial data.

        Returns dict with keys covering:
            valuation (pe_trailing, pe_forward, peg_ratio, ...),
            profitability (profit_margin, operating_margin, ...),
            growth (revenue_growth, earnings_growth),
            income (revenue, ebitda, net_income),
            balance sheet (total_debt, total_cash, debt_to_equity, ...),
            dividends, analyst targets
        """
        ...

    @abstractmethod
    def get_info(self, ticker: str) -> dict[str, Any]:
        """Get the raw info dict for a ticker.

        This powers peer comparison and any other tool that needs
        the full set of key-value pairs from the data source.
        """
        ...

    @abstractmethod
    def get_insider_transactions(self, ticker: str) -> Any:
        """Return insider transaction data (DataFrame or list).

        Returns None or empty if unavailable.
        """
        ...

    @abstractmethod
    def get_earnings_history(self, ticker: str) -> Any:
        """Return earnings history data (DataFrame or list).

        Returns None or empty if unavailable.
        """
        ...

    @abstractmethod
    def get_quarterly_earnings(self, ticker: str) -> Any:
        """Return quarterly earnings (revenue/earnings) as fallback.

        Returns None or empty if unavailable.
        """
        ...

    @abstractmethod
    def get_history(
        self, ticker: str, period: str = "6mo"
    ) -> Any:
        """Return historical price DataFrame (pandas).

        Columns: Open, High, Low, Close, Volume (at minimum).
        """
        ...
