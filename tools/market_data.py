"""
Market data tool — provider-agnostic.

Provides price history, company info, and fundamental data
for agents to consume as structured inputs.

Data source is configurable via the provider abstraction layer:
    - Yahoo Finance (default, no special requirements)
    - Bloomberg Terminal (requires blpapi)
    - Interactive Brokers (requires ib_insync + TWS/Gateway)

Usage:
    # Default (Yahoo Finance)
    tool = MarketDataTool()

    # Bloomberg
    tool = MarketDataTool(provider_name="Bloomberg")

    # Interactive Brokers
    tool = MarketDataTool(provider_name="Interactive Brokers")
"""

from __future__ import annotations

import logging
from typing import Any

from tools.data_providers.factory import get_provider_safe
from tools.data_providers.base import MarketDataProvider

logger = logging.getLogger(__name__)

# Module-level default provider (lazy-initialized)
_default_provider: MarketDataProvider | None = None


def _get_default_provider() -> MarketDataProvider:
    """Return the module-level default provider (Yahoo Finance)."""
    global _default_provider
    if _default_provider is None:
        _default_provider = get_provider_safe()
    return _default_provider


def set_provider(name: str = "Yahoo Finance", **kwargs: Any) -> None:
    """Switch the default provider used by all MarketDataTool static methods.

    Args:
        name: Provider name ("Yahoo Finance", "Bloomberg", "Interactive Brokers")
        **kwargs: Provider-specific args (e.g. host, port for Bloomberg/IB)
    """
    global _default_provider
    _default_provider = get_provider_safe(name, **kwargs)
    logger.info("MarketDataTool provider set to: %s", _default_provider.name)


class MarketDataTool:
    """Fetches and structures market data from the active provider.

    By default uses Yahoo Finance. Call ``set_provider()`` at startup
    to switch to Bloomberg or Interactive Brokers, or pass a provider
    instance directly.
    """

    def __init__(self, provider: MarketDataProvider | None = None):
        self._provider = provider

    @property
    def provider(self) -> MarketDataProvider:
        return self._provider or _get_default_provider()

    @staticmethod
    def get_company_overview(ticker: str) -> dict[str, Any]:
        """Get a structured company overview."""
        return _get_default_provider().get_company_overview(ticker)

    @staticmethod
    def get_price_data(ticker: str, period: str = "6mo") -> dict[str, Any]:
        """Get price data and performance metrics."""
        return _get_default_provider().get_price_data(ticker, period)

    @staticmethod
    def get_fundamentals(ticker: str) -> dict[str, Any]:
        """Get fundamental financial data: valuations, income, balance sheet."""
        return _get_default_provider().get_fundamentals(ticker)


def _format_large_number(n: int | float | None) -> str | None:
    """Format large numbers into readable strings (e.g., 1.5T, 230B, 45M)."""
    if n is None:
        return None
    n = float(n)
    if abs(n) >= 1e12:
        return f"${n/1e12:.2f}T"
    if abs(n) >= 1e9:
        return f"${n/1e9:.2f}B"
    if abs(n) >= 1e6:
        return f"${n/1e6:.1f}M"
    return f"${n:,.0f}"


def _pct(value: float | None) -> str | None:
    """Convert decimal to percentage string."""
    if value is None:
        return None
    return f"{value * 100:.1f}%"
