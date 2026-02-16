"""Yahoo Finance data provider (default).

Re-exports from bds-data-providers shared package.
Includes helper functions used by bloomberg_provider and ib_provider.
"""

from bds_data_providers.yahoo_market import (
    YahooMarketProvider as YahooProvider,
    _format_large_number,
    _pct,
)

# Re-export is_available for factory compatibility
def is_available() -> bool:
    """Yahoo Finance is always available if yfinance is installed."""
    return True

__all__ = ["YahooProvider", "is_available", "_format_large_number", "_pct"]
