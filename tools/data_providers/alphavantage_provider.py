"""Alpha Vantage data provider.

Re-exports from bds-data-providers shared package.
"""

from bds_data_providers.alphavantage_market import (
    AlphaVantageMarketProvider as AlphaVantageProvider,
)


def is_available() -> bool:
    """Return True if requests is installed and API key is set."""
    from bds_data_providers.alphavantage_market import is_available as _is_available
    return _is_available()


__all__ = ["AlphaVantageProvider", "is_available"]
