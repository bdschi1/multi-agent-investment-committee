"""
Data provider abstraction layer.

Allows seamless switching between Yahoo Finance (default), Bloomberg,
Interactive Brokers, and Alpha Vantage as the market data source.  All
providers return identical dict schemas so the rest of the tool layer is
data-source agnostic.

All logic lives in the shared ``bds-data-providers`` package.  This
sub-package provides backward-compatible import paths for MAIC consumers.

Quick start:
    from tools.data_providers import get_provider
    provider = get_provider()              # Yahoo Finance (default)
    provider = get_provider("Bloomberg")   # Bloomberg Terminal (requires blpapi)
    provider = get_provider("Interactive Brokers")  # IB TWS/Gateway (requires ib_insync)
"""

from tools.data_providers.factory import (
    available_providers,
    clear_cache,
    get_provider,
    get_provider_safe,
)

__all__ = [
    "available_providers",
    "clear_cache",
    "get_provider",
    "get_provider_safe",
]
