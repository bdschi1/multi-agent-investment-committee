"""Provider factory — thin shim delegating to shared financial-data-providers package.

All factory logic now lives in bds_data_providers.market_factory.
This module re-exports the public API with backward-compatible names
(get_provider, get_provider_safe, available_providers, clear_cache).

Usage:
    from tools.data_providers.factory import get_provider, available_providers

    providers = available_providers()   # e.g. ["Yahoo Finance", "Bloomberg"]
    provider  = get_provider()          # Yahoo Finance (default)
    provider  = get_provider("Bloomberg")
    provider  = get_provider_safe("Bloomberg")  # falls back to Yahoo on error
"""

from bds_data_providers.market_factory import (  # noqa: I001
    available_market_providers as available_providers,
    clear_cache,
    get_market_provider as get_provider,
    get_market_provider_safe as get_provider_safe,
)

__all__ = ["get_provider", "get_provider_safe", "available_providers", "clear_cache"]
