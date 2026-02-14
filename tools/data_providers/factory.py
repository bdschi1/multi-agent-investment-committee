"""Provider factory — dynamic discovery, instantiation, and fallback.

Auto-detects which data providers are available based on installed
packages and exposes a simple ``get_provider(name)`` interface.

Usage:
    from tools.data_providers.factory import get_provider, available_providers

    providers = available_providers()   # e.g. ["Yahoo Finance", "Bloomberg"]
    provider  = get_provider()          # Yahoo Finance (default)
    provider  = get_provider("Bloomberg")
    provider  = get_provider_safe("Bloomberg")  # falls back to Yahoo on error
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

from tools.data_providers.base import MarketDataProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider registry
#
# Each entry: (display_name, module_path, class_name, availability_func_or_None)
#
# - availability_func is a dotted path to a ``is_available() -> bool`` function.
#   If None the provider is assumed to always be available.
# ---------------------------------------------------------------------------

_PROVIDER_REGISTRY: list[tuple[str, str, str, str | None]] = [
    (
        "Yahoo Finance",
        "tools.data_providers.yahoo_provider",
        "YahooProvider",
        None,  # always available
    ),
    (
        "Bloomberg",
        "tools.data_providers.bloomberg_provider",
        "BloombergProvider",
        "tools.data_providers.bloomberg_provider.is_available",
    ),
    (
        "Interactive Brokers",
        "tools.data_providers.ib_provider",
        "IBProvider",
        "tools.data_providers.ib_provider.is_available",
    ),
]

# Singleton cache so repeated calls return the same instance
_provider_cache: dict[str, MarketDataProvider] = {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def available_providers() -> list[str]:
    """Return names of providers whose dependencies are installed.

    Always includes "Yahoo Finance". Bloomberg and IB appear only when
    their respective packages (blpapi, ib_insync) are importable.
    """
    names: list[str] = []
    for display_name, module_path, _cls, avail_path in _PROVIDER_REGISTRY:
        if avail_path is None:
            names.append(display_name)
            continue
        try:
            mod_path, func_name = avail_path.rsplit(".", 1)
            mod = importlib.import_module(mod_path)
            avail_fn = getattr(mod, func_name)
            if avail_fn():
                names.append(display_name)
        except Exception:
            # Module couldn't even be imported — provider not available
            pass
    return names


def get_provider(
    name: str = "Yahoo Finance",
    **kwargs: Any,
) -> MarketDataProvider:
    """Instantiate (or return cached) provider by display name.

    Args:
        name: One of the names returned by ``available_providers()``.
        **kwargs: Forwarded to the provider constructor (e.g. host, port).

    Raises:
        ValueError: If the provider name is not in the registry.
        ImportError: If the required package is not installed.
        ConnectionError: If the provider can't connect (Bloomberg/IB).
    """
    # Return cached instance if no custom kwargs
    if not kwargs and name in _provider_cache:
        return _provider_cache[name]

    for display_name, module_path, cls_name, _avail in _PROVIDER_REGISTRY:
        if display_name == name:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, cls_name)
            instance = cls(**kwargs)
            if not kwargs:
                _provider_cache[name] = instance
            return instance

    available = available_providers()
    msg = (
        f"Unknown provider '{name}'. "
        f"Available: {available}"
    )
    raise ValueError(msg)


def get_provider_safe(
    name: str = "Yahoo Finance",
    **kwargs: Any,
) -> MarketDataProvider:
    """Like ``get_provider`` but falls back to Yahoo Finance on any error.

    Use this in production paths where you want graceful degradation
    rather than a hard failure if Bloomberg/IB are down.
    """
    try:
        return get_provider(name, **kwargs)
    except Exception as exc:
        if name != "Yahoo Finance":
            logger.warning(
                "Provider '%s' unavailable (%s), falling back to Yahoo Finance",
                name, exc,
            )
            return get_provider("Yahoo Finance")
        raise


def clear_cache() -> None:
    """Clear the singleton provider cache (useful for testing)."""
    _provider_cache.clear()
