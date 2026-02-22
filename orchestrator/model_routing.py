"""
Per-node model routing for the investment committee pipeline.

Different pipeline stages can use different LLM models:

    Task Type          Default Model        Reasoning
    -----------------  -------------------  ----------------------------------
    Analyst reasoning  (shared model)       Can override with cheaper/faster model
    Debate             (shared model)       Can override with cheaper/faster model
    PM synthesis       (shared model)       Can override with premium model

Usage:
    from orchestrator.model_routing import build_model_cache

    cache = build_model_cache(default_model, create_model)
    # Pass cache into graph config; nodes check it via _get_model()

Users configure via .env or settings.task_models:

    TASK_MODELS={"run_sector_analyst": "google:gemini-2.0-flash",
                 "run_portfolio_manager": "anthropic:claude-sonnet-4-20250514"}

Value format is "provider:model_name".  If no colon, uses the current provider.
"""

from __future__ import annotations

import logging
from typing import Any

from config.settings import LLMProvider, settings

logger = logging.getLogger(__name__)


def parse_model_spec(spec: str) -> tuple[str | None, str]:
    """Parse a 'provider:model_name' spec into (provider, model_name).

    If no colon, returns (None, spec) meaning 'use current provider'.

    >>> parse_model_spec("anthropic:claude-sonnet-4-20250514")
    ('anthropic', 'claude-sonnet-4-20250514')
    >>> parse_model_spec("gemini-2.0-flash")
    (None, 'gemini-2.0-flash')
    """
    if ":" in spec:
        provider, model = spec.split(":", 1)
        return provider.strip(), model.strip()
    return None, spec.strip()


def get_node_model_spec(node_name: str) -> str | None:
    """Get the model spec for a specific pipeline node, or None if no override.

    Checks user overrides in settings.task_models.
    """
    user_overrides = getattr(settings, "task_models", {})
    if user_overrides and node_name in user_overrides:
        return user_overrides[node_name]
    return None


def build_model_cache(
    default_model: Any,
    create_model_fn: Any,
) -> dict[str, Any]:
    """Pre-create all per-node model overrides at graph invocation time.

    Returns a dict of {node_name: model_callable} for nodes that have
    overrides.  Nodes not in the dict use the default_model.

    Models with the same spec share a single instance (and rate limiter).

    Args:
        default_model: The shared model callable (fallback).
        create_model_fn: Factory ``create_model(provider, model_name=...)``
    """
    cache: dict[str, Any] = {}
    user_overrides = getattr(settings, "task_models", {})

    if not user_overrides:
        return cache

    # Group by spec to share rate limiters
    spec_to_model: dict[str, Any] = {}

    for node_name, spec in user_overrides.items():
        if spec in spec_to_model:
            cache[node_name] = spec_to_model[spec]
            continue

        provider_str, model_name = parse_model_spec(spec)
        try:
            if provider_str:
                provider = LLMProvider(provider_str)
                model = create_model_fn(provider, model_name=model_name)
            else:
                model = create_model_fn(settings.llm_provider, model_name=model_name)
            spec_to_model[spec] = model
            cache[node_name] = model
            logger.info(f"Model override for {node_name}: {spec}")
        except Exception as e:
            logger.warning(
                f"Failed to create model override for {node_name} ({spec}): {e}. "
                f"Using default."
            )

    return cache
