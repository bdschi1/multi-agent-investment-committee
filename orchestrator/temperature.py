"""
Per-node temperature routing for the investment committee pipeline.

Different pipeline stages benefit from different temperature settings:

    Task Type          Temperature  Reasoning
    ─────────────────  ───────────  ──────────────────────────────────────
    Data extraction    0.1          Factual retrieval — one "right" answer
    Risk assessment    0.5          Explore unlikely-but-possible tail risks
    Analyst reasoning  0.5          Balanced exploration of thesis points
    Debate (rebuttals) 0.5          Adversarial — consider edge arguments
    PM synthesis       0.7          Narrative flow, connecting themes
    Math / valuation   0.0          Deterministic computation (BL optimizer)

Usage:
    model = _get_model(state, config)
    model = with_temperature(model, get_node_temperature("run_sector_analyst"))
    agent = SectorAnalystAgent(model=model)

The with_temperature() wrapper calls model(prompt, temperature=X). If the
underlying model doesn't support the kwarg (e.g., a mock in tests), it falls
back to model(prompt) silently.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from config.settings import settings

logger = logging.getLogger(__name__)

# Default temperature by pipeline node.
# Users can override via settings.task_temperatures (config/settings.py).
_DEFAULT_TASK_TEMPERATURES: dict[str, float] = {
    # Phase 0: data gathering
    "gather_data": 0.1,
    # Phase 1: parallel analysis
    "run_sector_analyst": 0.5,
    "run_short_analyst": 0.5,
    "run_risk_manager": 0.5,
    "run_macro_analyst": 0.5,
    # Phase 1 reporters (no LLM calls, but included for completeness)
    "report_phase1": 0.0,
    # Phase 2: debate
    "run_debate_round": 0.5,
    "report_debate_complete": 0.0,
    # Phase 3: PM synthesis
    "run_portfolio_manager": 0.7,
    # Phase 3b: optimizer (no LLM calls — pure math)
    "run_optimizer": 0.0,
    # Finalize (no LLM calls)
    "finalize": 0.0,
}


def get_node_temperature(node_name: str) -> float:
    """
    Get the temperature for a specific pipeline node.

    Checks user overrides in settings.task_temperatures first,
    then falls back to the built-in defaults.
    """
    user_overrides = getattr(settings, 'task_temperatures', {})
    if user_overrides and node_name in user_overrides:
        return user_overrides[node_name]
    return _DEFAULT_TASK_TEMPERATURES.get(node_name, settings.temperature)


def with_temperature(model: Any, temperature: float) -> Callable[[str], str]:
    """
    Wrap a model callable to use a specific temperature.

    Returns a new callable that passes temperature=X to the underlying
    model. If the model doesn't support the kwarg (e.g., test mocks),
    falls back gracefully to calling without it.

    Args:
        model: The LLM callable (str) -> str
        temperature: Temperature to use for this call

    Returns:
        Wrapped callable with the same (str) -> str interface.
    """
    def wrapped(prompt: str) -> str:
        try:
            return model(prompt, temperature=temperature)
        except TypeError:
            # Model doesn't support temperature kwarg — call without it
            return model(prompt)

    # Preserve any attributes on the original model (e.g., RateLimitedModel internals)
    wrapped._original_model = model
    wrapped._temperature = temperature

    return wrapped
