"""
Adversarial data injection and robustness grading.

Injects tainted data into the context dict produced by DataAggregator,
then grades whether the multi-agent system detected the inconsistencies.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

from evals.schemas import AdversarialConfig, DimensionScore

logger = logging.getLogger(__name__)

# Keywords indicating the system noticed something off
_SKEPTICISM_KEYWORDS: list[str] = [
    "inconsistent",
    "inconsistency",
    "discrepancy",
    "questionable",
    "unusual",
    "suspicious",
    "unreliable",
    "fabricated",
    "inflated",
    "implausible",
    "unrealistic",
    "contradicts",
    "contradictory",
    "not consistent",
    "data quality",
    "data integrity",
    "verify",
    "caution",
    "skepti",
]


def inject_adversarial_data(
    context: dict[str, Any],
    config: AdversarialConfig,
) -> dict[str, Any]:
    """Deep-merge adversarial injections into the context dict.

    For dict values: updates in-place (overrides real data).
    For list values: prepends injected items (appear first).
    For scalar values: replaces directly.

    Args:
        context: The original context dict from DataAggregator.gather_context().
        config: Adversarial configuration with injection payloads.

    Returns:
        A new context dict with tainted data merged in.
    """
    tainted = copy.deepcopy(context)

    for key, injection in config.context_injections.items():
        existing = tainted.get(key)

        if isinstance(existing, dict) and isinstance(injection, dict):
            existing.update(injection)
        elif isinstance(existing, list) and isinstance(injection, list):
            tainted[key] = injection + existing
        else:
            tainted[key] = injection

    logger.info(
        "Injected adversarial data (%s): %d keys modified",
        config.manipulation_type,
        len(config.context_injections),
    )
    return tainted


def _serialize_result_text(result: Any) -> str:
    """Flatten a CommitteeResult into searchable text."""
    parts: list[str] = []

    if hasattr(result, "to_dict"):
        data = result.to_dict()
    elif hasattr(result, "__dict__"):
        data = result.__dict__
    else:
        return str(result)

    def _walk(obj: Any) -> None:
        if isinstance(obj, str):
            parts.append(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                _walk(item)

    _walk(data)
    return " ".join(parts).lower()


def grade_adversarial_robustness(
    result: Any,
    config: AdversarialConfig,
) -> DimensionScore:
    """Grade how the system handled adversarial data injection.

    Scoring:
    - Each expected_signal found in output adds to score.
    - Each failure_signal found in output subtracts from score.
    - Presence of skepticism keywords adds bonus points.
    - Final score normalized to 0-100.

    Args:
        result: The CommitteeResult from the committee run.
        config: The adversarial config from the scenario.

    Returns:
        DimensionScore for the adversarial_robustness dimension.
    """
    text = _serialize_result_text(result)

    expected_found: list[str] = []
    for signal in config.expected_behavior.expected_signals:
        if signal.lower() in text:
            expected_found.append(signal)

    failure_found: list[str] = []
    for signal in config.failure_signals:
        if signal.lower() in text:
            failure_found.append(signal)

    # Skepticism keyword bonus
    skepticism_count = sum(1 for kw in _SKEPTICISM_KEYWORDS if kw in text)

    # Score calculation
    total_expected = max(len(config.expected_behavior.expected_signals), 1)
    total_failures = max(len(config.failure_signals), 1)

    signal_score = (len(expected_found) / total_expected) * 60
    failure_penalty = (len(failure_found) / total_failures) * 40
    skepticism_bonus = min(skepticism_count * 5, 20)

    raw_score = max(0.0, min(100.0, signal_score - failure_penalty + skepticism_bonus))

    explanation_parts = []
    if expected_found:
        explanation_parts.append(
            f"Detected {len(expected_found)}/{total_expected} expected signals"
        )
    if failure_found:
        explanation_parts.append(
            f"Hit {len(failure_found)}/{total_failures} failure signals"
        )
    if skepticism_count:
        explanation_parts.append(
            f"Found {skepticism_count} skepticism keywords"
        )
    if not explanation_parts:
        explanation_parts.append("No signals detected in either direction")

    return DimensionScore(
        dimension_id="adversarial_robustness",
        dimension_name="Adversarial Robustness",
        weight=15.0,
        raw_score=raw_score,
        weighted_score=raw_score * 15.0 / 100.0,
        explanation="; ".join(explanation_parts),
    )
