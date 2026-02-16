"""
Scenario and rubric loading from YAML files.

Discovers scenario files, validates against Pydantic schemas,
and provides filtering by type/tag/ticker.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from evals.schemas import EvalScenario

logger = logging.getLogger(__name__)

_EVALS_DIR = Path(__file__).parent
_SCENARIOS_DIR = _EVALS_DIR / "scenarios"
_RUBRICS_DIR = _EVALS_DIR / "rubrics"


def load_scenario(path: Path) -> EvalScenario:
    """Load and validate a single scenario YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return EvalScenario(**data)


def discover_scenarios(
    scenarios_dir: Path | None = None,
    filter_type: str | None = None,
    filter_tags: list[str] | None = None,
    filter_scenario: str | None = None,
) -> list[EvalScenario]:
    """Discover and load all scenario YAML files.

    Skips files with ``_`` prefix (templates).

    Args:
        scenarios_dir: Directory to scan. Defaults to ``evals/scenarios/``.
        filter_type: Only return scenarios of this type ("ground_truth" or "adversarial").
        filter_tags: Only return scenarios containing at least one of these tags.
        filter_scenario: Only return the scenario with this ID (partial match).

    Returns:
        List of validated ``EvalScenario`` objects sorted by ID.
    """
    scenarios_dir = scenarios_dir or _SCENARIOS_DIR
    scenarios: list[EvalScenario] = []

    for path in sorted(scenarios_dir.glob("*.yaml")):
        if path.name.startswith("_"):
            continue
        try:
            scenario = load_scenario(path)
        except Exception:
            logger.warning("Failed to load scenario: %s", path.name, exc_info=True)
            continue

        # Apply filters
        if filter_type and scenario.type != filter_type:
            continue
        if filter_tags and not set(filter_tags) & set(scenario.tags):
            continue
        if filter_scenario and filter_scenario not in scenario.id:
            continue

        scenarios.append(scenario)

    return scenarios


def load_rubric(
    rubric_id: str,
    rubrics_dir: Path | None = None,
) -> dict[str, Any]:
    """Load a rubric YAML by its ID.

    Args:
        rubric_id: The rubric identifier (filename without .yaml).
        rubrics_dir: Directory to search. Defaults to ``evals/rubrics/``.

    Returns:
        Parsed rubric dict.

    Raises:
        FileNotFoundError: If the rubric file doesn't exist.
    """
    rubrics_dir = rubrics_dir or _RUBRICS_DIR
    path = rubrics_dir / f"{rubric_id}.yaml"
    if not path.exists():
        msg = f"Rubric not found: {path}"
        raise FileNotFoundError(msg)
    with open(path) as f:
        return yaml.safe_load(f)
