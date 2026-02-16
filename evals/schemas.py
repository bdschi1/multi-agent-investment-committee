"""
Eval harness data models.

Pydantic schemas for scenario definitions, ground truth,
adversarial configs, and grading results.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Scenario input models
# ---------------------------------------------------------------------------

class GroundTruth(BaseModel):
    """Expert-provided ground truth for a scenario."""

    expected_direction: int = Field(
        description="Expected position direction: +1 (long), -1 (short), 0 (flat/avoid)",
    )
    expected_recommendation_bucket: str = Field(
        description="Expected recommendation from the committee",
    )
    conviction_range: tuple[float, float] = Field(
        default=(0.0, 10.0),
        description="Acceptable conviction range [min, max]",
    )
    actual_outcome: str | None = Field(
        default=None,
        description="What actually happened (expert fills in)",
    )
    actual_return_pct: float | None = Field(
        default=None,
        description="Actual return over eval horizon",
    )
    must_find_facts: list[str] = Field(
        default_factory=list,
        description="Key facts the system should surface",
    )
    must_find_risks: list[str] = Field(
        default_factory=list,
        description="Key risks the system should identify",
    )
    must_not_claim: list[str] = Field(
        default_factory=list,
        description="Statements that would be hallucinations",
    )
    expert_reasoning_sketch: str | None = Field(
        default=None,
        description="Expert's brief thesis for comparison",
    )

    @field_validator("expected_direction")
    @classmethod
    def _validate_direction(cls, v: int) -> int:
        if v not in (-1, 0, 1):
            msg = f"expected_direction must be -1, 0, or 1, got {v}"
            raise ValueError(msg)
        return v


class AdversarialExpectedBehavior(BaseModel):
    """What the system should do when encountering adversarial data."""

    should_flag_inconsistency: bool = True
    should_question_data_source: bool = False
    expected_signals: list[str] = Field(
        default_factory=list,
        description="Phrases/concepts the system should surface",
    )


class AdversarialConfig(BaseModel):
    """Configuration for adversarial data injection."""

    manipulation_type: str = Field(
        description="Category: inflated_fundamentals, fake_news, "
        "contradictory_data, authority_anchor, stale_data_injection",
    )
    description: str = Field(
        description="Human-readable description of the manipulation",
    )
    context_injections: dict[str, Any] = Field(
        default_factory=dict,
        description="Key-value pairs merged into DataAggregator context",
    )
    expected_behavior: AdversarialExpectedBehavior = Field(
        default_factory=AdversarialExpectedBehavior,
    )
    failure_signals: list[str] = Field(
        default_factory=list,
        description="Phrases indicating the system was fooled",
    )


class EvaluationCriteria(BaseModel):
    """Which rubric and overrides to apply."""

    rubric: str = Field(
        default="committee_standard",
        description="Rubric ID referencing a YAML file in rubrics/",
    )
    dimension_overrides: dict[str, float] = Field(
        default_factory=dict,
        description="Override dimension weights for this scenario",
    )
    critical_failures: list[str] = Field(
        default_factory=list,
        description="Scenario-specific auto-fail conditions",
    )


class ScenarioMetadata(BaseModel):
    """Provenance and notes."""

    author: str = ""
    created_at: str = ""
    notes: str = ""


class EvalScenario(BaseModel):
    """A single evaluation scenario loaded from YAML."""

    id: str
    title: str
    version: str = "1.0.0"
    type: Literal["ground_truth", "adversarial"] = "ground_truth"
    difficulty: str = "moderate"
    tags: list[str] = Field(default_factory=list)

    ticker: str
    as_of_date: str = ""
    as_of_context: str = ""
    user_context: str = ""

    ground_truth: GroundTruth
    adversarial: AdversarialConfig | None = None
    evaluation_criteria: EvaluationCriteria = Field(
        default_factory=EvaluationCriteria,
    )
    metadata: ScenarioMetadata = Field(default_factory=ScenarioMetadata)


# ---------------------------------------------------------------------------
# Grading output models
# ---------------------------------------------------------------------------

RECOMMENDATION_ORDER: list[str] = [
    "STRONG BUY",
    "BUY",
    "HOLD",
    "UNDERWEIGHT",
    "SELL",
    "ACTIVE SHORT",
    "AVOID",
]


class LikertLevel(BaseModel):
    """Ordinal quality level mapped from a continuous 0-100 score.

    Provides a human-readable assessment layer on top of numeric scores.
    Each level carries an anchor description defining what that level
    means for the specific dimension being evaluated.

    Standard 5-point scale:
        5 = Excellent, 4 = Good, 3 = Adequate, 2 = Poor, 1 = Fail
    """

    level: int = Field(
        ge=1, le=5,
        description="Ordinal level (1=Fail, 2=Poor, 3=Adequate, 4=Good, 5=Excellent)",
    )
    label: str = Field(
        description="Human-readable label: Fail | Poor | Adequate | Good | Excellent",
    )
    anchor: str = Field(
        default="",
        description="Dimension-specific anchor text defining this level",
    )


# Canonical labels for each Likert point
LIKERT_LABELS: dict[int, str] = {
    1: "Fail",
    2: "Poor",
    3: "Adequate",
    4: "Good",
    5: "Excellent",
}


class DimensionScore(BaseModel):
    """Score for a single rubric dimension."""

    dimension_id: str
    dimension_name: str
    weight: float
    raw_score: float = Field(
        ge=0.0, le=100.0,
        description="Score within this dimension (0-100)",
    )
    weighted_score: float = Field(
        description="raw_score * weight / 100",
    )
    explanation: str = ""
    likert_level: LikertLevel | None = Field(
        default=None,
        description="Ordinal quality assessment with dimension-specific anchor",
    )


class GradingResult(BaseModel):
    """Complete grading output for one scenario run."""

    scenario_id: str
    ticker: str
    scenario_type: str
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
    )
    dimension_scores: list[DimensionScore] = Field(default_factory=list)
    total_score: float = 0.0
    passed: bool = False
    critical_failures: list[str] = Field(default_factory=list)
    adversarial_result: dict[str, Any] | None = None
    committee_result_summary: dict[str, Any] = Field(default_factory=dict)
    run_metadata: dict[str, Any] = Field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        """Serialize for JSON output."""
        data = self.model_dump()
        data["timestamp"] = self.timestamp.isoformat()
        return data
