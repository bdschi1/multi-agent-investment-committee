"""Tests for eval schema validation and serialization."""

from __future__ import annotations

import pytest

from evals.schemas import (
    LIKERT_LABELS,
    RECOMMENDATION_ORDER,
    DimensionScore,
    EvalScenario,
    GradingResult,
    GroundTruth,
    LikertLevel,
)

# ---------------------------------------------------------------------------
# GroundTruth
# ---------------------------------------------------------------------------

class TestGroundTruth:
    def test_valid_directions(self):
        for d in (-1, 0, 1):
            gt = GroundTruth(
                expected_direction=d,
                expected_recommendation_bucket="HOLD",
            )
            assert gt.expected_direction == d

    def test_invalid_direction_rejected(self):
        with pytest.raises(ValueError, match="must be -1, 0, or 1"):
            GroundTruth(
                expected_direction=2,
                expected_recommendation_bucket="HOLD",
            )

    def test_defaults(self):
        gt = GroundTruth(
            expected_direction=1,
            expected_recommendation_bucket="BUY",
        )
        assert gt.conviction_range == (0.0, 10.0)
        assert gt.must_find_facts == []
        assert gt.must_find_risks == []
        assert gt.must_not_claim == []
        assert gt.actual_outcome is None
        assert gt.actual_return_pct is None


# ---------------------------------------------------------------------------
# EvalScenario
# ---------------------------------------------------------------------------

class TestEvalScenario:
    def _minimal(self, **overrides) -> dict:
        defaults = {
            "id": "test_scenario",
            "title": "Test",
            "ticker": "AAPL",
            "ground_truth": {
                "expected_direction": 1,
                "expected_recommendation_bucket": "BUY",
            },
        }
        defaults.update(overrides)
        return defaults

    def test_minimal_valid(self):
        s = EvalScenario(**self._minimal())
        assert s.id == "test_scenario"
        assert s.type == "ground_truth"
        assert s.difficulty == "moderate"
        assert s.tags == []

    def test_adversarial_type(self):
        s = EvalScenario(**self._minimal(
            type="adversarial",
            adversarial={
                "manipulation_type": "fake_news",
                "description": "Inject fake headlines",
            },
        ))
        assert s.type == "adversarial"
        assert s.adversarial.manipulation_type == "fake_news"

    def test_invalid_type_rejected(self):
        with pytest.raises(ValueError):
            EvalScenario(**self._minimal(type="unknown"))

    def test_with_full_ground_truth(self):
        s = EvalScenario(**self._minimal(ground_truth={
            "expected_direction": -1,
            "expected_recommendation_bucket": "SELL",
            "conviction_range": [7.0, 9.0],
            "actual_outcome": "Stock dropped 50%",
            "actual_return_pct": -50.0,
            "must_find_facts": ["Revenue declining"],
            "must_find_risks": ["Liquidity crisis"],
            "must_not_claim": ["Company is profitable"],
            "expert_reasoning_sketch": "Classic short setup",
        }))
        assert s.ground_truth.expected_direction == -1
        assert s.ground_truth.conviction_range == (7.0, 9.0)
        assert len(s.ground_truth.must_find_risks) == 1


# ---------------------------------------------------------------------------
# DimensionScore
# ---------------------------------------------------------------------------

class TestDimensionScore:
    def test_valid_score(self):
        ds = DimensionScore(
            dimension_id="direction_accuracy",
            dimension_name="Direction Accuracy",
            weight=25.0,
            raw_score=80.0,
            weighted_score=20.0,
            explanation="Good match",
        )
        assert ds.raw_score == 80.0
        assert ds.weighted_score == 20.0

    def test_score_bounds(self):
        with pytest.raises(ValueError):
            DimensionScore(
                dimension_id="test",
                dimension_name="Test",
                weight=10.0,
                raw_score=101.0,
                weighted_score=10.0,
            )


# ---------------------------------------------------------------------------
# GradingResult
# ---------------------------------------------------------------------------

class TestGradingResult:
    def test_to_json_roundtrip(self):
        gr = GradingResult(
            scenario_id="test",
            ticker="AAPL",
            scenario_type="ground_truth",
            total_score=75.5,
            passed=True,
        )
        data = gr.to_json()
        assert data["scenario_id"] == "test"
        assert data["total_score"] == 75.5
        assert isinstance(data["timestamp"], str)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestLikertLevel:
    def test_valid_levels(self):
        for level in range(1, 6):
            lk = LikertLevel(level=level, label=LIKERT_LABELS[level])
            assert lk.level == level
            assert lk.label == LIKERT_LABELS[level]
            assert lk.anchor == ""

    def test_with_anchor(self):
        lk = LikertLevel(
            level=5,
            label="Excellent",
            anchor="Direction correct, exact bucket match",
        )
        assert lk.anchor == "Direction correct, exact bucket match"

    def test_level_below_1_rejected(self):
        with pytest.raises(ValueError):
            LikertLevel(level=0, label="Invalid")

    def test_level_above_5_rejected(self):
        with pytest.raises(ValueError):
            LikertLevel(level=6, label="Invalid")

    def test_dimension_score_with_likert(self):
        lk = LikertLevel(level=4, label="Good", anchor="Most risks covered")
        ds = DimensionScore(
            dimension_id="risk_identification",
            dimension_name="Risk Identification",
            weight=20.0,
            raw_score=80.0,
            weighted_score=16.0,
            explanation="Found 3/4 risks",
            likert_level=lk,
        )
        assert ds.likert_level.level == 4
        assert ds.likert_level.label == "Good"

    def test_dimension_score_likert_defaults_to_none(self):
        ds = DimensionScore(
            dimension_id="test",
            dimension_name="Test",
            weight=10.0,
            raw_score=50.0,
            weighted_score=5.0,
        )
        assert ds.likert_level is None

    def test_likert_labels_complete(self):
        assert len(LIKERT_LABELS) == 5
        assert LIKERT_LABELS[1] == "Fail"
        assert LIKERT_LABELS[5] == "Excellent"


class TestConstants:
    def test_recommendation_order_length(self):
        assert len(RECOMMENDATION_ORDER) == 7

    def test_recommendation_order_starts_with_strong_buy(self):
        assert RECOMMENDATION_ORDER[0] == "STRONG BUY"

    def test_recommendation_order_ends_with_avoid(self):
        assert RECOMMENDATION_ORDER[-1] == "AVOID"
