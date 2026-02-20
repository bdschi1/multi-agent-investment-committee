"""Tests for the multi-dimensional grading engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from evals.adversarial import grade_adversarial_robustness, inject_adversarial_data
from evals.grader import (
    _assign_likert_level,
    _check_must_not_claim,
    _fuzzy_match,
    _grade_conviction_calibration,
    _grade_direction,
    _grade_fact_coverage,
    _grade_reasoning_quality,
    _grade_risk_identification,
    _score_to_likert_level,
    grade_result,
)
from evals.schemas import (
    AdversarialConfig,
    AdversarialExpectedBehavior,
    DimensionScore,
    EvalScenario,
    GroundTruth,
)

# ---------------------------------------------------------------------------
# Mock CommitteeResult and sub-models
# ---------------------------------------------------------------------------

@dataclass
class MockMemo:
    recommendation: str = "BUY"
    position_direction: int = 1
    conviction: float = 7.0
    raw_confidence: float = 0.7
    t_signal: float = 0.5
    thesis_summary: str = "Strong momentum"
    key_factors: list[str] = field(default_factory=lambda: ["AI demand"])
    bull_points_accepted: list[str] = field(default_factory=lambda: ["Revenue growth"])
    bear_points_accepted: list[str] = field(default_factory=lambda: ["Valuation risk"])
    dissenting_points: list[str] = field(default_factory=list)
    risk_mitigants: list[str] = field(default_factory=list)

    def model_dump(self) -> dict:
        return self.__dict__


@dataclass
class MockBullCase:
    thesis: str = "AI datacenter demand driving growth"
    supporting_evidence: list[str] = field(
        default_factory=lambda: ["Data center revenue up 200%"],
    )
    catalysts: list[str] = field(
        default_factory=lambda: ["Blackwell architecture launch"],
    )
    conviction_score: float = 8.0

    def model_dump(self) -> dict:
        return self.__dict__


@dataclass
class MockBearCase:
    risks: list[str] = field(
        default_factory=lambda: [
            "Customer concentration in hyperscalers",
            "AMD MI300X competitive threat",
        ],
    )
    second_order_effects: list[str] = field(
        default_factory=lambda: ["Capex cycle slowdown"],
    )
    worst_case_scenario: str = "AI bubble bursts, capex cuts"
    bearish_conviction: float = 5.0

    def model_dump(self) -> dict:
        return self.__dict__


@dataclass
class MockRebuttal:
    argument: str = "Strong counter-argument here"

    def model_dump(self) -> dict:
        return self.__dict__


@dataclass
class MockConvictionSnapshot:
    phase: str = "Initial"
    agent: str = "Sector Analyst"
    score: float = 7.0
    score_type: str = "conviction"


@dataclass
class MockStep:
    content: str = "Reasoning step content"


@dataclass
class MockTrace:
    steps: list[MockStep] = field(
        default_factory=lambda: [MockStep(), MockStep()],
    )


@dataclass
class MockResult:
    ticker: str = "NVDA"
    bull_case: Any = None
    bear_case: Any = None
    short_case: Any = None
    macro_view: Any = None
    long_rebuttal: Any = None
    short_rebuttal: Any = None
    risk_rebuttal: Any = None
    committee_memo: Any = None
    traces: dict = field(default_factory=dict)
    conviction_timeline: list = field(default_factory=list)
    parsing_failures: list = field(default_factory=list)
    total_duration_ms: float = 5000.0
    total_tokens: int = 10000

    def to_dict(self) -> dict:
        result = {}
        for k, v in self.__dict__.items():
            if hasattr(v, "model_dump"):
                result[k] = v.model_dump()
            elif isinstance(v, list):
                result[k] = [
                    item.model_dump() if hasattr(item, "model_dump") else item
                    for item in v
                ]
            elif isinstance(v, dict):
                result[k] = {
                    dk: dv.model_dump() if hasattr(dv, "model_dump") else dv
                    for dk, dv in v.items()
                }
            else:
                result[k] = v
        return result


def _make_result(**overrides) -> MockResult:
    defaults = {
        "committee_memo": MockMemo(),
        "bull_case": MockBullCase(),
        "bear_case": MockBearCase(),
        "long_rebuttal": MockRebuttal(),
        "short_rebuttal": MockRebuttal(),
        "risk_rebuttal": MockRebuttal(),
        "traces": {
            "sector_analyst": MockTrace(),
            "short_analyst": MockTrace(),
            "risk_manager": MockTrace(),
            "portfolio_manager": MockTrace(),
        },
        "conviction_timeline": [
            MockConvictionSnapshot(phase="Initial", agent="Long Analyst", score=7.0),
            MockConvictionSnapshot(phase="Initial", agent="Short Analyst", score=4.5),
            MockConvictionSnapshot(phase="Post-Debate", agent="Long Analyst", score=6.5),
            MockConvictionSnapshot(phase="PM Decision", agent="Portfolio Manager", score=7.2),
        ],
    }
    defaults.update(overrides)
    return MockResult(**defaults)


def _make_truth(**overrides) -> GroundTruth:
    defaults = {
        "expected_direction": 1,
        "expected_recommendation_bucket": "BUY",
        "conviction_range": (6.0, 9.0),
        "must_find_facts": ["AI datacenter demand", "data center revenue"],
        "must_find_risks": ["customer concentration", "AMD competition"],
    }
    defaults.update(overrides)
    return GroundTruth(**defaults)


def _make_scenario(**overrides) -> EvalScenario:
    defaults = {
        "id": "test",
        "title": "Test Scenario",
        "ticker": "NVDA",
        "ground_truth": _make_truth().model_dump(),
        "evaluation_criteria": {"rubric": "committee_standard"},
    }
    defaults.update(overrides)
    return EvalScenario(**defaults)


# ---------------------------------------------------------------------------
# Fuzzy matching
# ---------------------------------------------------------------------------

class TestFuzzyMatch:
    def test_exact_match(self):
        assert _fuzzy_match("customer concentration", "customer concentration risk")

    def test_keyword_overlap(self):
        assert _fuzzy_match(
            "AMD MI300X competitive threat",
            "The AMD MI300X poses a significant competitive challenge",
        )

    def test_no_match(self):
        assert not _fuzzy_match(
            "cryptocurrency regulation",
            "AI datacenter demand is strong",
        )

    def test_empty_needle(self):
        assert _fuzzy_match("", "anything")


# ---------------------------------------------------------------------------
# Direction grading
# ---------------------------------------------------------------------------

class TestGradeDirection:
    def test_exact_match(self):
        result = _make_result()
        truth = _make_truth()
        score = _grade_direction(result, truth)
        assert score.raw_score == 100.0

    def test_direction_correct_adjacent_bucket(self):
        result = _make_result(committee_memo=MockMemo(recommendation="STRONG BUY"))
        truth = _make_truth(expected_recommendation_bucket="BUY")
        score = _grade_direction(result, truth)
        assert score.raw_score == 80.0

    def test_wrong_direction_low_conviction(self):
        result = _make_result(
            committee_memo=MockMemo(position_direction=-1, conviction=4.0),
        )
        truth = _make_truth(expected_direction=1)
        score = _grade_direction(result, truth)
        assert score.raw_score == 20.0

    def test_wrong_direction_high_conviction(self):
        result = _make_result(
            committee_memo=MockMemo(
                position_direction=-1,
                conviction=9.0,
                recommendation="ACTIVE SHORT",
            ),
        )
        truth = _make_truth(expected_direction=1)
        score = _grade_direction(result, truth)
        assert score.raw_score == 0.0

    def test_no_memo(self):
        result = _make_result(committee_memo=None)
        truth = _make_truth()
        score = _grade_direction(result, truth)
        assert score.raw_score == 0.0


# ---------------------------------------------------------------------------
# Conviction calibration
# ---------------------------------------------------------------------------

class TestGradeConvictionCalibration:
    def test_in_range(self):
        result = _make_result(committee_memo=MockMemo(conviction=7.0))
        truth = _make_truth(conviction_range=(6.0, 9.0))
        score = _grade_conviction_calibration(result, truth)
        assert score.raw_score == 100.0

    def test_near_range(self):
        result = _make_result(committee_memo=MockMemo(conviction=5.5))
        truth = _make_truth(conviction_range=(6.0, 9.0))
        score = _grade_conviction_calibration(result, truth)
        assert score.raw_score == 70.0

    def test_far_from_range(self):
        result = _make_result(committee_memo=MockMemo(conviction=2.0))
        truth = _make_truth(conviction_range=(6.0, 9.0))
        score = _grade_conviction_calibration(result, truth)
        assert score.raw_score == 40.0

    def test_wrong_direction_high_conviction(self):
        result = _make_result(
            committee_memo=MockMemo(position_direction=-1, conviction=9.0),
        )
        truth = _make_truth(expected_direction=1, conviction_range=(6.0, 9.0))
        score = _grade_conviction_calibration(result, truth)
        assert score.raw_score == 0.0


# ---------------------------------------------------------------------------
# Risk identification
# ---------------------------------------------------------------------------

class TestGradeRiskIdentification:
    def test_all_risks_found(self):
        result = _make_result()
        truth = _make_truth(must_find_risks=["customer concentration", "AMD competition"])
        score = _grade_risk_identification(result, truth)
        assert score.raw_score >= 80.0

    def test_no_risks_specified(self):
        result = _make_result()
        truth = _make_truth(must_find_risks=[])
        score = _grade_risk_identification(result, truth)
        assert score.raw_score == 100.0

    def test_risks_not_found(self):
        result = _make_result()
        truth = _make_truth(
            must_find_risks=["cryptocurrency regulation", "quantum computing disruption"],
        )
        score = _grade_risk_identification(result, truth)
        assert score.raw_score <= 30.0


# ---------------------------------------------------------------------------
# Fact coverage
# ---------------------------------------------------------------------------

class TestGradeFactCoverage:
    def test_facts_found(self):
        result = _make_result()
        truth = _make_truth(must_find_facts=["AI datacenter demand"])
        score = _grade_fact_coverage(result, truth)
        assert score.raw_score >= 80.0

    def test_no_facts_specified(self):
        result = _make_result()
        truth = _make_truth(must_find_facts=[])
        score = _grade_fact_coverage(result, truth)
        assert score.raw_score == 100.0


# ---------------------------------------------------------------------------
# Reasoning quality
# ---------------------------------------------------------------------------

class TestGradeReasoningQuality:
    def test_full_quality(self):
        result = _make_result()
        score = _grade_reasoning_quality(result)
        assert score.raw_score >= 75.0

    def test_no_memo(self):
        result = _make_result(committee_memo=None)
        score = _grade_reasoning_quality(result)
        # No memo → loses PM-synthesis points (25), but keeps timeline + rebuttal + traces
        assert score.raw_score == 75.0

    def test_no_rebuttals(self):
        result = _make_result(
            long_rebuttal=None,
            short_rebuttal=None,
            risk_rebuttal=None,
        )
        score = _grade_reasoning_quality(result)
        # No rebuttals → drops from 25 to 5, but keeps all other components
        assert score.raw_score == 80.0


# ---------------------------------------------------------------------------
# Critical failures
# ---------------------------------------------------------------------------

class TestMustNotClaim:
    def test_no_violation(self):
        result = _make_result()
        violations = _check_must_not_claim(result, ["cryptocurrency is the future"])
        assert violations == []

    def test_violation_found(self):
        result = _make_result(
            committee_memo=MockMemo(thesis_summary="AI datacenter demand is strong"),
        )
        violations = _check_must_not_claim(result, ["ai datacenter demand is strong"])
        assert len(violations) == 1


# ---------------------------------------------------------------------------
# Adversarial
# ---------------------------------------------------------------------------

class TestAdversarial:
    def test_inject_dict(self):
        context = {"financial_metrics": {"pe": 25.0, "revenue_growth": "5%"}}
        config = AdversarialConfig(
            manipulation_type="inflated_fundamentals",
            description="Inflate revenue",
            context_injections={"financial_metrics": {"revenue_growth": "80%"}},
        )
        tainted = inject_adversarial_data(context, config)
        assert tainted["financial_metrics"]["revenue_growth"] == "80%"
        assert tainted["financial_metrics"]["pe"] == 25.0  # unchanged

    def test_inject_list_prepends(self):
        context = {"news": [{"headline": "Real news"}]}
        config = AdversarialConfig(
            manipulation_type="fake_news",
            description="Inject fake headline",
            context_injections={"news": [{"headline": "FAKE breaking news"}]},
        )
        tainted = inject_adversarial_data(context, config)
        assert len(tainted["news"]) == 2
        assert tainted["news"][0]["headline"] == "FAKE breaking news"
        assert tainted["news"][1]["headline"] == "Real news"

    def test_grade_robustness_with_signals(self):
        result = _make_result(
            bear_case=MockBearCase(
                risks=["Data inconsistency detected", "Questionable metrics"],
            ),
        )
        config = AdversarialConfig(
            manipulation_type="inflated_fundamentals",
            description="Test",
            expected_behavior=AdversarialExpectedBehavior(
                expected_signals=["inconsistency", "questionable"],
            ),
            failure_signals=["accepted at face value"],
        )
        score = grade_adversarial_robustness(result, config)
        assert score.raw_score > 50.0

    def test_grade_robustness_fooled(self):
        result = _make_result(
            committee_memo=MockMemo(
                thesis_summary="Strong buy based solely on fabricated metrics",
            ),
        )
        config = AdversarialConfig(
            manipulation_type="inflated_fundamentals",
            description="Test",
            expected_behavior=AdversarialExpectedBehavior(
                expected_signals=["flagged inconsistency"],
            ),
            failure_signals=["strong buy based solely on fabricated metrics"],
        )
        score = grade_adversarial_robustness(result, config)
        assert score.raw_score < 50.0


# ---------------------------------------------------------------------------
# Likert mapping
# ---------------------------------------------------------------------------

class TestScoreToLikertLevel:
    def test_boundaries_default(self):
        assert _score_to_likert_level(0.0) == 1    # Fail
        assert _score_to_likert_level(10.0) == 1   # still Fail
        assert _score_to_likert_level(24.9) == 1   # just below Poor
        assert _score_to_likert_level(25.0) == 2   # Poor
        assert _score_to_likert_level(49.9) == 2   # still Poor
        assert _score_to_likert_level(50.0) == 3   # Adequate
        assert _score_to_likert_level(69.9) == 3   # still Adequate
        assert _score_to_likert_level(70.0) == 4   # Good
        assert _score_to_likert_level(79.9) == 4   # still Good
        assert _score_to_likert_level(80.0) == 5   # Excellent
        assert _score_to_likert_level(100.0) == 5  # max Excellent

    def test_custom_boundaries(self):
        # Stricter scale: [Fail, Poor, Adequate, Good, Excellent]
        bounds = [0, 30, 60, 80, 90]
        assert _score_to_likert_level(25.0, bounds) == 1
        assert _score_to_likert_level(55.0, bounds) == 2
        assert _score_to_likert_level(75.0, bounds) == 3
        assert _score_to_likert_level(85.0, bounds) == 4
        assert _score_to_likert_level(95.0, bounds) == 5


class TestAssignLikertLevel:
    def test_assigns_level_and_label(self):
        score = DimensionScore(
            dimension_id="direction_accuracy",
            dimension_name="Direction Accuracy",
            weight=25.0,
            raw_score=100.0,
            weighted_score=25.0,
            explanation="Exact match",
        )
        rubric = {
            "likert_scale": {"boundaries": [0, 25, 50, 70, 80]},
            "dimensions": [
                {
                    "id": "direction_accuracy",
                    "likert_anchors": {
                        5: "Direction correct, exact match",
                        4: "Direction correct, off by one",
                        3: "Direction correct, off by two+",
                        2: "Wrong direction, low conviction",
                        1: "Wrong direction, high conviction",
                    },
                },
            ],
        }
        _assign_likert_level(score, rubric)
        assert score.likert_level is not None
        assert score.likert_level.level == 5
        assert score.likert_level.label == "Excellent"
        assert score.likert_level.anchor == "Direction correct, exact match"

    def test_assigns_fail_for_zero(self):
        score = DimensionScore(
            dimension_id="direction_accuracy",
            dimension_name="Direction Accuracy",
            weight=25.0,
            raw_score=0.0,
            weighted_score=0.0,
            explanation="Wrong direction",
        )
        rubric = {
            "dimensions": [
                {
                    "id": "direction_accuracy",
                    "likert_anchors": {1: "Completely wrong"},
                },
            ],
        }
        _assign_likert_level(score, rubric)
        assert score.likert_level.level == 1
        assert score.likert_level.label == "Fail"
        assert score.likert_level.anchor == "Completely wrong"

    def test_works_without_rubric_anchors(self):
        score = DimensionScore(
            dimension_id="unknown_dim",
            dimension_name="Unknown",
            weight=10.0,
            raw_score=75.0,
            weighted_score=7.5,
        )
        rubric = {"dimensions": []}
        _assign_likert_level(score, rubric)
        assert score.likert_level is not None
        assert score.likert_level.level == 4
        assert score.likert_level.label == "Good"
        assert score.likert_level.anchor == ""

    def test_works_with_empty_rubric(self):
        score = DimensionScore(
            dimension_id="test",
            dimension_name="Test",
            weight=10.0,
            raw_score=55.0,
            weighted_score=5.5,
        )
        _assign_likert_level(score, {})
        assert score.likert_level.level == 3
        assert score.likert_level.label == "Adequate"


# ---------------------------------------------------------------------------
# Full grading
# ---------------------------------------------------------------------------

class TestGradeResult:
    def test_passing_result(self):
        result = _make_result()
        scenario = _make_scenario()
        rubric = {
            "pass_threshold": 60,
            "critical_failures": [],
        }
        grading = grade_result(result, scenario, rubric)
        assert grading.total_score > 0
        assert grading.scenario_id == "test"
        assert len(grading.dimension_scores) >= 5

    def test_likert_levels_populated(self):
        result = _make_result()
        scenario = _make_scenario()
        rubric = {
            "pass_threshold": 60,
            "critical_failures": [],
            "likert_scale": {"boundaries": [0, 25, 50, 70, 80]},
            "dimensions": [
                {"id": "direction_accuracy", "likert_anchors": {5: "Exact match"}},
                {"id": "conviction_calibration", "likert_anchors": {5: "In range"}},
                {"id": "risk_identification", "likert_anchors": {5: "All risks"}},
                {"id": "fact_coverage", "likert_anchors": {5: "All facts"}},
                {"id": "reasoning_quality", "likert_anchors": {5: "Full quality"}},
            ],
        }
        grading = grade_result(result, scenario, rubric)
        for ds in grading.dimension_scores:
            assert ds.likert_level is not None, f"{ds.dimension_id} missing Likert"
            assert 1 <= ds.likert_level.level <= 5

    def test_likert_excellent_on_perfect_direction(self):
        result = _make_result()
        scenario = _make_scenario()
        rubric = {
            "pass_threshold": 60,
            "critical_failures": [],
            "likert_scale": {"boundaries": [0, 25, 50, 70, 80]},
            "dimensions": [
                {"id": "direction_accuracy", "likert_anchors": {5: "Perfect"}},
            ],
        }
        grading = grade_result(result, scenario, rubric)
        direction_score = next(
            ds for ds in grading.dimension_scores
            if ds.dimension_id == "direction_accuracy"
        )
        assert direction_score.likert_level.level == 5
        assert direction_score.likert_level.label == "Excellent"

    def test_critical_failure_fails(self):
        result = _make_result(
            committee_memo=MockMemo(
                position_direction=-1,
                conviction=9.5,
                recommendation="ACTIVE SHORT",
            ),
        )
        scenario = _make_scenario()
        rubric = {"pass_threshold": 60, "critical_failures": []}
        grading = grade_result(result, scenario, rubric)
        assert not grading.passed
        assert any("Wrong direction" in f for f in grading.critical_failures)
