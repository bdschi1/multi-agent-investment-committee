"""Tests for the LLM-as-judge scoring module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

from evals.llm_judge import _build_prompt, _parse_scores, llm_judge_score
from evals.schemas import EvalScenario, GroundTruth


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@dataclass
class MockMemo:
    recommendation: str = "BUY"
    position_direction: int = 1
    conviction: float = 7.0
    thesis_summary: str = "Strong momentum"
    bull_points_accepted: list[str] = field(default_factory=lambda: ["Revenue growth"])
    bear_points_accepted: list[str] = field(default_factory=lambda: ["Valuation risk"])

    def model_dump(self) -> dict:
        return self.__dict__


@dataclass
class MockResult:
    ticker: str = "NVDA"
    committee_memo: Any = None

    def to_dict(self) -> dict:
        result = {}
        for k, v in self.__dict__.items():
            if hasattr(v, "model_dump"):
                result[k] = v.model_dump()
            else:
                result[k] = v
        return result


def _make_scenario() -> EvalScenario:
    return EvalScenario(
        id="test",
        title="Test Scenario",
        ticker="NVDA",
        ground_truth=GroundTruth(
            expected_direction=1,
            expected_recommendation_bucket="BUY",
            must_find_facts=["AI demand"],
            must_find_risks=["competition"],
        ).model_dump(),
    )


def _make_rubric() -> dict:
    return {
        "dimensions": [
            {"id": "direction_accuracy", "name": "Direction Accuracy", "weight": 25,
             "description": "Directional call", "likert_anchors": {5: "Exact", 1: "Wrong"}},
            {"id": "fact_recall", "name": "Fact Recall", "weight": 8,
             "description": "Key facts found", "likert_anchors": {5: "All", 1: "None"}},
        ],
    }


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def test_includes_dimensions(self):
        result = MockResult(committee_memo=MockMemo())
        scenario = _make_scenario()
        rubric = _make_rubric()
        prompt = _build_prompt(result, scenario, rubric)
        assert "direction_accuracy" in prompt
        assert "fact_recall" in prompt

    def test_includes_ground_truth(self):
        result = MockResult(committee_memo=MockMemo())
        scenario = _make_scenario()
        rubric = _make_rubric()
        prompt = _build_prompt(result, scenario, rubric)
        assert "AI demand" in prompt
        assert "competition" in prompt

    def test_includes_result_data(self):
        result = MockResult(committee_memo=MockMemo())
        scenario = _make_scenario()
        rubric = _make_rubric()
        prompt = _build_prompt(result, scenario, rubric)
        assert "NVDA" in prompt
        assert "BUY" in prompt


# ---------------------------------------------------------------------------
# Score parsing
# ---------------------------------------------------------------------------

class TestParseScores:
    def test_valid_json(self):
        response = '{"direction_accuracy": 0.85, "conviction_calibration": 0.7, "risk_identification": 0.9, "fact_recall": 0.6, "fact_precision": 0.95, "reasoning_quality": 0.75}'
        scores = _parse_scores(response)
        assert scores["direction_accuracy"] == 0.85
        assert scores["fact_recall"] == 0.6
        assert len(scores) == 6

    def test_markdown_wrapped_json(self):
        response = '```json\n{"direction_accuracy": 0.9, "fact_recall": 0.8}\n```'
        scores = _parse_scores(response)
        assert scores["direction_accuracy"] == 0.9
        assert scores["fact_recall"] == 0.8

    def test_clamps_values(self):
        response = '{"direction_accuracy": 1.5, "fact_recall": -0.3}'
        scores = _parse_scores(response)
        assert scores["direction_accuracy"] == 1.0
        assert scores["fact_recall"] == 0.0

    def test_malformed_json(self):
        response = "This is not JSON at all"
        scores = _parse_scores(response)
        assert scores == {}

    def test_partial_dimensions(self):
        response = '{"direction_accuracy": 0.8}'
        scores = _parse_scores(response)
        assert len(scores) == 1
        assert scores["direction_accuracy"] == 0.8

    def test_ignores_unknown_dimensions(self):
        response = '{"direction_accuracy": 0.8, "unknown_dim": 0.5}'
        scores = _parse_scores(response)
        assert "unknown_dim" not in scores
        assert scores["direction_accuracy"] == 0.8


# ---------------------------------------------------------------------------
# Full scoring flow (mocked)
# ---------------------------------------------------------------------------

class TestLLMJudgeScore:
    def test_with_mock_model(self):
        mock_response = '{"direction_accuracy": 0.9, "conviction_calibration": 0.7, "risk_identification": 0.85, "fact_recall": 0.8, "fact_precision": 0.95, "reasoning_quality": 0.75}'

        def mock_model(prompt: str) -> str:
            return mock_response

        result = MockResult(committee_memo=MockMemo())
        scenario = _make_scenario()
        rubric = _make_rubric()

        scores = llm_judge_score(result, scenario, rubric, model=mock_model)
        assert len(scores) == 6
        assert scores["direction_accuracy"] == 0.9

    def test_handles_model_failure(self):
        def failing_model(prompt: str) -> str:
            raise RuntimeError("API error")

        result = MockResult(committee_memo=MockMemo())
        scenario = _make_scenario()
        rubric = _make_rubric()

        scores = llm_judge_score(result, scenario, rubric, model=failing_model)
        assert scores == {}

    def test_handles_malformed_response(self):
        def bad_model(prompt: str) -> str:
            return "I cannot evaluate this."

        result = MockResult(committee_memo=MockMemo())
        scenario = _make_scenario()
        rubric = _make_rubric()

        scores = llm_judge_score(result, scenario, rubric, model=bad_model)
        assert scores == {}
