"""Tests for scenario and rubric loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from evals.loader import discover_scenarios, load_rubric, load_scenario

_SCENARIOS_DIR = Path(__file__).parent.parent / "evals" / "scenarios"
_RUBRICS_DIR = Path(__file__).parent.parent / "evals" / "rubrics"


class TestLoadScenario:
    def test_load_svb(self):
        path = _SCENARIOS_DIR / "svb_2023_collapse.yaml"
        if not path.exists():
            pytest.skip("Scenario file not found")
        s = load_scenario(path)
        assert s.id == "svb_2023_collapse"
        assert s.ticker == "SIVB"
        assert s.type == "ground_truth"

    def test_load_adversarial(self):
        path = _SCENARIOS_DIR / "adversarial_inflated_earnings.yaml"
        if not path.exists():
            pytest.skip("Scenario file not found")
        s = load_scenario(path)
        assert s.type == "adversarial"
        assert s.adversarial is not None
        assert s.adversarial.manipulation_type == "inflated_fundamentals"


class TestDiscoverScenarios:
    def test_discovers_scenarios(self):
        scenarios = discover_scenarios(_SCENARIOS_DIR)
        assert len(scenarios) >= 3  # at least nvda, svb, meta, adversarial
        ids = {s.id for s in scenarios}
        assert "svb_2023_collapse" in ids

    def test_skips_templates(self):
        scenarios = discover_scenarios(_SCENARIOS_DIR)
        ids = {s.id for s in scenarios}
        # Templates start with _ and should be skipped
        for s_id in ids:
            assert not s_id.startswith("_")

    def test_filter_by_type(self):
        adversarial = discover_scenarios(_SCENARIOS_DIR, filter_type="adversarial")
        for s in adversarial:
            assert s.type == "adversarial"

    def test_filter_by_type_ground_truth(self):
        gt = discover_scenarios(_SCENARIOS_DIR, filter_type="ground_truth")
        for s in gt:
            assert s.type == "ground_truth"

    def test_filter_by_tag(self):
        ai_scenarios = discover_scenarios(_SCENARIOS_DIR, filter_tags=["AI"])
        for s in ai_scenarios:
            assert "AI" in s.tags

    def test_filter_by_scenario_id(self):
        results = discover_scenarios(_SCENARIOS_DIR, filter_scenario="svb")
        assert len(results) >= 1
        assert all("svb" in s.id for s in results)


class TestLoadRubric:
    def test_load_standard_rubric(self):
        rubric = load_rubric("committee_standard", _RUBRICS_DIR)
        assert rubric["id"] == "committee_standard"
        assert rubric["total_points"] == 100
        assert rubric["pass_threshold"] == 60
        assert len(rubric["dimensions"]) == 8

    def test_rubric_has_likert_scale(self):
        rubric = load_rubric("committee_standard", _RUBRICS_DIR)
        assert "likert_scale" in rubric
        assert rubric["likert_scale"]["levels"] == 5
        assert len(rubric["likert_scale"]["boundaries"]) == 5
        assert rubric["likert_scale"]["labels"] == [
            "Fail", "Poor", "Adequate", "Good", "Excellent",
        ]

    def test_rubric_dimensions_have_likert_anchors(self):
        rubric = load_rubric("committee_standard", _RUBRICS_DIR)
        for dim in rubric["dimensions"]:
            assert "likert_anchors" in dim, f"{dim['id']} missing likert_anchors"
            anchors = dim["likert_anchors"]
            # All 5 levels should have anchors
            for level in range(1, 6):
                assert level in anchors, (
                    f"{dim['id']} missing anchor for level {level}"
                )

    def test_missing_rubric_raises(self):
        with pytest.raises(FileNotFoundError):
            load_rubric("nonexistent_rubric", _RUBRICS_DIR)
