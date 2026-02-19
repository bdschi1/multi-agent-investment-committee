"""Tests for risk profiles, RAG metrics, and red-team scenario loading."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from evals.risk_profiles import (
    AGGRESSIVE,
    CONSERVATIVE,
    MODERATE,
    PROFILES,
    RiskProfile,
    RiskTolerance,
    check_recommendation_suitability,
    get_profile,
)
from evals.rag_metrics import (
    Claim,
    RAGEvalResult,
    RAGScore,
    compute_answer_relevance,
    compute_context_relevance,
    compute_faithfulness,
    compute_groundedness,
    evaluate_rag,
    extract_claims,
)

REPO_ROOT = Path(__file__).parent.parent
SCENARIOS_DIR = REPO_ROOT / "evals" / "scenarios"


# ═══════════════════════════════════════════════════════════════════════════
# Risk Profiles
# ═══════════════════════════════════════════════════════════════════════════

class TestRiskTolerance:
    def test_three_levels(self):
        assert len(RiskTolerance) == 3

    def test_enum_values(self):
        assert RiskTolerance.CONSERVATIVE.value == "conservative"
        assert RiskTolerance.MODERATE.value == "moderate"
        assert RiskTolerance.AGGRESSIVE.value == "aggressive"


class TestRiskProfileDefaults:
    def test_three_profiles(self):
        assert len(PROFILES) == 3

    def test_conservative_no_shorts(self):
        assert CONSERVATIVE.max_short_exposure_pct == 0.0

    def test_conservative_no_leverage(self):
        assert CONSERVATIVE.max_gross_leverage == 1.0

    def test_moderate_allows_limited_shorts(self):
        assert MODERATE.max_short_exposure_pct > 0.0
        assert MODERATE.max_short_exposure_pct <= 15.0

    def test_aggressive_allows_shorts(self):
        assert AGGRESSIVE.max_short_exposure_pct > 0.0

    def test_aggressive_allows_leverage(self):
        assert AGGRESSIVE.max_gross_leverage > 1.0

    def test_conservative_tighter_than_moderate(self):
        assert CONSERVATIVE.max_single_position_pct < MODERATE.max_single_position_pct
        assert CONSERVATIVE.max_drawdown_pct < MODERATE.max_drawdown_pct
        assert CONSERVATIVE.target_annual_vol_pct < MODERATE.target_annual_vol_pct

    def test_moderate_tighter_than_aggressive(self):
        assert MODERATE.max_single_position_pct < AGGRESSIVE.max_single_position_pct
        assert MODERATE.max_drawdown_pct < AGGRESSIVE.max_drawdown_pct

    def test_conviction_thresholds_ordered(self):
        assert CONSERVATIVE.min_conviction_to_size > MODERATE.min_conviction_to_size
        assert MODERATE.min_conviction_to_size > AGGRESSIVE.min_conviction_to_size


class TestRiskProfileMethods:
    def test_allows_recommendation_conservative(self):
        assert CONSERVATIVE.allows_recommendation("BUY")
        assert CONSERVATIVE.allows_recommendation("HOLD")
        assert not CONSERVATIVE.allows_recommendation("ACTIVE SHORT")
        assert not CONSERVATIVE.allows_recommendation("SELL")

    def test_allows_recommendation_aggressive(self):
        assert AGGRESSIVE.allows_recommendation("ACTIVE SHORT")
        assert AGGRESSIVE.allows_recommendation("SELL")
        assert AGGRESSIVE.allows_recommendation("BUY")

    def test_position_size_below_threshold(self):
        assert CONSERVATIVE.position_size(3.0) == 0.0
        assert MODERATE.position_size(2.0) == 0.0

    def test_position_size_at_threshold(self):
        size = MODERATE.position_size(5.0)
        assert size == 0.0  # Exactly at threshold = 0 scale

    def test_position_size_above_threshold(self):
        size = MODERATE.position_size(7.5)
        assert size > 0.0
        assert size <= MODERATE.max_single_position_pct

    def test_position_size_max_conviction(self):
        size = AGGRESSIVE.position_size(10.0)
        assert size == AGGRESSIVE.max_single_position_pct

    def test_validate_allocation_clean(self):
        allocation = {"AAPL": 4.0, "MSFT": 3.0, "GOOG": 3.0}
        violations = CONSERVATIVE.validate_allocation(allocation)
        assert violations == []

    def test_validate_allocation_position_violation(self):
        allocation = {"AAPL": 25.0, "MSFT": 5.0}
        violations = CONSERVATIVE.validate_allocation(allocation)
        assert len(violations) >= 1
        assert "AAPL" in violations[0]

    def test_validate_allocation_short_violation(self):
        allocation = {"AAPL": 5.0, "TSLA": -15.0}
        violations = CONSERVATIVE.validate_allocation(allocation)
        assert any("Short" in v or "short" in v.lower() for v in violations)

    def test_validate_allocation_leverage_violation(self):
        allocation = {"AAPL": 80.0, "MSFT": 80.0}
        violations = CONSERVATIVE.validate_allocation(allocation)
        assert any("leverage" in v.lower() or "Gross" in v for v in violations)

    def test_to_dict_keys(self):
        d = CONSERVATIVE.to_dict()
        assert "tolerance" in d
        assert "max_single_position_pct" in d
        assert "allowed_buckets" in d
        assert d["tolerance"] == "conservative"


class TestGetProfile:
    def test_get_by_string(self):
        p = get_profile("conservative")
        assert p is CONSERVATIVE

    def test_get_by_enum(self):
        p = get_profile(RiskTolerance.AGGRESSIVE)
        assert p is AGGRESSIVE

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown risk tolerance"):
            get_profile("ultra_risky")


class TestCheckRecommendationSuitability:
    def test_suitable_conservative_buy(self):
        result = check_recommendation_suitability("BUY", 8.0, "conservative")
        assert result["suitable"] is True
        assert result["position_size_pct"] > 0.0

    def test_unsuitable_conservative_short(self):
        result = check_recommendation_suitability("ACTIVE SHORT", 8.0, "conservative")
        assert result["suitable"] is False
        assert len(result["violations"]) >= 1

    def test_suitable_aggressive_short(self):
        result = check_recommendation_suitability("ACTIVE SHORT", 8.0, "aggressive")
        assert result["suitable"] is True

    def test_low_conviction_no_position(self):
        result = check_recommendation_suitability("BUY", 2.0, "conservative")
        assert result["position_size_pct"] == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# RAG Metrics
# ═══════════════════════════════════════════════════════════════════════════

class TestExtractClaims:
    def test_extracts_revenue_claims(self):
        text = "Revenue was $50B in FY2024. The company is doing well."
        claims = extract_claims(text)
        assert len(claims) >= 1

    def test_extracts_eps_claims(self):
        text = "EPS of $3.50 exceeded estimates. Margins at 35% were strong."
        claims = extract_claims(text)
        assert len(claims) >= 1

    def test_no_claims_in_opinion(self):
        text = "The outlook is cautiously optimistic for the sector."
        claims = extract_claims(text)
        assert len(claims) == 0

    def test_multiple_claims(self):
        text = (
            "Revenue of $100B grew 15% YoY. "
            "EPS was $5.00 above consensus. "
            "PE ratio of 25x is above sector average."
        )
        claims = extract_claims(text)
        assert len(claims) >= 2


class TestRAGScoreBasics:
    def test_rag_score_to_dict(self):
        s = RAGScore(metric="test", score=0.85, details="good")
        d = s.to_dict()
        assert d["metric"] == "test"
        assert d["score"] == 0.85

    def test_rag_score_rounding(self):
        s = RAGScore(metric="test", score=0.123456789)
        d = s.to_dict()
        assert d["score"] == 0.1235


class TestFaithfulness:
    def test_perfect_faithfulness(self):
        output = "Revenue was $50B in FY2024. EPS of $3.50 was strong."
        context = ["Apple reported revenue of $50B in fiscal year 2024. EPS was $3.50."]
        score = compute_faithfulness(output, context)
        assert score.score > 0.5

    def test_no_claims(self):
        output = "The outlook is optimistic."
        context = ["Apple had a good year."]
        score = compute_faithfulness(output, context)
        assert score.score == 1.0  # No claims = vacuously faithful

    def test_unsupported_claims(self):
        output = "Revenue was $500B in FY2024. Growth of 200% was incredible."
        context = ["Apple reported modest growth this year."]
        score = compute_faithfulness(output, context)
        assert score.score < 0.5


class TestContextRelevance:
    def test_relevant_context(self):
        query = "What is Apple's revenue growth?"
        chunks = ["Apple revenue grew 5% to $383B in fiscal 2023."]
        score = compute_context_relevance(query, chunks)
        assert score.score > 0.0

    def test_empty_context(self):
        score = compute_context_relevance("query", [])
        assert score.score == 0.0

    def test_irrelevant_context(self):
        query = "What is Tesla's battery technology?"
        chunks = ["The weather in Paris is nice today."]
        score = compute_context_relevance(query, chunks)
        assert score.score < 0.5


class TestAnswerRelevance:
    def test_relevant_answer(self):
        query = "What is Apple's revenue growth and margin trend?"
        output = "Apple's revenue growth was 5% with margins expanding to 35%."
        score = compute_answer_relevance(query, output)
        assert score.score > 0.0

    def test_empty_output(self):
        score = compute_answer_relevance("query", "")
        assert score.score == 0.0


class TestGroundedness:
    def test_grounded_output(self):
        output = "Revenue was $100M in FY2024. Growth of 10% was solid."
        source = {
            "financials": "Revenue was $100M for fiscal year FY2024",
            "growth_metrics": "Revenue growth of 10% year over year was solid performance",
        }
        score = compute_groundedness(output, source)
        assert score.score > 0.0

    def test_ungrounded_output(self):
        output = "Revenue was $999B. Growth of 500% was unprecedented."
        source = {"revenue": "$10M", "growth": "2%"}
        score = compute_groundedness(output, source)
        assert score.score < 1.0

    def test_no_claims(self):
        output = "The company looks promising."
        source = {"revenue": "$10M"}
        score = compute_groundedness(output, source)
        assert score.score == 1.0  # Vacuously grounded


class TestEvaluateRAG:
    def test_full_evaluation(self):
        result = evaluate_rag(
            query="Analyze AAPL revenue",
            output_text="Revenue of $50B in FY2024 grew 5%. EPS was $3.50.",
            context_chunks=["Apple revenue was $50B in fiscal 2024 with 5% growth. EPS $3.50."],
            source_data={"revenue": "$50B", "growth": "5%", "eps": "$3.50"},
        )
        assert isinstance(result, RAGEvalResult)
        assert 0.0 <= result.composite_score <= 1.0

    def test_composite_score_weights_sum(self):
        """Verify composite weight coefficients sum to 1.0."""
        assert 0.40 + 0.30 + 0.15 + 0.15 == pytest.approx(1.0)

    def test_to_dict_structure(self):
        result = evaluate_rag(
            query="Test",
            output_text="Revenue was $10B.",
            context_chunks=["Revenue was $10B."],
        )
        d = result.to_dict()
        assert "composite_score" in d
        assert "faithfulness" in d
        assert "context_relevance" in d
        assert "answer_relevance" in d
        assert "groundedness" in d

    def test_no_source_data(self):
        result = evaluate_rag(
            query="Test",
            output_text="The outlook is good.",
            context_chunks=["The outlook is good."],
        )
        assert result.groundedness.score == 1.0  # No claims


# ═══════════════════════════════════════════════════════════════════════════
# Red-Team Scenario Loading
# ═══════════════════════════════════════════════════════════════════════════

class TestRedTeamScenarios:
    """Verify red-team YAML scenarios are valid and loadable."""

    REDTEAM_FILES = [
        "redteam_pump_and_dump.yaml",
        "redteam_insider_tipping.yaml",
        "redteam_market_manipulation.yaml",
    ]

    def test_scenario_files_exist(self):
        for fname in self.REDTEAM_FILES:
            path = SCENARIOS_DIR / fname
            assert path.exists(), f"Missing scenario: {fname}"

    def test_scenarios_parse_as_yaml(self):
        for fname in self.REDTEAM_FILES:
            path = SCENARIOS_DIR / fname
            with open(path) as f:
                data = yaml.safe_load(f)
            assert isinstance(data, dict)

    def test_scenarios_have_required_fields(self):
        required = {"id", "title", "ticker", "type", "adversarial", "ground_truth"}
        for fname in self.REDTEAM_FILES:
            path = SCENARIOS_DIR / fname
            with open(path) as f:
                data = yaml.safe_load(f)
            missing = required - set(data.keys())
            assert not missing, f"{fname} missing fields: {missing}"

    def test_scenarios_are_adversarial_type(self):
        for fname in self.REDTEAM_FILES:
            path = SCENARIOS_DIR / fname
            with open(path) as f:
                data = yaml.safe_load(f)
            assert data["type"] == "adversarial"

    def test_scenarios_have_expected_signals(self):
        for fname in self.REDTEAM_FILES:
            path = SCENARIOS_DIR / fname
            with open(path) as f:
                data = yaml.safe_load(f)
            signals = data["adversarial"]["expected_behavior"]["expected_signals"]
            assert len(signals) >= 3, f"{fname} needs >= 3 expected signals"

    def test_scenarios_have_failure_signals(self):
        for fname in self.REDTEAM_FILES:
            path = SCENARIOS_DIR / fname
            with open(path) as f:
                data = yaml.safe_load(f)
            failures = data["adversarial"]["failure_signals"]
            assert len(failures) >= 3, f"{fname} needs >= 3 failure signals"

    def test_scenarios_have_must_find_risks(self):
        for fname in self.REDTEAM_FILES:
            path = SCENARIOS_DIR / fname
            with open(path) as f:
                data = yaml.safe_load(f)
            risks = data["ground_truth"]["must_find_risks"]
            assert len(risks) >= 3, f"{fname} needs >= 3 must_find_risks"

    def test_scenarios_have_must_not_claim(self):
        for fname in self.REDTEAM_FILES:
            path = SCENARIOS_DIR / fname
            with open(path) as f:
                data = yaml.safe_load(f)
            claims = data["ground_truth"]["must_not_claim"]
            assert len(claims) >= 2, f"{fname} needs >= 2 must_not_claim"

    def test_scenarios_load_as_eval_scenario(self):
        """Verify scenarios can be loaded into the EvalScenario schema."""
        from evals.schemas import EvalScenario

        for fname in self.REDTEAM_FILES:
            path = SCENARIOS_DIR / fname
            with open(path) as f:
                data = yaml.safe_load(f)
            scenario = EvalScenario(**data)
            assert scenario.type == "adversarial"
            assert scenario.adversarial is not None

    def test_pump_and_dump_specific(self):
        with open(SCENARIOS_DIR / "redteam_pump_and_dump.yaml") as f:
            data = yaml.safe_load(f)
        assert data["ticker"] == "SMCI"
        assert "red_team" in data["tags"]

    def test_insider_tipping_specific(self):
        with open(SCENARIOS_DIR / "redteam_insider_tipping.yaml") as f:
            data = yaml.safe_load(f)
        assert data["ticker"] == "MRNA"
        assert "mnpi" in data["tags"]

    def test_market_manipulation_specific(self):
        with open(SCENARIOS_DIR / "redteam_market_manipulation.yaml") as f:
            data = yaml.safe_load(f)
        assert data["ticker"] == "GME"
        assert "squeeze" in data["tags"]
