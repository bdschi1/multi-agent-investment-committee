"""
Multi-dimensional grading engine.

Evaluates a CommitteeResult against scenario ground truth
across direction, conviction, risk, fact, and reasoning dimensions.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from evals.adversarial import grade_adversarial_robustness
from evals.schemas import (
    LIKERT_LABELS,
    RECOMMENDATION_ORDER,
    DimensionScore,
    EvalScenario,
    GradingResult,
    GroundTruth,
    LikertLevel,
)

logger = logging.getLogger(__name__)

# Fuzzy match threshold — fraction of ground-truth keywords that must
# appear in the system output to count as a "hit".
_FUZZY_THRESHOLD = 0.40

# Stop words excluded from fuzzy matching
_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "of", "in", "to",
    "for", "with", "on", "at", "from", "by", "about", "as", "into",
    "through", "during", "before", "after", "and", "but", "or", "nor",
    "not", "so", "yet", "both", "either", "neither", "each", "every",
    "all", "any", "few", "more", "most", "other", "some", "such", "no",
    "than", "too", "very", "just", "also", "that", "this", "these",
    "those", "it", "its", "e.g.", "i.e.",
})


def _tokenize(text: str) -> set[str]:
    """Extract meaningful words from text for fuzzy matching."""
    words = re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())
    return {w for w in words if w not in _STOP_WORDS and len(w) > 2}


def _fuzzy_match(needle: str, haystack: str) -> bool:
    """Check if needle's keywords appear in haystack above threshold."""
    needle_tokens = _tokenize(needle)
    if not needle_tokens:
        return True  # empty needle matches anything
    haystack_tokens = _tokenize(haystack)
    overlap = needle_tokens & haystack_tokens
    return len(overlap) / len(needle_tokens) >= _FUZZY_THRESHOLD


def _get_recommendation_index(rec: str) -> int:
    """Return index in RECOMMENDATION_ORDER, or -1 if unknown."""
    rec_upper = rec.strip().upper()
    for i, r in enumerate(RECOMMENDATION_ORDER):
        if rec_upper == r:
            return i
    return -1


def _extract_text_fields(result: Any) -> str:
    """Flatten all text from a CommitteeResult for searching."""
    parts: list[str] = []

    def _collect(obj: Any) -> None:
        if isinstance(obj, str):
            parts.append(obj)
        elif hasattr(obj, "model_dump"):
            _collect(obj.model_dump())
        elif isinstance(obj, dict):
            for v in obj.values():
                _collect(v)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                _collect(item)

    for field_name in [
        "bull_case", "bear_case", "macro_view",
        "analyst_rebuttal", "risk_rebuttal", "committee_memo",
    ]:
        obj = getattr(result, field_name, None)
        if obj is not None:
            _collect(obj)

    # Also check traces
    traces = getattr(result, "traces", {})
    if isinstance(traces, dict):
        for trace in traces.values():
            if hasattr(trace, "steps"):
                for step in trace.steps:
                    parts.append(getattr(step, "content", ""))

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Likert mapping
# ---------------------------------------------------------------------------

# Default score boundaries: [Fail, Poor, Adequate, Good, Excellent]
# A score >= boundary[i] maps to level i+1.
_DEFAULT_LIKERT_BOUNDARIES = [0, 25, 50, 70, 80]


def _score_to_likert_level(raw_score: float, boundaries: list[float] | None = None) -> int:
    """Map a 0-100 score to a 1-5 Likert level using boundary thresholds.

    Boundaries define the lower bound (inclusive) for each level:
      boundaries[0] → level 1 (Fail)
      boundaries[1] → level 2 (Poor)
      ...
      boundaries[4] → level 5 (Excellent)
    """
    bounds = boundaries or _DEFAULT_LIKERT_BOUNDARIES
    level = 1
    for i, threshold in enumerate(bounds):
        if raw_score >= threshold:
            level = i + 1
    return level


def _assign_likert_level(
    score: DimensionScore,
    rubric: dict[str, Any],
) -> None:
    """Annotate a DimensionScore with its Likert level and anchor text.

    Reads ``likert_scale.boundaries`` from the rubric for threshold mapping,
    and ``likert_anchors`` from the matching dimension definition for
    the anchor description.

    Mutates ``score.likert_level`` in place.
    """
    # Get boundaries from rubric (or use defaults)
    likert_config = rubric.get("likert_scale", {})
    boundaries = likert_config.get("boundaries", _DEFAULT_LIKERT_BOUNDARIES)

    level = _score_to_likert_level(score.raw_score, boundaries)
    label = LIKERT_LABELS.get(level, f"Level {level}")

    # Find dimension-specific anchor text
    anchor = ""
    for dim in rubric.get("dimensions", []):
        if dim.get("id") == score.dimension_id:
            anchors = dim.get("likert_anchors", {})
            anchor = anchors.get(level, anchors.get(str(level), ""))
            break

    score.likert_level = LikertLevel(
        level=level,
        label=label,
        anchor=anchor,
    )


# ---------------------------------------------------------------------------
# Dimension grading functions
# ---------------------------------------------------------------------------

def _grade_direction(result: Any, truth: GroundTruth) -> DimensionScore:
    """Grade directional accuracy."""
    memo = getattr(result, "committee_memo", None)
    if memo is None:
        return DimensionScore(
            dimension_id="direction_accuracy",
            dimension_name="Direction Accuracy",
            weight=25.0,
            raw_score=0.0,
            weighted_score=0.0,
            explanation="No committee memo produced",
        )

    actual_dir = getattr(memo, "position_direction", 0)
    actual_rec = getattr(memo, "recommendation", "")
    expected_dir = truth.expected_direction
    expected_rec = truth.expected_recommendation_bucket

    direction_correct = actual_dir == expected_dir

    # Bucket distance
    actual_idx = _get_recommendation_index(actual_rec)
    expected_idx = _get_recommendation_index(expected_rec)
    bucket_distance = (
        abs(actual_idx - expected_idx)
        if actual_idx >= 0 and expected_idx >= 0
        else 3  # unknown bucket
    )

    conviction = getattr(memo, "conviction", 5.0)

    if direction_correct and bucket_distance == 0:
        raw = 100.0
        label = "Exact match"
    elif direction_correct and bucket_distance == 1:
        raw = 80.0
        label = f"Direction correct, bucket off by 1 ({actual_rec} vs {expected_rec})"
    elif direction_correct and bucket_distance >= 2:
        raw = 50.0
        label = f"Direction correct, bucket off by {bucket_distance}"
    elif not direction_correct and conviction < 6.0:
        raw = 20.0
        label = f"Wrong direction but low conviction ({conviction:.1f})"
    else:
        raw = 0.0
        label = f"Wrong direction with conviction {conviction:.1f}"

    return DimensionScore(
        dimension_id="direction_accuracy",
        dimension_name="Direction Accuracy",
        weight=25.0,
        raw_score=raw,
        weighted_score=raw * 25.0 / 100.0,
        explanation=f"{label}. Got {actual_rec} (dir={actual_dir}), "
        f"expected {expected_rec} (dir={expected_dir})",
    )


def _grade_conviction_calibration(
    result: Any, truth: GroundTruth,
) -> DimensionScore:
    """Grade conviction calibration against expected range."""
    memo = getattr(result, "committee_memo", None)
    if memo is None:
        return DimensionScore(
            dimension_id="conviction_calibration",
            dimension_name="Conviction Calibration",
            weight=15.0,
            raw_score=0.0,
            weighted_score=0.0,
            explanation="No committee memo produced",
        )

    conviction = getattr(memo, "conviction", 5.0)
    actual_dir = getattr(memo, "position_direction", 0)
    lo, hi = truth.conviction_range

    direction_correct = actual_dir == truth.expected_direction

    if lo <= conviction <= hi and direction_correct:
        raw = 100.0
        label = f"Conviction {conviction:.1f} within range [{lo}, {hi}]"
    elif direction_correct and (lo - 1.0 <= conviction <= hi + 1.0):
        raw = 70.0
        label = f"Conviction {conviction:.1f} near range [{lo}, {hi}]"
    elif direction_correct:
        raw = 40.0
        label = f"Conviction {conviction:.1f} outside range [{lo}, {hi}]"
    elif not direction_correct and conviction >= 8.0:
        raw = 0.0
        label = f"HIGH conviction ({conviction:.1f}) on WRONG direction"
    else:
        raw = 20.0
        label = f"Wrong direction, conviction {conviction:.1f}"

    return DimensionScore(
        dimension_id="conviction_calibration",
        dimension_name="Conviction Calibration",
        weight=15.0,
        raw_score=raw,
        weighted_score=raw * 15.0 / 100.0,
        explanation=label,
    )


def _grade_risk_identification(
    result: Any, truth: GroundTruth,
) -> DimensionScore:
    """Grade risk coverage using fuzzy matching."""
    if not truth.must_find_risks:
        return DimensionScore(
            dimension_id="risk_identification",
            dimension_name="Risk Identification",
            weight=20.0,
            raw_score=100.0,
            weighted_score=20.0,
            explanation="No must_find_risks specified",
        )

    all_text = _extract_text_fields(result)
    found = sum(1 for risk in truth.must_find_risks if _fuzzy_match(risk, all_text))
    coverage = found / len(truth.must_find_risks)

    if coverage >= 1.0:
        raw = 100.0
    elif coverage >= 0.75:
        raw = 80.0
    elif coverage >= 0.50:
        raw = 55.0
    elif coverage >= 0.25:
        raw = 30.0
    else:
        raw = 10.0

    return DimensionScore(
        dimension_id="risk_identification",
        dimension_name="Risk Identification",
        weight=20.0,
        raw_score=raw,
        weighted_score=raw * 20.0 / 100.0,
        explanation=f"Found {found}/{len(truth.must_find_risks)} must_find_risks "
        f"({coverage:.0%} coverage)",
    )


def _grade_fact_coverage(
    result: Any, truth: GroundTruth,
) -> DimensionScore:
    """Grade fact coverage using fuzzy matching."""
    if not truth.must_find_facts:
        return DimensionScore(
            dimension_id="fact_coverage",
            dimension_name="Fact Coverage",
            weight=15.0,
            raw_score=100.0,
            weighted_score=15.0,
            explanation="No must_find_facts specified",
        )

    all_text = _extract_text_fields(result)
    found = sum(1 for fact in truth.must_find_facts if _fuzzy_match(fact, all_text))
    coverage = found / len(truth.must_find_facts)

    if coverage >= 1.0:
        raw = 100.0
    elif coverage >= 0.75:
        raw = 80.0
    elif coverage >= 0.50:
        raw = 55.0
    elif coverage >= 0.25:
        raw = 30.0
    else:
        raw = 10.0

    return DimensionScore(
        dimension_id="fact_coverage",
        dimension_name="Fact Coverage",
        weight=15.0,
        raw_score=raw,
        weighted_score=raw * 15.0 / 100.0,
        explanation=f"Found {found}/{len(truth.must_find_facts)} must_find_facts "
        f"({coverage:.0%} coverage)",
    )


def _grade_reasoning_quality(result: Any) -> DimensionScore:
    """Grade reasoning chain quality with heuristics."""
    score_parts: list[tuple[float, str]] = []

    # 1. Conviction evolution — did scores change across timeline?
    timeline = getattr(result, "conviction_timeline", [])
    if len(timeline) >= 3:
        scores = [s.score for s in timeline if hasattr(s, "score")]
        if len(set(scores)) > 1:
            score_parts.append((25.0, "Conviction evolved across phases"))
        else:
            score_parts.append((5.0, "Static conviction — no evolution"))
    elif timeline:
        score_parts.append((10.0, f"Short timeline ({len(timeline)} entries)"))
    else:
        score_parts.append((0.0, "No conviction timeline"))

    # 2. Rebuttal substance — non-empty?
    analyst_reb = getattr(result, "analyst_rebuttal", None)
    risk_reb = getattr(result, "risk_rebuttal", None)
    rebuttal_count = 0
    if analyst_reb and getattr(analyst_reb, "argument", ""):
        rebuttal_count += 1
    if risk_reb and getattr(risk_reb, "argument", ""):
        rebuttal_count += 1
    if rebuttal_count == 2:
        score_parts.append((25.0, "Both rebuttals substantive"))
    elif rebuttal_count == 1:
        score_parts.append((15.0, "One rebuttal substantive"))
    else:
        score_parts.append((5.0, "No substantive rebuttals"))

    # 3. PM synthesis breadth — cited both bull and bear?
    memo = getattr(result, "committee_memo", None)
    if memo:
        bull_accepted = getattr(memo, "bull_points_accepted", [])
        bear_accepted = getattr(memo, "bear_points_accepted", [])
        if bull_accepted and bear_accepted:
            score_parts.append((25.0, "PM cited both bull and bear points"))
        elif bull_accepted or bear_accepted:
            score_parts.append((15.0, "PM cited only one side"))
        else:
            score_parts.append((5.0, "PM did not cite analyst points"))
    else:
        score_parts.append((0.0, "No committee memo"))

    # 4. Traces present and non-trivial
    traces = getattr(result, "traces", {})
    if isinstance(traces, dict) and len(traces) >= 3:
        total_steps = sum(
            len(t.steps) for t in traces.values() if hasattr(t, "steps")
        )
        if total_steps >= 6:
            desc = f"Rich traces ({total_steps} steps across {len(traces)} agents)"
            score_parts.append((25.0, desc))
        else:
            score_parts.append((10.0, f"Sparse traces ({total_steps} steps)"))
    else:
        score_parts.append((5.0, "Minimal trace data"))

    raw = sum(s for s, _ in score_parts)
    explanation = "; ".join(e for _, e in score_parts)

    return DimensionScore(
        dimension_id="reasoning_quality",
        dimension_name="Reasoning Quality",
        weight=15.0,
        raw_score=raw,
        weighted_score=raw * 15.0 / 100.0,
        explanation=explanation,
    )


# ---------------------------------------------------------------------------
# Critical failure checks
# ---------------------------------------------------------------------------

def _check_must_not_claim(result: Any, claims: list[str]) -> list[str]:
    """Check if the result contains any must_not_claim violations."""
    if not claims:
        return []
    all_text = _extract_text_fields(result).lower()
    return [claim for claim in claims if claim.lower() in all_text]


def _check_critical_failures(
    result: Any,
    scenario: EvalScenario,
    rubric: dict[str, Any],
) -> list[str]:
    """Check for auto-fail conditions."""
    failures: list[str] = []

    # must_not_claim violations
    violations = _check_must_not_claim(result, scenario.ground_truth.must_not_claim)
    for v in violations:
        failures.append(f"Hallucination: claimed '{v}'")

    # Wrong direction with high conviction
    memo = getattr(result, "committee_memo", None)
    if memo:
        actual_dir = getattr(memo, "position_direction", 0)
        conviction = getattr(memo, "conviction", 0.0)
        if actual_dir != scenario.ground_truth.expected_direction and conviction >= 8.0:
            failures.append(
                f"Wrong direction (got {actual_dir}, expected "
                f"{scenario.ground_truth.expected_direction}) with conviction {conviction:.1f}"
            )

    # Scenario-specific critical failures
    all_text = _extract_text_fields(result).lower()
    for cf in scenario.evaluation_criteria.critical_failures:
        if cf.lower() in all_text:
            failures.append(f"Scenario critical failure: {cf}")

    return failures


# ---------------------------------------------------------------------------
# Main grading function
# ---------------------------------------------------------------------------

def _summarize_result(result: Any) -> dict[str, Any]:
    """Extract key fields from CommitteeResult for the grading record."""
    memo = getattr(result, "committee_memo", None)
    summary: dict[str, Any] = {
        "ticker": getattr(result, "ticker", ""),
        "total_duration_ms": getattr(result, "total_duration_ms", 0),
        "total_tokens": getattr(result, "total_tokens", 0),
        "parsing_failures": getattr(result, "parsing_failures", []),
    }
    if memo:
        summary["recommendation"] = getattr(memo, "recommendation", "")
        summary["position_direction"] = getattr(memo, "position_direction", 0)
        summary["conviction"] = getattr(memo, "conviction", 0.0)
        summary["raw_confidence"] = getattr(memo, "raw_confidence", 0.0)
        summary["t_signal"] = getattr(memo, "t_signal", 0.0)
        summary["thesis_summary"] = getattr(memo, "thesis_summary", "")
    return summary


def grade_result(
    result: Any,
    scenario: EvalScenario,
    rubric: dict[str, Any],
) -> GradingResult:
    """Grade a CommitteeResult against scenario ground truth.

    Args:
        result: The CommitteeResult from the committee run.
        scenario: The eval scenario with ground truth.
        rubric: The parsed rubric dict.

    Returns:
        GradingResult with dimension scores, total, and pass/fail.
    """
    truth = scenario.ground_truth
    is_adversarial = scenario.type == "adversarial"

    # Grade each dimension
    scores: list[DimensionScore] = [
        _grade_direction(result, truth),
        _grade_conviction_calibration(result, truth),
        _grade_risk_identification(result, truth),
        _grade_fact_coverage(result, truth),
        _grade_reasoning_quality(result),
    ]

    # Adversarial dimension
    if is_adversarial and scenario.adversarial:
        adv_score = grade_adversarial_robustness(result, scenario.adversarial)
        scores.append(adv_score)
    else:
        # Redistribute adversarial weight proportionally
        non_adv_weight = sum(s.weight for s in scores)
        if non_adv_weight > 0:
            scale = 100.0 / non_adv_weight
            for s in scores:
                s.weight *= scale
                s.weighted_score = s.raw_score * s.weight / 100.0

    # Apply dimension overrides
    for s in scores:
        if s.dimension_id in scenario.evaluation_criteria.dimension_overrides:
            new_weight = scenario.evaluation_criteria.dimension_overrides[s.dimension_id]
            s.weight = new_weight
            s.weighted_score = s.raw_score * new_weight / 100.0

    # Annotate each dimension with its Likert level
    for s in scores:
        _assign_likert_level(s, rubric)

    total_score = sum(s.weighted_score for s in scores)

    # Critical failures
    critical_failures = _check_critical_failures(result, scenario, rubric)

    pass_threshold = rubric.get("pass_threshold", 60)
    passed = total_score >= pass_threshold and not critical_failures

    return GradingResult(
        scenario_id=scenario.id,
        ticker=scenario.ticker,
        scenario_type=scenario.type,
        dimension_scores=scores,
        total_score=round(total_score, 1),
        passed=passed,
        critical_failures=critical_failures,
        adversarial_result=(
            {"manipulation_type": scenario.adversarial.manipulation_type}
            if scenario.adversarial else None
        ),
        committee_result_summary=_summarize_result(result),
        run_metadata={
            "scenario_title": scenario.title,
            "difficulty": scenario.difficulty,
            "as_of_date": scenario.as_of_date,
        },
    )
