"""
LLM-as-judge scoring — parallel pathway for automated eval.

Scores each rubric dimension on a continuous 0.0–1.0 scale using an LLM,
providing automated evaluation at scale alongside the deterministic grader.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from evals.schemas import EvalScenario

logger = logging.getLogger(__name__)

# Dimensions the judge scores (matches grader dimension IDs)
_JUDGE_DIMENSIONS = [
    "direction_accuracy",
    "conviction_calibration",
    "risk_identification",
    "fact_recall",
    "fact_precision",
    "reasoning_quality",
]

_SYSTEM_PROMPT = """You are an expert investment analyst evaluating the output of an AI investment committee system.

You will be given:
1. A rubric with dimension definitions and Likert anchor descriptions
2. A scenario with ground truth (expert-provided expected outcomes)
3. The AI system's committee output

Score each dimension on a continuous 0.0 to 1.0 scale where:
- 0.0 = Fail (Likert 1)
- 0.25 = Poor (Likert 2)
- 0.50 = Adequate (Likert 3)
- 0.70 = Good (Likert 4)
- 0.80+ = Excellent (Likert 5)

Respond with ONLY a JSON object mapping dimension IDs to float scores.
Example: {"direction_accuracy": 0.85, "conviction_calibration": 0.70, ...}

Do not include explanations — only the JSON object."""


def _build_prompt(
    result: Any,
    scenario: EvalScenario,
    rubric: dict[str, Any],
) -> str:
    """Build the evaluation prompt for the LLM judge."""
    # Extract rubric dimensions
    dimensions_text = ""
    for dim in rubric.get("dimensions", []):
        dim_id = dim.get("id", "")
        if dim_id not in _JUDGE_DIMENSIONS:
            continue
        dimensions_text += f"\n### {dim.get('name', dim_id)} ({dim_id})\n"
        dimensions_text += f"Weight: {dim.get('weight', 0)}\n"
        dimensions_text += f"Description: {dim.get('description', '').strip()}\n"
        anchors = dim.get("likert_anchors", {})
        for level in sorted(anchors.keys(), key=lambda x: int(x), reverse=True):
            dimensions_text += f"  {level}: {anchors[level]}\n"

    # Extract ground truth
    truth = scenario.ground_truth
    truth_text = (
        f"Expected direction: {truth.expected_direction}\n"
        f"Expected recommendation: {truth.expected_recommendation_bucket}\n"
        f"Conviction range: {truth.conviction_range}\n"
        f"Must-find facts: {truth.must_find_facts}\n"
        f"Must-find risks: {truth.must_find_risks}\n"
        f"Must-not-claim: {truth.must_not_claim}\n"
    )

    # Serialize the committee result
    if hasattr(result, "model_dump"):
        result_text = json.dumps(result.model_dump(), indent=2, default=str)
    elif hasattr(result, "to_dict"):
        result_text = json.dumps(result.to_dict(), indent=2, default=str)
    elif hasattr(result, "__dict__"):
        result_text = json.dumps(
            {k: str(v) for k, v in result.__dict__.items()},
            indent=2,
        )
    else:
        result_text = str(result)

    # Truncate if too long
    if len(result_text) > 15000:
        result_text = result_text[:15000] + "\n... [truncated]"

    return f"""{_SYSTEM_PROMPT}

## Rubric Dimensions
{dimensions_text}

## Ground Truth
{truth_text}

## Committee Output
{result_text}

Score each of these dimensions: {', '.join(_JUDGE_DIMENSIONS)}
Respond with JSON only."""


def _parse_scores(response: str) -> dict[str, float]:
    """Parse LLM response into dimension scores, with validation."""
    # Try to extract JSON from the response
    # Handle cases where LLM wraps JSON in markdown code blocks
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
    if json_match:
        raw = json_match.group(1)
    else:
        # Try to find a bare JSON object
        json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
        if json_match:
            raw = json_match.group(0)
        else:
            logger.warning("LLM judge returned no parseable JSON: %s", response[:200])
            return {}

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("LLM judge JSON parse error: %s", exc)
        return {}

    # Validate and clamp scores to [0, 1]
    scores: dict[str, float] = {}
    for dim_id in _JUDGE_DIMENSIONS:
        if dim_id in data:
            try:
                val = float(data[dim_id])
                scores[dim_id] = max(0.0, min(1.0, val))
            except (TypeError, ValueError):
                logger.warning("Invalid score for %s: %s", dim_id, data[dim_id])

    return scores


def llm_judge_score(
    result: Any,
    scenario: EvalScenario,
    rubric: dict[str, Any],
    model: Any | None = None,
    provider: str | None = None,
    model_name: str | None = None,
) -> dict[str, float]:
    """Score each dimension using an LLM as judge.

    Args:
        result: The CommitteeResult from the committee run.
        scenario: The eval scenario with ground truth.
        rubric: The parsed rubric dict.
        model: LLM callable (str -> str). If None, creates one.
        provider: Provider name (used only if model is None).
        model_name: Model name override (used only if model is None).

    Returns:
        Dict mapping dimension_id → float score (0.0-1.0).
        Empty dict if scoring fails.
    """
    if model is None:
        model = _create_judge_model(provider, model_name)

    prompt = _build_prompt(result, scenario, rubric)

    try:
        response = model(prompt)
        scores = _parse_scores(response)
        if not scores:
            logger.warning("LLM judge returned no valid scores for %s", scenario.id)
        else:
            logger.info(
                "LLM judge scored %s: %s",
                scenario.id,
                {k: f"{v:.2f}" for k, v in scores.items()},
            )
        return scores
    except Exception as exc:
        logger.error("LLM judge failed for %s: %s", scenario.id, exc)
        return {}


def _create_judge_model(
    provider: str | None = None,
    model_name: str | None = None,
) -> Any:
    """Create an LLM callable for the judge, using the repo's provider system."""
    from config.settings import LLMProvider, settings

    if provider:
        prov = LLMProvider(provider)
    else:
        prov = settings.resolve_provider()

    # Import runner's _create_model pattern — avoids pulling in Gradio/app.py
    if prov == LLMProvider.ANTHROPIC:
        from anthropic import Anthropic

        client = Anthropic(api_key=settings.anthropic_api_key)
        model = model_name or settings.anthropic_model

        def call(prompt: str) -> str:
            response = client.messages.create(
                model=model,
                max_tokens=2048,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

        return call

    elif prov == LLMProvider.OPENAI:
        from openai import OpenAI

        client = OpenAI(api_key=settings.openai_api_key)
        model = model_name or settings.openai_model

        def call(prompt: str) -> str:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2048,
            )
            return response.choices[0].message.content

        return call

    elif prov == LLMProvider.GOOGLE:
        from google import genai

        client = genai.Client(api_key=settings.google_api_key)
        model = model_name or settings.google_model

        def call(prompt: str) -> str:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config={"temperature": 0.3, "max_output_tokens": 2048},
            )
            return response.text

        return call

    elif prov == LLMProvider.OLLAMA:
        import ollama as ollama_lib

        model = model_name or settings.ollama_model
        client = ollama_lib.Client(host=settings.ollama_base_url)

        def call(prompt: str) -> str:
            response = client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.3, "num_predict": 2048},
                format="json",
            )
            return response["message"]["content"]

        return call

    else:
        msg = f"Unsupported provider for LLM judge: {prov}"
        raise ValueError(msg)
