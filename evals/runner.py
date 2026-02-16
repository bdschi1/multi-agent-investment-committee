"""
Eval runner — orchestrates scenario execution and grading.

Loads scenarios, runs the investment committee, grades results,
and saves output to the results directory.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from evals.adversarial import inject_adversarial_data
from evals.grader import grade_result
from evals.loader import discover_scenarios, load_rubric
from evals.reporter import save_result_json
from evals.schemas import EvalScenario, GradingResult

logger = logging.getLogger(__name__)

_EVALS_DIR = Path(__file__).parent
_RESULTS_DIR = _EVALS_DIR / "results"


def _create_model() -> Any:
    """Create an LLM callable using the repo's existing provider system.

    Imports lazily to avoid pulling in Gradio at import time.
    """
    from config.settings import settings

    provider = settings.resolve_provider()
    logger.info("Using LLM provider: %s", provider.value)

    # Import the factory map and create the model
    # We replicate the minimal factory logic to avoid importing app.py
    # (which triggers Gradio initialization)
    from config.settings import LLMProvider

    if provider == LLMProvider.OLLAMA:
        import ollama as ollama_lib

        model_name = settings.ollama_model
        client = ollama_lib.Client(host=settings.ollama_base_url)

        def call(prompt: str) -> str:
            response = client.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": settings.temperature,
                    "num_predict": settings.max_tokens_per_agent,
                },
            )
            return response["message"]["content"]

        return call

    elif provider == LLMProvider.ANTHROPIC:
        from anthropic import Anthropic

        client = Anthropic(api_key=settings.anthropic_api_key)
        model_name = settings.anthropic_model

        def call(prompt: str) -> str:
            response = client.messages.create(
                model=model_name,
                max_tokens=settings.max_tokens_per_agent,
                temperature=settings.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

        return call

    elif provider == LLMProvider.GOOGLE:
        from google import genai

        client = genai.Client(api_key=settings.google_api_key)
        model_name = settings.google_model

        def call(prompt: str) -> str:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={
                    "temperature": settings.temperature,
                    "max_output_tokens": settings.max_tokens_per_agent,
                },
            )
            return response.text

        return call

    elif provider == LLMProvider.OPENAI:
        from openai import OpenAI

        client = OpenAI(api_key=settings.openai_api_key)
        model_name = settings.openai_model

        def call(prompt: str) -> str:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=settings.temperature,
                max_tokens=settings.max_tokens_per_agent,
            )
            return response.choices[0].message.content

        return call

    elif provider == LLMProvider.DEEPSEEK:
        from openai import OpenAI

        client = OpenAI(
            api_key=settings.deepseek_api_key,
            base_url="https://api.deepseek.com",
        )
        model_name = settings.deepseek_model

        def call(prompt: str) -> str:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=settings.temperature,
                max_tokens=settings.max_tokens_per_agent,
            )
            return response.choices[0].message.content

        return call

    else:
        msg = f"Unsupported provider for eval runner: {provider}"
        raise ValueError(msg)


def run_scenario(
    scenario: EvalScenario,
    model: Any | None = None,
    results_dir: Path | None = None,
) -> GradingResult:
    """Run a single eval scenario end-to-end.

    1. Gather live market data
    2. Inject adversarial data (if applicable)
    3. Run the investment committee
    4. Grade the result
    5. Save JSON output

    Args:
        scenario: The eval scenario to run.
        model: LLM callable. If None, creates one from settings.
        results_dir: Where to save results. Defaults to evals/results/.

    Returns:
        GradingResult with scores and metadata.
    """
    results_dir = results_dir or _RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    if model is None:
        model = _create_model()

    logger.info(
        "Running scenario: %s (%s) — %s",
        scenario.id, scenario.ticker, scenario.type,
    )
    start = time.time()

    # 1. Gather context
    from tools.data_aggregator import DataAggregator

    user_ctx = scenario.user_context or scenario.as_of_context
    context = DataAggregator.gather_context(scenario.ticker, user_ctx)

    # 2. Adversarial injection
    if scenario.type == "adversarial" and scenario.adversarial:
        context = inject_adversarial_data(context, scenario.adversarial)

    # 3. Run the committee
    from orchestrator.committee import InvestmentCommittee

    committee = InvestmentCommittee(model=model)
    result = committee.run(ticker=scenario.ticker, context=context)

    elapsed = time.time() - start
    logger.info("Scenario %s completed in %.1fs", scenario.id, elapsed)

    # 4. Grade
    rubric = load_rubric(scenario.evaluation_criteria.rubric)
    grading = grade_result(result, scenario, rubric)
    grading.run_metadata["elapsed_s"] = round(elapsed, 1)

    # 5. Save
    save_result_json(grading, results_dir)

    return grading


def run_all(
    scenarios_dir: Path | None = None,
    results_dir: Path | None = None,
    filter_type: str | None = None,
    filter_tags: list[str] | None = None,
    filter_scenario: str | None = None,
    model: Any | None = None,
) -> list[GradingResult]:
    """Run all matching scenarios.

    Args:
        scenarios_dir: Scenario YAML directory.
        results_dir: Output directory.
        filter_type: Filter by type.
        filter_tags: Filter by tags.
        filter_scenario: Filter by scenario ID (partial match).
        model: LLM callable. Created once and reused.

    Returns:
        List of GradingResults.
    """
    scenarios = discover_scenarios(
        scenarios_dir=scenarios_dir,
        filter_type=filter_type,
        filter_tags=filter_tags,
        filter_scenario=filter_scenario,
    )

    if not scenarios:
        logger.warning("No scenarios found matching filters")
        return []

    if model is None:
        model = _create_model()

    logger.info("Running %d scenario(s)", len(scenarios))

    results: list[GradingResult] = []
    for scenario in scenarios:
        try:
            grading = run_scenario(scenario, model=model, results_dir=results_dir)
            results.append(grading)
            status = "PASS" if grading.passed else "FAIL"
            logger.info(
                "  %s — %s (%.1f/100)%s",
                scenario.id,
                status,
                grading.total_score,
                f" [{', '.join(grading.critical_failures)}]" if grading.critical_failures else "",
            )
        except Exception:
            logger.error("Scenario %s failed", scenario.id, exc_info=True)

    return results
