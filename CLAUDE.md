# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A multi-agent investment committee that runs four AI agents (Sector Analyst, Risk Manager, Macro Strategist, Portfolio Manager) in parallel via LangGraph, debates bull/bear theses, and produces a structured committee memo with a T signal and Black-Litterman optimized portfolio weights.

## Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env  # then add API key(s)

# Run
python app.py                    # Gradio UI at http://localhost:7860

# Tests
pytest tests/ -v                 # full suite (~260 tests)
pytest tests/test_optimizer.py -v    # optimizer tests only
pytest tests/test_temperature.py -v  # temperature routing tests only
pytest tests/ -k "test_name"     # single test by name

# Evals
python -m evals list
python -m evals run --scenario nvda_2024
python -m evals report

# Lint
ruff check .
ruff check --fix .
```

## Architecture

### Pipeline Flow (LangGraph StateGraph)

```
gather_data → [sector_analyst, risk_manager, macro_analyst] (parallel)
           → adversarial_debate (conditional, up to N rounds)
           → portfolio_manager → optimizer → finalize
```

- **Phase 1**: Three analyst agents run in parallel via LangGraph `Send()` fan-out
- **Phase 2**: Bull (sector) and bear (risk) debate with conviction updates
- **Phase 3**: PM synthesizes all evidence into `CommitteeMemo` with T signal
- **Phase 4**: Black-Litterman optimizer computes real portfolio weights from PM output

### Key Patterns

- **LLM interface**: All models are `callable(str) -> str` with optional `temperature` kwarg. Provider-agnostic — same pipeline works with Anthropic, Google, OpenAI, DeepSeek, HuggingFace, Ollama.
- **Model factories** (`app.py`): Each provider has a factory closure. All accept `temperature` kwarg for per-node routing.
- **`RateLimitedModel`** (`app.py`): Wraps model callables with RPM/TPM rate limiting. Passes `**kwargs` through.
- **Per-node temperature** (`orchestrator/temperature.py`): `with_temperature(model, temp)` wrapper. Falls back gracefully if model doesn't support the kwarg.
- **Graceful fallback**: Optimizer returns `OptimizerFallback` on any exception; pipeline never breaks.
- **JSON extraction** (`agents/base.py`): `_extract_json()` handles markdown fences, partial JSON, common LLM formatting errors.

### State Management

- `CommitteeState` (`orchestrator/state.py`): LangGraph TypedDict with `Annotated` reducers for list fields
- `CommitteeResult` (`orchestrator/committee.py`): Final output dataclass with all agent outputs + optimization result
- `CommitteeMemo` (`agents/base.py`): Pydantic model for PM's structured output

### Optimizer Package

`optimizer/` is a self-contained package:
- `bl_optimizer.py`: Main pipeline (universe → covariance → views → BL model → efficient frontier → analytics)
- `node.py`: LangGraph node that wraps the pipeline
- `views.py`: Extracts BL views (P, Q, omega) from PM output with regex parsing + confidence scaling
- `universe.py`: Builds 7-8 asset universe from sector peers + ETFs via yfinance
- `covariance.py`: Ledoit-Wolf shrinkage via `pypfopt.CovarianceShrinkage`
- `analytics.py`: Sharpe/Sortino, OLS factor betas, MCTR risk decomposition

### Data Providers

`tools/data_providers/`: `BaseDataProvider` ABC → `YahooProvider` (default), `BloombergProvider`, `IBProvider`. Factory pattern via `get_provider()`.

## Testing Conventions

- Tests use mock LLM fixtures (no API keys needed): `mock_model` returns canned JSON
- Optimizer tests mock `build_universe()` with synthetic price data
- Temperature tests verify kwarg passthrough and graceful fallback
- Run `pytest tests/ -v` — expect ~260 passing, 1 pre-existing skip in `test_tools_phase_b.py`

## Settings

All config via `config/settings.py` (Pydantic `BaseSettings`), loaded from `.env`:
- `LLM_PROVIDER`: anthropic/google/openai/deepseek/huggingface/ollama
- `TEMPERATURE`: global default (0.7)
- `task_temperatures`: per-node overrides (dict)
- Rate limiting: `RATE_LIMIT_RPM`, `RATE_LIMIT_INPUT_TPM`, `RATE_LIMIT_OUTPUT_TPM`
