# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A multi-agent investment committee that runs four AI agents (Sector Analyst, Risk Manager, Macro Strategist, Portfolio Manager) in parallel via LangGraph, preceded by an XAI pre-screen with Shapley value explanations, debates bull/bear theses, and produces a structured committee memo with a T signal and Black-Litterman optimized portfolio weights.

## Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env  # then add API key(s)

# Run
python app.py                    # Gradio UI at http://localhost:7860

# Tests
pytest tests/ -v                 # full suite (~358 tests)
pytest tests/test_optimizer.py -v    # optimizer tests only
pytest tests/test_xai.py -v         # XAI module tests
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
gather_data → run_xai_analysis → [sector_analyst, risk_manager, macro_analyst] (parallel)
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

### XAI Package

`xai/` is a self-contained explainable AI module:
- `features.py`: Extracts 12 numeric features from fundamentals dict (handles % string parsing)
- `distress.py`: AltmanZScoreModel (always available) + XGBoostDistressModel (optional artifact)
- `shapley.py`: Built-in Shapley calculators — ExactLinearShapley (analytical for Z-Score) + PermutationShapley (sampling-based for any model)
- `explainer.py`: SHAP wrapper that auto-falls back to built-in Shapley when `shap` not installed
- `returns.py`: Distress screening + expected return computation (ER = (1-PFD) * earnings_yield)
- `pipeline.py`: XAIPipeline orchestrates all 5 steps into XAIResult
- `node.py`: LangGraph node (between gather_data and analyst fan-out)
- `train.py`: Optional CLI for training XGBoost model on labeled data
- `artifacts/`: Trained models (.gitignored)

Based on: Sotic & Radovanovic (2024), "Explainable AI in Finance" (doi:10.20935/AcadAI8017)

### Data Providers

`tools/data_providers/`: `BaseDataProvider` ABC → `YahooProvider` (default), `BloombergProvider`, `IBProvider`. Factory pattern via `get_provider()`.

### Backtest + Analytics Package

`backtest/` is a self-contained analytics stack:
- `database.py`: SQLite persistence — 3 tables (signals, portfolio_snapshots, backtest_results) with indexes
- `runner.py`: Historical backtest engine — fills realized returns from yfinance, computes P&L/Sharpe/win rate
- `calibration.py`: Conviction → realized return mapping in configurable buckets
- `alpha_decay.py`: IC at multiple horizons, optimal holding period identification
- `benchmark.py`: IC signals vs SPY, always-long, and momentum strategies
- `portfolio.py`: Multi-asset portfolio from latest signals (T-signal, equal, conviction weighting)
- `explainability.py`: Agent-level attribution decomposition
- CLI: `python -m backtest [stats|run|calibration|decay|benchmark|portfolio|explain|report]`

### REST API

`api/main.py`: FastAPI app with endpoints for `/analyze`, `/signals`, `/backtest`, `/portfolio`, `/health`. Run with `uvicorn api.main:app --reload`.

### Structured Output Hardening

- `retry_extract_json()` in `agents/base.py`: re-prompts model with error feedback when initial parse fails
- Ollama JSON mode: prompt heuristic detection triggers `format: "json"` constraint
- All 4 agents use retry in their `act()` methods before falling back to defaults

## Testing Conventions

- Tests use mock LLM fixtures (no API keys needed): `mock_model` returns canned JSON
- Optimizer tests mock `build_universe()` with synthetic price data
- Backtest tests use `tmp_path` fixture for temporary SQLite databases
- Temperature tests verify kwarg passthrough and graceful fallback
- XAI tests use mock fundamentals dicts (no API keys needed), 2 XGBoost tests skip if xgboost not installed
- Run `pytest tests/ -v` — expect ~358 passing, 2 skipped (xgboost optional)

## Settings

All config via `config/settings.py` (Pydantic `BaseSettings`), loaded from `.env`:
- `LLM_PROVIDER`: anthropic/google/openai/deepseek/huggingface/ollama
- `TEMPERATURE`: global default (0.7)
- `task_temperatures`: per-node overrides (dict)
- Rate limiting: `RATE_LIMIT_RPM`, `RATE_LIMIT_INPUT_TPM`, `RATE_LIMIT_OUTPUT_TPM`
