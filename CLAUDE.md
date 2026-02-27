# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A multi-agent investment committee that runs five AI agents (Long Analyst, Short Analyst, Risk Manager, Macro Strategist, Portfolio Manager) in parallel via LangGraph, preceded by an XAI pre-screen with Shapley value explanations, debates long/short theses, and produces a structured committee memo with a T signal and multi-strategy optimized portfolio weights (Black-Litterman, HRP, Mean-Variance, Min Variance, Risk Parity, Equal Weight, or Ensemble).

## Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env  # then add API key(s)

# Run
python app.py                    # Gradio UI at http://localhost:7860

# Tests
pytest tests/ -v                 # full suite (~687 tests)
pytest tests/test_optimizer.py tests/test_optimizer_strategies.py -v  # optimizer tests
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
gather_data → run_xai_analysis → [sector_analyst, short_analyst, risk_manager, macro_analyst] (parallel)
           → adversarial_debate (Long vs Short, up to N rounds)
           → portfolio_manager → optimizer → finalize
```

- **Phase 1**: Four analyst agents run in parallel via LangGraph `Send()` fan-out (Long, Short, Risk, Macro)
- **Phase 2**: Long vs Short adversarial debate with Risk Manager sizing commentary
- **Phase 3**: PM synthesizes all evidence into `CommitteeMemo` with T signal
- **Phase 4**: Multi-strategy optimizer computes real portfolio weights (BL default, 5 alternatives, or Ensemble meta-strategy via UI dropdown)

### Key Patterns

- **LLM interface**: All models are `callable(str) -> str` with optional `temperature` kwarg. Provider-agnostic — same pipeline works with Anthropic, Google, OpenAI, HuggingFace, Ollama.
- **Model factories** (`app_lib/model_factory.py`): Each provider has a factory closure. All accept `temperature` kwarg for per-node routing.
- **`RateLimitedModel`** (`app_lib/model_factory.py`): Wraps model callables with RPM/TPM rate limiting. Passes `**kwargs` through.
- **Result formatters** (`app_lib/formatters.py`): All `_format_*` functions for Gradio UI output.
- **Per-node temperature** (`orchestrator/temperature.py`): `with_temperature(model, temp)` wrapper. Falls back gracefully if model doesn't support the kwarg.
- **Graceful fallback**: Optimizer returns `OptimizerFallback` on any exception; pipeline never breaks.
- **JSON extraction** (`agents/base.py`): `_extract_json()` handles markdown fences, partial JSON, common LLM formatting errors.

### State Management

- `CommitteeState` (`orchestrator/state.py`): LangGraph TypedDict with `Annotated` reducers for list fields
- `CommitteeResult` (`orchestrator/committee.py`): Final output dataclass with all agent outputs + optimization result
- `CommitteeMemo` (`agents/base.py`): Pydantic model for PM's structured output

### Optimizer Package

`optimizer/` is a self-contained package with 6 pluggable strategies + 1 ensemble meta-strategy:
- `strategies.py`: `OptimizerStrategy` ABC + 6 implementations (Black-Litterman, HRP, Mean-Variance, Min Variance, Risk Parity, Equal Weight) + `STRATEGY_REGISTRY` + `get_strategy()` factory
- `ensemble.py`: Ensemble meta-strategy — runs all 6 strategies on shared universe/covariance, produces cross-strategy analytics (consensus matrix, divergence flags, blended weights, layered narrative). Role assignments: BL=Core Allocation (35%), RP=Sanity Check (25%), MinVar=Defensive Overlay (20%), HRP/MV/EW=reference.
- `bl_optimizer.py`: Shared pipeline dispatcher — `run_optimization()` (universe → covariance → strategy dispatch → analytics). `run_black_litterman()` kept as backward-compatible wrapper.
- `node.py`: LangGraph node — reads `optimizer_method` from config, dispatches to `run_ensemble()` or `run_optimization()`
- `models.py`: `OptimizationResult`, `OptimizerFallback`, `EnsembleResult`, `StrategyComparison`, `TickerConsensus`, `DivergenceFlag`, `FactorExposure`, `RiskContribution`
- `views.py`: Extracts BL views (P, Q, omega) from PM output with regex parsing + confidence scaling
- `universe.py`: Builds 7-8 asset universe from sector peers + ETFs via yfinance
- `covariance.py`: Ledoit-Wolf shrinkage via `pypfopt.CovarianceShrinkage`
- `analytics.py`: Sharpe/Sortino, OLS factor betas, MCTR risk decomposition

Strategy selection: Gradio UI dropdown → `config["configurable"]["optimizer_method"]` → `run_optimization()` (or `run_ensemble()` for "ensemble"). Non-BL strategies skip view construction and run without PM memo. Ensemble builds universe/covariance once, loops all 6 strategies, and produces `EnsembleResult` with blended allocation + divergence analysis.

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

### Volatility Surface

`tools/volatility_surface.py`: Implied vol surface generation for analyst agents. Two registered tools:
- `get_vol_surface(spot, r, model, ...)`: Full strike x maturity IV grid (Heston or CEV model)
- `get_vol_smile(spot, maturity, model, ...)`: Single-maturity smile for quick skew assessment

Models: Heston stochastic vol (COS method pricing), CEV local vol (non-central chi-squared). Returns structured dicts with `iv_surface_pct`, `atm_term_structure`, `skew_by_maturity`, and plain-English `summary.interpretation` for agent consumption. Numerical methods adapted from Oosterlee & Grzelak (2019) under BSD 3-Clause — see `THIRD_PARTY_NOTICES.md`.

### Vol Intelligence Pipeline

`tools/vol_intelligence.py`: Bridge between numerical vol methods and agent reasoning. Auto-runs during `gather_context()` and injects structured vol data into all agent prompts.

Pipeline: `daily prices (yfinance) → realized vol → Heston calibration → implied surface → signals → agent summary`

Signals produced:
- **Realized vol** (`tools/financial_metrics.py:compute_realized_vol`): Multi-window HV (10/30/60/90d), downside vol, vol ratio, percentile rank, regime classification (low/normal/elevated/crisis)
- **IV vs HV**: Compares 3m ATM implied vol to 30d realized. 5 signal levels: `iv_elevated`, `iv_slight_premium`, `iv_fair`, `iv_cheap`, `iv_very_cheap`
- **Skew flags**: Squeeze risk from 25-delta skew (elevated if inverted), extreme put skew flags, vol backwardation detection
- **Regime sizing multiplier**: Maps vol regime to BL confidence scaling [0.3, 1.3] for position sizing

Agent injection (in each agent's `act()` method):
- **Risk Manager**: Vol for sizing, structuring, and tail risk assessment
- **Short Analyst**: Vol with squeeze risk and borrow assessment context
- **Macro Analyst**: Computed regime and sizing multiplier for vol_budget_guidance field
- **Portfolio Manager**: IV vs HV signal, downside vol, regime — feeds required `implied_vol_assessment` output field

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
- All 5 agents use retry in their `act()` methods before falling back to defaults

## Testing Conventions

- Tests use mock LLM fixtures (no API keys needed): `mock_model` returns canned JSON
- Optimizer tests mock `build_universe()` with synthetic price data
- Backtest tests use `tmp_path` fixture for temporary SQLite databases
- Temperature tests verify kwarg passthrough and graceful fallback
- XAI tests use mock fundamentals dicts (no API keys needed), 2 XGBoost tests skip if xgboost not installed
- Vol intelligence tests use synthetic GBM prices (no API keys needed), 51 tests covering realized vol, IV vs HV, skew flags, regime multiplier, agent injection
- Optimizer strategy tests use synthetic price data (7 tickers, 504 days), 43 tests covering all 6 strategies + dispatcher + backward compat
- Run `pytest tests/ -v` — expect ~687 passing, 0 skipped

## Settings

All config via `config/settings.py` (Pydantic `BaseSettings`), loaded from `.env`:
- `LLM_PROVIDER`: anthropic/google/openai/huggingface/ollama
- `TEMPERATURE`: global default (0.7)
- `task_temperatures`: per-node overrides (dict)
- `OPTIMIZER_METHOD`: optimizer strategy key (default `black_litterman`; options: `hrp`, `mean_variance`, `min_variance`, `risk_parity`, `equal_weight`, `ensemble`)
- Rate limiting: `RATE_LIMIT_RPM`, `RATE_LIMIT_INPUT_TPM`, `RATE_LIMIT_OUTPUT_TPM`
