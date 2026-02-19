# Changelog

All notable changes to this project will be documented in this file.

## [3.8.0] - 2026-02-18

### Added
- **Signal persistence** (`backtest/database.py`) — SQLite database stores every IC signal with conviction, direction, T signal, BL optimizer output, and realized returns at 5 horizons (1d/5d/10d/20d/60d)
- **Historical backtest engine** (`backtest/runner.py`) — fills realized returns from yfinance, computes directional P&L, Sharpe, Sortino, win rate, direction accuracy, max drawdown, information coefficients
- **Calibration analysis** (`backtest/calibration.py`) — bins signals by conviction level, measures hit rate and average return per bucket, conviction-return rank correlation
- **Alpha decay curves** (`backtest/alpha_decay.py`) — information coefficient at each forward horizon, identifies optimal holding period and signal half-life, t-statistics for significance
- **Benchmark comparison** (`backtest/benchmark.py`) — IC signals vs SPY buy-and-hold, always-long, and momentum (T-signal ranked) strategies
- **Multi-asset portfolio construction** (`backtest/portfolio.py`) — aggregates latest signal per ticker, constructs weighted portfolio (by T signal, equal weight, or conviction), computes gross/net exposure
- **Explainability / agent attribution** (`backtest/explainability.py`) — decomposes each T signal into bull/bear/macro/debate agent contributions, identifies dominant agent per signal, computes dominance rates
- **Backtest CLI** (`python -m backtest`) — stats, fill-returns, run, calibration, decay, benchmark, portfolio, explain, report subcommands
- **FastAPI REST endpoint** (`api/main.py`) — POST /analyze (run pipeline + persist signal), GET /signals, POST /backtest, GET /portfolio, GET /health
- **Retry-with-feedback JSON extraction** (`agents/base.py: retry_extract_json`) — when initial parse fails, re-prompts the model with the error and truncated response; typically recovers 80%+ of parse failures
- **Ollama JSON mode** — prompts requesting JSON output automatically trigger Ollama's `format: "json"` constraint via prompt heuristic detection
- **Signal persistence in Gradio** — every committee run automatically stores a signal in `store/signals.db` for later backtesting
- 50 new tests (37 backtest/analytics + 10 API + 3 retry extraction)

### Changed
- All 4 agent `act()` methods now use `retry_extract_json` with 1 retry before falling back to defaults
- Ollama model factory detects JSON-requesting prompts and enables constrained output mode
- `pyproject.toml` version bumped to 3.8.0, added `fastapi` and `uvicorn` to dependencies
- `backtest/` and `api/` packages added to wheel build targets

### Dependencies
- Added `fastapi>=0.115.0`, `uvicorn>=0.30.0` to base dependencies

## [3.7.0] - 2026-02-18

### Added
- **Black-Litterman portfolio optimizer** (`optimizer/` package) — real pypfopt BL model takes PM conviction as a view, computes covariance/weights/risk metrics, and displays computed values alongside LLM heuristics
  - Universe construction: target + 5 sector peers + sector ETF + SPY
  - Ledoit-Wolf covariance shrinkage via `pypfopt.CovarianceShrinkage`
  - Absolute view extraction from PM output with confidence scaling from bull/bear conviction spread
  - Max-Sharpe optimization via `pypfopt.EfficientFrontier`
  - Post-optimization analytics: Sharpe/Sortino, OLS factor betas with t-stats, MCTR risk decomposition
  - Side-by-side comparison table (LLM heuristic vs BL computed) in committee memo
  - Graceful fallback — pipeline never breaks if optimizer fails
- **Per-node temperature routing** (`orchestrator/temperature.py`) — each pipeline stage uses a task-appropriate temperature (data: 0.1, analysis: 0.5, PM synthesis: 0.7, math: 0.0) instead of a single global setting
  - Works across all 6 providers (Anthropic, Google, OpenAI, DeepSeek, HuggingFace, Ollama)
  - User overrides via `settings.task_temperatures`
  - `with_temperature()` wrapper with graceful fallback for models that don't support the kwarg
- 45 new tests (32 optimizer + 13 temperature routing)

### Changed
- All model factory closures now accept optional `temperature` kwarg
- `RateLimitedModel` passes `**kwargs` through to wrapped model
- Pipeline graph: `run_portfolio_manager → run_optimizer → finalize` (optimizer inserted)
- `CommitteeState` and `CommitteeResult` include `optimization_result` field

### Dependencies
- Added `pyportfolioopt>=1.5.0`, `scipy>=1.10.0`, `scikit-learn>=1.3.0` to base dependencies

## [3.6.0] - 2026-02-15

### Added
- **Eval harness** (`evals/` package) — ground-truth evaluation framework with 6 grading dimensions (100 total points)
- **Likert scoring** — 5-point Likert levels (Fail/Poor/Adequate/Good/Excellent) mapped to continuous 0-100 scores
- **Adversarial testing** — context injection scenarios to test robustness against tainted data
- 4 ground-truth scenarios: NVDA mid-2024, SVB pre-collapse, META late 2022, AAPL adversarial
- **Data provider abstraction** (`tools/data_providers/`) — `BaseDataProvider` ABC with factory pattern
- Bloomberg (`blpapi`) and Interactive Brokers (`ib_insync`) provider implementations
- Risk-unit framework, smart PDF page scoring, prompt injection defense (ported from llm-long-short-arena)

### Changed
- All tools refactored to use provider abstraction
- License standardized to MIT

## [3.5.0] - 2026-02-10

### Added
- T signal with citation — PM now cites specific evidence supporting the T signal value
- Conviction tracker redesign — interactive Plotly chart with LLM-generated rationale at each conviction update

## [3.4.0] - 2026-02-08

### Added
- Small-LLM resilience — graceful degradation for models with limited instruction following
- Model selection dropdown in UI
- Ollama improvements — better local model support

## [3.3.0] - 2026-02-06

### Added
- Conviction timeline with LLM-generated rationale at each update point

## [3.2.0] - 2026-02-04

### Added
- Document KB upload — up to 5 research documents (PDF, DOCX, TXT) chunked and injected as context

## [3.1.0] - 2026-02-02

### Added
- Return decomposition (industry return + idiosyncratic alpha)
- Heuristic Sharpe and Sortino ratio estimation
- NMV (net market value) sizing in PM output

## [3.0.0] - 2026-01-30

### Added
- Sentiment extraction with per-headline signal strength
- Portfolio construction heuristics in PM synthesis
- T signal — entropy-adjusted scalar in [-1, 1]

## [2.0.0] - 2026-01-25

### Changed
- Migrated from ThreadPoolExecutor to LangGraph StateGraph orchestration
- Added dynamic tool calling with budget enforcement
- Added human-in-the-loop (HITL) two-phase mode

## [1.0.0] - 2026-01-20

### Added
- Initial multi-agent investment committee
- Four agents: Sector Analyst, Risk Manager, Macro Strategist, Portfolio Manager
- ThreadPoolExecutor parallelism
- Gradio UI
- yfinance market data, RSS news feeds
