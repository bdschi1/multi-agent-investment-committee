---
title: Multi-Agent Investment Committee
emoji: "\U0001F4CA"
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.0.0
app_file: app.py
pinned: false
license: mit
tags:
  - multi-agent
  - agentic-ai
  - investment
  - reasoning
  - langgraph
  - reinforcement-learning
  - black-litterman
  - explainable-ai
  - shapley-values
---

# Multi-Agent Investment Committee

Four AI agents analyze a ticker in parallel, debate each other's theses, and produce a structured committee memo with a trading signal — preceded by an XAI pre-screen with Shapley value explanations and validated by a real Black-Litterman portfolio optimizer.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-358%20passing-brightgreen.svg)](tests/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **Not financial advice.** All output is AI-generated. Do not use for investment decisions.

---

## How It Works

The pipeline runs in six phases:

1. **XAI Pre-Screen** -- Before any LLM calls, a quantitative pre-screen estimates the Probability of Financial Distress (PFD) using an Altman Z-Score model (or optional trained XGBoost), computes risk-adjusted expected return, and identifies the top risk/return drivers via Shapley value decomposition. This creates **dual explainability**: feature-level Shapley (quantitative) + agent-level attribution (qualitative).

2. **Parallel Analysis** -- Three agents analyze the same ticker simultaneously, with XAI context injected:
   - **Sector Analyst** (bull case) -- sentiment extraction, return decomposition, catalyst identification
   - **Risk Manager** (bear case) -- causal chain reasoning (1st/2nd/3rd order effects), alpha challenge
   - **Macro Strategist** (top-down) -- vol regime classification, net exposure, sizing frameworks

3. **Adversarial Debate** -- Bull and bear challenge each other through structured rebuttals. Convictions update based on evidence.

4. **PM Synthesis** -- Portfolio Manager weighs all evidence, applies sizing heuristics, produces a committee memo with recommendation and T signal.

5. **Portfolio Optimization** -- Black-Litterman optimizer takes the PM's conviction as a view, computes actual covariance/weights/risk metrics, and displays computed values alongside the LLM heuristics.

6. **Signal Persistence + Analytics** -- Every signal is stored in SQLite for historical backtesting, calibration analysis, alpha decay measurement, benchmark comparison, and multi-asset portfolio construction.

```
User Input (ticker + guidance + docs)
       |
       v
  Data Gathering (yfinance, RSS, derived metrics, uploaded docs)
       |
       v
  XAI Pre-Screen (PFD, Shapley values, expected return)
       |
   +---+---+--------+
   v   v            v
 Sector  Risk     Macro         (XAI context injected)
Analyst Manager  Strategist
   |      |          |
   v      v          |
  Adversarial        |
    Debate            |
       |              |
       v              v
  Portfolio Manager
  (Synthesis + T Signal)
       |
       v
  Black-Litterman Optimizer
  (pypfopt: BL weights, Sharpe, MCTR)
       |
       v
  Committee Memo
```

Each agent follows a THINK -> PLAN -> ACT -> REFLECT loop. Every step is captured in a reasoning trace visible in the UI. Each pipeline node runs at a task-appropriate temperature (data extraction: 0.1, analysis: 0.5, PM synthesis: 0.7, math: 0.0).

~20 LLM API calls per analysis. 90-120 seconds wall clock depending on provider.

---

## Agents

**Sector Analyst** -- Builds the bull case. Extracts per-headline sentiment (bullish/bearish/neutral with signal strength). Decomposes expected return into industry return and idiosyncratic alpha. Estimates heuristic Sharpe and Sortino ratios.

**Risk Manager** -- Builds the bear case. Traces how primary risks cascade into 2nd and 3rd order effects. Challenges whether claimed alpha is genuine idiosyncratic return or disguised factor/sector beta. Produces active short pitches when warranted.

**Macro Strategist** -- Sets portfolio-level guardrails. Classifies vol regime (low/normal/elevated/crisis). Recommends net exposure direction, sizing method (proportional / risk parity / mean-variance / shrunk mean-variance), and portfolio vol target. The PM operates within these constraints.

**Portfolio Manager** -- Chairs the committee. Weighs bull thesis, bear risks, macro context, and debate outcomes. Validates return decomposition after hearing both sides. Computes the T signal. Outputs a structured memo with recommendation, sizing rationale, conviction triggers, and factor exposures.

---

## Black-Litterman Optimizer

After the PM produces its qualitative sizing heuristics, the `optimizer/` package runs a real [pypfopt](https://github.com/robertmartin8/PyPortfolioOpt) Black-Litterman model:

1. **Universe construction** -- target stock + 5 sector peers + sector ETF + SPY (7-8 assets)
2. **Covariance estimation** -- Ledoit-Wolf shrinkage on 2 years of daily returns
3. **View extraction** -- PM's alpha estimate becomes the BL absolute view; bull/bear conviction spread scales view confidence (omega)
4. **Posterior returns** -- BL model combines market-cap equilibrium prior with PM's view
5. **Max-Sharpe optimization** -- efficient frontier produces optimal weights
6. **Analytics** -- computed Sharpe/Sortino, OLS factor betas with t-stats, marginal contribution to risk (MCTR)

The report displays a **side-by-side comparison** of LLM heuristic estimates vs. computed values. If the optimizer fails (e.g., insufficient price data), the pipeline continues gracefully with heuristics as primary reference.

---

## Explainable AI (XAI) Pre-Screen

The `xai/` package implements a five-step explainable AI procedure that runs before the LLM agents, giving them quantitative context. Inspired by Sotic & Radovanovic (2024), "Explainable AI in Finance" ([doi:10.20935/AcadAI8017](https://doi.org/10.20935/AcadAI8017)), which proposes combining XGBoost prediction with SHAP explanations for investment decision-making.

| Step | What | Method |
|------|------|--------|
| 1 | **PFD Estimation** | Altman Z-Score (zero-config) or trained XGBoost |
| 2 | **Risk Factor Identification** | Shapley value decomposition of distress drivers |
| 3 | **Distress Screening** | PFD threshold flag (configurable, default 50%) |
| 4 | **Expected Return** | ER = (1 - PFD) x earnings yield proxy |
| 5 | **Return Driver Analysis** | Shapley value decomposition of profitability factors |

**Two-tier model architecture**: Altman Z-Score always works with zero configuration using proxy variables from fundamentals. When a trained XGBoost model exists in `xai/artifacts/`, it activates automatically (mirrors the Yahoo/Bloomberg provider pattern).

**Shapley values without dependencies**: The built-in Shapley calculator (`xai/shapley.py`) provides exact analytical Shapley values for the Z-Score linear model and permutation-based approximate Shapley for any other model. The optional `shap` library enhances this with waterfall plots, but explanations always work even in a minimal install.

**Dual explainability**: Feature-level Shapley values (quantitative: "debt_to_equity contributed +0.04 to distress risk") plus agent-level attribution (qualitative: "the risk manager's bear case shifted conviction by -1.2").

```bash
pip install -e ".[xai]"           # Optional: XGBoost + shap + matplotlib
python -m xai.train --data data.csv --target is_distressed  # Train custom model
```

---

## Backtesting + Signal Analytics

The `backtest/` package provides a full analytics stack for evaluating signal quality over time:

| Module | What it does |
|--------|-------------|
| **Signal Persistence** | SQLite database stores every signal with conviction, direction, T signal, and BL optimizer output |
| **Historical Backtest** | Fills realized 1d/5d/10d/20d/60d returns, computes directional P&L, Sharpe, win rate, max drawdown |
| **Calibration** | Bins signals by conviction level, measures hit rate and avg return per bucket -- answers "is conviction 8/10 better than 6/10?" |
| **Alpha Decay** | Information coefficient at each horizon -- identifies optimal holding period and signal half-life |
| **Benchmark Comparison** | IC signals vs SPY buy-and-hold, always-long, and momentum (T-signal ranked) |
| **Multi-Asset Portfolio** | Aggregates latest signal per ticker, constructs weighted portfolio with exposure metrics |
| **Explainability** | Decomposes each T signal into bull/bear/macro/debate agent contributions |

```bash
python -m backtest stats                      # Signal and portfolio statistics
python -m backtest fill-returns               # Fill realized returns from yfinance
python -m backtest run --ticker NVDA          # Run backtest for a ticker
python -m backtest calibration                # Conviction-return calibration
python -m backtest decay                      # Alpha decay curve
python -m backtest benchmark                  # Compare vs SPY, momentum
python -m backtest portfolio                  # Build portfolio snapshot
python -m backtest explain                    # Agent attribution analysis
python -m backtest report                     # Full analytics report
```

---

## REST API

The `api/` package provides a FastAPI endpoint for programmatic access:

```bash
pip install -e "."
uvicorn api.main:app --reload --port 8000
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analyze` | POST | Run full IC pipeline, returns signal + stores in DB |
| `/signals` | GET | List stored signals (filter by ticker) |
| `/signals/{id}` | GET | Get specific signal by ID |
| `/backtest` | POST | Run backtest on stored signals |
| `/portfolio` | GET | Build portfolio snapshot from latest signals |
| `/fill-returns` | POST | Fill realized returns for stored signals |
| `/health` | GET | Health check with signal stats |

---

## Structured Output Hardening

All agents use a progressive JSON extraction pipeline with 7 repair strategies (markdown block removal, brace boundary, trailing comma fix, quote repair, unbalanced brace repair). v3.8 adds:

- **Retry-with-feedback** -- When extraction fails, the model receives the error and its truncated response, and is prompted to produce clean JSON. One retry typically recovers 80%+ of parse failures.
- **Ollama JSON mode** -- Prompts requesting JSON output automatically trigger Ollama's `format: "json"` constraint, preventing free-form text responses from smaller models.

---

## T Signal

Single scalar in [-1, 1] encoding direction and confidence:

```
T = direction x C

direction in {-1, +1}     (-1 = short, +1 = long)
C = e + (1 - e)(1 - H)    (entropy-adjusted certainty)
H = 1 - raw_confidence     (proxy for entropy)
e = 0.01                   (floor)
```

`T = +0.85` means strong long conviction with high certainty. `T = -0.40` means moderate short conviction with moderate certainty.

Since the LLM interface is provider-agnostic (`callable(str) -> str`), token-level entropy is unavailable. The PM's self-reported confidence serves as proxy. Approach adapted from Darmanin & Vella, [arXiv:2508.02366](https://arxiv.org/abs/2508.02366).

---

## Per-Node Temperature Routing

Each pipeline node runs at a task-appropriate temperature instead of a single global setting:

| Node | Temperature | Reasoning |
|------|------------|-----------|
| `gather_data` | 0.1 | Factual retrieval — one "right" answer |
| `run_xai_analysis` | 0.0 | Deterministic computation (no LLM calls) |
| `run_sector_analyst` | 0.5 | Balanced exploration of thesis points |
| `run_risk_manager` | 0.5 | Explore unlikely-but-possible tail risks |
| `run_macro_analyst` | 0.5 | Balanced macro analysis |
| `run_debate_round` | 0.5 | Adversarial — consider edge arguments |
| `run_portfolio_manager` | 0.7 | Narrative flow, connecting themes |
| `run_optimizer` | 0.0 | Deterministic computation (no LLM calls) |

Works across all providers (Anthropic, Google, OpenAI, DeepSeek, HuggingFace, Ollama). Override per-node via `settings.task_temperatures`.

---

## Eval Harness

`evals/` contains a ground-truth evaluation framework for grading committee output against known historical scenarios.

**Grading dimensions** (6 dimensions, 100 total points):
- Direction accuracy (25) -- did the committee get the direction right?
- Conviction calibration (15) -- was conviction appropriately scaled?
- Risk identification (20) -- did the bear case find the key risks?
- Fact coverage (15) -- did the bull case surface the important evidence?
- Reasoning quality (15) -- did agents update convictions, produce substantive rebuttals?
- Adversarial robustness (10) -- did agents detect injected/tainted data?

Each dimension maps to a 5-point **Likert level** (Fail/Poor/Adequate/Good/Excellent) with dimension-specific anchor definitions for interpretability. The 0-100 continuous scores remain the scoring backbone; Likert is the human-readable layer.

**Scenarios included** (ground truth filled, ready to run):
- NVDA mid-2024 (AI datacenter thesis, +25% actual)
- SVB pre-collapse (bank run risk, -100% actual)
- META late 2022 (deep value rerating, +450% actual)
- AAPL adversarial (fabricated 80% revenue growth injection)

Adding a scenario: copy `evals/scenarios/_template.yaml`, fill the fields, run.

```bash
python -m evals list                          # list scenarios
python -m evals run --scenario nvda_2024      # run one
python -m evals run --type adversarial        # adversarial only
python -m evals report                        # generate report from results/
```

---

## Quick Start

### Prerequisites
- Python 3.11+
- At least one LLM API key, or Ollama for local execution

### Providers

| Provider | Default Model | Setup |
|----------|--------------|-------|
| Anthropic | `claude-sonnet-4-20250514` | `ANTHROPIC_API_KEY` in `.env` |
| Google | `gemini-2.0-flash` | `GOOGLE_API_KEY` in `.env` |
| OpenAI | `gpt-4o-mini` | `pip install -e ".[openai]"` + `OPENAI_API_KEY` |
| DeepSeek | `deepseek-chat` | `pip install -e ".[deepseek]"` + `DEEPSEEK_API_KEY` |
| HuggingFace | `Qwen/Qwen2.5-72B-Instruct` | `HF_TOKEN` in `.env` |
| Ollama | `llama3.1:8b` | `pip install -e ".[ollama]"` + `ollama serve` |

Switch providers at runtime via the UI dropdown or `LLM_PROVIDER` in `.env`.

### Install

```bash
git clone https://github.com/bdschi1/multi-agent-investment-committee.git
cd multi-agent-investment-committee

python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"

cp .env.example .env
# Add API key(s) to .env
```

### Run

```bash
python app.py
# http://localhost:7860
```

### Test

```bash
pytest tests/ -v   # 358 tests
```

---

## Execution Modes

| Mode | Description |
|------|-------------|
| Full Auto | Runs entire pipeline end-to-end |
| Review Before PM | Two-phase HITL -- review analyst outputs + debate, optionally add PM guidance, then PM synthesizes |

---

## Data Providers

Market data defaults to yfinance. The `tools/data_providers/` package implements a provider abstraction (`BaseDataProvider` ABC) with a factory pattern. Bloomberg (`blpapi`) and Interactive Brokers (`ib_insync`) providers are available as optional installs:

```bash
pip install -e ".[bloomberg]"   # Bloomberg Terminal
pip install -e ".[ibkr]"        # Interactive Brokers
```

---

## Document Upload

Upload up to 5 research documents (PDF, DOCX, TXT). Files are token-aware chunked (800 tokens/chunk) and injected into all agents as supplementary context. Requires `pip install -e ".[docs]"` for PDF/DOCX support.

---

## Project Structure

```
multi-agent-investment-committee/
├── app.py                       # Gradio UI + report formatting
├── agents/
│   ├── base.py                  # BaseAgent ABC + schemas + JSON extraction
│   ├── sector_analyst.py        # Bull case + sentiment
│   ├── risk_manager.py          # Bear case + causal chains
│   ├── macro_analyst.py         # Macro context + portfolio strategy
│   └── portfolio_manager.py     # PM synthesis + T signal
├── orchestrator/
│   ├── graph.py                 # LangGraph StateGraph (full + phase1 + phase2)
│   ├── nodes.py                 # Node functions + T signal computation
│   ├── state.py                 # CommitteeState TypedDict with reducers
│   ├── committee.py             # CommitteeResult + ConvictionSnapshot
│   ├── temperature.py           # Per-node temperature routing
│   ├── memory.py                # Session memory store
│   └── reasoning_trace.py       # Trace rendering
├── optimizer/
│   ├── bl_optimizer.py          # Main BL pipeline (universe → cov → views → BL → frontier)
│   ├── node.py                  # LangGraph node wrapper with graceful fallback
│   ├── models.py                # OptimizationResult, OptimizerFallback, FactorExposure, RiskContribution
│   ├── views.py                 # Extract BL views (P, Q, omega) from PM output
│   ├── universe.py              # Build asset universe (target + peers + sector ETF + SPY)
│   ├── covariance.py            # Ledoit-Wolf covariance shrinkage via pypfopt
│   └── analytics.py             # Sharpe/Sortino, factor betas (OLS), MCTR risk decomposition
├── tools/
│   ├── registry.py              # ToolRegistry + budget enforcement
│   ├── market_data.py           # yfinance wrapper
│   ├── news_retrieval.py        # RSS feeds
│   ├── financial_metrics.py     # Derived metrics + quality scoring
│   ├── earnings_data.py         # Earnings beat/miss
│   ├── insider_data.py          # SEC Form 4
│   ├── peer_comparison.py       # Peer valuation
│   ├── data_aggregator.py       # Unified data pipeline
│   ├── doc_chunker.py           # Document ingestion + chunking
│   └── data_providers/          # Provider abstraction (yahoo/bloomberg/ib)
├── evals/
│   ├── schemas.py               # EvalScenario, GroundTruth, GradingResult, LikertLevel
│   ├── grader.py                # 6-dimension scoring engine
│   ├── adversarial.py           # Context injection + robustness grading
│   ├── loader.py                # YAML scenario discovery
│   ├── runner.py                # Scenario execution
│   ├── reporter.py              # JSON/markdown report generation
│   ├── scenarios/               # Ground-truth + adversarial YAML scenarios
│   └── rubrics/                 # Scoring rubric definitions
├── backtest/
│   ├── database.py              # SQLite signal persistence (3 tables, indexes)
│   ├── models.py                # SignalRecord, BacktestResult, CalibrationBucket, etc.
│   ├── runner.py                # Historical backtest engine (fill returns, compute P&L)
│   ├── calibration.py           # Conviction-return calibration analysis
│   ├── alpha_decay.py           # IC at multiple horizons, optimal holding period
│   ├── benchmark.py             # IC vs SPY, always-long, momentum
│   ├── portfolio.py             # Multi-asset portfolio construction
│   ├── explainability.py        # Agent attribution decomposition
│   └── __main__.py              # CLI: python -m backtest
├── xai/
│   ├── models.py                # XAIResult, DistressAssessment, ReturnDecomposition schemas
│   ├── features.py              # 12-feature extraction from fundamentals (% string parsing)
│   ├── distress.py              # AltmanZScoreModel + optional XGBoostDistressModel
│   ├── shapley.py               # Built-in Shapley: ExactLinearShapley + PermutationShapley
│   ├── explainer.py             # SHAP wrapper with built-in fallback
│   ├── returns.py               # Distress screening + expected return computation
│   ├── pipeline.py              # XAIPipeline: orchestrates all 5 steps
│   ├── node.py                  # LangGraph node (between gather_data and analysts)
│   ├── train.py                 # CLI: python -m xai.train (optional XGBoost training)
│   └── artifacts/               # Trained models (.gitignored)
├── api/
│   ├── main.py                  # FastAPI app with /analyze, /signals, /backtest, /portfolio
│   └── models.py                # Request/response Pydantic models
├── config/
│   └── settings.py              # Pydantic settings with .env + per-node temperature overrides + XAI config
├── tests/                       # 358 tests
├── .env.example                 # Environment variable template
├── CHANGELOG.md                 # Version history
└── pyproject.toml
```

---

## Tech Stack

- [LangGraph](https://github.com/langchain-ai/langgraph) -- StateGraph orchestration with fan-out/fan-in and conditional edges
- [Gradio](https://www.gradio.app/) -- UI with 8 analysis tabs, T signal gauge, document upload
- [Pydantic](https://docs.pydantic.dev/) -- Output validation for all agent schemas
- [pypfopt](https://github.com/robertmartin8/PyPortfolioOpt) -- Black-Litterman model, efficient frontier, covariance shrinkage
- [FastAPI](https://fastapi.tiangolo.com/) -- REST API for programmatic access
- [yfinance](https://github.com/ranaroussi/yfinance) -- Market data
- [pytest](https://docs.pytest.org/) -- 358 tests with mock LLM fixtures
- [SHAP](https://github.com/shap/shap) -- Shapley value explanations (optional; built-in fallback included)
- [XGBoost](https://xgboost.readthedocs.io/) -- Optional trained distress model (Altman Z-Score always available)

LLM providers: Anthropic, Google, OpenAI, DeepSeek, HuggingFace, Ollama

---

## Version History

- v1: Multi-agent committee with ThreadPoolExecutor parallelism
- v2: LangGraph orchestration, dynamic tool calling, HITL
- v3: Sentiment extraction, portfolio construction heuristics, T signal
- v3.1: Return decomposition, Sharpe/Sortino, NMV sizing
- v3.2: Document KB upload
- v3.3: Conviction timeline with LLM-generated rationale
- v3.4: Small-LLM resilience, model selection, Ollama improvements
- v3.5: T signal with citation, conviction tracker redesign
- v3.6: Eval harness, adversarial testing framework, Likert scoring, data provider abstraction
- v3.7: Black-Litterman portfolio optimizer, per-node temperature routing
- v3.8: Signal persistence (SQLite), historical backtesting, calibration analysis, alpha decay, benchmark comparison, multi-asset portfolio, explainability/attribution, FastAPI REST endpoint, structured output hardening (retry + Ollama JSON mode)
- v3.9: XAI pre-screen module — Altman Z-Score/XGBoost distress estimation, built-in Shapley value decomposition (exact linear + permutation-based), expected return computation, dual explainability (feature-level + agent-level). Based on Sotic & Radovanovic (2024)

---

## License

MIT

---

![Python](https://img.shields.io/badge/python-3.11+-3776AB?style=flat&logo=python&logoColor=white)

![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C?style=flat&logo=langchain&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-F97316?style=flat&logo=gradio&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-E92063?style=flat&logo=pydantic&logoColor=white)
![pypfopt](https://img.shields.io/badge/pypfopt-Black--Litterman-2C5F2D?style=flat)
![Anthropic](https://img.shields.io/badge/Anthropic-191919?style=flat&logo=anthropic&logoColor=white)
![Google](https://img.shields.io/badge/Gemini-4285F4?style=flat&logo=google&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=flat&logo=openai&logoColor=white)
![Yahoo Finance](https://img.shields.io/badge/Yahoo_Finance-6001D2?style=flat&logo=yahoo&logoColor=white)
![Bloomberg](https://img.shields.io/badge/Bloomberg-000000?style=flat&logo=bloomberg&logoColor=white)
![Interactive Brokers](https://img.shields.io/badge/Interactive_Brokers-D71920?style=flat)
