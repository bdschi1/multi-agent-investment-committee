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
  - black-litterman
  - explainable-ai
  - shapley-values
  - long-short
---

# Multi-Agent Investment Committee

An AI system that replicates an institutional investment committee. You enter a stock ticker, and five AI agents — playing the roles of a long analyst, short analyst, risk manager, macro strategist, and portfolio manager — independently analyze the stock, debate each other, and produce a structured recommendation with a position-sizing signal.

Under the hood, a quantitative pre-screen estimates financial distress risk and expected returns before the agents run. After the agents reach a conclusion, a portfolio optimizer converts their qualitative views into concrete portfolio weights. Every signal is stored for historical backtesting.

This is a continually developed project. Features, interfaces, and test coverage expand over time as new research ideas and workflow needs arise.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-687%20passing-brightgreen.svg)](tests/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **Not financial advice.** All output is AI-generated. Do not use for investment decisions.

---

## Pipeline at a Glance

```
╔═══════════════════════════════════════════════════════╗
║                    USER INPUT                         ║
║            ticker · guidance · documents              ║
╚═══════════════════════════╤═══════════════════════════╝
                            ▼
┌─ PHASE 1 ─────────────────────────────────────────────┐
│  Data Gathering ──► XAI Pre-Screen                    │
│  (yfinance, RSS)    (distress, Shapley, exp. return)  │
└───────────────────────────┬───────────────────────────┘
            ┌───────┬───────┴───────┬───────┐
            ▼       ▼               ▼       ▼
┌─ PHASE 2 ─────────────────────────────────────────────┐
│   Long      Short        Risk        Macro            │
│   Analyst   Analyst      Manager     Strategist       │
│      └────┬────┘           │            │             │
│           ▼                │            │             │
│     Adversarial Debate ◄───┘            │             │
└───────────┬─────────────────────────────┼─────────────┘
            └─────────────┬───────────────┘
                          ▼
┌─ PHASE 3 ─────────────────────────────────────────────┐
│  Portfolio Manager ──► Portfolio Optimizer              │
│  (T signal, memo)      (BL / ensemble / 4 others)     │
└───────────────────────────┬───────────────────────────┘
                            ▼
                    ┌───────────────┐
                    │ COMMITTEE MEMO│
                    └───────────────┘
```

Each agent follows a **THINK → PLAN → ACT → REFLECT** loop — similar to how a human analyst would frame the question, plan the analysis, write the conclusion, then reconsider. Every step is captured in a reasoning trace visible in the UI. A typical run makes ~25 LLM calls and takes 90-120 seconds depending on provider.

---

## Quick Start

```bash
git clone https://github.com/bdschi1/multi-agent-investment-committee.git
cd multi-agent-investment-committee

cp .env.example .env   # add API key(s)
./run.sh               # setup + launch Gradio UI at http://localhost:7860
```

Or manually:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python app.py           # http://localhost:7860
pytest tests/ -v        # 687 tests
```

Run `./run.sh help` for all commands (`setup`, `app`, `api`, `test`).

| Provider | Default Model | Setup |
|----------|--------------|-------|
| Anthropic | `claude-sonnet-4-20250514` | `ANTHROPIC_API_KEY` in `.env` |
| Google | `gemini-2.0-flash` | `GOOGLE_API_KEY` in `.env` |
| OpenAI | `gpt-4o-mini` | `pip install -e ".[openai]"` + `OPENAI_API_KEY` |
| HuggingFace | `Qwen/Qwen2.5-72B-Instruct` | `HF_TOKEN` in `.env` |
| Ollama | `llama3.1:8b` | `pip install -e ".[ollama]"` + `ollama serve` |

Switch providers at runtime via the UI dropdown or `LLM_PROVIDER` in `.env`.

---

<div align="center">

[Agents](#agents) · [How It Works](#how-it-works) · [Vol Intelligence](#volatility-intelligence) · [Optimizer](#portfolio-optimizer) · [Backtest](#signal-analytics--backtesting) · [Agent Memory](#agent-memory--post-trade-reflection) · [API](#rest-api) · [Evals](#eval-harness) · [Architecture](#architecture)

</div>

---

## Agents

```
  Long Analyst ◄── Adversarial Debate ──► Short Analyst
       │                  ▲                     │
       │            Risk Manager                │
       │         (sizing commentary)            │
       │                                        │
       ▼                  ▼                     ▼
  ┌─────────────────────────────────────────────────┐
  │              Portfolio Manager                   │
  │       (synthesis + T signal + conviction map)    │
  └─────────────────────────────────────────────────┘
                          ▲
                          │
                   Macro Strategist
              (vol regime, net exposure)
```

| Agent | Role | Key Output |
|-------|------|------------|
| **Long Analyst** | Bull case — sentiment, return decomposition, catalysts | `bull_case` with conviction 1-10 |
| **Short Analyst** | Short thesis — alpha/beta classification, borrow, event path | `short_case` with thesis type |
| **Risk Manager** | Sizing/structuring — stop-loss, stress scenarios, correlation | `bear_case` with position structure |
| **Macro Strategist** | Top-down — vol regime, net exposure, sizing framework | `macro_view` with regime classification |
| **Portfolio Manager** | Synthesis — weighs all evidence, conviction change map | `CommitteeMemo` with T signal |

<details>
<summary><b>Agent details</b></summary>

**Long Analyst** (`sector_analyst.py`) — Builds the bull case. Extracts per-headline sentiment (bullish/bearish/neutral with signal strength). Decomposes expected return into industry return and idiosyncratic alpha. Estimates heuristic Sharpe and Sortino ratios. Debates the Short Analyst.

**Short Analyst** (`short_analyst.py`) — Independent short-side analysis. Classifies thesis type (alpha short, hedge, pair leg, or no position). Builds event path for the short to work. Decomposes expected short return into systematic vs. idiosyncratic. Assesses borrow cost and liquidity. Debates the Long Analyst.

**Risk Manager** (`risk_manager.py`) — Sizes and structures the position. Traces how primary risks cascade into 2nd and 3rd order effects. Recommends position structure (outright, hedged, paired), stop-loss levels, max risk allocation, and stress scenarios. Flags correlation and crowding risks. Does not generate an independent thesis — stress-tests the long and short theses.

**Macro Strategist** (`macro_analyst.py`) — Sets portfolio-level guardrails. Classifies vol regime (low/normal/elevated/crisis). Recommends net exposure direction, sizing method (proportional / risk parity / mean-variance / shrunk mean-variance), and portfolio vol target. Assesses how the position interacts with the existing book.

**Portfolio Manager** (`portfolio_manager.py`) — Chairs the committee. Weighs bull thesis, short thesis, risk assessment, macro context, and debate outcomes. Produces a conviction change map — for each event, which agent's conviction shifts and by how much. Computes the T signal. Outputs a structured memo with recommendation, sizing rationale, and factor exposures.

</details>

---

## How It Works

The system follows the same workflow as a real investment committee: gather data, let each specialist analyze independently, have the bull and bear sides challenge each other, then have the PM make the final call with quantitative validation.

1. **XAI Pre-Screen** — Quantitative distress estimation + Shapley explanations before any LLM calls
2. **Parallel Analysis** — Four agents analyze simultaneously with XAI context injected
3. **Adversarial Debate** — Long vs Short with structured rebuttals; Risk Manager provides sizing commentary
4. **PM Synthesis** — Portfolio Manager weighs all evidence, produces T signal + committee memo
5. **Portfolio Optimization** — Selected optimizer (BL, HRP, Mean-Var, Min-Var, Risk Parity, or Equal Weight) computes real weights
6. **Signal Persistence** — Every signal stored in SQLite for backtesting and analytics

Each pipeline node runs at a task-appropriate "temperature" — an LLM setting that controls how creative vs. deterministic the output is. Data extraction uses low temperature (stick to facts), while the PM uses higher temperature (connect themes, weigh trade-offs).

---

### Explainable AI (XAI) Pre-Screen

<details>
<summary>Five-step explainable AI procedure that runs before the LLM agents, giving them quantitative context</summary>

Before any AI agent writes a word, the system runs a quantitative check: how likely is this company to face financial distress, and what's driving that risk? The results — along with which financial metrics matter most and why — are passed to the agents so their analysis starts from a data-grounded baseline rather than pure LLM reasoning.

```
┌─ PHASE 1 ── Estimation ─────────────────────────────────┐
│  Fundamentals ──► PFD Estimation ──► Risk Factors        │
│  (12 features)    (Altman Z /        (Shapley            │
│                    XGBoost)           decomposition)      │
└───────────────────────────────┬───────────────────────────┘
                                ▼
┌─ PHASE 2 ── Screening + Return ─────────────────────────┐
│  Distress Screen ──► Expected Return ──► Return Drivers  │
│  (PFD > threshold?)  (ER = (1-PFD)      (Shapley        │
│                       × yield)            decomposition)  │
└───────────────────────────────┬───────────────────────────┘
                                ▼
                        ┌───────────────┐
                        │    AGENTS     │
                        └───────────────┘
```

Based on Sotic & Radovanovic (2024), "Explainable AI in Finance" ([doi:10.20935/AcadAI8017](https://doi.org/10.20935/AcadAI8017)).

| Step | What | Method |
|------|------|--------|
| 1 | **PFD Estimation** | Altman Z-Score (zero-config) or trained XGBoost |
| 2 | **Risk Factor Identification** | Shapley value decomposition of distress drivers |
| 3 | **Distress Screening** | PFD threshold flag (configurable, default 50%) |
| 4 | **Expected Return** | ER = (1 - PFD) x earnings yield proxy |
| 5 | **Return Driver Analysis** | Shapley value decomposition of profitability factors |

In plain terms, the system estimates financial distress likelihood, identifies which metrics drive that estimate, and passes this context to the analyst agents before they begin.

**Two-tier model architecture**: Altman Z-Score always works with zero configuration using proxy variables from fundamentals. When a trained XGBoost model exists in `xai/artifacts/`, it activates automatically.

**Shapley values without dependencies**: The built-in Shapley calculator (`xai/shapley.py`) provides exact analytical Shapley values for the Z-Score linear model and permutation-based approximate Shapley for any other model. The optional `shap` library enhances this with waterfall plots, but explanations always work even in a minimal install.

**Dual explainability**: Feature-level Shapley values (quantitative: "debt_to_equity contributed +0.04 to distress risk") plus agent-level attribution (qualitative: "the risk manager's bear case shifted conviction by -1.2").

```bash
pip install -e ".[xai]"           # Optional: XGBoost + shap + matplotlib
python -m xai.train --data data.csv --target is_distressed  # Train custom model
```

</details>

---

### Adversarial Debate

<details>
<summary>Structured Long vs Short debate with Risk Manager sizing commentary</summary>

Long and Short analysts challenge each other through structured rebuttals over configurable rounds (default: 2). The Risk Manager provides sizing commentary on each round — adjusting position structure, stop-loss levels, and stress scenarios based on the arguments presented.

Convictions update based on evidence. The PM's conviction change map tracks exactly which argument shifted which agent's score and by how much. Debate always runs (convergence is noted, not skipped) so the user can observe the reasoning process in the Debate tab.

</details>

---

### Volatility Intelligence

<details>
<summary>Quantitative vol analysis auto-injected into all agent prompts</summary>

During data gathering, the system computes a volatility profile for the stock — how much the price has been moving historically, what the options market implies about future moves, and whether current conditions look calm or stressed. These signals are injected into each agent's prompt so they can reason about vol quantitatively rather than guessing.

| Signal | What it measures |
|--------|-----------------|
| **Realized vol** | Multi-window historical vol (10/30/60/90d), downside vol, vol ratio, percentile rank, regime classification (low/normal/elevated/crisis) |
| **Implied vol surface** | Strike-by-maturity IV grid from Heston stochastic vol model, ATM term structure, skew by maturity |
| **IV vs HV** | Compares 3-month implied vol to 30-day realized. Signals: `iv_elevated`, `iv_slight_premium`, `iv_fair`, `iv_cheap`, `iv_very_cheap` |
| **Skew flags** | Squeeze risk from inverted 25-delta skew, extreme put skew, vol backwardation |
| **Regime multiplier** | Maps vol regime to a position-sizing scalar [0.3 - 1.3] — the optimizer scales conviction confidence accordingly |

Each agent receives the relevant subset: the Risk Manager uses vol for sizing and tail risk, the Short Analyst gets squeeze and borrow context, the Macro Analyst gets regime classification, and the PM gets IV vs HV and downside vol for its required output fields.

</details>

---

### Portfolio Optimizer

<details>
<summary>Seven optimization strategies selectable via UI dropdown — from views-based (Black-Litterman) to ensemble cross-strategy analytics</summary>

After the PM produces its recommendation, the optimizer converts qualitative conviction into concrete portfolio weights across the target stock, its sector peers, and broad market indices. You choose the optimization method from a dropdown in the UI — each uses the same universe and covariance infrastructure but differs in how weights are computed.

| Strategy | How it works | Uses PM views? |
|----------|-------------|----------------|
| **Black-Litterman** (default) | Adjusts market-implied expected returns using PM's alpha view and conviction-scaled confidence, then runs max-Sharpe optimization | Yes |
| **Hierarchical Risk Parity** | Clusters assets by correlation structure and allocates via inverse-variance weighting down the tree — no covariance inversion, more robust to estimation error | No |
| **Mean-Variance (Max Sharpe)** | Classic Markowitz — uses historical expected returns directly, finds the tangency portfolio on the efficient frontier | No |
| **Minimum Variance** | Minimizes portfolio variance regardless of expected returns — purely risk-based, no return estimates needed | No |
| **Risk Parity (ERC)** | Equal risk contribution — each asset contributes the same fraction of total portfolio risk, solved via SLSQP | No |
| **Equal Weight (1/N)** | Simple 1/N allocation — surprisingly hard to beat out-of-sample and useful as a baseline | No |
| **Ensemble (All Strategies)** | Runs all 6 strategies on shared inputs, produces blended allocation + cross-strategy analytics | Mixed |

In plain terms, Black-Litterman incorporates the PM's conviction into market-consensus expectations, HRP uses correlation clustering instead of covariance inversion, and Risk Parity equalizes each asset's contribution to portfolio risk. The simpler methods (min-variance, equal weight) serve as useful benchmarks. Ensemble runs everything and compares.

```
┌─ SHARED ── Universe + Covariance (built once) ────────────┐
│  Universe Construction    Covariance Estimation             │
│  (target + 5 peers        (Ledoit-Wolf shrinkage)          │
│   + ETF + SPY)                                             │
└──────────────┬────────────────┬────────────────────────────┘
               └────────┬───────┘
                        ▼
┌─ STRATEGY ── (selected via UI dropdown) ──────────────────┐
│  BL / HRP / Mean-Var / Min-Var / Risk Parity / Equal Wt   │
│  ── OR ── Ensemble (runs all 6 on shared inputs)           │
└───────────────────────────────┬────────────────────────────┘
                                ▼
┌─ SHARED ── Analytics ─────────────────────────────────────┐
│  Sharpe / Sortino / Factor Betas (OLS) / MCTR             │
│  Ensemble adds: Consensus Matrix, Divergence Flags,        │
│  Blended Allocation, Layered Narrative                     │
└───────────────────────────────────────────────────────────┘
```

**Ensemble mode** is what most sophisticated funds use — layered approaches where different strategies serve different roles:

| Strategy | Ensemble Role | Blend Weight |
|----------|--------------|-------------|
| Black-Litterman | Core Allocation — PM views-driven, highest information content | 35% |
| Risk Parity | Sanity Check — risk-balanced, no return assumptions | 25% |
| Minimum Variance | Defensive Overlay — lowest-vol positioning | 20% |
| HRP | Reference — correlation-clustered alternative | 10% |
| Mean-Variance | Reference — classical Markowitz benchmark | 5% |
| Equal Weight | Reference — naive diversification baseline | 5% |

Ensemble output includes: a strategy comparison table (side-by-side vol, Sharpe, Sortino, HHI), a weight consensus matrix showing where strategies agree or disagree, divergence flags for tickers with high agreement (std < 2pp) or disagreement (std > 10pp), a blended allocation with risk contributions, and a layered narrative comparing core vs sanity check vs defensive overlay positioning.

All strategies share post-optimization analytics: Sharpe/Sortino ratios, OLS factor betas with t-stats, and marginal contribution to risk (MCTR). The report adapts to the selected method — BL results include a heuristic-vs-computed comparison table; non-BL results show a simpler risk metrics view.

If any optimizer fails (e.g., insufficient price data, solver non-convergence), the pipeline continues gracefully with heuristics as primary reference. Mean-Variance falls back to min-volatility when no asset's expected return exceeds the risk-free rate. In ensemble mode, individual strategy failures are logged and skipped — the ensemble continues with remaining strategies.

Configure the default via `.env` (`OPTIMIZER_METHOD=ensemble`) or select at runtime from the UI dropdown.

</details>

---

### Signal Analytics + Backtesting

<details>
<summary>Full analytics stack for evaluating signal quality over time</summary>

Every recommendation the system produces is stored in a database. Over time, you can measure whether higher-conviction calls actually produce better returns, how quickly the signal decays, and which agents contributed most to the final call — the same kind of post-trade attribution a fund would run on its analysts.

```
┌─ PERSISTENCE ────────────────────────────────────────────┐
│  Committee Signal ──► SQLite DB                           │
│  (T, conviction, direction)                               │
└───────────────────────────────┬───────────────────────────┘
         ┌──────┬──────┬────────┼────────┬──────┐
         ▼      ▼      ▼        ▼        ▼      ▼
┌─ ANALYTICS ──────────────────────────────────────────────┐
│  Backtest       Calibration    Alpha Decay                │
│  (returns,      (conviction    (IC by                     │
│   P&L)           buckets)       horizon)                  │
│  Benchmark      Portfolio      Attribution                │
│  (vs SPY,       (multi-asset   (agent                     │
│   momentum)      weights)       decomposition)            │
└──────────────────────────────────────────────────────────┘
```

| Module | What it does |
|--------|-------------|
| **Signal Persistence** | SQLite database stores every signal with conviction, direction, T signal, and BL optimizer output |
| **Historical Backtest** | Fills realized 1d/5d/10d/20d/60d returns, computes directional P&L, Sharpe, win rate, max drawdown |
| **Calibration** | Bins signals by conviction level, measures hit rate and avg return per bucket — answers "is conviction 8/10 better than 6/10?" |
| **Alpha Decay** | Information coefficient (IC) at each horizon — identifies optimal holding period and signal half-life |
| **Benchmark Comparison** | Committee signals vs SPY buy-and-hold, always-long, and momentum (T-signal ranked) |
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

</details>

---

### Agent Memory + Post-Trade Reflection

<details>
<summary>Agents learn from past calls via a reflection-and-retrieval feedback loop</summary>

After realized returns come in, the system generates a reflection for each agent on each signal — was the call correct, what worked, what failed, and was conviction appropriately calibrated? These reflections are stored in SQLite alongside the signals.

On future runs, each agent's prompt is augmented with relevant past reflections retrieved via BM25 similarity search. If the Risk Manager is analyzing a biotech stock, it retrieves its own prior reflections on similar tickers — lessons like "high conviction was misplaced on XYZ" or "correctly flagged tail risk on ABC." This gives agents a form of institutional memory across sessions.

Reflections can be generated rule-based (no LLM cost) or LLM-powered (richer lessons). The retrieval uses `rank_bm25` and degrades gracefully if the library is not installed.

</details>

---

### T Signal

<details>
<summary>Single scalar in [-1, 1] encoding direction and confidence</summary>

```
T = direction x C

direction in {-1, +1}     (-1 = short, +1 = long)
C = e + (1 - e)(1 - H)    (entropy-adjusted certainty)
H = 1 - raw_confidence     (proxy for entropy)
e = 0.01                   (floor)
```

In plain terms, T combines the direction of the recommendation (long or short) with conviction strength into a single number between −1 and +1.

`T = +0.85` means strong long conviction with high certainty. `T = -0.40` means moderate short conviction with moderate certainty.

Since the LLM interface is provider-agnostic (`callable(str) -> str`), token-level entropy is unavailable. The PM's self-reported confidence serves as proxy. Approach adapted from Darmanin & Vella, [arXiv:2508.02366](https://arxiv.org/abs/2508.02366).

</details>

---

### Per-Node Temperature Routing

<details>
<summary>Task-appropriate temperature per pipeline node instead of a single global setting</summary>

| Node | Temperature | Reasoning |
|------|------------|-----------|
| `gather_data` | 0.1 | Factual retrieval — one "right" answer |
| `run_xai_analysis` | 0.0 | Deterministic computation (no LLM calls) |
| `run_sector_analyst` | 0.5 | Balanced exploration of thesis points |
| `run_short_analyst` | 0.5 | Explore short thesis scenarios |
| `run_risk_manager` | 0.5 | Explore unlikely-but-possible tail risks |
| `run_macro_analyst` | 0.5 | Balanced macro analysis |
| `run_debate_round` | 0.5 | Adversarial — consider edge arguments |
| `run_portfolio_manager` | 0.7 | Narrative flow, connecting themes |
| `run_optimizer` | 0.0 | Deterministic computation (no LLM calls) |

Works across all providers (Anthropic, Google, OpenAI, HuggingFace, Ollama). Override per-node via `settings.task_temperatures`.

</details>

---

### Per-Node Model Routing

<details>
<summary>Run different LLM models at different pipeline stages to balance cost and quality</summary>

Independent of temperature, each pipeline node can use a different LLM model — or even a different provider. A common setup: use a fast, cheap model (e.g., `gemini-2.0-flash`) for data gathering and analyst agents, but route the PM synthesis to a more capable model (e.g., `claude-sonnet-4-20250514`) where reasoning quality matters most.

Configure via `.env` or `settings.task_models`:

```
TASK_MODELS={"run_sector_analyst": "google:gemini-2.0-flash", "run_portfolio_manager": "anthropic:claude-sonnet-4-20250514"}
```

Format is `provider:model_name`. If no colon, the current provider is used with just the model name swapped. Models with the same spec share a single rate limiter instance.

</details>

---

## Architecture

```
app.py (Gradio UI) ──┬──► app_lib/        (Model Factory + Formatters)
                     ├──► orchestrator/   (LangGraph StateGraph)
                     │      ├──► agents/      (5 AI Agents) ──► tools/
                     │      ├──► xai/         (Explainable AI)
                     │      └──► optimizer/   (BL / Ensemble / 4 others) ──► tools/
                     └──► tools/          (Data + Aggregation)

api/ (FastAPI) ──► orchestrator/
backtest/ (Analytics) ──► tools/
config/ (Settings) ···► app.py, orchestrator/, agents/
```

<details>
<summary><b>Project structure</b></summary>

```
multi-agent-investment-committee/
├── app.py                       # Gradio UI + pipeline orchestration
├── app_lib/
│   ├── model_factory.py         # LLM provider factories, rate limiting, timeout
│   └── formatters.py            # Report formatting + preview builders
├── agents/
│   ├── base.py                  # BaseAgent ABC + schemas + JSON extraction
│   ├── sector_analyst.py        # Long Analyst — bull case + sentiment
│   ├── short_analyst.py         # Short Analyst — short thesis + alpha/beta classification
│   ├── risk_manager.py          # Risk Manager — sizing, structuring, stress scenarios
│   ├── macro_analyst.py         # Macro Strategist — portfolio context + vol regime
│   └── portfolio_manager.py     # PM synthesis + conviction change map + T signal
├── orchestrator/
│   ├── graph.py                 # LangGraph StateGraph (full + phase1 + phase2)
│   ├── nodes.py                 # Node functions + T signal computation
│   ├── state.py                 # CommitteeState TypedDict with reducers
│   ├── committee.py             # CommitteeResult + ConvictionSnapshot
│   ├── temperature.py           # Per-node temperature routing
│   ├── model_routing.py         # Per-node model selection
│   ├── memory.py                # Session memory store + BM25 retrieval
│   └── reasoning_trace.py       # Trace rendering
├── optimizer/
│   ├── bl_optimizer.py          # Shared pipeline (universe → cov → strategy → analytics) + BL wrapper
│   ├── strategies.py            # 6 strategies: BL, HRP, Mean-Var, Min-Var, Risk Parity, Equal Weight
│   ├── ensemble.py              # Ensemble meta-strategy — runs all 6, blends, consensus + divergence
│   ├── node.py                  # LangGraph node — reads optimizer_method from config
│   ├── models.py                # OptimizationResult, EnsembleResult, OptimizerFallback, + data models
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
│   ├── volatility_surface.py    # Implied vol surface (Heston/CEV models)
│   ├── vol_intelligence.py      # Vol regime signals + agent injection
│   └── data_providers/          # Provider abstraction (yahoo/bloomberg/ib)
├── evals/
│   ├── schemas.py               # EvalScenario, GroundTruth, GradingResult, LikertLevel
│   ├── grader.py                # 6-dimension scoring engine
│   ├── adversarial.py           # Context injection + robustness grading
│   ├── llm_judge.py             # LLM-as-judge evaluation
│   ├── loader.py                # YAML scenario discovery
│   ├── runner.py                # Scenario execution
│   ├── reporter.py              # JSON/markdown report generation
│   ├── risk_profiles.py         # Risk tolerance profiles for eval scenarios
│   ├── rag_metrics.py           # RAG retrieval quality metrics
│   ├── scenarios/               # Ground-truth + adversarial YAML scenarios
│   └── rubrics/                 # Scoring rubric definitions
├── backtest/
│   ├── database.py              # SQLite signal persistence (3 tables, indexes)
│   ├── models.py                # SignalRecord, BacktestResult, CalibrationBucket, etc.
│   ├── runner.py                # Historical backtest engine (fill returns, compute P&L)
│   ├── calibration.py           # Conviction-return calibration analysis
│   ├── alpha_decay.py           # Information coefficient at multiple horizons
│   ├── benchmark.py             # Signals vs SPY, always-long, momentum
│   ├── portfolio.py             # Multi-asset portfolio construction
│   ├── explainability.py        # Agent attribution decomposition
│   ├── persist.py               # Shared signal persistence utility
│   ├── reflection.py            # Post-trade reflection + agent memory
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
├── scripts/
│   ├── accumulate_signals.py    # Batch signal accumulation across tickers
│   ├── inter_rater_reliability.py # Inter-rater reliability analysis
│   └── run_permutations.py      # Agent permutation testing
├── config/
│   └── settings.py              # Pydantic settings with .env + per-node temperature overrides + XAI config
├── tests/                       # 687 tests
├── .env.example                 # Environment variable template
├── CHANGELOG.md                 # Version history
└── pyproject.toml
```

</details>

<details>
<summary><b>Data providers</b></summary>

```
            BaseDataProvider (ABC)
           ┌────────┼────────┐
           │        │        │
           ▼        ▼        ▼
    YahooProvider  BloombergProvider  IBProvider
    (default,      (requires          (requires
     zero config)   blpapi)            ib_insync)

    ProviderFactory.get_provider(name) → BaseDataProvider
```

Market data defaults to yfinance. The `tools/data_providers/` package implements a provider abstraction (`BaseDataProvider` ABC) with a factory pattern. Bloomberg and Interactive Brokers providers are available as optional installs:

```bash
pip install -e ".[bloomberg]"   # Bloomberg Terminal
pip install -e ".[ibkr]"        # Interactive Brokers
```

</details>

<details>
<summary><b>Structured output hardening</b></summary>

LLMs sometimes return malformed output — broken JSON, extra text outside the structure, or incomplete responses. The system handles this with multiple layers of repair so the pipeline doesn't break mid-run.

All agents use a progressive JSON extraction pipeline with 7 repair strategies (markdown block removal, brace boundary, trailing comma fix, quote repair, unbalanced brace repair).

- **Retry-with-feedback** — When extraction fails, the model receives the error and its truncated response, and is prompted to produce clean JSON.
- **Ollama JSON mode** — Prompts requesting JSON output automatically trigger Ollama's `format: "json"` constraint, preventing free-form text responses from smaller models.

</details>

---

## REST API

<details>
<summary>FastAPI endpoint for programmatic access</summary>

```bash
pip install -e "."
uvicorn api.main:app --reload --port 8000
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analyze` | POST | Run full committee pipeline, returns signal + stores in DB |
| `/signals` | GET | List stored signals (filter by ticker) |
| `/signals/{id}` | GET | Get specific signal by ID |
| `/backtest` | POST | Run backtest on stored signals |
| `/portfolio` | GET | Build portfolio snapshot from latest signals |
| `/fill-returns` | POST | Fill realized returns for stored signals |
| `/health` | GET | Health check with signal stats |

</details>

---

## Eval Harness

<details>
<summary>Ground-truth evaluation framework for grading committee output against known historical scenarios</summary>

The eval harness answers the question: "If we had run this system before a known event (e.g., SVB collapse, NVDA's AI run), would it have gotten the direction right and identified the key risks?" Scenarios with known outcomes are replayed and graded on six dimensions.

**Grading dimensions** (6 dimensions, 100 total points):
- Direction accuracy (25) — did the committee get the direction right?
- Conviction calibration (15) — was conviction appropriately scaled?
- Risk identification (20) — did the bear case find the key risks?
- Fact coverage (15) — did the bull case surface the important evidence?
- Reasoning quality (15) — did agents update convictions, produce substantive rebuttals?
- Adversarial robustness (10) — did agents detect injected/tainted data?

Each dimension maps to a 5-point scale (Fail/Poor/Adequate/Good/Excellent) with dimension-specific anchor definitions for interpretability.

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

</details>

---

## Execution Modes

| Mode | Description |
|------|-------------|
| **Full Auto** | Runs entire pipeline end-to-end |
| **Review Before PM** | Two-phase HITL — review analyst outputs + debate, optionally add PM guidance, then PM synthesizes |

**Document upload:** Upload up to 5 research documents (PDF, DOCX, TXT) as supplementary context for all agents. Files are token-aware chunked and injected into agent prompts. Requires `pip install -e ".[docs]"` for PDF/DOCX support.

---

## Tech Stack

[LangGraph](https://github.com/langchain-ai/langgraph) · [Gradio](https://www.gradio.app/) · [Pydantic](https://docs.pydantic.dev/) · [pypfopt](https://github.com/robertmartin8/PyPortfolioOpt) · [FastAPI](https://fastapi.tiangolo.com/) · [yfinance](https://github.com/ranaroussi/yfinance) · [pytest](https://docs.pytest.org/) · [SHAP](https://github.com/shap/shap) · [XGBoost](https://xgboost.readthedocs.io/)

LLM providers: Anthropic, Google, OpenAI, HuggingFace, Ollama

---

<details>
<summary><b>Related Work</b></summary>

This system draws on academic research in multi-agent financial AI, explainability, and adversarial robustness:

- **AlphaAgents** (Zhao et al., 2025) — BlackRock's multi-agent LLM framework for equity portfolio construction using debate between Fundamental, Sentiment, and Valuation agents. Directly informs our committee debate architecture and risk tolerance profile design (`evals/risk_profiles.py`). [arXiv:2508.11152](https://arxiv.org/abs/2508.11152)
- **AuditAgent** (Bai et al., 2025) — Multi-agent framework for cross-document fraud evidence discovery with Bayesian prior modeling for audit focus narrowing. Informs our multi-expert reasoning pattern and eval grading methodology. [arXiv:2510.00156](https://arxiv.org/abs/2510.00156)
- **Red-Teaming Financial AI / FinJailbreak** (Li, 2026) — 1,250 adversarial prompts across 5 financial malfeasance categories, with Financial Constitutional Fine-Tuning (FCFT) defense. Inspires our red-team adversarial scenarios in `evals/scenarios/redteam_*.yaml`. [AAAI 2026]
- **Explainable AI in Finance** (Sotic & Radovanovic, 2024) — Comprehensive taxonomy of XAI methods for financial applications. Directly informs the `xai/` module's Shapley value architecture and multi-stakeholder explainability design. [doi:10.20935/AcadAI8017]
- **XAI for SME Investment** (Babaei & Giudici, 2025) — Dual-component XAI framework combining credit risk and expected return models with SHAP explanations. Validates our approach of integrating Shapley-based explainability into the investment pipeline. [Expert Systems with Applications, 2025]
- **FLaME** (Matlin et al., 2025) — Holistic financial NLP evaluation across 6 task categories. Provides taxonomy for RAG metrics evaluation (`evals/rag_metrics.py`). [arXiv:2506.15846](https://arxiv.org/abs/2506.15846)

</details>

<details>
<summary><b>Version History</b></summary>

- v1: Multi-agent committee with ThreadPoolExecutor parallelism
- v2: LangGraph orchestration, dynamic tool calling, HITL
- v3: Sentiment extraction, portfolio construction heuristics, T signal
- v3.1: Return decomposition, Sharpe/Sortino, NMV sizing
- v3.2: Document KB upload
- v3.3: Conviction timeline with LLM-generated rationale
- v3.4: Small-LLM resilience, model selection, Ollama improvements
- v3.5: T signal with citation, conviction tracker redesign
- v3.6: Eval harness, adversarial testing framework, 5-level grading scale, data provider abstraction
- v3.7: Black-Litterman portfolio optimizer, per-node temperature routing
- v3.8: Signal persistence (SQLite), historical backtesting, calibration analysis, alpha decay, benchmark comparison, multi-asset portfolio, explainability/attribution, FastAPI REST endpoint, structured output hardening (retry + Ollama JSON mode)
- v3.9: XAI pre-screen module — Altman Z-Score/XGBoost distress estimation, built-in Shapley value decomposition (exact linear + permutation-based), expected return computation, dual explainability (feature-level + agent-level). Based on Sotic & Radovanovic (2024)
- v4.0: Five-agent architecture — dedicated Short Analyst (alpha/beta short classification, borrow assessment, event path), Risk Manager refocused on sizing/structuring (position structure, stop-loss, stress scenarios, correlation flags), Long vs Short adversarial debate with Risk Manager commentary, PM conviction change map, 4-way parallel fan-out
- v4.1: Per-node model routing, agent memory with BM25 retrieval, volatility surface generation (Heston/CEV), vol intelligence pipeline, realized vol signals, IV vs HV assessment, app.py modularization (`app_lib/` package), shared context injection helper, Alpha Vantage provider
- v4.2: Ensemble multi-strategy optimizer — runs all 6 strategies on shared universe/covariance, produces blended allocation (BL 35% / RP 25% / MinVar 20% / HRP 10% / MV 5% / EW 5%), weight consensus matrix, divergence flags, layered narrative (core vs sanity check vs defensive overlay), strategy comparison table with HHI concentration metric. 687 tests.

</details>

---

## Contributing

Contributions welcome. Areas for improvement:
- Additional agent roles and analytical perspectives
- New optimizer strategies and portfolio construction methods
- Extended eval scenarios and ground truth cases
- Enhanced vol intelligence and signal analytics

## Status

This project is under active, ongoing development. Core pipeline, agent architecture, optimizer, and XAI module are stable. New agent capabilities, optimization strategies, and evaluation scenarios are added as research needs evolve.

## License

MIT
