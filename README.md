---
title: Multi-Agent Investment Committee
emoji: "\U0001F4CA"
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.0.0
app_file: app.py
pinned: false
license: apache-2.0
tags:
  - multi-agent
  - agentic-ai
  - investment
  - reasoning
  - langgraph
  - reinforcement-learning
---

# Multi-Agent Investment Committee

Four AI agents analyze a ticker in parallel, debate each other's theses, and produce a structured committee memo with a trading signal.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-200%20passing-brightgreen.svg)](tests/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

> **Not financial advice.** All output is AI-generated. Do not use for investment decisions.

---

## How It Works

The pipeline runs in three phases:

1. **Parallel Analysis** -- Three agents analyze the same ticker simultaneously:
   - **Sector Analyst** (bull case) -- sentiment extraction, return decomposition, catalyst identification
   - **Risk Manager** (bear case) -- causal chain reasoning (1st/2nd/3rd order effects), alpha challenge
   - **Macro Strategist** (top-down) -- vol regime classification, net exposure, sizing frameworks

2. **Adversarial Debate** -- Bull and bear challenge each other through structured rebuttals. Convictions update based on evidence.

3. **PM Synthesis** -- Portfolio Manager weighs all evidence, applies sizing heuristics, produces a committee memo with recommendation and T signal.

```
User Input (ticker + guidance + docs)
       |
       v
  Data Gathering (yfinance, RSS, derived metrics, uploaded docs)
       |
   +---+---+--------+
   v   v            v
 Sector  Risk     Macro
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
  Committee Memo
```

Each agent follows a THINK -> PLAN -> ACT -> REFLECT loop. Every step is captured in a reasoning trace visible in the UI.

~20 LLM API calls per analysis. 90-120 seconds wall clock depending on provider.

---

## Agents

**Sector Analyst** -- Builds the bull case. Extracts per-headline sentiment (bullish/bearish/neutral with signal strength). Decomposes expected return into industry return and idiosyncratic alpha. Estimates heuristic Sharpe and Sortino ratios.

**Risk Manager** -- Builds the bear case. Traces how primary risks cascade into 2nd and 3rd order effects. Challenges whether claimed alpha is genuine idiosyncratic return or disguised factor/sector beta. Produces active short pitches when warranted.

**Macro Strategist** -- Sets portfolio-level guardrails. Classifies vol regime (low/normal/elevated/crisis). Recommends net exposure direction, sizing method (proportional / risk parity / mean-variance / shrunk mean-variance), and portfolio vol target. The PM operates within these constraints.

**Portfolio Manager** -- Chairs the committee. Weighs bull thesis, bear risks, macro context, and debate outcomes. Validates return decomposition after hearing both sides. Computes the T signal. Outputs a structured memo with recommendation, sizing rationale, conviction triggers, and factor exposures.

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
pytest tests/ -v   # 200 tests
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
├── app.py                       # Gradio UI
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
│   ├── memory.py                # Session memory store
│   └── reasoning_trace.py       # Trace rendering
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
├── config/
│   └── settings.py              # Pydantic settings with .env
├── tests/                       # 200 tests
└── pyproject.toml
```

---

## Tech Stack

- [LangGraph](https://github.com/langchain-ai/langgraph) -- StateGraph orchestration with fan-out/fan-in and conditional edges
- [Gradio](https://www.gradio.app/) -- UI with 8 analysis tabs, T signal gauge, document upload
- [Pydantic](https://docs.pydantic.dev/) -- Output validation for all agent schemas
- [yfinance](https://github.com/ranaroussi/yfinance) -- Market data
- [pytest](https://docs.pytest.org/) -- 200 tests with mock LLM fixtures

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

---

## License

Apache 2.0

---

![Python](https://img.shields.io/badge/python-3.11+-3776AB?style=flat&logo=python&logoColor=white)

![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C?style=flat&logo=langchain&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-F97316?style=flat&logo=gradio&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-E92063?style=flat&logo=pydantic&logoColor=white)
![Anthropic](https://img.shields.io/badge/Anthropic-191919?style=flat&logo=anthropic&logoColor=white)
![Google](https://img.shields.io/badge/Gemini-4285F4?style=flat&logo=google&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=flat&logo=openai&logoColor=white)
![Yahoo Finance](https://img.shields.io/badge/Yahoo_Finance-6001D2?style=flat&logo=yahoo&logoColor=white)
![Bloomberg](https://img.shields.io/badge/Bloomberg-000000?style=flat&logo=bloomberg&logoColor=white)
![Interactive Brokers](https://img.shields.io/badge/Interactive_Brokers-D71920?style=flat)
