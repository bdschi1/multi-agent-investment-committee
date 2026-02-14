# Multi-Agent Investment Committee

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-111%20passing-brightgreen.svg)](tests/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Gradio](https://img.shields.io/badge/ui-Gradio%206.0-orange.svg)](https://www.gradio.app/)

![Committee Analysis UI](/assets/ui-screenshot.png)

Four autonomous AI agents that reason, debate, and synthesize investment theses through structured adversarial analysis. Not a chatbot wrapper — agents that *think, plan, act, and reflect* through multi-step reasoning chains with real-time tool calling, portfolio construction heuristics, and an RL-ready trading signal.

> [!TIP]
> **Try it live** on [Hugging Face Spaces](https://huggingface.co/spaces/bdschi1/multi-agent-investment-committee) — no setup required.
> Enter a ticker, pick a provider, and watch four agents analyze, debate, and decide in real time.

The project provides a complete multi-agent orchestration pipeline built on LangGraph, with a Gradio UI that exposes every step of the reasoning process. It supports five LLM providers out of the box and can run fully local with Ollama.

The pipeline is divided into three phases:

**Phase 1: Parallel Analysis** — three specialist agents analyze the same ticker simultaneously, each from a different angle:
- **Sector Analyst** builds the bull case with sentiment extraction, return decomposition, and catalyst identification
- **Risk Manager** builds the bear case with causal chain reasoning (1st → 2nd → 3rd order effects) and active short pitches
- **Macro Strategist** provides top-down context with vol regime classification, portfolio directionality, and sizing frameworks

**Phase 2: Adversarial Debate** — the bull and bear agents challenge each other's theses through structured rebuttals, forcing conviction updates backed by evidence

**Phase 3: PM Synthesis** — the Portfolio Manager weighs all evidence, applies quantitative sizing heuristics, and produces a structured committee memo with a trading signal

In addition to this, a working [Gradio](https://www.gradio.app/) UI is provided with 8 analysis tabs, conviction timeline tracking, document upload, copy/download export, and a human-in-the-loop review mode.

---

## 🎞 Overview

> [!WARNING]
> This project is a **demonstration of multi-agent AI architecture** — not financial advice.
> All analyses are AI-generated. Do not use for actual investment decisions.

### Why this exists

Most "AI agent" demos are thin wrappers around a single prompt. They don't show what makes agentic systems interesting: agents that *disagree*, update their beliefs under pressure, and produce structured outputs you can actually use downstream.

This project was built to demonstrate what production-grade multi-agent orchestration looks like when applied to a domain that demands rigor — investment analysis. Every agent follows a think → plan → act → reflect loop, calls tools dynamically, and produces Pydantic-validated structured output. The bull and bear agents then debate each other before a Portfolio Manager synthesizes the final decision.

The result is a system where you can trace every step of the reasoning, see exactly why an agent changed its conviction, and feed the output directly into a downstream RL agent via the T signal.

### How it evolved

The first version was a straightforward multi-agent pipeline with ThreadPoolExecutor parallelism and manual orchestration. It worked, but the control flow was implicit and hard to extend.

v2 ported everything to **LangGraph StateGraph** — parallel fan-out via conditional edges, a debate loop with convergence detection, and two-phase human-in-the-loop execution. Dynamic tool calling was added with per-agent budgets and a 10-tool registry.

v3 deepened the domain expertise: news sentiment extraction, portfolio construction heuristics (return decomposition, Sharpe/Sortino, NMV sizing methods), a trading-fluent PM who thinks in vol surfaces and factor tilts, and the T signal — a single scalar that compresses the committee's view into an RL-consumable feature.

> The current version runs **~20 LLM API calls** per analysis, completing in **90-120 seconds** depending on provider and model.

---

## 📐 Architecture

```
User Input (ticker + expert guidance + research docs)
       │
       ▼
┌─────────────────┐
│  Data Gathering  │  ← yfinance, RSS feeds, derived metrics
│  + KB Processing │  ← PDF/DOCX/TXT → chunk → inject
└────────┬────────┘
         │
    ┌────┼────────────┐
    ▼    ▼            ▼
┌────────┐ ┌────────┐ ┌────────────┐
│ Sector │ │ Risk   │ │ Macro      │   ← Parallel (3 agents)
│Analyst │ │Manager │ │Strategist  │
│ (Bull) │ │ (Bear) │ │(Top-Down)  │
└───┬────┘ └───┬────┘ └───┬────────┘
    │          │           │
    ▼          ▼           │
┌──────────────────┐       │
│ Adversarial      │       │
│ Debate           │       │
│ (Rebuttals)      │       │
└────────┬─────────┘       │
         │                 │
         ▼                 ▼
┌────────────────────────────┐
│ Portfolio Manager          │   ← Weighs all evidence
│ (Synthesis + T Signal)     │   ← BUY/SELL/HOLD + T ∈ [-1, 1]
└────────────┬───────────────┘
             │
             ▼
       Committee Memo
       (Structured Output)
```

The design of the committee allows easy extension of both the agent roster and the orchestration logic. Some key architectural decisions:

* **LangGraph StateGraph** for orchestration — parallel fan-out, conditional debate routing, and typed state with reducer functions for clean fan-in merging
* **Provider abstraction** — the model interface is `callable(str) → str`, making it trivial to swap between Anthropic, Google, OpenAI, HuggingFace, or Ollama at runtime
* **Config injection via RunnableConfig** — non-serializable objects (model, callbacks, tool registry) flow through LangGraph's config system, keeping the state serializable
* **Pydantic output schemas** — every agent output is validated against strict schemas with backward-compatible defaults, so downstream consumers never get surprises

Main building blocks:

* **Agents** are defined in `agents/<role>.py`. Each agent extends `BaseAgent` with a think → plan → act → reflect loop. Agents are model-agnostic and tool-aware but don't depend on the orchestration layer.
* **Nodes** are defined in `orchestrator/nodes.py`. Each node wraps an agent's `.run()` method and handles state reads/writes. The T signal computation also lives here.
* **The graph** is defined in `orchestrator/graph.py` with three variants: full pipeline, phase-1-only (for HITL), and phase-2-only (PM synthesis after human review).
* **Tools** are registered in `tools/registry.py` with per-agent call budgets and automatic argument coercion.

---

## 🧠 The Four Agents

### Sector Analyst — Bull Case + Sentiment

Builds the affirmative thesis with conviction scoring, catalysts, and evidence. Processes every news headline through a sentiment extraction pipeline — classifying each as bullish/bearish/neutral with signal strength, computing aggregate sentiment, and flagging sentiment-price divergences (an alpha signal). Performs heuristic return decomposition: price target, total return, industry return stripped out, and idiosyncratic alpha isolated — with Sharpe and Sortino estimates.

### Risk Manager — Bear Case + Causal Chains

Adversarial analyst with causal chain reasoning — traces how primary risks cascade into 2nd and 3rd order effects. Seeks non-obvious risks the market is underpricing. Produces active short pitches when warranted. Challenges the bull case's return decomposition — stress-tests whether claimed "alpha" is genuine idiosyncratic return or disguised factor/sector beta.

### Macro Strategist — Top-Down + Portfolio Guardrails

Global macro context and portfolio-level strategy. Classifies the vol regime (low/normal/elevated/crisis), recommends net exposure direction, assesses sector rotation and correlation regime, and provides quantitative sizing frameworks (proportional, risk parity, mean-variance, or shrunk mean-variance). Sets guardrails the PM must operate within.

### Portfolio Manager — Synthesis + Trading Signal

Chairs the committee. A fundamental expert who also uses quantitative tools. Weighs all evidence — bull thesis, bear risks, macro context, debate outcomes — and produces a structured committee memo. Thinks in vol surfaces, factor tilts, and event paths. Computes the T signal: direction times entropy-adjusted confidence, a single scalar for downstream RL consumption.

---

## 📊 T Signal — RL Input Feature

The T signal compresses the committee's output into a single scalar in [-1, 1]:

```
T = direction × C

Where:
  direction ∈ {-1, +1}     (-1 = short, +1 = long)
  C = ε + (1 - ε)(1 - H)   (entropy-adjusted certainty)
  H = normalized entropy     (proxy: 1 - raw_confidence)
  ε = 0.01                  (floor to avoid zero)
```

| T Value | Interpretation |
|---------|---------------|
| `+0.85` | Strong long conviction, high certainty |
| `-0.40` | Moderate short conviction, moderate certainty |
| `+0.01` | Long but extremely uncertain (high entropy) |

> The entropy-weighted confidence approach is adapted from Darmanin & Vella, ["Language Model Guided Reinforcement Learning in Quantitative Trading"](https://arxiv.org/abs/2508.02366) (arXiv:2508.02366v3, Oct 2025). Since the LLM interface is provider-agnostic (`callable(str) → str`), actual token entropy isn't available — the PM's self-reported confidence serves as the proxy.

---

## 🔧 Quick Start

### Prerequisites
- Python 3.11+
- At least one LLM API key — or Ollama for fully local execution

### Supported Providers

| Provider | Default Model | Setup |
|----------|--------------|-------|
| **Anthropic** | `claude-sonnet-4-20250514` | Set `ANTHROPIC_API_KEY` in `.env` |
| **Google** | `gemini-2.0-flash` | Set `GOOGLE_API_KEY` in `.env` |
| **OpenAI** | `gpt-4o-mini` | `pip install -e ".[openai]"` + `OPENAI_API_KEY` |
| **HuggingFace** | `Qwen/Qwen2.5-72B-Instruct` | Set `HF_TOKEN` in `.env` |
| **Ollama** | `llama3.2:3b` | `pip install -e ".[ollama]"` + `ollama serve` |

Switch providers at runtime via the UI dropdown, or set `LLM_PROVIDER` in `.env`.

### Installation

```bash
git clone https://github.com/bdschi1/multi-agent-investment-committee.git
cd multi-agent-investment-committee

python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"

cp .env.example .env
# Add your API key(s) to .env
```

### Run

```bash
python app.py
# Opens at http://localhost:7860
```

### Test

```bash
pytest tests/ -v   # 111 tests
```

---

## 🧩 Execution Modes

| Mode | How It Works |
|------|-------------|
| **Full Auto** | Single click runs the entire pipeline end-to-end |
| **Review Before PM** | Two-phase HITL — see analyst outputs + debate results, optionally add PM guidance, then the PM synthesizes |

In HITL mode, you review what the analysts produced and can steer the PM before it makes its decision (e.g., *"Weight the bear case more heavily — the valuation risk is underappreciated. Consider a half position."*).

---

## 📁 Project Structure

```
multi-agent-investment-committee/
├── app.py                       # Gradio UI (8 tabs + T signal + doc upload)
├── agents/
│   ├── base.py                  # BaseAgent ABC + schemas + JSON extraction
│   ├── sector_analyst.py        # Bull case + sentiment extraction
│   ├── risk_manager.py          # Bear case + causal chains
│   ├── macro_analyst.py         # Macro context + portfolio strategy
│   └── portfolio_manager.py     # PM synthesis + T signal inputs
├── orchestrator/
│   ├── graph.py                 # LangGraph StateGraph (full + phase1 + phase2)
│   ├── nodes.py                 # Node functions + T signal computation
│   ├── state.py                 # CommitteeState TypedDict with reducers
│   ├── committee.py             # CommitteeResult + ConvictionSnapshot
│   ├── memory.py                # Session memory store
│   └── reasoning_trace.py       # Trace rendering for UI
├── tools/
│   ├── registry.py              # ToolRegistry + budget enforcement
│   ├── market_data.py           # yfinance wrapper
│   ├── news_retrieval.py        # RSS feed aggregation
│   ├── financial_metrics.py     # Derived metrics + quality scoring
│   ├── earnings_data.py         # Earnings beat/miss tracking
│   ├── insider_data.py          # SEC Form 4 insider transactions
│   ├── peer_comparison.py       # Peer valuation comparison
│   ├── data_aggregator.py       # Unified data pipeline
│   └── doc_chunker.py           # PDF/DOCX/TXT ingestion + chunking
├── config/
│   └── settings.py              # Pydantic settings with .env support
├── tests/                       # 111 tests
└── pyproject.toml
```

---

## 💡 Design Philosophy

**Variant views over consensus.** Every agent identifies what "the street" thinks before presenting their own view. Generic observations ("AI is growing", "competition risk") are flagged as already priced in. Agents must identify where their view *differs* from consensus and why — supported by 2-3 independent, converging data points.

**Structured reasoning, not prompt engineering.** Each agent follows think → plan → act → reflect with tool calls between plan and act. Every step is captured in a trace visible in the UI. The value isn't in clever prompts — it's in the architecture that forces agents through a disciplined analytical process.

**Heuristic quant, not pretend quant.** Agents reason through portfolio construction frameworks (Sharpe, Sortino, return decomposition, NMV sizing) as mental models. They can't run optimizers, but the structured reasoning produces better-calibrated outputs than unstructured prompts. The Risk Manager's role of challenging whether claimed alpha is genuine or disguised factor beta is particularly valuable.

**Adversarial pressure improves output.** The debate phase isn't theater — agents update their convictions based on challenges. The PM sees the original theses *and* the debate outcomes, producing decisions that have been stress-tested before synthesis.

**RL-ready from day one.** The T signal compresses the entire committee's output into a single scalar consumable by a reinforcement learning agent. Direction and confidence are encoded together, with entropy adjustment to penalize uncertain outputs.

---

## 🗺 Roadmap

- [x] v1: Core multi-agent committee with structured reasoning
- [x] v2: LangGraph orchestration + dynamic tool calling + HITL
- [x] v3: Sentiment extraction, trading-fluent PM, portfolio strategist, T signal
- [x] v3.1: Quantitative portfolio construction heuristics
- [x] v3.2: Document KB upload (PDF/DOCX/TXT)
- [x] v3.3: Conviction timeline with LLM-generated rationale
- [x] v3.4: Small-LLM resilience, model selection, Ollama improvements
- [x] v3.5: T signal with citation, conviction tracker redesign
- [ ] Token-level entropy measurement (provider-specific logprobs API)
- [ ] RAG integration for SEC filings and earnings transcripts
- [ ] Persistent cross-session memory with vector store
- [ ] RL agent that consumes T signal for position sizing

---

## 🤗 Tech Stack

This project is built on top of:

* [LangGraph](https://github.com/langchain-ai/langgraph) — StateGraph orchestration with conditional edges and fan-out/fan-in
* [Gradio](https://www.gradio.app/) — UI with 8-tab interface, T signal gauge, and document upload
* [Pydantic](https://docs.pydantic.dev/) — Strict output validation for all agent schemas
* [yfinance](https://github.com/ranaroussi/yfinance) — Market data retrieval
* [pytest](https://docs.pytest.org/) — 111 tests with mock LLM fixtures

LLM providers: [Anthropic](https://www.anthropic.com/) (Claude), [Google](https://ai.google.dev/) (Gemini), [OpenAI](https://openai.com/) (GPT), [HuggingFace](https://huggingface.co/), [Ollama](https://ollama.ai/) (local)

---

## License

Apache 2.0
