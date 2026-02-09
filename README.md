---
title: Multi-Agent Investment Committee
emoji: 📊
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
  - smolagents
---

# Multi-Agent Investment Committee

**Four autonomous AI agents that reason, debate, and synthesize investment theses.**

This project demonstrates production-grade multi-agent orchestration applied to investment analysis — not a chatbot wrapper, but agents that *think, plan, act, and reflect* through structured reasoning chains with adversarial debate and variant-view seeking.

[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/YOUR_USERNAME/multi-agent-investment-committee)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

---

## Architecture

```
User Input (ticker + expert guidance)
       │
       ▼
┌─────────────────┐
│  Data Gathering  │  ← yfinance, RSS feeds, derived metrics
└────────┬────────┘
         │
    ┌────┼────────────┐
    ▼    ▼            ▼
┌────────┐ ┌────────┐ ┌────────┐
│ Sector │ │ Risk   │ │ Macro  │   ← Parallel execution (3 agents)
│Analyst │ │Manager │ │Analyst │
│ (Bull) │ │ (Bear) │ │(Top-Dn)│
└───┬────┘ └───┬────┘ └───┬────┘
    │          │           │
    ▼          ▼           │ (no debate)
┌──────────────────┐       │
│ Adversarial      │       │
│ Debate           │       │
│ (Rebuttals)      │       │
└────────┬─────────┘       │
         │                 │
         ▼                 ▼
┌────────────────────────────┐
│ Portfolio Manager           │   ← Weighs evidence + macro context
│ (Synthesis)                 │
└────────────┬───────────────┘
             │
             ▼
       Committee Memo
       (Structured Output)
```

## What Makes This Different

| Feature | Typical AI Demo | This Project |
|---------|----------------|--------------|
| Reasoning | Single prompt → answer | think → plan → act → reflect loop |
| Multi-agent | Sequential prompts | Parallel execution + adversarial debate |
| Output | Unstructured text | Pydantic-validated schemas |
| Observability | Black box | Full reasoning trace per agent |
| Domain | Generic | Investment-specific with real market data |
| Analysis Quality | Restates obvious | Seeks **variant/alpha views**, flags consensus |

## The Four Agents

### Sector Analyst (Bull Case)
Builds the affirmative investment thesis with conviction scoring, catalysts, and evidence-backed arguments. Seeks **variant perceptions** — non-consensus views backed by triangulated data — rather than restating widely-known narratives.

### Risk Manager (Bear Case)
Adversarial analysis with **causal chain reasoning** — traces how primary risks cascade into 2nd and 3rd order effects. Actively seeks **non-obvious risks** the market is underpricing, and produces active short pitches when warranted:

```
Primary Risk: "Rising interest rates"
  → 2nd Order: "Higher borrowing costs compress margins by 200bps"
    → 3rd Order: "Forced R&D cuts weaken competitive moat over 18-24 months"
```

### Macro Analyst (Top-Down Context)
Global macro strategist providing economic cycle assessment, rate environment, sector rotation signals, geopolitical risk mapping, and cross-asset signal analysis. Runs in parallel with bull/bear (zero added latency). Feeds the PM — does not debate.

### Portfolio Manager (Synthesizer)
Chairs the committee. Weighs bull vs. bear evidence with macro context, assesses variant view quality, and produces a final recommendation with explicit reasoning for why one side was weighted over the other. Every recommendation includes conviction sensitivity: what would change the PM's mind.

## Alpha-Seeking Philosophy

All agents are prompted to avoid the obvious and seek variant views:

- **Consensus awareness**: Each agent states what "the street" thinks before presenting their own view
- **Anti-obvious filter**: Generic points like "AI is growing" or "competition risk" are flagged as already priced in
- **Variant requirement**: Agents must identify where their view *differs* from consensus and why
- **Triangulation**: Theses should be supported by 2-3 independent, converging data points
- **Conviction sensitivity**: Every agent answers "what would change my score by 2+ points?"

## Reasoning Loop

Every agent follows a structured protocol:

1. **THINK** — Assess the situation, form hypotheses, identify consensus vs. variant views
2. **PLAN** — Decide what data to analyze and how to find non-obvious insights
3. **ACT** — Execute analysis, call tools, build thesis with variant view requirements
4. **REFLECT** — Evaluate output quality, test conviction sensitivity, check for obvious-ness

Each step is captured in a structured reasoning trace visible in the UI.

## Performance & API Calls

Each committee analysis makes **~20 LLM API calls** (with the default 2 debate rounds):

| Phase | API Calls | Wall-Clock Steps | What Happens |
|-------|-----------|-----------------|--------------|
| Phase 1: Parallel Analysis | 12 | 4 | Each agent runs think→plan→act→reflect (4 calls × 3 agents, parallel) |
| Phase 2: Adversarial Debate | 4 | 2 | Bull and bear produce rebuttals per round (2 agents × 2 rounds, parallel) |
| Phase 3: PM Synthesis | 4 | 4 | Portfolio Manager runs think→plan→act→reflect |

**Total: ~20 API calls, ~10 wall-clock steps** (default) → expect **90-120 seconds** depending on provider and model.

### Adjusting Debate Rounds

Debate rounds are adjustable from the Gradio UI slider (1-20), or via `.env`:

```bash
MAX_DEBATE_ROUNDS=2    # Default: 2 rounds → ~20 API calls total
# MAX_DEBATE_ROUNDS=1  # Faster: 1 round → ~18 API calls total
# MAX_DEBATE_ROUNDS=5  # Deeper: 5 rounds → ~24 API calls total
```

Formula: **Total API calls = 12 (analysts) + 2 × debate_rounds (rebuttals) + 4 (PM)**

## Quick Start

### Prerequisites
- Python 3.11+
- At least one LLM API key (Anthropic, Google, OpenAI, or HuggingFace) — or Ollama for local models

### Supported Providers

| Provider | Model (default) | Setup |
|----------|----------------|-------|
| **Anthropic** (Claude) | `claude-sonnet-4-20250514` | Set `ANTHROPIC_API_KEY` in `.env` |
| **Google** (Gemini) | `gemini-2.0-flash` | Set `GOOGLE_API_KEY` in `.env` |
| **OpenAI** (GPT) | `gpt-4o-mini` | `pip install -e ".[openai]"` + set `OPENAI_API_KEY` |
| **HuggingFace** | `Qwen/Qwen2.5-72B-Instruct` | Set `HF_TOKEN` in `.env` |
| **Ollama** (local) | `llama3.1:8b` | `pip install -e ".[ollama]"` + `ollama serve` |

Switch providers at runtime via the dropdown in the UI, or set `LLM_PROVIDER` in `.env`.

### Setup

```bash
git clone https://github.com/YOUR_USERNAME/multi-agent-investment-committee.git
cd multi-agent-investment-committee

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Configure
cp .env.example .env
# Edit .env with your API keys
```

### Run Locally

```bash
python app.py
# Auto-opens at http://localhost:7860
```

> **Note:** Each analysis takes 90-120 seconds (~20 API calls with default settings). The UI shows real-time progress, and the terminal displays detailed phase logs.

### Run Tests

```bash
pytest tests/ -v
```

## Project Structure

```
multi-agent-investment-committee/
├── app.py                       # Gradio UI entry point (8 tabs + export)
├── agents/
│   ├── base.py                  # BaseAgent ABC + output schemas (BullCase, BearCase, MacroView, etc.)
│   ├── sector_analyst.py        # Bull case agent (variant-seeking)
│   ├── risk_manager.py          # Bear case + causal chains + active shorts
│   ├── macro_analyst.py         # Top-down macro context (cycle, rates, rotation)
│   └── portfolio_manager.py     # Synthesis + decision + conviction sensitivity
├── orchestrator/
│   ├── committee.py             # Multi-agent orchestration (ThreadPoolExecutor)
│   └── reasoning_trace.py       # Trace rendering for UI
├── tools/
│   ├── market_data.py           # yfinance wrapper
│   ├── news_retrieval.py        # RSS feed aggregation
│   ├── financial_metrics.py     # Derived metrics + quality scoring
│   └── data_aggregator.py       # Unified data pipeline
├── prompts/
│   ├── sector_analyst.yaml      # Externalized prompt documentation
│   ├── risk_manager.yaml
│   ├── macro_analyst.yaml
│   └── portfolio_manager.yaml
├── config/
│   └── settings.py              # Pydantic settings with .env support
└── tests/
    ├── test_agents.py
    ├── test_orchestrator.py
    └── test_tools.py
```

## Design Decisions

- **Variant-view philosophy:** Every agent is prompted to identify consensus first, then seek non-obvious, differentiated insights. Generic points are flagged as "already priced in." This mirrors how the best fundamental analysts actually think.

- **Conviction sensitivity:** All agents answer "what would change my score by 2+ points?" in their reflect step. This forces genuine uncertainty quantification, not just a number.

- **Externalized prompts (YAML):** Prompt templates are versioned and separated from code — a best practice for production agentic systems where prompt iteration is continuous.

- **Pydantic output schemas:** Every agent output is validated against a strict schema. This catches malformed LLM outputs and ensures downstream consumers (the PM agent, the UI) receive structured data.

- **Parallel execution:** All Phase 1 agents (bull, bear, macro) run concurrently via `ThreadPoolExecutor(max_workers=3)`. Debate rounds also parallelize both rebuttals. Zero wall-clock overhead from the macro agent.

- **Adversarial debate:** This isn't four prompts in sequence. Agents see each other's outputs and produce structured rebuttals with concessions — mimicking how real investment committees work.

- **Provider abstraction:** The model layer is a simple `callable(str) → str`, making it trivial to swap between Anthropic, Google, OpenAI, HuggingFace, or local Ollama models — switchable at runtime via the UI dropdown.

## Tech Stack

- **Orchestration:** ThreadPoolExecutor-based parallel orchestrator (v1), designed for LangGraph state-machine migration (v2)
- **LLM:** Anthropic (Claude), Google (Gemini), OpenAI, HuggingFace, Ollama — switchable at runtime
- **UI:** Gradio 6.0 with 8-tab interface (Memo, Bull, Bear, Macro, Debate, Conviction, Trace, Session) + copy/download export
- **Data:** yfinance (market data), feedparser (RSS news), derived financial metrics
- **Validation:** Pydantic v2 with strict output schemas
- **Config:** pydantic-settings with `.env` support
- **Testing:** pytest with mock LLM fixtures

## Roadmap

- [x] v1: Core multi-agent committee with smolagents-style architecture
- [x] 4th agent: Macro Analyst (top-down economic context)
- [x] Variant-view / alpha-seeking prompt philosophy
- [x] Active short pitches in bear case
- [x] Conviction sensitivity tracking
- [ ] v2: LangGraph state-machine orchestration with dynamic tool calling
- [ ] RAG integration for SEC filings and earnings transcripts
- [ ] Persistent memory across sessions

---

## Disclaimer

This is a demonstration of multi-agent AI reasoning architecture. It is **not** financial advice. All analyses are AI-generated and should not be used for actual investment decisions. Always consult qualified financial professionals.

## License

Apache 2.0
