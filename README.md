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
  - langgraph
  - reinforcement-learning
---

# Multi-Agent Investment Committee

**Four autonomous AI agents that reason, debate, and synthesize investment theses — with document KB upload, sentiment extraction, volatility-aware trading logic, quantitative portfolio construction heuristics, and an RL-ready T signal.**

This project demonstrates production-grade multi-agent orchestration applied to investment analysis — not a chatbot wrapper, but agents that *think, plan, act, and reflect* through structured reasoning chains with adversarial debate, user-uploaded research documents, news sentiment processing, return decomposition, factor-aware positioning, risk-adjusted sizing, and entropy-adjusted confidence signaling.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-111%20passing-brightgreen.svg)](tests/)

---

## Architecture

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
│ Sector │ │ Risk   │ │ Macro      │   ← Parallel execution (3 agents)
│Analyst │ │Manager │ │Strategist  │
│ (Bull  │ │ (Bear) │ │(Top-Down + │
│  +Sent)│ │        │ │ Portfolio) │
└───┬────┘ └───┬────┘ └───┬────────┘
    │          │           │
    ▼          ▼           │ (no debate — provides guardrails)
┌──────────────────┐       │
│ Adversarial      │       │
│ Debate           │       │
│ (Rebuttals)      │       │
└────────┬─────────┘       │
         │                 │
         ▼                 ▼
┌────────────────────────────┐
│ Portfolio Manager          │   ← Weighs evidence + macro + vol + factors
│ (Synthesis + T Signal)     │   ← Outputs: recommendation + T ∈ [-1, 1]
└────────────┬───────────────┘
             │
             ▼
       Committee Memo
       (Structured Output + T Signal for RL)
```

## What Makes This Different

| Feature | Typical AI Demo | This Project |
|---------|----------------|--------------|
| Reasoning | Single prompt → answer | think → plan → act → reflect loop |
| Multi-agent | Sequential prompts | Parallel execution + adversarial debate |
| Output | Unstructured text | Pydantic-validated schemas |
| Observability | Black box | Full reasoning trace per agent |
| Sentiment | None | Per-headline extraction + divergence detection |
| Volatility | None | IV/HV assessment, vol regime, sizing guardrails |
| Portfolio Mgmt | None | Net exposure, factor tilts, correlation regime |
| Quant Heuristics | None | Return decomposition, Sharpe/Sortino, NMV sizing methods |
| Document KB | None | Upload PDF/DOCX/TXT research docs → auto-chunk → feed all agents |
| RL Integration | None | T signal: direction × entropy-adjusted confidence |
| Orchestration | ThreadPool | LangGraph StateGraph with conditional edges |
| Tool Calling | None | Dynamic tool calling with per-agent budgets |
| Human-in-Loop | None | Two-phase HITL with PM guidance injection |

## The Four Agents

### Sector Analyst (Bull Case + Sentiment + Return Decomposition)
Builds the affirmative investment thesis with conviction scoring, catalysts, and evidence-backed arguments. Processes every news headline through a sentiment extraction pipeline — classifying each as bullish/bearish/neutral with signal strength and catalyst type, computing aggregate sentiment, and flagging sentiment-price divergences. Performs heuristic return decomposition — setting a price target, estimating total return, stripping out industry return to isolate idiosyncratic alpha, and computing heuristic Sharpe and Sortino ratios (with sign-flip for short theses). Each estimate includes reasoning (e.g., methodology, assumptions, and what drives the number).

### Risk Manager (Bear Case + Alpha Challenge)
Adversarial analysis with **causal chain reasoning** — traces how primary risks cascade into 2nd and 3rd order effects. Actively seeks **non-obvious risks** the market is underpricing, and produces active short pitches when warranted. **New:** challenges the bull case's return decomposition — stress-tests whether claimed "alpha" is genuine idiosyncratic return or disguised factor/sector beta, stress-tests Sortino with elevated downside vol assumptions, and flags factor-as-alpha risk where returns can be replicated cheaper with ETFs:

```
Primary Risk: "Rising interest rates"
  → 2nd Order: "Higher borrowing costs compress margins by 200bps"
    → 3rd Order: "Forced R&D cuts weaken competitive moat over 18-24 months"
```

### Macro Analyst → Portfolio Strategist
Global macro strategist **and** portfolio strategist (not a PM — sets guardrails). Provides portfolio-level strategy guidance and **quantitative sizing frameworks**:

| Dimension | What It Does |
|-----------|-------------|
| **Vol Regime** | Classifies current environment (low/normal/elevated/crisis) with VIX-referenced sizing constraints |
| **Net Exposure** | Recommends portfolio directionality (net long / market neutral / net short) based on cycle + vol |
| **Sector & Style** | Agnostic assessment of growth vs value, large vs small, defensive vs cyclical rotation |
| **Correlation Regime** | Identifies macro-driven vs stock-picking environments for diversification guidance |
| **Vol Budget** | Specific position sizing constraints (e.g., "max 3% per name, reduce gross by 20% in elevated vol") |
| **Sector Vol** | Estimates sector annualized vol and compares to stock-level vol for relative sizing |
| **Sizing Method** | Recommends one of: proportional, risk parity, mean-variance, or shrunk mean-variance |
| **Vol Target** | Sets portfolio annualized vol target (vol targeting > GMV targeting) |

### Portfolio Manager (Fundamental + Quantitative)
Chairs the committee. A fundamental expert who also uses quantitative metrics and tools to size and manage the portfolio. Most names are fundamental thesis-driven, but several are statistical positions that help manage portfolio risk metrics. Speaks to his head trader daily and thinks fluently in vol surfaces, factor tilts, and event paths:

| Capability | Description |
|-----------|-------------|
| **IV/HV Assessment** | Compares implied vs historical vol, reads the vol surface (skew, term structure) |
| **Event Path** | Maps ordered sequence of binary events with conviction impact at each node |
| **Factor Exposures** | Identifies momentum, value, quality, size, volatility tilts the position creates |
| **Conviction Triggers** | Precise, actionable triggers for sizing up, cutting, or reversing the thesis |
| **Idio Return Validation** | Validates analyst's return decomposition after hearing the bear challenge |
| **Sharpe/Sortino Synthesis** | Computes heuristic risk-adjusted returns (sign-flipped for shorts) |
| **NMV Sizing** | Applies chosen sizing method (proportional/risk-parity/MV/shrunk-MV) with rationale |
| **Vol Targeting** | Ensures position vol contribution fits within portfolio vol budget |
| **T Signal** | Outputs direction × entropy-adjusted confidence for downstream RL consumption |

## T Signal — RL Input Feature

The T signal is a single scalar in [-1, 1] that encodes both the LLM committee's directional view and its confidence:

```
T = direction × C

Where:
  direction ∈ {-1, +1}     (-1 = short, +1 = long)
  C = ε + (1 - ε)(1 - H)   (entropy-adjusted certainty)
  H = normalized entropy     (proxy: 1 - raw_confidence)
  ε = 0.01                  (floor to avoid zero)
```

**Interpretation:**
- `T = +0.85` → Strong long conviction, high certainty
- `T = -0.40` → Moderate short conviction, moderate certainty
- `T = +0.01` → Long but extremely uncertain (high entropy)

The T signal is computed in `orchestrator/nodes.py::_compute_t_signal()` after the PM produces its memo, and is stored in `CommitteeMemo.t_signal`. It's logged in the run JSONL and displayed in the UI with a gauge visualization.

> **Note on entropy:** Since the LLM interface is `callable(str) → str` (provider-agnostic), we cannot access actual token-level entropy. The PM's self-reported `raw_confidence` serves as a proxy for certainty, reflecting how decisive the committee's analysis was.
>
> The entropy-weighted confidence approach is adapted from Darmanin & Vella, ["Language Model Guided Reinforcement Learning in Quantitative Trading"](https://arxiv.org/abs/2508.02366) (arXiv:2508.02366v3, Oct 2025).

## Quantitative Portfolio Construction Heuristics

All agents reason through quantitative portfolio construction frameworks as **heuristic mental models** — not precise computations. The LLM agents cannot run optimizers, but they can reason through the logic of these frameworks to produce better-calibrated outputs.

### Return Decomposition
The Sector Analyst decomposes expected return into sector/industry return and idiosyncratic (alpha) return. The Risk Manager challenges whether the claimed alpha is genuine or disguised factor/sector beta:

```
Total Return = Industry Return + Idiosyncratic Return (alpha)

If idiosyncratic return ≈ 0, the thesis is a sector bet, not a stock pick.
```

### Risk-Adjusted Return (Sharpe & Sortino)
Both the analyst and PM estimate heuristic Sharpe and Sortino ratios. For short positions, the return sign is flipped:

```
Sharpe  ≈ idio_return / vol           (symmetric risk)
Sortino ≈ idio_return / downside_vol  (loss-focused risk)

For shorts: use -1 × expected_return

Sharpe < 0.3 → weak risk-adjusted return, may not justify the position
Sharpe-Sortino divergence → asymmetric tail risk, size accordingly
```

### Position Sizing Methods
The Macro Analyst recommends a sizing method; the PM applies it:

| Method | Formula | Best When |
|--------|---------|-----------|
| **Proportional** | NMV = κ × α | Alpha estimates are trusted, vol is stable |
| **Risk Parity** | NMV = κ × α / σ | Vol dispersion is wide across names |
| **Mean-Variance** | NMV = κ × α / σ² | Vol-conscious, tilt toward lower-risk alpha |
| **Shrunk Mean-Variance** | NMV = κ × α / [p×σ² + (1-p)×σ²_sector] | Noisy vol estimates or limited history |

### Vol Targeting
Vol targeting controls portfolio risk better than gross market value (GMV) targeting because volatility is persistent and partially predictable. The Macro Analyst sets a portfolio vol target; the PM sizes positions to stay within it.

> **Note:** These are heuristic reasoning frameworks, not optimizer outputs. The agents reason through the logic qualitatively using available market data. The value is in the structured thinking process — forcing agents to decompose returns, challenge alpha claims, and justify sizing — not in the precision of the numbers.

## Document Knowledge Base

Upload up to **5 research documents** (PDF, DOCX, TXT) — broker reports, 10-Ks, internal memos, sector notes — and the system automatically processes them for all agents:

| Step | What Happens |
|------|-------------|
| **Ingest** | Reads PDF (pypdf), DOCX (python-docx), or plain text |
| **Decide** | Files ≤ 3K tokens → inject whole; larger files → chunk |
| **Chunk** | Token-aware splitting (800 tokens/chunk) with section-break awareness and equity research boilerplate removal |
| **Inject** | Formatted as `SUPPLEMENTARY RESEARCH` section in every agent's prompt (12K token budget) |

Agents treat uploaded documents as **one resource among many** — they rely on their own analytical judgment. The KB may contain valuable data points, but agents are instructed to weigh it critically alongside market data, news, and their own reasoning.

> **Optional dependencies:** `pip install -e ".[docs]"` for PDF/DOCX support (tiktoken, pypdf, python-docx). TXT works with no extra dependencies.

## Alpha-Seeking Philosophy

All agents are prompted to avoid the obvious and seek variant views:

- **Consensus awareness**: Each agent states what "the street" thinks before presenting their own view
- **Anti-obvious filter**: Generic points like "AI is growing" or "competition risk" are flagged as already priced in
- **Variant requirement**: Agents must identify where their view *differs* from consensus and why
- **Triangulation**: Theses should be supported by 2-3 independent, converging data points
- **Conviction sensitivity**: Every agent answers "what would change my score by 2+ points?"
- **Sentiment divergence**: Analyst flags when news sentiment conflicts with price action (alpha signal)

## Reasoning Loop

Every agent follows a structured protocol:

1. **THINK** — Assess the situation, form hypotheses, identify consensus vs. variant views
2. **PLAN** — Decide what data to analyze, request dynamic tools, plan non-obvious analysis
3. **[EXECUTE TOOLS]** — Dynamic tool calling with per-agent budget enforcement
4. **ACT** — Execute analysis, build thesis with variant view requirements, extract sentiment
5. **REFLECT** — Evaluate output quality, test conviction sensitivity, check for obvious-ness

Each step is captured in a structured reasoning trace visible in the UI.

## Performance & API Calls

Each committee analysis makes **~20 LLM API calls** (with the default 2 debate rounds):

| Phase | API Calls | Wall-Clock Steps | What Happens |
|-------|-----------|-----------------|--------------|
| Phase 1: Parallel Analysis | 12 | 4 | Each agent runs think→plan→[tools]→act→reflect (4 calls × 3 agents, parallel) |
| Phase 2: Adversarial Debate | 4 | 2 | Bull and bear produce rebuttals per round (2 agents × 2 rounds, parallel) |
| Phase 3: PM Synthesis | 4 | 4 | PM runs think→plan→[tools]→act→reflect + T signal computation |

**Total: ~20 API calls, ~10 wall-clock steps** (default) → expect **90-120 seconds** depending on provider and model.

Formula: **Total API calls = 12 (analysts) + 2 × debate_rounds (rebuttals) + 4 (PM)**

> **Rate limiting (Anthropic Tier 1):** A built-in rate limiter wraps Anthropic calls with a 60-second sliding window enforcing token budgets (default: 45 RPM, 25K input TPM — 85% of Tier 1 limits). This adds ~30-60s to Sonnet runs but prevents 429 errors. Configure via `.env` (`RATE_LIMIT_RPM`, `RATE_LIMIT_INPUT_TPM`). Other providers (Google, OpenAI, Ollama) are unaffected.

## Execution Modes

| Mode | Description |
|------|-------------|
| **Full Auto** | Single button runs the entire pipeline end-to-end |
| **Review Before PM** | Two-phase HITL — review analyst outputs + debate, add PM guidance, then PM synthesizes |

In HITL mode, you can steer the PM after seeing what the analysts produced (e.g., "Weight the bear case more heavily, focus on the valuation risk, consider a half position").

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
| **Ollama** (local) | `llama3.2:3b` | `pip install -e ".[ollama]"` + `ollama serve` |

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
pytest tests/ -v   # 111 tests, all passing
```

## Project Structure

```
multi-agent-investment-committee/
├── app.py                       # Gradio UI (8 tabs + export + T signal display)
├── agents/
│   ├── base.py                  # BaseAgent ABC + schemas + extract_json() + clean_json_artifacts() + brace-aware arg parser
│   ├── sector_analyst.py        # Bull case + news sentiment extraction
│   ├── risk_manager.py          # Bear case + causal chains + active shorts
│   ├── macro_analyst.py         # Macro context + portfolio strategy (vol, L/S, style)
│   └── portfolio_manager.py     # PM synthesis + quant sizing + T signal inputs
├── orchestrator/
│   ├── graph.py                 # LangGraph StateGraph (full + phase1 + phase2)
│   ├── nodes.py                 # Node functions + T signal computation
│   ├── state.py                 # CommitteeState TypedDict with reducers
│   ├── committee.py             # CommitteeResult + ConvictionSnapshot (with rationale)
│   ├── memory.py                # Session memory store
│   └── reasoning_trace.py       # Trace rendering for UI
├── tools/
│   ├── registry.py              # ToolRegistry + dynamic tool calling + budget + arg coercion
│   ├── market_data.py           # yfinance wrapper
│   ├── news_retrieval.py        # RSS feed aggregation
│   ├── financial_metrics.py     # Derived metrics + quality scoring
│   ├── earnings_data.py         # Earnings beat/miss tracking
│   ├── insider_data.py          # SEC Form 4 insider transactions
│   ├── peer_comparison.py       # Peer valuation comparison
│   ├── data_aggregator.py       # Unified data pipeline
│   └── doc_chunker.py           # Document KB: PDF/DOCX/TXT ingestion + chunking
├── config/
│   └── settings.py              # Pydantic settings with .env support
├── tests/                       # 111 tests (agents, orchestrator, tools, parsing, phase_b, phase_c)
│   ├── test_agents.py
│   ├── test_graph.py
│   ├── test_json_extraction.py  # JSON repair + artifact cleaning + sentinel detection
│   ├── test_orchestrator.py
│   ├── test_phase_c.py
│   ├── test_tools.py
│   └── test_tools_phase_b.py    # Includes dict-arg parsing + registry coercion tests
├── .env.example                 # Environment template
├── pyproject.toml               # Build config + dependencies
└── requirements.txt             # Core deps
```

## Schema Extensions (v3)

### BullCase (Sector Analyst)
```python
# Sentiment extraction
sentiment_factors: list[dict]      # Per-headline: {headline, sentiment, signal_strength, catalyst_type}
aggregate_news_sentiment: str       # strongly_bullish / bullish / neutral / bearish / strongly_bearish
sentiment_divergence: str           # Where sentiment diverges from price action

# Quantitative heuristics — estimate & reasoning (LLM-estimated, not computed)
price_target: str                   # Target + methodology (e.g. "$185 in 12m — DCF with 12% WACC")
forecasted_total_return: str        # Return + derivation (e.g. "22% — (185/152)-1, includes 1.2% div")
estimated_industry_return: str      # Sector return + reasoning (e.g. "8% — semis track GDP+4%")
idiosyncratic_return: str           # Alpha + drivers (e.g. "14% — share gains + margin expansion")
estimated_sharpe: str               # Sharpe + reasoning (e.g. "0.47 — 14% idio / 30% vol")
estimated_sortino: str              # Sortino + reasoning (e.g. "0.64 — 14% idio / 22% downside vol")
```

### MacroView (Portfolio Strategist)
```python
# Portfolio strategy
annualized_vol_regime: str          # low / normal / elevated / crisis (VIX-referenced)
vol_budget_guidance: str            # Position sizing constraints for the PM
portfolio_directionality: str       # net long / market neutral / net short
sector_style_assessment: str        # Growth vs value, large vs small, defensive vs cyclical
correlation_regime: str             # Macro-driven vs stock-picking environment

# Quantitative sizing framework (LLM-estimated, not computed)
sector_avg_volatility: str          # Sector annualized vol estimate vs stock vol
recommended_sizing_method: str      # proportional / risk_parity / mean_variance / shrunk_mean_variance
portfolio_vol_target: str           # Recommended portfolio annualized vol target
```

### CommitteeMemo (Portfolio Manager)
```python
# Trading-fluent PM fields
implied_vol_assessment: str         # IV vs HV, vol surface, sizing implication
event_path: list[str]              # Ordered near-term events with conviction impact
conviction_change_triggers: dict    # {size_up, cut_position, reverse_thesis}
factor_exposures: dict              # {momentum, value, quality, size, volatility}

# Quantitative sizing synthesis (LLM-estimated, not computed)
idio_return_estimate: str           # PM's validated alpha after weighing bull/bear
sharpe_estimate: str                # Heuristic Sharpe (sign-flipped for shorts)
sortino_estimate: str               # Heuristic Sortino (sign-flipped for shorts)
sizing_method_used: str             # Chosen sizing method with rationale
target_nmv_rationale: str           # How alpha, vol, and method combine to set NMV
vol_target_rationale: str           # How position fits portfolio vol budget

# T signal
position_direction: int             # +1 (long), -1 (short), 0 (flat)
raw_confidence: float               # [0, 1] — PM's self-reported certainty
t_signal: float                     # T = direction * C, computed post-hoc in nodes.py
```

## Design Decisions

- **Variant-view philosophy:** Every agent identifies consensus first, then seeks non-obvious, differentiated insights. Generic points are flagged as "already priced in."

- **Sentiment as alpha:** The analyst doesn't just summarize news — it extracts structured sentiment per headline and flags divergences between sentiment and price action, which are among the strongest short-term alpha signals.

- **Trading-fluent PM:** The PM is a fundamental expert who also uses quantitative tools for sizing and risk management. He speaks to his head trader daily and thinks fluently in vol surfaces, factor tilts, and event paths — bridging strategic research and executable trades. Most positions are fundamental thesis-driven, but several are statistical to manage portfolio risk metrics.

- **Portfolio strategist guardrails:** The macro agent constrains the PM by setting vol budgets, net exposure direction, and correlation-aware sizing. The PM operates *within* these guardrails.

- **Quant heuristics, not optimizers:** Agents reason through quantitative frameworks (Sharpe, Sortino, return decomposition, NMV sizing) as heuristic mental models. They can't run optimizers, but the structured reasoning produces better-calibrated outputs than unstructured prompts. The Risk Manager's alpha-challenge role (is this genuine idio return or disguised factor beta?) is particularly valuable.

- **T signal for RL:** The single scalar T ∈ [-1, 1] compresses the entire committee's output into a form consumable by a reinforcement learning agent. It encodes both direction and confidence, with entropy adjustment to penalize uncertain outputs.

- **Pydantic output schemas:** Every agent output is validated against a strict schema with backward-compatible defaults. All 111 tests pass without modification after v3 additions.

- **Document KB:** Users can upload research docs (PDF, DOCX, TXT) that are automatically chunked and injected into all agents as supplementary research. Agents are instructed to treat uploaded documents as one input among many — they rely on their own judgment, not the KB as gospel.

- **LangGraph orchestration:** Parallel fan-out via Send, debate loop (always runs — convergence is noted, not skipped), and two-phase HITL execution with PM guidance injection.

- **Provider abstraction:** The model layer is `callable(str) → str`, making it trivial to swap between Anthropic, Google, OpenAI, HuggingFace, or local Ollama models at runtime.

## Tech Stack

- **Orchestration:** LangGraph StateGraph with conditional edges, fan-out/fan-in, debate loop
- **LLM:** Anthropic (Claude), Google (Gemini), OpenAI, HuggingFace, Ollama — switchable at runtime
- **UI:** Gradio 6.0 with 8-tab interface + T signal gauge + document upload + copy/download export
- **Data:** yfinance (market data), feedparser (RSS news), derived financial metrics
- **Document KB:** PDF/DOCX/TXT ingestion with token-aware chunking (tiktoken + pypdf + python-docx)
- **Tools:** Dynamic tool calling with per-agent budget (10 tools: earnings, peers, insiders, etc.)
- **Validation:** Pydantic v2 with strict output schemas
- **Config:** pydantic-settings with `.env` support
- **Testing:** pytest with mock LLM fixtures (111 tests)

## Roadmap

- [x] v1: Core multi-agent committee with structured reasoning
- [x] v2: LangGraph orchestration + dynamic tool calling + HITL
- [x] v3: Sentiment extraction, head-trader PM, portfolio strategist, T signal
- [x] v3.1: Quantitative portfolio construction heuristics (return decomposition, Sharpe/Sortino, NMV sizing, vol targeting)
- [x] v3.2: Document KB upload (PDF/DOCX/TXT → auto-chunk → feed all agents), estimate & reasoning column, compact UI, debate always runs
- [x] v3.3: Bearish conviction rename (consistent scoring), conviction timeline rationale (LLM-generated per-case reasoning)
- [x] v3.4: Small-LLM resilience (robust JSON parsing, parsing failure tracking, UI warnings), model selection dropdown (Haiku, Flash, etc.), Ollama improvements
- [x] v3.4.1: Brace-aware tool arg parser, resilient doc chunker, session history in .md export, registry arg coercion safety net, rate limiter for Anthropic Tier 1 compliance
- [x] v3.5: T signal defined as "Trading Signal" with citation (Darmanin & Vella), conviction tracker redesign (4-section dual-bar map, 4-subsection interpretation), input lock during execution, scrolling fix, Expert/PM guidance labels clarified
- [ ] Actual token entropy measurement (requires provider-specific logprobs API)
- [ ] RAG integration for SEC filings and earnings transcripts
- [ ] Persistent cross-session memory with vector store
- [ ] RL agent that consumes T signal for position sizing optimization

---

## Disclaimer

This is a demonstration of multi-agent AI reasoning architecture. It is **not** financial advice. All analyses are AI-generated and should not be used for actual investment decisions. Always consult qualified financial professionals.

## License

Apache 2.0
