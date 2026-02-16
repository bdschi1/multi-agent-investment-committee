# Architecture

## Overview

Four agents analyze a ticker through a LangGraph StateGraph. Three run in parallel (sector analyst, risk manager, macro strategist), bull and bear debate, then the portfolio manager synthesizes a committee memo with a T signal.

## Orchestration

The pipeline is a LangGraph StateGraph defined in `orchestrator/graph.py`. Three graph variants exist:

- **Full pipeline** -- data gathering -> parallel analysis -> debate -> PM synthesis
- **Phase 1 only** -- stops after debate for human-in-the-loop review
- **Phase 2 only** -- PM synthesis after human provides guidance

Parallel fan-out uses LangGraph `Send`. The debate loop always runs (convergence is noted in output, not used to skip rounds). Non-serializable objects (model callable, callbacks, tool registry) flow through LangGraph's `RunnableConfig`, keeping `CommitteeState` serializable.

## Agent Reasoning Protocol

Every agent follows a 4-step loop:

```
THINK -> PLAN -> ACT -> REFLECT
```

1. **THINK** -- form hypotheses, identify consensus view
2. **PLAN** -- decide what to analyze, request tools
3. **ACT** -- execute analysis against real data, produce structured output
4. **REFLECT** -- self-critique, identify gaps, test conviction sensitivity

Each step is captured in a `ReasoningTrace` with timestamps and content. Traces are visible in the UI and stored in run output.

## Data Flow

```
User Input (ticker + guidance + optional docs)
    |
    v
DataAggregator.gather_context()
    |-- MarketDataTool (price, fundamentals, overview)
    |-- NewsRetrievalTool (RSS feeds)
    |-- FinancialMetricsTool (valuation, quality score)
    |-- EarningsDataTool (beat/miss history)
    |-- InsiderDataTool (SEC Form 4)
    |-- PeerComparisonTool (peer valuation)
    |-- DocChunker (uploaded PDF/DOCX/TXT)
    |
    v
CommitteeState (typed dict with reducers)
    |
    +---> Sector Analyst (parallel)
    +---> Risk Manager (parallel)
    +---> Macro Strategist (parallel)
    |
    v
Debate (bull vs bear rebuttals, configurable rounds)
    |
    v
Portfolio Manager (sequential -- needs all prior output)
    |
    v
CommitteeResult + T Signal
```

Agents invoke tools dynamically through the `ToolRegistry` with per-agent call budgets. The registry handles argument coercion (string args parsed to dicts/lists) and budget enforcement.

## State Management

`CommitteeState` is a TypedDict in `orchestrator/state.py`. Key fields:

- `ticker`, `user_context`, `context` -- input data
- `bull_case`, `bear_case`, `macro_view` -- agent outputs (Pydantic models)
- `bull_rebuttal`, `bear_rebuttal` -- debate outputs
- `committee_memo` -- PM synthesis
- `conviction_timeline` -- list of ConvictionSnapshot objects tracking how convictions evolve
- `reasoning_traces` -- per-agent traces

Reducer functions handle fan-in merging from parallel nodes. All agent outputs are Pydantic models with backward-compatible defaults.

## Tool System

Tools are registered in `tools/registry.py`. Each tool is a callable with metadata (name, description, parameter schema). Agents request tools by name during the PLAN step. The registry:

- Validates tool exists
- Checks per-agent budget
- Coerces arguments (string -> dict/list parsing)
- Returns structured results

10 tools available: market data, news, financial metrics, earnings, insider transactions, peer comparison, plus document chunks.

## Data Providers

`tools/data_providers/` implements a provider abstraction:

- `BaseDataProvider` (ABC) -- interface for price, fundamentals, overview
- `YahooProvider` -- default, uses yfinance
- `BloombergProvider` -- optional, uses blpapi
- `IBProvider` -- optional, uses ib_insync
- `ProviderFactory` -- runtime selection

All tools use the provider abstraction. Switch with `set_provider("Bloomberg")` or install optional deps.

## Output Schemas

All agent outputs are Pydantic models in `agents/base.py`:

- `BullCase` -- thesis, evidence, catalysts, conviction, sentiment factors, return decomposition
- `BearCase` -- thesis, risks, causal chains, conviction, short pitch
- `MacroView` -- vol regime, net exposure, sizing method, portfolio strategy
- `CommitteeMemo` -- recommendation, conviction, T signal inputs, factor exposures, sizing rationale

Schemas have strict validation (e.g., conviction 0.0-10.0) with backward-compatible defaults for fields added in later versions. This guarantees downstream consumers (UI, eval harness, exports) never get partial data.

## Eval Harness

`evals/` provides a ground-truth evaluation framework:

- `schemas.py` -- EvalScenario, GroundTruth, GradingResult, LikertLevel
- `grader.py` -- 6-dimension scoring (direction, conviction, risks, facts, reasoning, adversarial)
- `adversarial.py` -- context injection engine that merges tainted data into real DataAggregator output
- `loader.py` -- YAML scenario discovery with type/tag filtering
- `runner.py` -- runs committee against scenarios, grades output
- `reporter.py` -- JSON/markdown report generation

Scoring uses 0-100 continuous scale per dimension with configurable weights. Each score maps to a Likert level (1-5) with dimension-specific anchor text from the rubric YAML.

## Error Handling

LLM output is unreliable. The system handles this through:

1. JSON extraction with fallbacks: direct parse -> markdown code block -> brace matching
2. Fallback schema construction: if parsing fails, a valid but minimal schema is returned
3. Parse failure logging with raw response for debugging
4. No crashes: the committee always produces output, even if individual agents produce lower-quality results

## Testing

200 tests, all run without API calls:

- MockLLM returns pre-defined JSON based on prompt keywords
- Schema tests validate Pydantic models independently
- Integration tests run the full committee with mocks
- Eval tests validate grading, loading, and schema logic
- Network-dependent tests marked with `skipif` for CI
