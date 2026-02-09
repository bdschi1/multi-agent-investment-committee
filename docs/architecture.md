# Architecture Deep-Dive

## Overview

The Multi-Agent Investment Committee implements a structured multi-agent workflow where autonomous AI agents reason through an investment decision collaboratively, with adversarial tension built into the process.

This document explains the architectural choices, the reasoning protocol, and how the system is designed for extensibility.

## Agent Reasoning Protocol

Every agent follows a 4-step reasoning loop inspired by cognitive architectures:

```
THINK → PLAN → ACT → REFLECT
```

### Why This Matters

Most "agent" demos are just prompt chains — input goes in, output comes out, with no visibility into *how* the model reached its conclusion. Our protocol forces:

1. **Explicit hypothesis formation** (THINK) — the agent must articulate what it believes before gathering data
2. **Structured planning** (PLAN) — the agent decides what to analyze before doing it
3. **Grounded execution** (ACT) — analysis is performed against real data with structured outputs
4. **Self-evaluation** (REFLECT) — the agent critiques its own work, identifying gaps and biases

Every step is captured in a `ReasoningTrace` object with timestamps, token counts, and content — providing full observability into the agent's reasoning chain.

## Orchestration Design

### v1: Async Orchestrator (Current)

The current implementation uses a custom async orchestrator that coordinates agents via `asyncio`:

```python
# Phase 1: Parallel analysis
analyst_result, risk_result = await asyncio.gather(
    analyst.run(ticker, context),
    risk_mgr.run(ticker, context),
)

# Phase 2: Adversarial debate
analyst_rebuttal, risk_rebuttal = await asyncio.gather(
    analyst.rebut(ticker, bear_case, bull_case),
    risk_mgr.rebut(ticker, bull_case, bear_case),
)

# Phase 3: PM synthesis (sequential — needs debate results)
pm_result = await pm.run(ticker, synthesis_context)
```

### v2: LangGraph State Machine (Planned)

The v2 migration will convert this into a LangGraph state graph where:
- Each agent is a node
- Edges represent data flow and conditional routing
- Tool calls happen within graph nodes (not pre-gathered)
- The debate can be modeled as a cycle with termination conditions

The current architecture is designed for this migration:
- Agents implement a consistent interface (`run`, `rebut`)
- Output schemas are Pydantic models (serializable for graph state)
- The model is injected as a callable (swappable)

## Data Flow

```
User Input
    │
    ▼
DataAggregator.gather_context()
    ├── MarketDataTool.get_company_overview()
    ├── MarketDataTool.get_price_data()
    ├── MarketDataTool.get_fundamentals()
    ├── NewsRetrievalTool.get_news()
    ├── FinancialMetricsTool.compute_valuation_assessment()
    └── FinancialMetricsTool.compute_quality_score()
    │
    ▼
Unified Context Dict
    │
    ├──► Sector Analyst (parallel)
    ├──► Risk Manager (parallel)
    │
    ▼
Debate Round (parallel rebuttals)
    │
    ▼
Portfolio Manager (sequential)
    │
    ▼
CommitteeResult
```

### v1 vs v2 Data Strategy

**v1 (current):** Data is pre-gathered by `DataAggregator` and passed as context. Agents receive structured data but don't invoke tools dynamically. This is simpler and sufficient for demonstration.

**v2 (planned):** Agents will invoke tools dynamically through LangGraph tool nodes. The Sector Analyst might decide mid-analysis that it needs peer comparison data and request it. This shows more sophisticated tool use but requires careful state management.

## Output Schema Design

All agent outputs are Pydantic models with strict validation:

```python
class BullCase(BaseModel):
    ticker: str
    thesis: str
    supporting_evidence: list[str]
    catalysts: list[str]
    conviction_score: float = Field(ge=0.0, le=10.0)
    time_horizon: str
    key_metrics: dict[str, Any]
```

This ensures:
- **Downstream reliability:** The PM agent can always access `.conviction_score` without error handling
- **UI consistency:** Gradio formatters always receive complete, typed data
- **Testability:** Schemas can be validated in unit tests without LLM calls

## Prompt Management

Prompts are externalized as YAML files in `/prompts/`:

```yaml
version: "1.0.0"
role: sector_analyst
persona: |
  You are a senior sector analyst...
constraints:
  - Always ground analysis in available data
  - Provide specific numbers...
```

This separation enables:
- Prompt iteration without code changes
- Version tracking for A/B testing
- Team collaboration on prompt design
- Clear audit trail of prompt evolution

## Error Handling Strategy

LLM outputs are inherently unreliable. The system handles this through:

1. **JSON extraction with fallbacks:** `_extract_json()` tries direct parse → markdown code block → brace matching
2. **Fallback schema construction:** If parsing fails entirely, a valid but minimal schema is returned
3. **Logging:** All parse failures are logged with the raw response for debugging
4. **No crashes:** The committee will always produce *some* output, even if individual agents produce lower-quality results

## Testing Strategy

Tests are designed to run without API calls:

- **MockLLM:** Returns pre-defined JSON responses based on prompt keywords
- **Schema tests:** Validate Pydantic models independently
- **Integration tests:** Run the full committee with mocks
- **Network tests:** Marked with `skipif` for CI environments

This ensures CI runs fast and free while still validating the full pipeline logic.
