# v2: LangGraph State-Machine Orchestration

**Status:** Planning
**Migration from:** v1 ThreadPoolExecutor-based parallel orchestrator
**Migration to:** LangGraph state-machine with dynamic tool calling, conditional edges, and persistent state

---

## Why v2

v1 works but has architectural ceilings:

1. **No dynamic tool calling.** Agents receive pre-fetched data. They can't decide mid-reasoning to fetch additional data (e.g., pull peer comparison after discovering valuation is extreme).
2. **Fixed execution graph.** The 3-phase flow (parallel analysis, debate, synthesis) is hardcoded. Can't add conditional paths like "skip debate if bull and bear agree within 2 points."
3. **No memory.** Each run is stateless. The PM can't reference a prior analysis of the same ticker or track how conviction evolved across multiple sessions.
4. **No human-in-the-loop.** There's no breakpoint where a user can intervene mid-workflow (e.g., redirect the debate, inject new information after Phase 1).
5. **Scaling agents is manual.** Adding the Macro Analyst required touching 8 files. A graph-based approach makes agent addition declarative.

---

## Architecture

```
                          ┌─────────────────────┐
                          │   START              │
                          │   (validate input)   │
                          └──────────┬──────────┘
                                     │
                          ┌──────────▼──────────┐
                          │   DATA GATHERING     │
                          │   (tool node)        │
                          │   market, news,      │
                          │   financials, SEC     │
                          └──────────┬──────────┘
                                     │
                  ┌──────────────────┼──────────────────┐
                  │                  │                   │
         ┌───────▼───────┐ ┌───────▼───────┐ ┌────────▼────────┐
         │ SECTOR ANALYST │ │ RISK MANAGER  │ │ MACRO ANALYST   │
         │ (agent node)   │ │ (agent node)  │ │ (agent node)    │
         │ think→plan→    │ │ think→plan→   │ │ think→plan→     │
         │ tool→act→      │ │ tool→act→     │ │ tool→act→       │
         │ reflect        │ │ reflect       │ │ reflect         │
         └───────┬───────┘ └───────┬───────┘ └────────┬────────┘
                  │                  │                   │
                  └────────┬─────────┘                   │
                           │                             │
                ┌──────────▼──────────┐                  │
                │  CONVERGENCE CHECK  │◄─────────────────┘
                │  (conditional edge) │
                │  if |bull - (10-bear)| < 2             │
                │  → skip debate      │
                └──────────┬──────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
    ┌─────────▼─────────┐    ┌─────────▼─────────┐
    │  DEBATE (loop)     │    │  SKIP TO PM       │
    │  bull rebuts bear  │    │  (conditional)    │
    │  bear rebuts bull  │    │                   │
    │  ──────────────    │    │                   │
    │  convergence_check │    │                   │
    │  if converged or   │    │                   │
    │  max_rounds → exit │    │                   │
    └─────────┬─────────┘    └─────────┬─────────┘
              │                         │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │  HUMAN CHECKPOINT       │
              │  (optional breakpoint)  │
              │  user can inject context│
              │  or redirect debate     │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │  PORTFOLIO MANAGER      │
              │  (agent node)           │
              │  reads full state       │
              │  synthesizes + decides  │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │  END                    │
              │  (emit CommitteeResult) │
              └─────────────────────────┘
```

---

## State Schema

LangGraph uses a typed state dict that flows through every node. This replaces v1's `CommitteeResult` dataclass.

```python
from typing import TypedDict, Annotated, Optional
from langgraph.graph import add_messages

class CommitteeState(TypedDict):
    # Input
    ticker: str
    user_context: str
    provider: str
    debate_rounds: int

    # Data (populated by data-gathering node)
    market_data: dict
    financial_metrics: dict
    news: list[dict]
    valuation_assessment: dict
    quality_score: dict
    sec_filings: list[dict]         # NEW in v2

    # Agent outputs
    bull_case: Optional[dict]
    bear_case: Optional[dict]
    macro_view: Optional[dict]

    # Debate
    debate_history: Annotated[list[dict], add_messages]  # append-only
    debate_round: int
    convergence_reached: bool

    # PM
    committee_memo: Optional[dict]

    # Traces
    reasoning_traces: dict[str, list]
    conviction_timeline: list[dict]

    # Meta
    errors: list[str]
    total_duration_ms: float
```

---

## Node Definitions

### 1. `data_gathering_node`
Replaces v1's `DataAggregator.gather_context()`. Same logic but writes to state instead of returning a dict.

**v2 addition:** SEC filing retrieval via EDGAR API or a RAG pipeline over 10-K/10-Q.

```python
def data_gathering_node(state: CommitteeState) -> CommitteeState:
    ticker = state["ticker"]
    # existing: yfinance, RSS, derived metrics
    # new: SEC EDGAR fetch, optional RAG over filings
    return {"market_data": ..., "financial_metrics": ..., ...}
```

### 2. Agent Nodes (x3, parallel via `Send`)

Each agent node wraps the existing `BaseInvestmentAgent.run()` but now agents can **invoke tools mid-reasoning** via LangGraph's tool-calling pattern.

```python
def sector_analyst_node(state: CommitteeState) -> CommitteeState:
    agent = SectorAnalystAgent(model=get_model(state))
    # Agent can call tools dynamically:
    #   - fetch_peer_comparison(ticker, peers)
    #   - get_insider_transactions(ticker)
    #   - search_sec_filings(ticker, query)
    result = agent.run(state["ticker"], state)
    return {"bull_case": result["output"].model_dump(), ...}
```

**Parallel execution:** LangGraph's `Send` API replaces ThreadPoolExecutor:

```python
def fan_out_analysts(state: CommitteeState):
    return [
        Send("sector_analyst", state),
        Send("risk_manager", state),
        Send("macro_analyst", state),
    ]
```

### 3. `convergence_check_node` (conditional edge)

New in v2. Checks whether bull and bear are close enough to skip debate:

```python
def should_debate(state: CommitteeState) -> str:
    bull_score = state["bull_case"]["conviction_score"]
    bear_score = state["bear_case"]["risk_score"]
    spread = abs(bull_score - (10 - bear_score))
    if spread < 2.0:
        return "skip_to_pm"  # strong agreement, debate adds little
    return "debate"
```

### 4. `debate_node` (loop with exit condition)

Replaces v1's `for round_num in range(...)` loop. LangGraph manages the loop via conditional edges.

```python
def debate_node(state: CommitteeState) -> CommitteeState:
    round_num = state["debate_round"] + 1
    # Bull rebuts bear, bear rebuts bull (parallel via Send)
    ...
    return {
        "debate_history": [bull_rebuttal, bear_rebuttal],
        "debate_round": round_num,
        "convergence_reached": check_convergence(...)
    }

def should_continue_debate(state: CommitteeState) -> str:
    if state["convergence_reached"]:
        return "pm"
    if state["debate_round"] >= state["debate_rounds"]:
        return "pm"
    return "debate"  # loop back
```

### 5. `human_checkpoint_node`

New in v2. Optional breakpoint before PM synthesis. User can:
- Inject additional context after seeing analyst outputs
- Override debate conclusions
- Force a re-run of a specific agent

```python
from langgraph.checkpoint import MemorySaver

# Graph compiled with checkpointer
graph = builder.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["portfolio_manager"],  # pause here
)
```

### 6. `portfolio_manager_node`

Same as v1 but reads the full state graph instead of a manually-assembled context dict.

---

## Dynamic Tool Calling

The biggest upgrade in v2. Agents can invoke tools mid-reasoning instead of receiving pre-fetched data.

### New Tools for v2

| Tool | Description | Which Agents Use It |
|------|-------------|-------------------|
| `fetch_peer_comparison` | Compare valuation multiples vs. 5 closest peers | Sector Analyst, PM |
| `get_insider_transactions` | Recent insider buys/sells from SEC Form 4 | Risk Manager |
| `search_sec_filings` | RAG search over 10-K, 10-Q, 8-K text | All |
| `get_options_flow` | Unusual options activity, put/call ratio | Risk Manager |
| `get_institutional_ownership` | Top holders, recent 13F changes | Sector Analyst, PM |
| `get_earnings_transcript` | Latest earnings call Q&A excerpts | All |
| `get_macro_indicators` | FRED data: GDP, CPI, unemployment, ISM | Macro Analyst |
| `get_sector_etf_flows` | ETF fund flow data for sector rotation | Macro Analyst |
| `calculate_dcf` | Run a simplified DCF with provided assumptions | Sector Analyst |
| `correlation_analysis` | Cross-asset correlation matrix | Macro Analyst |

### Tool-Calling Pattern

```python
from langchain_core.tools import tool

@tool
def fetch_peer_comparison(ticker: str, peers: list[str]) -> dict:
    """Compare valuation multiples (P/E, EV/EBITDA, P/S) across peers."""
    ...

# Agent node with tool binding
tools = [fetch_peer_comparison, search_sec_filings, ...]
model_with_tools = model.bind_tools(tools)
```

Agents decide in their **plan** step which tools to call. The LangGraph `ToolNode` handles execution and feeds results back into the agent's context for the **act** step.

---

## Persistent Memory

### Session Memory (SQLite via LangGraph Checkpointer)

Every run is checkpointed. Benefits:
- Resume interrupted analyses
- Compare current run to previous run of same ticker
- PM can reference historical conviction evolution

```python
from langgraph.checkpoint.sqlite import SqliteSaver

checkpointer = SqliteSaver.from_conn_string("runs/committee_memory.db")
graph = builder.compile(checkpointer=checkpointer)
```

### Cross-Session Memory (Vector Store)

Store completed analyses in a vector store. When analyzing a new ticker, agents can retrieve:
- Prior analyses of the same ticker (temporal comparison)
- Analyses of peer companies (cross-reference)
- Historical accuracy tracking (was the PM right last time?)

```python
# Pseudocode for memory retrieval
prior_analyses = vector_store.similarity_search(
    f"investment analysis {ticker}",
    k=3,
    filter={"ticker": ticker}
)
```

---

## Graph Construction

```python
from langgraph.graph import StateGraph, END

builder = StateGraph(CommitteeState)

# Nodes
builder.add_node("data_gathering", data_gathering_node)
builder.add_node("sector_analyst", sector_analyst_node)
builder.add_node("risk_manager", risk_manager_node)
builder.add_node("macro_analyst", macro_analyst_node)
builder.add_node("debate", debate_node)
builder.add_node("portfolio_manager", portfolio_manager_node)

# Edges
builder.set_entry_point("data_gathering")
builder.add_conditional_edges(
    "data_gathering",
    fan_out_analysts,  # parallel via Send
)
builder.add_edge("sector_analyst", "convergence_check")
builder.add_edge("risk_manager", "convergence_check")
builder.add_edge("macro_analyst", "convergence_check")
builder.add_conditional_edges(
    "convergence_check",
    should_debate,
    {"debate": "debate", "skip_to_pm": "portfolio_manager"},
)
builder.add_conditional_edges(
    "debate",
    should_continue_debate,
    {"debate": "debate", "pm": "portfolio_manager"},
)
builder.add_edge("portfolio_manager", END)

# Compile with checkpointer
graph = builder.compile(
    checkpointer=SqliteSaver.from_conn_string("runs/memory.db"),
    interrupt_before=["portfolio_manager"],  # optional human-in-the-loop
)
```

---

## File Changes from v1

| v1 File | v2 Change |
|---------|-----------|
| `orchestrator/committee.py` | Replaced entirely by `orchestrator/graph.py` (LangGraph StateGraph) |
| `orchestrator/nodes.py` | NEW — all node functions (data gathering, agents, debate, PM) |
| `orchestrator/state.py` | NEW — `CommitteeState` TypedDict |
| `orchestrator/tools_v2.py` | NEW — LangChain `@tool` definitions for dynamic calling |
| `orchestrator/memory.py` | NEW — checkpointer + vector store setup |
| `agents/base.py` | Updated: `BaseInvestmentAgent` gets `tools` parameter, tool-calling in plan/act |
| `agents/*.py` | Updated: agent prompts include tool descriptions, agents can request tool calls |
| `tools/sec_filings.py` | NEW — EDGAR API + RAG over SEC filings |
| `tools/options_flow.py` | NEW — options data (cboe or similar) |
| `tools/macro_indicators.py` | NEW — FRED API wrapper |
| `config/settings.py` | Add: `enable_memory`, `memory_db_path`, `enable_human_checkpoint` |
| `app.py` | Update: stream graph execution via `graph.astream()`, add human checkpoint UI |
| `pyproject.toml` | Add: `langgraph`, `langchain-core`, `langchain-anthropic`, `faiss-cpu` or `chromadb` |

---

## Migration Strategy

### Phase A: LangGraph Core (no new tools)

Port the existing 3-phase flow to LangGraph without adding new capabilities. This validates the graph works identically to v1.

1. Define `CommitteeState` TypedDict
2. Port each agent's `run()` into a LangGraph node
3. Implement `Send` for parallel analyst execution
4. Implement debate loop via conditional edges
5. Add convergence check (new behavior, low risk)
6. Wire up `graph.compile()` and verify outputs match v1
7. Update `app.py` to call `graph.invoke()` instead of `committee.run()`
8. All 9+ existing tests should pass against the new graph

**Success criteria:** Same outputs, same wall-clock performance, same test results.

### Phase B: Dynamic Tool Calling

Add tool-calling capability to agents. They keep the think-plan-act-reflect loop but can now invoke tools between plan and act.

1. Define tools as `@tool` decorated functions
2. Bind tools to model via `model.bind_tools()`
3. Add `ToolNode` to the graph for each agent
4. Update agent prompts to describe available tools
5. Implement tool-calling loop: agent decides tools in plan, ToolNode executes, results feed act

**Success criteria:** Agents make at least 1 dynamic tool call per run. Quality of analysis improves measurably (more specific data points, peer comparisons).

### Phase C: Memory + Human-in-the-Loop

1. Add SQLite checkpointer for session persistence
2. Add vector store for cross-session memory
3. Implement `interrupt_before` for human checkpoint
4. Update Gradio UI with checkpoint controls (approve/redirect/inject)
5. Add historical accuracy tracking

**Success criteria:** Users can pause, inject context, and resume. PM references prior analyses when available.

### Phase D: New Data Sources

1. SEC EDGAR integration (10-K, 10-Q, 8-K)
2. Earnings transcript retrieval
3. Options flow data
4. Institutional ownership (13F)
5. FRED macro indicators

**Success criteria:** Each new data source is accessible via tool calling and demonstrably improves analysis quality.

---

## Dependencies

```toml
[project.optional-dependencies]
langgraph = [
    "langgraph>=0.2.0",
    "langchain-core>=0.3.0",
    "langchain-anthropic>=0.3.0",
    "langchain-google-genai>=2.0.0",
    "langchain-openai>=0.3.0",
]
memory = [
    "chromadb>=0.5.0",        # or faiss-cpu
    "langchain-chroma>=0.2.0",
]
sec = [
    "sec-edgar-downloader>=5.0.0",
    "unstructured>=0.15.0",   # for PDF/HTML parsing
]
```

---

## Open Questions

1. **LangGraph streaming vs Gradio:** LangGraph's `astream_events()` emits fine-grained events. Can we pipe these directly to Gradio's streaming interface, or do we need an intermediary?

2. **Cost control with dynamic tools:** If agents can call tools freely, how do we prevent runaway API costs? Likely need a token/call budget per agent.

3. **Vector store choice:** ChromaDB (simpler, embedded) vs. FAISS (faster, but no metadata filtering) vs. Pinecone (hosted, but adds dependency). ChromaDB is the likely choice for a self-contained demo.

4. **Backwards compatibility:** Should v2 support running without LangGraph installed (falling back to v1 ThreadPoolExecutor)? This adds complexity but keeps the project accessible.

5. **SEC EDGAR rate limits:** EDGAR has strict rate limiting (10 req/sec). Need to cache filings locally and implement respectful crawling.

---

## Timeline Estimate

| Phase | Scope | Estimate |
|-------|-------|----------|
| A — LangGraph Core | Port existing flow to state graph | 2-3 days |
| B — Dynamic Tools | Tool calling, ToolNode integration | 2-3 days |
| C — Memory + HITL | Checkpointer, vector store, UI | 3-4 days |
| D — New Data Sources | SEC, earnings, options, FRED | 3-5 days |
| **Total** | | **10-15 days** |

---

## Success Metrics

- **Functional parity:** All v1 test cases pass against v2 graph
- **Dynamic tool usage:** Agents make 2+ tool calls per run on average
- **Analysis quality:** Blind evaluation — v2 analyses should score higher on specificity, non-consensus insights, and data backing
- **Latency:** No more than 20% wall-clock regression from v1 (parallel execution preserved)
- **Memory utility:** PM demonstrably references prior analyses when available
- **Human-in-the-loop:** Users can pause and redirect in under 2 clicks
