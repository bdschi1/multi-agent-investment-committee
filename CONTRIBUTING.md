# Contributing

## Development Setup

```bash
git clone https://github.com/bdschi1/multi-agent-investment-committee.git
cd multi-agent-investment-committee
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env  # add your API key(s)
```

### Optional dependency groups

- **Live Bloomberg data:** `pip install -e ".[bloomberg]"`
- **Live IBKR data:** `pip install -e ".[ibkr]"`

These are not required for development or testing -- the default Yahoo provider works without extra installs.

## Architecture Overview

The pipeline is a **LangGraph StateGraph** following a fan-out/fan-in pattern:

```
gather_data -> run_xai_analysis -> [sector_analyst, short_analyst, risk_manager, macro_analyst] (parallel)
    -> adversarial_debate (Long vs Short, up to N rounds)
    -> portfolio_manager -> optimizer -> finalize
```

Each agent follows a **Think-Plan-Act-Reflect** cycle implemented in its `act()` method. Agents reason about their domain, plan analysis steps, produce structured JSON output, then self-check for consistency before returning.

Key packages: `agents/`, `orchestrator/`, `optimizer/`, `xai/`, `backtest/`, `tools/`, `api/`.

## Code Style

- Lint with `ruff check .`
- Format with `ruff format .`
- Auto-fix lint issues: `ruff check --fix .`
- Type hints encouraged

## Testing

```bash
pytest tests/ -v          # full suite (~595 tests)
pytest tests/ -k "test_name"  # single test by name
```

Tests use **mock LLM fixtures** -- no API keys are needed to run the test suite. The `mock_model` fixture returns canned JSON responses. Optimizer tests mock `build_universe()` with synthetic price data, and vol intelligence tests use synthetic GBM prices.

## Pull Requests

1. Create a feature branch from `main`
2. Make focused, single-purpose commits
3. Ensure all tests pass and `ruff check .` is clean before submitting
4. Open a PR with a clear description of changes
