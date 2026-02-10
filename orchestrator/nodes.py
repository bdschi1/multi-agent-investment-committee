"""
Node functions for the LangGraph investment committee graph.

Each node wraps an existing agent's .run() or .rebut() method.
Agents are NOT modified — they receive the same inputs as in v1.

Node contract: (state: dict, config: dict) -> dict (partial state update)

Phase C: All nodes accept an optional `config` parameter for
RunnableConfig injection. Non-serializable objects (model, on_status,
tool_registry) are read from config["configurable"] first, falling
back to state for backward compatibility with Phase A/B tests.
"""

from __future__ import annotations

import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional

from langgraph.types import RunnableConfig

from agents.base import clean_json_artifacts
from agents.sector_analyst import SectorAnalystAgent
from agents.risk_manager import RiskManagerAgent
from agents.macro_analyst import MacroAnalystAgent
from agents.portfolio_manager import PortfolioManagerAgent
from orchestrator.committee import ConvictionSnapshot

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config helpers — read from config first, fall back to state
# ---------------------------------------------------------------------------

def _get_model(state: dict, config: Optional[RunnableConfig] = None) -> Any:
    """Get LLM model callable from config (preferred) or state (fallback)."""
    if config:
        cfg = config.get("configurable", {})
        model = cfg.get("model")
        if model is not None:
            return model
    return state.get("model")


def _get_on_status(state: dict, config: Optional[RunnableConfig] = None) -> Optional[Callable]:
    """Get on_status callback from config (preferred) or state (fallback)."""
    if config:
        cfg = config.get("configurable", {})
        cb = cfg.get("on_status")
        if cb is not None:
            return cb
    return state.get("on_status")


def _get_tool_registry(state: dict, config: Optional[RunnableConfig] = None) -> Any:
    """Get tool registry from config (preferred) or state (fallback)."""
    if config:
        cfg = config.get("configurable", {})
        registry = cfg.get("tool_registry")
        if registry is not None:
            return registry
    return state.get("tool_registry")


# ---------------------------------------------------------------------------
# Status helper
# ---------------------------------------------------------------------------

def _status(state: dict, msg: str, config: Optional[RunnableConfig] = None) -> None:
    """Fire the on_status callback if present, and log."""
    cb = _get_on_status(state, config)
    if cb:
        cb(msg)
    logger.info(msg)


# ---------------------------------------------------------------------------
# Parsing failure sentinel detection
# ---------------------------------------------------------------------------

_PARSING_FAILED_SENTINEL = "structured parsing failed"


def _is_parsing_degraded_bull(bull_case) -> bool:
    """Check if the bull case was a parsing-fallback object."""
    evidence = getattr(bull_case, "supporting_evidence", [])
    return any(_PARSING_FAILED_SENTINEL in str(e) for e in evidence)


def _is_parsing_degraded_bear(bear_case) -> bool:
    """Check if the bear case was a parsing-fallback object."""
    risks = getattr(bear_case, "risks", [])
    return any(_PARSING_FAILED_SENTINEL in str(r) for r in risks)


def _is_parsing_degraded_macro(macro_view) -> bool:
    """Check if the macro view was a parsing-fallback object."""
    phase = getattr(macro_view, "economic_cycle_phase", "")
    return _PARSING_FAILED_SENTINEL in str(phase)


def _is_parsing_degraded_memo(memo) -> bool:
    """Check if the committee memo was a parsing-fallback object."""
    factors = getattr(memo, "key_factors", [])
    return any(_PARSING_FAILED_SENTINEL in str(f) for f in factors)


# ---------------------------------------------------------------------------
# Rationale helpers — extract concise rationales for conviction snapshots
# ---------------------------------------------------------------------------

def _clean_rationale_text(text: str) -> str:
    """
    Clean potential JSON artifacts from text before extracting rationale.

    If text looks like JSON (starts with { or contains ":), clean it
    using the shared artifact cleaner. Otherwise return as-is.
    """
    if not text:
        return text
    stripped = text.strip()
    if stripped.startswith("{") or stripped.startswith("[") or ('"' in stripped[:50] and '":' in stripped[:100]):
        cleaned = clean_json_artifacts(stripped)
        return cleaned if cleaned else text
    return text


def _first_sentence(text: str, max_len: int = 200) -> str:
    """Extract the first sentence from text, capped at max_len chars."""
    text = _clean_rationale_text(text)
    if not text:
        return ""
    if "." in text:
        return text.split(".")[0].strip() + "."
    return text[:max_len]


def _bull_rationale(bull_case) -> str:
    """Extract a concise rationale from the bull case output."""
    parts = []
    thesis = getattr(bull_case, "thesis", "")
    if thesis:
        parts.append(_first_sentence(thesis))
    pt = getattr(bull_case, "price_target", "")
    if pt:
        cleaned_pt = _clean_rationale_text(pt)
        parts.append(f"PT: {cleaned_pt[:80]}")
    return " ".join(parts)[:300] if parts else "Initial bull thesis established."


def _bear_rationale(bear_case) -> str:
    """Extract a concise rationale from the bear case output."""
    parts = []
    risks = getattr(bear_case, "risks", [])
    if risks:
        cleaned_risk = _clean_rationale_text(str(risks[0]))
        parts.append(f"Key risk: {cleaned_risk[:120]}")
    rec = getattr(bear_case, "actionable_recommendation", "")
    if rec:
        parts.append(f"Rec: {_clean_rationale_text(rec)}")
    short = getattr(bear_case, "short_thesis", "")
    if short:
        parts.append(f"Short thesis: {_clean_rationale_text(short)[:100]}")
    return " | ".join(parts)[:300] if parts else "Initial bear thesis established."


def _macro_rationale(macro_view) -> str:
    """Extract a concise rationale from the macro view output."""
    parts = []
    impact = getattr(macro_view, "macro_impact_on_stock", "")
    if impact:
        parts.append(_first_sentence(impact))
    cycle = getattr(macro_view, "economic_cycle_phase", "")
    rates = getattr(macro_view, "rate_environment", "")
    if cycle or rates:
        parts.append(f"Cycle: {cycle}, Rates: {rates}")
    return " ".join(parts)[:300] if parts else "Macro environment assessed."


def _debate_rationale(rebuttal, agent_name: str, initial_score: float | None = None) -> str:
    """Extract a concise rationale from a debate rebuttal."""
    parts = []
    revised = getattr(rebuttal, "revised_conviction", None)
    if revised is not None and initial_score is not None:
        delta = revised - initial_score
        direction = "raised" if delta > 0 else "lowered" if delta < 0 else "maintained"
        parts.append(f"{direction} from {initial_score} to {revised}")
    concessions = getattr(rebuttal, "concessions", [])
    if concessions:
        parts.append(f"Conceded: {_clean_rationale_text(str(concessions[0]))[:100]}")
    points = getattr(rebuttal, "points", [])
    if points:
        parts.append(f"Pressed: {_clean_rationale_text(str(points[0]))[:100]}")
    return " — ".join(parts)[:300] if parts else f"{agent_name} conviction updated after debate."


def _pm_rationale(memo) -> str:
    """Extract a concise rationale from the PM's committee memo."""
    parts = []
    summary = getattr(memo, "thesis_summary", "")
    if summary:
        parts.append(_first_sentence(summary))
    rec = getattr(memo, "recommendation", "")
    size = getattr(memo, "position_size", "")
    if rec:
        parts.append(f"{rec}, {size}" if size else rec)
    return " — ".join(parts)[:300] if parts else "PM synthesis complete."


# ---------------------------------------------------------------------------
# Phase 0: Data Gathering
# ---------------------------------------------------------------------------

def gather_data(state: dict, config: Optional[RunnableConfig] = None) -> dict:
    """
    Gather market data for the ticker.

    If context is already populated (pre-gathered by app.py), this
    is a passthrough. Emits the Phase 1 status message so the
    progress bar fires in app.py.
    """
    _status(
        state,
        "Phase 1/3: Sector Analyst, Risk Manager, and Macro Analyst "
        "analyzing in parallel...",
        config,
    )

    if state.get("context"):
        return {}

    from tools.data_aggregator import DataAggregator

    context = DataAggregator.gather_context(
        state["ticker"],
        state.get("user_context", ""),
    )
    return {"context": context}


# ---------------------------------------------------------------------------
# Phase 1: Parallel Analysis Nodes
# ---------------------------------------------------------------------------

def run_sector_analyst(state: dict, config: Optional[RunnableConfig] = None) -> dict:
    """Run the Sector Analyst agent (bull case)."""
    model = _get_model(state, config)
    tool_registry = _get_tool_registry(state, config)

    agent = SectorAnalystAgent(model=model, tool_registry=tool_registry)
    result = agent.run(state["ticker"], state["context"])

    bull_case = result["output"]
    trace = result["trace"]

    result_dict = {
        "bull_case": bull_case,
        "traces": {"sector_analyst": trace},
        "conviction_timeline": [
            ConvictionSnapshot(
                phase="Initial Analysis",
                agent="Sector Analyst",
                score=bull_case.conviction_score,
                score_type="conviction",
                rationale=_bull_rationale(bull_case),
            )
        ],
        "parsing_failures": ["sector_analyst"] if _is_parsing_degraded_bull(bull_case) else [],
    }
    return result_dict


def run_risk_manager(state: dict, config: Optional[RunnableConfig] = None) -> dict:
    """Run the Risk Manager agent (bear case)."""
    model = _get_model(state, config)
    tool_registry = _get_tool_registry(state, config)

    agent = RiskManagerAgent(model=model, tool_registry=tool_registry)
    result = agent.run(state["ticker"], state["context"])

    bear_case = result["output"]
    trace = result["trace"]

    result_dict = {
        "bear_case": bear_case,
        "traces": {"risk_manager": trace},
        "conviction_timeline": [
            ConvictionSnapshot(
                phase="Initial Analysis",
                agent="Risk Manager",
                score=bear_case.bearish_conviction,
                score_type="bearish",
                rationale=_bear_rationale(bear_case),
            )
        ],
        "parsing_failures": ["risk_manager"] if _is_parsing_degraded_bear(bear_case) else [],
    }
    return result_dict


def run_macro_analyst(state: dict, config: Optional[RunnableConfig] = None) -> dict:
    """Run the Macro Analyst agent (top-down context)."""
    model = _get_model(state, config)
    tool_registry = _get_tool_registry(state, config)

    agent = MacroAnalystAgent(model=model, tool_registry=tool_registry)
    result = agent.run(state["ticker"], state["context"])

    macro_view = result["output"]
    trace = result["trace"]

    result_dict = {
        "macro_view": macro_view,
        "traces": {"macro_analyst": trace},
        "conviction_timeline": [
            ConvictionSnapshot(
                phase="Initial Analysis",
                agent="Macro Analyst",
                score=macro_view.macro_favorability,
                score_type="favorability",
                rationale=_macro_rationale(macro_view),
            )
        ],
        "parsing_failures": ["macro_analyst"] if _is_parsing_degraded_macro(macro_view) else [],
    }
    return result_dict


# ---------------------------------------------------------------------------
# Phase 1 Reporter (fan-in convergence point)
# ---------------------------------------------------------------------------

def report_phase1(state: dict, config: Optional[RunnableConfig] = None) -> dict:
    """
    Log Phase 1 scores after all three analysts complete.
    Initializes debate_round counter.
    """
    bc = state.get("bull_case")
    bear = state.get("bear_case")
    mv = state.get("macro_view")

    bull_score = bc.conviction_score if bc else 0
    bear_score = bear.bearish_conviction if bear else 0
    macro_score = mv.macro_favorability if mv else 0

    _status(state, (
        f"  Bull conviction: {bull_score}/10 | "
        f"Bearish conviction: {bear_score}/10 | "
        f"Macro favorability: {macro_score}/10"
    ), config)

    return {"debate_round": 0, "debate_skipped": False}


# ---------------------------------------------------------------------------
# Debate skip marker
# ---------------------------------------------------------------------------

def mark_debate_skipped(state: dict, config: Optional[RunnableConfig] = None) -> dict:
    """Set flag when debate is skipped due to convergence."""
    return {"debate_skipped": True}


# ---------------------------------------------------------------------------
# Phase 2: Debate
# ---------------------------------------------------------------------------

def run_debate_round(state: dict, config: Optional[RunnableConfig] = None) -> dict:
    """
    Execute one debate round: analyst.rebut() and risk_mgr.rebut()
    in parallel via ThreadPoolExecutor, exactly as v1 does.
    """
    model = _get_model(state, config)

    current_round = state.get("debate_round", 0) + 1
    max_rounds = state.get("max_debate_rounds", 2)

    # Emit Phase 2 header on first round
    if current_round == 1:
        _status(
            state,
            "Phase 2/3: Adversarial debate — agents challenging each other...",
            config,
        )

    _status(state, f"  Debate round {current_round}/{max_rounds}...", config)

    analyst = SectorAnalystAgent(model=model)
    risk_mgr = RiskManagerAgent(model=model)

    bull_case = state["bull_case"]
    bear_case = state["bear_case"]

    analyst_rebuttal = None
    risk_rebuttal = None

    with ThreadPoolExecutor(max_workers=2) as executor:
        analyst_future = executor.submit(
            analyst.rebut, state["ticker"], bear_case, bull_case
        )
        risk_future = executor.submit(
            risk_mgr.rebut, state["ticker"], bull_case, bear_case
        )

        try:
            analyst_rebuttal = analyst_future.result()
        except Exception as e:
            logger.warning(f"Analyst rebuttal failed: {e}")
        try:
            risk_rebuttal = risk_future.result()
        except Exception as e:
            logger.warning(f"Risk rebuttal failed: {e}")

    # Build conviction timeline entries for post-debate
    # Look up initial scores for delta computation in rationale
    bull_initial = bull_case.conviction_score if bull_case else None
    bear_initial = bear_case.bearish_conviction if bear_case else None

    round_label = f"Debate Round {current_round}" if max_rounds > 1 else "Post-Debate"

    timeline_entries: list[ConvictionSnapshot] = []
    if analyst_rebuttal and analyst_rebuttal.revised_conviction is not None:
        timeline_entries.append(ConvictionSnapshot(
            phase=round_label,
            agent="Sector Analyst",
            score=analyst_rebuttal.revised_conviction,
            score_type="conviction",
            rationale=_debate_rationale(analyst_rebuttal, "Sector Analyst", bull_initial),
        ))
    if risk_rebuttal and risk_rebuttal.revised_conviction is not None:
        timeline_entries.append(ConvictionSnapshot(
            phase=round_label,
            agent="Risk Manager",
            score=risk_rebuttal.revised_conviction,
            score_type="bearish",
            rationale=_debate_rationale(risk_rebuttal, "Risk Manager", bear_initial),
        ))

    update: dict[str, Any] = {
        "debate_round": current_round,
        "analyst_rebuttal": analyst_rebuttal,
        "risk_rebuttal": risk_rebuttal,
    }

    if timeline_entries:
        update["conviction_timeline"] = timeline_entries

    return update


# ---------------------------------------------------------------------------
# Phase 2 Reporter
# ---------------------------------------------------------------------------

def report_debate_complete(state: dict, config: Optional[RunnableConfig] = None) -> dict:
    """Log debate completion and note convergence if applicable."""
    ar = state.get("analyst_rebuttal")
    rr = state.get("risk_rebuttal")

    _status(state, (
        f"  Debate complete | "
        f"Analyst revised conviction: "
        f"{ar.revised_conviction if ar else 'N/A'} | "
        f"Risk Mgr revised bearish conviction: "
        f"{rr.revised_conviction if rr else 'N/A'}"
    ), config)

    # Note convergence for informational display (debate still ran)
    bull = state.get("bull_case")
    bear = state.get("bear_case")
    if bull and bear:
        spread = abs(bull.conviction_score - bear.bearish_conviction)
        if spread < 2.0:
            _status(state, (
                f"  ℹ️ Convergence noted — bull/bear spread {spread:.1f} "
                f"(< 2.0 threshold)"
            ), config)

    return {}


# ---------------------------------------------------------------------------
# T Signal Computation
# ---------------------------------------------------------------------------

EPSILON = 0.01  # Small constant to avoid zero certainty


def _compute_t_signal(direction: int, raw_confidence: float) -> float:
    """
    Compute the T signal: T = direction * entropy-adjusted confidence.

    Entropy-weighted confidence approach adapted from:
      Darmanin & Vella, "Language Model Guided Reinforcement Learning
      in Quantitative Trading", arXiv:2508.02366v3 (Oct 2025).

    T = direction * C
    C = epsilon + (1 - epsilon) * (1 - H)

    Where:
      - direction in {-1, 0, +1}
      - raw_confidence in [0, 1] — the PM's self-reported confidence
      - H = normalized entropy proxy = 1 - raw_confidence
        (since we can't measure actual LLM token entropy through
         the callable interface, raw_confidence serves as the PM's
         assessment of its own certainty)
      - epsilon = 0.01 (floor to avoid zero)

    Returns:
      T in [-1, 1]. Positive = long with confidence, negative = short with confidence.
    """
    # Clamp inputs
    direction = max(-1, min(1, direction))
    raw_confidence = max(0.0, min(1.0, raw_confidence))

    # H = normalized entropy proxy (low confidence = high entropy)
    h = 1.0 - raw_confidence

    # C = certainty (entropy-adjusted)
    c = EPSILON + (1.0 - EPSILON) * (1.0 - h)

    # T = direction * certainty
    t = direction * c

    return round(t, 4)


# ---------------------------------------------------------------------------
# Phase 3: Portfolio Manager
# ---------------------------------------------------------------------------

def run_portfolio_manager(state: dict, config: Optional[RunnableConfig] = None) -> dict:
    """
    Run the PM synthesis — identical context assembly to v1
    (orchestrator/committee.py lines 238-247).

    Phase C additions:
    - pm_guidance: user-provided guidance from HITL review step
    - prior_analyses: session memory of prior runs for this ticker
    """
    model = _get_model(state, config)
    tool_registry = _get_tool_registry(state, config)

    _status(
        state,
        "Phase 3/3: Portfolio Manager synthesizing committee decision...",
        config,
    )

    pm = PortfolioManagerAgent(model=model, tool_registry=tool_registry)

    # Retrieve prior analyses from session memory
    prior_analyses: list[dict] = []
    try:
        from orchestrator.memory import get_prior_analyses
        prior_analyses = get_prior_analyses(state["ticker"])
    except Exception as e:
        logger.debug(f"Session memory unavailable: {e}")

    # Assemble PM context exactly as v1 does, plus Phase C additions
    pm_context: dict[str, Any] = {
        "bull_case": state["bull_case"],
        "bear_case": state["bear_case"],
        "macro_view": state.get("macro_view"),
        "debate": {
            "analyst_rebuttal": (
                state["analyst_rebuttal"].model_dump()
                if state.get("analyst_rebuttal") else {}
            ),
            "risk_rebuttal": (
                state["risk_rebuttal"].model_dump()
                if state.get("risk_rebuttal") else {}
            ),
        },
        "user_context": state.get("context", {}).get("user_context", ""),
        "pm_guidance": state.get("pm_guidance", ""),
        "prior_analyses": prior_analyses,
    }

    pm_result = pm.run(state["ticker"], pm_context)
    memo = pm_result["output"]
    trace = pm_result["trace"]

    # Compute T signal from PM's direction and confidence
    if memo:
        direction = getattr(memo, "position_direction", 0)
        raw_conf = getattr(memo, "raw_confidence", 0.5)
        t_signal = _compute_t_signal(direction, raw_conf)
        memo.t_signal = t_signal
        logger.info(
            f"T signal: direction={direction}, raw_confidence={raw_conf:.3f}, "
            f"T={t_signal:.4f}"
        )

    timeline_entries: list[ConvictionSnapshot] = []
    if memo:
        timeline_entries.append(ConvictionSnapshot(
            phase="PM Decision",
            agent="Portfolio Manager",
            score=memo.conviction,
            score_type="conviction",
            rationale=_pm_rationale(memo),
        ))

    update: dict[str, Any] = {
        "committee_memo": memo,
        "traces": {"portfolio_manager": trace},
    }

    if timeline_entries:
        update["conviction_timeline"] = timeline_entries

    if memo and _is_parsing_degraded_memo(memo):
        update["parsing_failures"] = ["portfolio_manager"]

    return update


# ---------------------------------------------------------------------------
# Finalize
# ---------------------------------------------------------------------------

def finalize(state: dict, config: Optional[RunnableConfig] = None) -> dict:
    """Compute final timing and token totals, log completion."""
    duration_ms = (time.time() - state["start_time"]) * 1000
    total_tokens = sum(t.total_tokens for t in state.get("traces", {}).values())

    memo = state.get("committee_memo")
    if memo:
        _status(state, (
            f"  Decision: {memo.recommendation} | "
            f"Conviction: {memo.conviction}/10 | "
            f"Size: {memo.position_size}"
        ), config)
    _status(state, f"\nCommittee complete in {duration_ms / 1000:.1f}s", config)

    return {
        "total_duration_ms": duration_ms,
        "total_tokens": total_tokens,
    }
