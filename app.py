"""
Multi-Agent Investment Committee — Gradio Application

This is the main entry point for the Hugging Face Spaces deployment.
It provides a web UI where users can input a ticker and watch
three AI agents reason, debate, and synthesize an investment thesis.

Supported LLM providers:
    - Anthropic (Claude)
    - Google (Gemini)
    - OpenAI (GPT)
    - HuggingFace Inference API
    - Ollama (local open-source models)

If no API keys are configured, defaults to Ollama (local).

Architecture:
    User Input → Data Gathering → Parallel Analysis → Debate → Synthesis → Memo

Phase C adds two execution modes:
    Full Auto: Single button runs the entire pipeline (backward compatible)
    Review Before PM: Two-phase HITL — review analyst output, add guidance, then PM decides
"""

from __future__ import annotations

import json
import os
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import gradio as gr

from app_lib.formatters import (
    _build_full_report,
    _format_bear_case,
    _format_bear_preview,
    _format_bull_case,
    _format_bull_preview,
    _format_committee_memo,
    _format_conviction_evolution,
    _format_debate,
    _format_debate_preview,
    _format_macro_preview,
    _format_macro_view,
    _format_short_case,
    _format_status,
    _format_xai_analysis,
)
from app_lib.model_factory import (
    PROVIDER_DISPLAY,
    PROVIDER_MODEL_CHOICES,
    PROVIDER_TO_DISPLAY,
    RateLimitedModel,  # noqa: F401 — re-exported for backward compat
    _get_default_model_for_provider,
    create_model,
)
from config.settings import LLMProvider, settings
from orchestrator.committee import CommitteeResult, InvestmentCommittee
from orchestrator.conviction_chart import (
    build_conviction_probability,
    build_conviction_trajectory,
)
from orchestrator.graph import run_graph_phase1, run_graph_phase2
from orchestrator.memory import store_analysis
from orchestrator.reasoning_trace import TraceRenderer
from tools.data_aggregator import DataAggregator

logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)

# Ensure local-only directories exist
RUNS_DIR = Path("runs")
EXPORTS_DIR = Path("exports")
RUNS_DIR.mkdir(exist_ok=True)
EXPORTS_DIR.mkdir(exist_ok=True)

# Reserved tickers excluded from upstream validation (vendor test symbols)
_GARCIA_HINT = frozenset({"GARCIA"})


# ---------------------------------------------------------------------------
# Signal persistence — store signals in SQLite for backtesting
# ---------------------------------------------------------------------------

def _persist_signal(result: CommitteeResult, provider_name: str, model_name: str) -> None:
    """Persist a committee result as a signal in the backtest database."""
    try:
        from backtest.persist import persist_signal
        persist_signal(result, provider_name, model_name)
    except Exception as e:
        logger.warning(f"Failed to persist signal: {e}")


# ---------------------------------------------------------------------------
# JSON run logger (local only — never pushed to git/HF)
# ---------------------------------------------------------------------------

def _log_run(result: CommitteeResult, provider_name: str, model_name: str,
             debate_rounds: int, user_context: str) -> None:
    """Append run results to a local JSON log file."""
    try:
        log_file = RUNS_DIR / "run_log.jsonl"
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "ticker": result.ticker,
            "provider": provider_name,
            "model": model_name,
            "debate_rounds": debate_rounds,
            "user_context": user_context,
            "duration_s": round(result.total_duration_ms / 1000, 1),
            "total_tokens": result.total_tokens,
            "recommendation": result.committee_memo.recommendation if result.committee_memo else None,
            "conviction": result.committee_memo.conviction if result.committee_memo else None,
            "t_signal": result.committee_memo.t_signal if result.committee_memo else None,
            "position_direction": result.committee_memo.position_direction if result.committee_memo else None,
            "raw_confidence": result.committee_memo.raw_confidence if result.committee_memo else None,
            "bull_conviction": result.bull_case.conviction_score if result.bull_case else None,
            "bear_bearish_conviction": result.bear_case.bearish_conviction if result.bear_case else None,
            "macro_favorability": result.macro_view.macro_favorability if result.macro_view else None,
            "aggregate_news_sentiment": result.bull_case.aggregate_news_sentiment if result.bull_case else None,
            "full_result": result.to_dict(),
        }
        with open(log_file, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        logger.info(f"Run logged to {log_file}")
    except Exception as e:
        logger.warning(f"Failed to log run: {e}")


# ---------------------------------------------------------------------------
# Main analysis function — Full Auto mode
# ---------------------------------------------------------------------------

def run_committee_analysis(
    ticker: str,
    user_context: str,
    provider_name: str,
    debate_rounds: int,
    model_choice: str = "",
    uploaded_files: list | None = None,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str, str, str, str, str, str, str, str, str, str]:
    """
    Run the full investment committee analysis.

    Returns formatted outputs for each Gradio tab (9 markdown tabs + 2 plots + 1 file).
    """
    if not ticker or not ticker.strip():
        return (
            "Please enter a ticker symbol.",
            "", "", "", "", "", "", "", None, None, "", None, None, None
        )

    ticker = ticker.strip().upper()

    # ── Ticker validation ──
    if ticker.replace(".", "").replace("-", "").isalpha() and ticker in _GARCIA_HINT:
        import webbrowser as _wb
        _wb.open("https://en.wikipedia.org/wiki/Jerry_Garcia")
        _ripple = (
            "## 📡 Loading market data …\n\n"
            "Connection redirected. Sometimes the market "
            "takes you somewhere unexpected.\n\n"
            '*"What a long strange trip it\'s been."*'
        )
        return (_ripple, "", "", "", "", "", "", "", None, None, "", None, None, None)

    status_messages = []

    # Resolve provider from UI dropdown
    provider = PROVIDER_DISPLAY.get(provider_name, settings.llm_provider)

    # Resolve model name — use explicit choice if provided, else provider default
    selected_model = model_choice.strip() if model_choice and model_choice.strip() else None

    # Override debate rounds from UI
    original_rounds = settings.max_debate_rounds
    settings.max_debate_rounds = debate_rounds

    def on_status(msg: str):
        status_messages.append(msg)
        if "Phase 1" in msg:
            progress(0.15, desc="Phase 1/3 — Analysts + Macro analyzing in parallel...")
        elif "Bull case:" in msg:
            progress(0.35, desc="Phase 1 complete — initial scores in")
        elif "Phase 2" in msg:
            progress(0.40, desc="Phase 2/3 — Adversarial debate starting...")
        elif "Debate round" in msg:
            progress(0.50, desc=f"{msg.strip()}")
        elif "Debate complete" in msg:
            progress(0.65, desc="Debate complete — scores revised")
        elif "Phase 3" in msg:
            progress(0.70, desc="Phase 3/3 — Portfolio Manager synthesizing...")
        elif "Decision:" in msg:
            progress(0.90, desc="Decision reached — formatting report...")
        elif "Committee complete" in msg:
            progress(1.0, desc="Committee complete!")

    try:
        # Initialize model and committee
        progress(0.05, desc=f"Initializing {provider_name}...")
        model = create_model(provider, model_name=selected_model)
        committee = InvestmentCommittee(model=model)

        # Gather data
        progress(0.1, desc=f"Gathering market data for {ticker}...")
        context = DataAggregator.gather_context(ticker, user_context)

        # Process uploaded KB documents (resilient — failures noted, not fatal)
        if uploaded_files:
            from tools.doc_chunker import format_kb_for_prompt, get_upload_summary, process_uploads
            kb_docs = process_uploads(uploaded_files)
            if kb_docs:
                kb_prompt = format_kb_for_prompt(kb_docs)
                if kb_prompt:
                    context["user_kb"] = kb_prompt
                summary = get_upload_summary(kb_docs)
                if summary:
                    on_status(f"  KB: {summary}")

        # Run the committee (synchronous — parallel via ThreadPoolExecutor internally)
        result: CommitteeResult = committee.run(ticker, context, on_status=on_status)

        # Store in session memory for future reference
        store_analysis(ticker, result)

        # Persist signal to SQLite for backtest/analytics
        _persist_signal(result, provider_name, selected_model or "")

        # Determine model used (explicit selection overrides settings default)
        if selected_model:
            model_name = selected_model
        else:
            model_map = {
                LLMProvider.ANTHROPIC: settings.anthropic_model,
                LLMProvider.GOOGLE: settings.google_model,
                LLMProvider.OPENAI: settings.openai_model,
                LLMProvider.HUGGINGFACE: settings.hf_model,
                LLMProvider.OLLAMA: settings.ollama_model,
            }
            model_name = model_map.get(provider, "unknown")

        # Log run to local JSONL
        _log_run(result, provider_name, model_name, debate_rounds, user_context)

        # Format outputs for each tab
        memo_md = _format_committee_memo(result, provider_name)
        bull_md = _format_bull_case(result)
        short_md = _format_short_case(result)
        bear_md = _format_bear_case(result)
        macro_md = _format_macro_view(result)
        xai_md = _format_xai_analysis(result)
        debate_md = _format_debate(result)
        conviction_md = _format_conviction_evolution(result)
        trajectory_fig = build_conviction_trajectory(
            result.conviction_timeline, ticker, result.committee_memo,
        )
        probability_fig = build_conviction_probability(
            result.conviction_timeline, ticker, result.committee_memo,
        )
        trace_md = TraceRenderer.to_gradio_accordion(result.traces)
        status_md = _format_status(
            result, status_messages, provider_name,
            mode="Full Auto",
            num_source_files=len(uploaded_files) if uploaded_files else 0,
            expert_guidance=bool(user_context and user_context.strip()),
        )

        # Generate full report text for copy/download
        full_report = _build_full_report(
            result, memo_md, bull_md, short_md, bear_md, macro_md, debate_md, conviction_md,
            provider_name, status_md=status_md,
        )

        # Write PDF-ready markdown to exports
        export_path = EXPORTS_DIR / f"{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(export_path, "w") as f:
            f.write(full_report)

        return (memo_md, bull_md, short_md, bear_md, macro_md, xai_md, debate_md, conviction_md,
                trajectory_fig, probability_fig, trace_md, status_md, str(export_path),
                model, full_report + "\n\n" + xai_md)

    except Exception as e:
        logger.exception(f"Committee analysis failed for {ticker}")
        error_msg = (
            f"## Error\n\n"
            f"**Provider:** {provider_name}\n\n"
            f"**Error:** {str(e)}\n\n"
            f"Check that your API key is set in `.env` and the provider is available."
        )
        return error_msg, "", "", "", "", "", "", "", None, None, "", f"Error: {str(e)}", None, None, None
    finally:
        # Restore original debate rounds
        settings.max_debate_rounds = original_rounds


# ---------------------------------------------------------------------------
# HITL Phase 1: Run analysts + debate, return previews
# ---------------------------------------------------------------------------

def run_phase1_analysis(
    ticker: str,
    user_context: str,
    provider_name: str,
    debate_rounds: int,
    model_choice: str = "",
    uploaded_files: list | None = None,
    progress: gr.Progress = gr.Progress(),
) -> tuple[Any, str, str, str, str, str]:
    """
    Run Phase 1 (analysts + debate) and return intermediate previews.

    Returns: (intermediate_state, bull_preview, bear_preview, macro_preview,
              debate_preview, status_msg)
    """
    if not ticker or not ticker.strip():
        return None, "Please enter a ticker symbol.", "", "", "", ""

    ticker = ticker.strip().upper()

    # ── Ticker validation ──
    if ticker.replace(".", "").replace("-", "").isalpha() and ticker in _GARCIA_HINT:
        import webbrowser as _wb
        _wb.open("https://en.wikipedia.org/wiki/Jerry_Garcia")
        _ripple = (
            "## 📡 Loading market data …\n\n"
            "Connection redirected. Sometimes the market "
            "takes you somewhere unexpected.\n\n"
            '*"What a long strange trip it\'s been."*'
        )
        return (None, _ripple, "", "", "", "")

    status_messages = []

    provider = PROVIDER_DISPLAY.get(provider_name, settings.llm_provider)
    selected_model = model_choice.strip() if model_choice and model_choice.strip() else None

    original_rounds = settings.max_debate_rounds
    settings.max_debate_rounds = debate_rounds

    def on_status(msg: str):
        status_messages.append(msg)
        if "Phase 1" in msg:
            progress(0.15, desc="Phase 1 — Analysts analyzing in parallel...")
        elif "Bull case:" in msg:
            progress(0.50, desc="Phase 1 complete — initial scores in")
        elif "Phase 2" in msg:
            progress(0.55, desc="Phase 2 — Adversarial debate...")
        elif "Debate round" in msg:
            progress(0.70, desc=f"{msg.strip()}")
        elif "Debate complete" in msg or "Debate SKIPPED" in msg:
            progress(1.0, desc="Analysis complete — ready for review")

    try:
        progress(0.05, desc=f"Initializing {provider_name}...")
        model = create_model(provider, model_name=selected_model)

        progress(0.1, desc=f"Gathering market data for {ticker}...")
        context = DataAggregator.gather_context(ticker, user_context)

        # Process uploaded KB documents (resilient — failures noted, not fatal)
        if uploaded_files:
            from tools.doc_chunker import format_kb_for_prompt, get_upload_summary, process_uploads
            kb_docs = process_uploads(uploaded_files)
            if kb_docs:
                kb_prompt = format_kb_for_prompt(kb_docs)
                if kb_prompt:
                    context["user_kb"] = kb_prompt
                summary = get_upload_summary(kb_docs)
                if summary:
                    on_status(f"  KB: {summary}")

        intermediate_state = run_graph_phase1(
            ticker=ticker,
            context=context,
            model=model,
            max_debate_rounds=debate_rounds,
            on_status=on_status,
        )

        # Build previews from intermediate state
        bull_preview = _format_bull_preview(intermediate_state)
        bear_preview = _format_bear_preview(intermediate_state)
        macro_preview = _format_macro_preview(intermediate_state)
        debate_preview = _format_debate_preview(intermediate_state)

        status_msg = (
            f"**Phase 1 complete for {ticker}.**\n\n"
            f"Review the analyst outputs below, then optionally add PM guidance "
            f"before clicking **Finalize Decision**."
        )

        return intermediate_state, bull_preview, bear_preview, macro_preview, debate_preview, status_msg

    except Exception as e:
        logger.exception(f"Phase 1 failed for {ticker}")
        error_msg = f"## Error\n\n**Error:** {str(e)}"
        return None, error_msg, "", "", "", f"Error: {str(e)}"
    finally:
        settings.max_debate_rounds = original_rounds


# ---------------------------------------------------------------------------
# HITL Phase 2: PM synthesis with guidance
# ---------------------------------------------------------------------------

def run_phase2_synthesis(
    intermediate_state: Any,
    pm_guidance: str,
    provider_name: str,
    model_choice: str = "",
    progress: gr.Progress = gr.Progress(),
) -> tuple:
    """
    Run Phase 2 (PM synthesis) and return full formatted results.

    Returns same tuple as run_committee_analysis for output compatibility.
    """
    if intermediate_state is None:
        return (
            "No Phase 1 results available. Run Phase 1 first.",
            "", "", "", "", "", "", None, None, "", "", None, None, None
        )

    status_messages = []
    provider = PROVIDER_DISPLAY.get(provider_name, settings.llm_provider)
    selected_model = model_choice.strip() if model_choice and model_choice.strip() else None

    def on_status(msg: str):
        status_messages.append(msg)
        if "Phase 3" in msg:
            progress(0.30, desc="Portfolio Manager synthesizing...")
        elif "Decision:" in msg:
            progress(0.80, desc="Decision reached — formatting report...")
        elif "Committee complete" in msg:
            progress(1.0, desc="Committee complete!")

    try:
        progress(0.05, desc=f"Initializing {provider_name} for PM synthesis...")
        model = create_model(provider, model_name=selected_model)

        progress(0.10, desc="Running Portfolio Manager with guidance...")
        result = run_graph_phase2(
            intermediate_state=intermediate_state,
            model=model,
            pm_guidance=pm_guidance or "",
            on_status=on_status,
        )

        ticker = result.ticker

        # Store in session memory
        store_analysis(ticker, result)

        # Determine model used (explicit selection overrides settings default)
        if selected_model:
            model_name = selected_model
        else:
            model_map = {
                LLMProvider.ANTHROPIC: settings.anthropic_model,
                LLMProvider.GOOGLE: settings.google_model,
                LLMProvider.OPENAI: settings.openai_model,
                LLMProvider.HUGGINGFACE: settings.hf_model,
                LLMProvider.OLLAMA: settings.ollama_model,
            }
            model_name = model_map.get(provider, "unknown")

        debate_rounds = intermediate_state.get("max_debate_rounds", 2)
        user_context = intermediate_state.get("context", {}).get("user_context", "")
        _log_run(result, provider_name, model_name, debate_rounds, user_context)

        # Format all outputs
        memo_md = _format_committee_memo(result, provider_name)
        bull_md = _format_bull_case(result)
        short_md = _format_short_case(result)
        bear_md = _format_bear_case(result)
        macro_md = _format_macro_view(result)
        xai_md = _format_xai_analysis(result)
        debate_md = _format_debate(result)
        conviction_md = _format_conviction_evolution(result)
        trajectory_fig = build_conviction_trajectory(
            result.conviction_timeline, ticker, result.committee_memo,
        )
        probability_fig = build_conviction_probability(
            result.conviction_timeline, ticker, result.committee_memo,
        )
        trace_md = TraceRenderer.to_gradio_accordion(result.traces)
        has_kb = bool(intermediate_state.get("context", {}).get("user_kb"))
        status_md = _format_status(
            result, status_messages, provider_name,
            mode="Review Before PM",
            num_source_files=1 if has_kb else 0,
            expert_guidance=bool(user_context and user_context.strip()),
            pm_guidance=bool(pm_guidance and pm_guidance.strip()),
        )

        full_report = _build_full_report(
            result, memo_md, bull_md, short_md, bear_md, macro_md, debate_md, conviction_md,
            provider_name, status_md=status_md,
        )

        export_path = EXPORTS_DIR / f"{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(export_path, "w") as f:
            f.write(full_report)

        return (memo_md, bull_md, short_md, bear_md, macro_md, xai_md, debate_md, conviction_md,
                trajectory_fig, probability_fig, trace_md, status_md, str(export_path),
                model, full_report + "\n\n" + xai_md)

    except Exception as e:
        logger.exception("Phase 2 failed")
        error_msg = f"## Error\n\n**Error:** {str(e)}"
        return error_msg, "", "", "", "", "", "", "", None, None, "", f"Error: {str(e)}", None, None, None


# ---------------------------------------------------------------------------
# Q&A Chat — post-analysis follow-up conversation
# ---------------------------------------------------------------------------

_QA_SYSTEM_PREAMBLE = """You are the investment committee's analyst assistant. The user has just \
reviewed a multi-agent investment analysis report. Answer questions ONLY from the analysis data \
below — do not speculate beyond what the report contains. If the report does not cover a topic, \
say so. Be concise and cite specific numbers from the report when possible.

--- BEGIN ANALYSIS REPORT ---
{report}
--- END ANALYSIS REPORT ---"""

_QA_MAX_HISTORY_TURNS = 10


def _extract_text(content) -> str:
    """Extract plain text from Gradio message content.

    Content may be a string, or a list of {text, type} dicts (Gradio 6 format).
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return " ".join(parts)
    return str(content) if content else ""


def _history_to_turns(history: list) -> list[tuple[str, str]]:
    """Convert messages-format history to (user, assistant) turn pairs."""
    turns: list[tuple[str, str]] = []
    i = 0
    while i < len(history):
        msg = history[i]
        if isinstance(msg, dict):
            if msg.get("role") == "user":
                user_text = _extract_text(msg.get("content", ""))
                assistant_text = ""
                if i + 1 < len(history):
                    nxt = history[i + 1]
                    if isinstance(nxt, dict) and nxt.get("role") == "assistant":
                        assistant_text = _extract_text(nxt.get("content", ""))
                        i += 1
                turns.append((user_text, assistant_text))
            i += 1
        elif isinstance(msg, (list, tuple)) and len(msg) == 2:
            turns.append((str(msg[0]), str(msg[1])))
            i += 1
        else:
            i += 1
    return turns


def handle_chat_message(
    user_message: str,
    chat_history: list,
    model_callable,
    report_text: str,
) -> tuple[str, list]:
    """Send a follow-up question grounded in the analysis report.

    Returns (cleared_textbox, updated_chat_history) using Gradio messages format.
    """
    if not user_message or not user_message.strip():
        return "", chat_history or []
    if model_callable is None or not report_text:
        chat_history = list(chat_history or [])
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": "Please run an analysis first — no report is available yet."})
        return "", chat_history

    chat_history = list(chat_history or [])

    # Convert history to turn pairs for prompt construction
    turns = _history_to_turns(chat_history)

    # Build single-turn prompt: system preamble + conversation history + new question
    system = _QA_SYSTEM_PREAMBLE.format(report=report_text)
    parts = [system, ""]
    # Include last N turns for context
    for human_text, assistant_text in turns[-_QA_MAX_HISTORY_TURNS:]:
        parts.append(f"User: {human_text}")
        parts.append(f"Assistant: {assistant_text}")
    parts.append(f"User: {user_message}")
    parts.append("Assistant:")

    prompt = "\n".join(parts)

    try:
        response = model_callable(prompt, temperature=0.3)
    except TypeError:
        # Model doesn't accept temperature kwarg
        response = model_callable(prompt)
    except Exception as e:
        response = f"Error generating response: {e}"

    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": response})
    return "", chat_history



# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    """Build the Gradio interface with Full Auto and Review Before PM modes."""

    resolved = settings.resolve_provider()
    default_display = PROVIDER_TO_DISPLAY.get(resolved, "Ollama (Local)")

    if resolved != settings.llm_provider:
        logger.info(
            "No API key for %s — auto-selected %s",
            settings.llm_provider.value, resolved.value,
        )

    with gr.Blocks(
        title="Multi-Agent Investment Committee",
    ) as app:
        gr.Markdown(
            """<div class="header">

# Multi-Agent Investment Committee

<span class="agent-chips">Sector Analysts &middot; Risk Manager &middot; Macro Strategist &middot; Portfolio Manager</span>
</div>""",
        )

        # ── Controls: all inputs + mode + button in two compact rows ──
        with gr.Group(elem_classes=["controls-group"]):
            with gr.Row(equal_height=True):
                ticker_input = gr.Textbox(
                    label="Ticker",
                    placeholder="e.g. NVDA",
                    max_lines=1,
                    scale=1,
                )
                provider_dropdown = gr.Dropdown(
                    choices=list(PROVIDER_DISPLAY.keys()),
                    value=default_display,
                    label="Provider",
                    interactive=True,
                    scale=1,
                )
                model_dropdown = gr.Dropdown(
                    choices=PROVIDER_MODEL_CHOICES.get(default_display, []),
                    value=_get_default_model_for_provider(default_display),
                    label="Model",
                    interactive=True,
                    allow_custom_value=True,
                    scale=1,
                )
                debate_rounds_input = gr.Slider(
                    minimum=1,
                    maximum=20,
                    step=1,
                    value=settings.max_debate_rounds,
                    label="Debate Rounds",
                    scale=1,
                )
                mode_selector = gr.Radio(
                    choices=["Full Auto", "Review Before PM"],
                    value="Review Before PM" if settings.enable_hitl else "Full Auto",
                    label="Mode",
                    scale=1,
                )
                with gr.Column(scale=1, min_width=140):
                    run_btn = gr.Button(
                        "Run",
                        variant="primary",
                        size="sm",
                        visible=not settings.enable_hitl,
                        elem_classes=["start-btn"],
                    )
                    phase1_btn = gr.Button(
                        "Run",
                        variant="primary",
                        size="sm",
                        visible=settings.enable_hitl,
                        elem_classes=["start-btn"],
                    )
            with gr.Row(equal_height=True):
                context_input = gr.Textbox(
                    label="Expert Guidance — steers all analysts (optional)",
                    placeholder=(
                        "Directs all 4 agents during analysis. e.g. 'focus on tariff exposure, "
                        "compare valuation vs. sector median, consider pharma rotation'"
                    ),
                    lines=4,
                    max_lines=6,
                    scale=3,
                )
                file_upload = gr.File(
                    label="Research Docs (max 10)",
                    file_count="multiple",
                    file_types=[".pdf", ".docx", ".doc", ".txt", ".xlsx", ".xls", ".csv"],
                    scale=2,
                )

        # ── HITL Review Section ──
        with gr.Accordion("Review Analyst Outputs", open=True, visible=False) as review_accordion:
            review_status = gr.Markdown("")

            with gr.Row(equal_height=True, elem_classes=["preview-row"]):
                with gr.Column(elem_classes=["preview-cell"]):
                    bull_preview = gr.Markdown(label="Bull")
                with gr.Column(elem_classes=["preview-cell"]):
                    bear_preview = gr.Markdown(label="Bear")
                with gr.Column(elem_classes=["preview-cell"]):
                    macro_preview = gr.Markdown(label="Macro")
                with gr.Column(elem_classes=["preview-cell"]):
                    debate_preview = gr.Markdown(label="Debate")

            with gr.Row(equal_height=True):
                pm_guidance_input = gr.Textbox(
                    label="PM Guidance — steers final synthesis only (optional)",
                    placeholder=(
                        "Directs only the PM's final decision. e.g. 'Weight bear case more, "
                        "cap position at 3%, focus on valuation risk'"
                    ),
                    max_lines=3,
                    scale=4,
                )
                phase2_btn = gr.Button(
                    "Go",
                    variant="primary",
                    size="sm",
                    scale=1,
                    elem_classes=["start-btn"],
                )

        # Hidden states
        report_path_state = gr.State(value=None)
        intermediate_state = gr.State(value=None)
        chat_model_state = gr.State(value=None)
        chat_report_state = gr.State(value=None)

        # ── Progress indicator (single timer) ──
        progress_status = gr.HTML(
            value="",
            elem_classes=["progress-bar"],
            show_label=False,
        )

        # ── Result tabs ──
        with gr.Tabs(elem_classes=["result-tabs"]):
            with gr.TabItem("Committee Memo"):
                memo_output = gr.Markdown(label="Investment Committee Memo")

            with gr.TabItem("Bull Case"):
                bull_output = gr.Markdown(label="Long Analyst — Bull Case")

            with gr.TabItem("Short Case"):
                short_output = gr.Markdown(label="Short Analyst — Short Case")

            with gr.TabItem("Bear Case"):
                bear_output = gr.Markdown(label="Risk Manager — Risk Assessment")

            with gr.TabItem("Macro View"):
                macro_output = gr.Markdown(label="Macro Analyst — Top-Down Environment")

            with gr.TabItem("XAI Pre-Screen"):
                xai_output = gr.Markdown(label="Explainable AI — Quantitative Pre-Screen")

            with gr.TabItem("Debate"):
                debate_output = gr.Markdown(label="Adversarial Debate Transcript")

            with gr.TabItem("Conviction Tracker"):
                conviction_trajectory_plot = gr.Plot(label="Conviction Trajectory (0-10)")
                conviction_probability_plot = gr.Plot(label="Conviction Probability (0-1)")
                conviction_output = gr.Markdown(label="Conviction Detail")

            with gr.TabItem("Reasoning Trace"):
                trace_output = gr.Markdown(label="Agent Reasoning Traces")

            with gr.TabItem("Session Info"):
                status_output = gr.Markdown(label="Session Summary")

            with gr.TabItem("Q&A Chat"):
                qa_chatbot = gr.Chatbot(
                    height=480,
                    placeholder="Run an analysis first, then ask follow-up questions here.",
                    label="Q&A Chat",
                )
                with gr.Row():
                    qa_textbox = gr.Textbox(
                        placeholder="Ask about the analysis...",
                        show_label=False,
                        scale=4,
                        interactive=False,
                    )
                    qa_send_btn = gr.Button("Send", scale=1, interactive=False)
                qa_clear_btn = gr.Button("Clear Chat", size="sm")

        # Export section
        with gr.Row(elem_classes=["export-row"]):
            copy_btn = gr.Button("Copy Report", size="sm", scale=1, min_width=120)
            download_btn = gr.Button("Download .md", size="sm", scale=1, min_width=120)
            copy_output = gr.Textbox(
                label="Full Report (select all + copy)",
                lines=3,
                max_lines=6,
                visible=False,
                scale=4,
            )
            download_output = gr.File(label="Download", visible=False, scale=2)

        # ── Provider → Model dynamic update ──
        def on_provider_change(provider_name):
            choices = PROVIDER_MODEL_CHOICES.get(provider_name, [])
            default = choices[0] if choices else ""
            return gr.update(choices=choices, value=default)

        provider_dropdown.change(
            fn=on_provider_change,
            inputs=[provider_dropdown],
            outputs=[model_dropdown],
        )

        # ── Mode switching ──
        def on_mode_change(mode):
            is_hitl = (mode == "Review Before PM")
            return (
                gr.update(visible=not is_hitl),  # run_btn
                gr.update(visible=is_hitl),       # phase1_btn
                gr.update(visible=False),          # review_accordion
            )

        mode_selector.change(
            fn=on_mode_change,
            inputs=[mode_selector],
            outputs=[run_btn, phase1_btn, review_accordion],
        )

        # ── Lock inputs during execution ──
        _lockable = [ticker_input, context_input, provider_dropdown,
                     model_dropdown, debate_rounds_input, mode_selector,
                     file_upload, pm_guidance_input,
                     qa_textbox, qa_send_btn,
                     run_btn, phase1_btn, phase2_btn]

        def _lock_inputs():
            updates = [gr.update(interactive=False) for _ in range(10)]  # input fields + chat
            updates.extend([
                gr.update(interactive=False, value="..."),  # run_btn
                gr.update(interactive=False, value="..."),  # phase1_btn
                gr.update(interactive=False, value="..."),  # phase2_btn
            ])
            return updates

        def _unlock_inputs():
            updates = [gr.update(interactive=True) for _ in range(10)]  # input fields + chat
            updates.extend([
                gr.update(interactive=True, value="Run"),   # run_btn
                gr.update(interactive=True, value="Run"),   # phase1_btn
                gr.update(interactive=True, value="Go"),    # phase2_btn
            ])
            return updates

        # ── Full Auto mode ──
        run_btn.click(
            fn=_lock_inputs,
            inputs=[],
            outputs=_lockable,
        ).then(
            fn=lambda: '<div style="background:#1a5c2a;color:#4ade80;padding:12px 20px;border-radius:8px;font-size:1.1rem;font-weight:600;text-align:center;animation:pulse 2s infinite">&#9881; Running committee analysis &mdash; agents thinking, debating, synthesizing...</div><style>@keyframes pulse{0%,100%{opacity:1}50%{opacity:.6}}</style>',
            inputs=[],
            outputs=[progress_status],
        ).then(
            fn=run_committee_analysis,
            inputs=[ticker_input, context_input, provider_dropdown, debate_rounds_input,
                    model_dropdown, file_upload],
            outputs=[memo_output, bull_output, short_output, bear_output, macro_output, xai_output,
                     debate_output, conviction_output, conviction_trajectory_plot,
                     conviction_probability_plot, trace_output, status_output,
                     report_path_state, chat_model_state, chat_report_state],
            show_progress="hidden",
        ).then(
            fn=lambda: (gr.update(value="", interactive=True), [], gr.update(interactive=True)),
            inputs=[],
            outputs=[qa_textbox, qa_chatbot, qa_send_btn],
        ).then(
            fn=lambda: "",
            inputs=[],
            outputs=[progress_status],
        ).then(
            fn=_unlock_inputs,
            inputs=[],
            outputs=_lockable,
        )

        # ── HITL Phase 1 ──
        def handle_phase1(ticker, user_context, provider_name, debate_rounds,
                          model_choice, uploaded_files, progress=gr.Progress()):
            state, bull_p, bear_p, macro_p, debate_p, status_msg = run_phase1_analysis(
                ticker, user_context, provider_name, debate_rounds,
                model_choice=model_choice, uploaded_files=uploaded_files, progress=progress,
            )
            return (
                state,
                bull_p,
                bear_p,
                macro_p,
                debate_p,
                status_msg,
                gr.update(visible=True),
            )

        phase1_btn.click(
            fn=_lock_inputs,
            inputs=[],
            outputs=_lockable,
        ).then(
            fn=lambda: '<div style="background:#1a5c2a;color:#4ade80;padding:12px 20px;border-radius:8px;font-size:1.1rem;font-weight:600;text-align:center;animation:pulse 2s infinite">&#9881; Running Phase 1 &mdash; analysts building bull, bear, and macro cases...</div><style>@keyframes pulse{0%,100%{opacity:1}50%{opacity:.6}}</style>',
            inputs=[],
            outputs=[progress_status],
        ).then(
            fn=handle_phase1,
            inputs=[ticker_input, context_input, provider_dropdown, debate_rounds_input,
                    model_dropdown, file_upload],
            outputs=[intermediate_state, bull_preview, bear_preview,
                     macro_preview, debate_preview, review_status, review_accordion],
            show_progress="hidden",
        ).then(
            fn=lambda: "",
            inputs=[],
            outputs=[progress_status],
        ).then(
            fn=_unlock_inputs,
            inputs=[],
            outputs=_lockable,
        )

        # ── HITL Phase 2 ──
        def handle_phase2(inter_state, pm_guidance, provider_name, model_choice,
                          progress=gr.Progress()):
            results = run_phase2_synthesis(
                inter_state, pm_guidance, provider_name,
                model_choice=model_choice, progress=progress,
            )
            return results + (gr.update(visible=False),)

        phase2_btn.click(
            fn=_lock_inputs,
            inputs=[],
            outputs=_lockable,
        ).then(
            fn=lambda: '<div style="background:#1a5c2a;color:#4ade80;padding:12px 20px;border-radius:8px;font-size:1.1rem;font-weight:600;text-align:center;animation:pulse 2s infinite">&#9881; Running PM synthesis &mdash; portfolio manager making final decision...</div><style>@keyframes pulse{0%,100%{opacity:1}50%{opacity:.6}}</style>',
            inputs=[],
            outputs=[progress_status],
        ).then(
            fn=handle_phase2,
            inputs=[intermediate_state, pm_guidance_input, provider_dropdown, model_dropdown],
            outputs=[memo_output, bull_output, short_output, bear_output, macro_output, xai_output,
                     debate_output, conviction_output, conviction_trajectory_plot,
                     conviction_probability_plot, trace_output, status_output,
                     report_path_state, chat_model_state, chat_report_state,
                     review_accordion],
            show_progress="hidden",
        ).then(
            fn=lambda: (gr.update(value="", interactive=True), [], gr.update(interactive=True)),
            inputs=[],
            outputs=[qa_textbox, qa_chatbot, qa_send_btn],
        ).then(
            fn=lambda: "",
            inputs=[],
            outputs=[progress_status],
        ).then(
            fn=_unlock_inputs,
            inputs=[],
            outputs=_lockable,
        )

        # ── Q&A Chat events ──
        _chat_inputs = [qa_textbox, qa_chatbot, chat_model_state, chat_report_state]
        _chat_outputs = [qa_textbox, qa_chatbot]

        qa_send_btn.click(
            fn=handle_chat_message,
            inputs=_chat_inputs,
            outputs=_chat_outputs,
        )
        qa_textbox.submit(
            fn=handle_chat_message,
            inputs=_chat_inputs,
            outputs=_chat_outputs,
        )
        qa_clear_btn.click(
            fn=lambda: ([], ""),
            inputs=[],
            outputs=[qa_chatbot, qa_textbox],
        )

        # ── Copy button ──
        def show_report_for_copy(*tab_outputs):
            memo, bull, bear, macro, xai, debate, conviction, trace, status = tab_outputs
            full = "\n\n---\n\n".join([
                s for s in [memo, bull, bear, macro, xai, debate, conviction, status] if s
            ])
            return gr.update(value=full, visible=True)

        copy_btn.click(
            fn=show_report_for_copy,
            inputs=[memo_output, bull_output, bear_output, macro_output, xai_output,
                    debate_output, conviction_output, trace_output, status_output],
            outputs=[copy_output],
        )

        # ── Download button ──
        def serve_download(path):
            if path and os.path.exists(path):
                return gr.update(value=path, visible=True)
            return gr.update(visible=False)

        download_btn.click(
            fn=serve_download,
            inputs=[report_path_state],
            outputs=[download_output],
        )

        gr.Markdown(
            """<div class="disclaimer">

**Disclaimer:** Demonstration of multi-agent AI reasoning — NOT financial advice.
All analyses are AI-generated. Consult qualified professionals for investment decisions.

**Architecture:** think → plan → act → reflect with adversarial debate.
*Full Auto* runs end-to-end. *Review Before PM* lets you steer the PM.
Providers: Claude | Gemini | GPT | HuggingFace | Ollama

</div>""",
        )

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft(),
        css="""
        .header { text-align: center; margin-bottom: 4px; }
        .header h1 { margin-bottom: 2px !important; font-size: 1.4rem !important; }
        .agent-chips { font-size: 0.8rem; color: #666; letter-spacing: 0.02em; }
        .controls-group { padding: 10px 16px !important; }
        .controls-group .gr-group { gap: 10px !important; }
        .controls-group .gr-row { margin-bottom: 8px !important; }
        .preview-cell { border-left: 1px solid #333 !important; padding-left: 14px !important; }
        .preview-row > .preview-cell:first-child { border-left: none !important; padding-left: 0 !important; }
        .result-tabs { margin-top: 12px !important; }
        .gr-accordion { margin-top: 10px !important; }
        .export-row { margin-top: 4px; }
        .disclaimer { font-size: 0.78em; color: #888; margin-top: 80px; padding: 10px 14px;
                       border: 1px solid #ddd; border-radius: 5px; line-height: 1.5; }
        .progress-bar { min-height: 0 !important; }
        .progress-bar .wrap, .progress-bar.wrap { min-height: 0 !important; max-height: 3rem !important; }
        .progress-text { font-size: 0.85rem !important; padding: 4px 8px !important; }
        """,
        js="""
        () => {
            function applyStyles() {
                // Shrink all controls-group text
                const cg = document.querySelectorAll('.controls-group');
                cg.forEach(group => {
                    group.querySelectorAll('input, button, select, textarea, span, label, li, div, option, a').forEach(el => {
                        // Skip start-btn children — they get separate styling
                        if (el.closest('.start-btn')) return;
                        el.style.setProperty('font-size', '0.72rem', 'important');
                    });
                    group.querySelectorAll('input, select, textarea').forEach(el => {
                        if (el.closest('.start-btn')) return;
                        el.style.setProperty('padding', '4px 6px', 'important');
                        el.style.setProperty('overflow', 'hidden', 'important');
                        el.style.setProperty('text-overflow', 'ellipsis', 'important');
                        el.style.setProperty('white-space', 'nowrap', 'important');
                    });
                });

                // Green circle run buttons
                document.querySelectorAll('.start-btn button').forEach(btn => {
                    btn.style.setProperty('border-radius', '50%', 'important');
                    btn.style.setProperty('width', '2.8rem', 'important');
                    btn.style.setProperty('height', '2.8rem', 'important');
                    btn.style.setProperty('min-width', '2.8rem', 'important');
                    btn.style.setProperty('max-width', '2.8rem', 'important');
                    btn.style.setProperty('min-height', '2.8rem', 'important');
                    btn.style.setProperty('max-height', '2.8rem', 'important');
                    btn.style.setProperty('padding', '0', 'important');
                    btn.style.setProperty('font-size', '0.6rem', 'important');
                    btn.style.setProperty('font-weight', '700', 'important');
                    btn.style.setProperty('line-height', '1', 'important');
                    btn.style.setProperty('text-align', 'center', 'important');
                    btn.style.setProperty('background', '#2ecc71', 'important');
                    btn.style.setProperty('background-color', '#2ecc71', 'important');
                    btn.style.setProperty('border', '2px solid #27ae60', 'important');
                    btn.style.setProperty('color', '#fff', 'important');
                    btn.style.setProperty('box-shadow', '0 2px 8px rgba(46,204,113,0.3)', 'important');
                });
            }

            // Run on load and re-run after Gradio finishes rendering
            applyStyles();
            setTimeout(applyStyles, 500);
            setTimeout(applyStyles, 1500);
            setTimeout(applyStyles, 3000);

            // Observe DOM changes to re-apply (dropdown opens, etc.)
            const observer = new MutationObserver(() => applyStyles());
            observer.observe(document.body, { childList: true, subtree: true });
        }
        """,
    )
