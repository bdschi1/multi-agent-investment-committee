"""
Multi-Agent Investment Committee — Gradio Application

This is the main entry point for the Hugging Face Spaces deployment.
It provides a web UI where users can input a ticker and watch
three AI agents reason, debate, and synthesize an investment thesis.

Supported LLM providers:
    - Anthropic (Claude)
    - Google (Gemini)
    - OpenAI (GPT)
    - DeepSeek
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
import logging
import os
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import gradio as gr

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
# Provider-specific model factories
# ---------------------------------------------------------------------------

def _create_anthropic_model(model_name: str | None = None) -> callable:
    """Create a Claude (Anthropic) callable."""
    from anthropic import Anthropic

    client = Anthropic(api_key=settings.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY"))
    model = model_name or settings.anthropic_model

    def call(prompt: str, *, temperature: float | None = None) -> str:
        response = client.messages.create(
            model=model,
            max_tokens=settings.max_tokens_per_agent,
            temperature=temperature if temperature is not None else settings.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    return call


def _create_google_model(model_name: str | None = None) -> callable:
    """Create a Gemini (Google) callable."""
    from google import genai

    client = genai.Client(api_key=settings.google_api_key or os.environ.get("GOOGLE_API_KEY"))
    model = model_name or settings.google_model

    def call(prompt: str, *, temperature: float | None = None) -> str:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config={
                "temperature": temperature if temperature is not None else settings.temperature,
                "max_output_tokens": settings.max_tokens_per_agent,
            },
        )
        return response.text

    return call


def _create_openai_model(model_name: str | None = None) -> callable:
    """Create an OpenAI callable."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "OpenAI package not installed. Run: pip install -e '.[openai]'"
        ) from exc

    client = OpenAI(api_key=settings.openai_api_key or os.environ.get("OPENAI_API_KEY"))
    model = model_name or settings.openai_model

    def call(prompt: str, *, temperature: float | None = None) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature if temperature is not None else settings.temperature,
            max_tokens=settings.max_tokens_per_agent,
        )
        return response.choices[0].message.content

    return call


def _create_huggingface_model(model_name: str | None = None) -> callable:
    """Create a HuggingFace Inference API callable."""
    from huggingface_hub import InferenceClient

    model = model_name or settings.hf_model
    client = InferenceClient(
        model=model,
        token=settings.hf_token or os.environ.get("HF_TOKEN"),
    )

    def call(prompt: str, *, temperature: float | None = None) -> str:
        temp = temperature if temperature is not None else settings.temperature
        response = client.text_generation(
            prompt,
            max_new_tokens=settings.max_tokens_per_agent,
            temperature=max(temp, 0.01),  # HF needs >0
            do_sample=True,
        )
        return response

    return call


def _create_ollama_model(model_name: str | None = None) -> callable:
    """Create an Ollama (local) callable for open-source models."""
    try:
        import ollama as ollama_lib
    except ImportError as exc:
        raise RuntimeError(
            "Ollama package not installed. Run: pip install -e '.[ollama]'\n"
            "Also ensure Ollama is running: ollama serve"
        ) from exc

    model = model_name or settings.ollama_model
    client = ollama_lib.Client(host=settings.ollama_base_url)

    # Verify the model is available before returning the callable
    try:
        client.show(model)
    except Exception as exc:
        raise RuntimeError(
            f"Ollama model '{model}' not found. Pull it first:\n\n"
            f"    ollama pull {model}\n\n"
            f"Make sure Ollama is running (ollama serve) and the model name is correct.\n"
            f"Available models can be listed with: ollama list"
        ) from exc

    def call(prompt: str, *, temperature: float | None = None, json_mode: bool = False) -> str:
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "options": {
                "temperature": temperature if temperature is not None else settings.temperature,
                "num_predict": settings.max_tokens_per_agent,
            },
        }
        # JSON mode: instruct Ollama to constrain output to valid JSON
        if json_mode or _prompt_requests_json(prompt):
            kwargs["format"] = "json"
        response = client.chat(**kwargs)
        return response["message"]["content"]

    return call


def _prompt_requests_json(prompt: str) -> bool:
    """Heuristic: detect if a prompt is asking for JSON output."""
    indicators = [
        "Respond ONLY with the JSON",
        "Respond in valid JSON",
        "respond with a JSON",
        "output as JSON",
        "JSON object",
        "JSON matching this",
    ]
    return any(ind.lower() in prompt.lower() for ind in indicators)


def _create_deepseek_model(model_name: str | None = None) -> callable:
    """Create a DeepSeek callable (OpenAI-compatible API)."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "OpenAI package required for DeepSeek (uses OpenAI-compatible API).\n"
            "Run: pip install -e '.[deepseek]'"
        ) from exc

    client = OpenAI(
        api_key=settings.deepseek_api_key or os.environ.get("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )
    model = model_name or settings.deepseek_model

    def call(prompt: str, *, temperature: float | None = None) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature if temperature is not None else settings.temperature,
            max_tokens=settings.max_tokens_per_agent,
        )
        return response.choices[0].message.content

    return call


# ---------------------------------------------------------------------------
# Unified model factory
# ---------------------------------------------------------------------------

PROVIDER_FACTORIES = {
    LLMProvider.ANTHROPIC: _create_anthropic_model,
    LLMProvider.GOOGLE: _create_google_model,
    LLMProvider.OPENAI: _create_openai_model,
    LLMProvider.DEEPSEEK: _create_deepseek_model,
    LLMProvider.HUGGINGFACE: _create_huggingface_model,
    LLMProvider.OLLAMA: _create_ollama_model,
}

# Display names for the UI dropdown
PROVIDER_DISPLAY = {
    "Claude (Anthropic)": LLMProvider.ANTHROPIC,
    "Gemini (Google)": LLMProvider.GOOGLE,
    "GPT (OpenAI)": LLMProvider.OPENAI,
    "DeepSeek": LLMProvider.DEEPSEEK,
    "HuggingFace API": LLMProvider.HUGGINGFACE,
    "Ollama (Local)": LLMProvider.OLLAMA,
}

# Reverse map for default selection
PROVIDER_TO_DISPLAY = {v: k for k, v in PROVIDER_DISPLAY.items()}

# Model choices per provider — first entry is the default
PROVIDER_MODEL_CHOICES: dict[str, list[str]] = {
    "Claude (Anthropic)": [
        "claude-sonnet-4-20250514",
        "claude-haiku-4-20250414",
    ],
    "Gemini (Google)": [
        "gemini-2.0-flash",
        "gemini-2.5-pro-preview-06-05",
        "gemini-2.5-flash-preview-04-17",
    ],
    "GPT (OpenAI)": [
        "gpt-4o-mini",
        "gpt-4o",
    ],
    "DeepSeek": [
        "deepseek-chat",
        "deepseek-reasoner",
    ],
    "HuggingFace API": [
        "Qwen/Qwen2.5-72B-Instruct",
    ],
    "Ollama (Local)": [
        "llama3.1:8b",
        "llama3.2:3b",
        "llama3:70b",
        "deepseek-r1-32b-abliterated",
    ],
}


def _get_default_model_for_provider(provider_display_name: str) -> str:
    """Return the default (first) model for a provider display name."""
    choices = PROVIDER_MODEL_CHOICES.get(provider_display_name, [])
    return choices[0] if choices else ""


# ---------------------------------------------------------------------------
# Rate limiter — wraps model callable to respect per-minute token budgets
# ---------------------------------------------------------------------------

# Provider-specific rate limits (configurable via settings / .env)
# Only providers listed here get wrapped. Others run unrestricted.
PROVIDER_RATE_LIMITS: dict[LLMProvider, dict[str, int]] = {
    LLMProvider.ANTHROPIC: {
        "max_rpm": settings.rate_limit_rpm,
        "max_input_tpm": settings.rate_limit_input_tpm,
        "max_output_tpm": settings.rate_limit_output_tpm,
    },
    # Google, OpenAI, DeepSeek, HF, Ollama: no wrapping needed (generous limits or local)
}


class RateLimitedModel:
    """
    Wraps a model callable with per-minute token budget enforcement.

    Uses a 60-second sliding window to track requests and token usage.
    Thread-safe — parallel agents (LangGraph Send / ThreadPoolExecutor)
    coordinate through a shared lock.

    Token estimation uses the same word-based heuristic as doc_chunker:
    ~1.33 tokens per word (no tiktoken dependency required).
    """

    def __init__(
        self,
        model_fn: callable,
        max_rpm: int = 50,
        max_input_tpm: int = 30_000,
        max_output_tpm: int = 8_000,
    ):
        self._model = model_fn
        self._max_rpm = max_rpm
        self._max_input_tpm = max_input_tpm
        self._max_output_tpm = max_output_tpm
        self._call_log: list[tuple[float, int, int]] = []  # (timestamp, in_tokens, out_tokens)
        self._lock = threading.Lock()

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Word-based token estimate: ~1.33 tokens per whitespace-separated word."""
        return int(len(text.split()) * 4 / 3)

    def _prune_old(self) -> None:
        """Remove entries older than 60 seconds from the sliding window."""
        cutoff = time.time() - 60
        self._call_log = [(t, i, o) for t, i, o in self._call_log if t > cutoff]

    def _wait_for_budget(self, input_tokens: int) -> None:
        """Block until the sliding window has enough budget for this call."""
        while True:
            with self._lock:
                self._prune_old()
                used_rpm = len(self._call_log)
                used_input = sum(i for _, i, _ in self._call_log)

                if (used_rpm < self._max_rpm
                        and used_input + input_tokens <= self._max_input_tpm):
                    return  # budget available

                # Calculate how long until enough budget frees up
                if self._call_log:
                    oldest = self._call_log[0][0]
                    wait_secs = max(0, 60 - (time.time() - oldest)) + 0.5
                else:
                    wait_secs = 2.0

            logger.debug(
                f"Rate limiter: {used_rpm} reqs, {used_input}+{input_tokens} input tokens "
                f"in window — waiting {wait_secs:.1f}s for budget"
            )
            time.sleep(min(wait_secs, 5.0))  # cap individual sleep at 5s for responsiveness

    def __call__(self, prompt: str, **kwargs) -> str:
        input_tokens = self._estimate_tokens(prompt)
        self._wait_for_budget(input_tokens)

        result = self._model(prompt, **kwargs)
        output_tokens = self._estimate_tokens(result)

        with self._lock:
            self._call_log.append((time.time(), input_tokens, output_tokens))

        return result


def create_model(
    provider: LLMProvider | None = None,
    model_name: str | None = None,
) -> callable:
    """
    Create the LLM callable for the given provider.

    Architecture note: Every provider returns a simple `callable(str) -> str`.
    This uniform interface means agents don't know or care which LLM they're
    talking to. Swapping providers is a one-line change.

    For providers with tight rate limits (Anthropic Tier 1), the callable is
    wrapped with a RateLimitedModel that enforces per-minute token budgets.

    Args:
        provider: Which LLM backend to use
        model_name: Specific model to use (overrides settings default)
    """
    provider = provider or settings.resolve_provider()
    factory = PROVIDER_FACTORIES.get(provider)
    if not factory:
        raise ValueError(f"Unknown provider: {provider}")

    model_fn = factory(model_name=model_name)

    # Wrap with rate limiter if provider has tight limits
    limits = PROVIDER_RATE_LIMITS.get(provider)
    if limits:
        logger.info(
            f"Rate limiter active for {provider.value}: "
            f"{limits['max_rpm']} RPM, {limits['max_input_tpm']} input TPM"
        )
        return RateLimitedModel(model_fn, **limits)

    return model_fn


# ---------------------------------------------------------------------------
# Signal persistence — store signals in SQLite for backtesting
# ---------------------------------------------------------------------------

def _persist_signal(result: CommitteeResult, provider_name: str, model_name: str) -> None:
    """Persist a committee result as a signal in the backtest database."""
    try:
        from backtest import SignalDatabase, SignalRecord

        if not result.committee_memo:
            return

        db = SignalDatabase()
        memo = result.committee_memo

        signal = SignalRecord(
            ticker=result.ticker,
            signal_date=datetime.now(UTC),
            provider=provider_name,
            model_name=model_name,
            recommendation=memo.recommendation,
            t_signal=memo.t_signal,
            conviction=memo.conviction,
            position_direction=memo.position_direction,
            raw_confidence=memo.raw_confidence,
            bull_conviction=result.bull_case.conviction_score if result.bull_case else 5.0,
            bear_conviction=result.bear_case.bearish_conviction if result.bear_case else 5.0,
            macro_favorability=result.macro_view.macro_favorability if result.macro_view else 5.0,
            duration_s=result.total_duration_ms / 1000,
            total_tokens=result.total_tokens,
        )

        # Add BL optimizer results if available
        if result.optimization_result and hasattr(result.optimization_result, 'success'):
            opt = result.optimization_result
            if opt.success:
                signal.bl_optimal_weight = opt.optimal_weight
                signal.bl_sharpe = opt.computed_sharpe
                signal.bl_sortino = opt.computed_sortino

        # Add XAI results if available
        xai = result.xai_result
        if xai and hasattr(xai, 'distress'):
            signal.xai_pfd = xai.distress.pfd
            signal.xai_z_score = xai.distress.z_score
            signal.xai_distress_zone = xai.distress.distress_zone
            signal.xai_expected_return = xai.returns.expected_return
            signal.xai_model_used = xai.distress.model_used
            if xai.distress.top_risk_factors:
                first_factor = xai.distress.top_risk_factors[0]
                if isinstance(first_factor, dict):
                    signal.xai_top_risk_factor = next(iter(first_factor), "")
                else:
                    signal.xai_top_risk_factor = str(first_factor)

        signal_id = db.store_signal(signal)
        db.close()
        logger.info(f"Signal persisted: {result.ticker} → id={signal_id}")
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
            "", "", "", "", "", "", "", None, None, "", None
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
        return (_ripple, "", "", "", "", "", "", "", None, None, "", None)

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
        status_md = _format_status(result, status_messages, provider_name)

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
                trajectory_fig, probability_fig, trace_md, status_md, str(export_path))

    except Exception as e:
        logger.exception(f"Committee analysis failed for {ticker}")
        error_msg = (
            f"## Error\n\n"
            f"**Provider:** {provider_name}\n\n"
            f"**Error:** {str(e)}\n\n"
            f"Check that your API key is set in `.env` and the provider is available."
        )
        return error_msg, "", "", "", "", "", "", "", None, None, "", f"Error: {str(e)}", None
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
            "", "", "", "", "", "", None, None, "", "", None
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
        status_md = _format_status(result, status_messages, provider_name)

        full_report = _build_full_report(
            result, memo_md, bull_md, short_md, bear_md, macro_md, debate_md, conviction_md,
            provider_name, status_md=status_md,
        )

        export_path = EXPORTS_DIR / f"{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(export_path, "w") as f:
            f.write(full_report)

        return (memo_md, bull_md, short_md, bear_md, macro_md, xai_md, debate_md, conviction_md,
                trajectory_fig, probability_fig, trace_md, status_md, str(export_path))

    except Exception as e:
        logger.exception("Phase 2 failed")
        error_msg = f"## Error\n\n**Error:** {str(e)}"
        return error_msg, "", "", "", "", "", "", "", None, None, "", f"Error: {str(e)}", None


# ---------------------------------------------------------------------------
# HITL Preview formatters (compact previews for the review step)
# ---------------------------------------------------------------------------

_PARSING_SENTINEL = "structured parsing failed"


def _format_bull_preview(state: dict) -> str:
    """Compact preview of bull case for HITL review."""
    bc = state.get("bull_case")
    if not bc:
        return "No bull case available."

    degraded = any(_PARSING_SENTINEL in str(e) for e in (bc.supporting_evidence or []))
    degraded_tag = " *(parsing degraded)*" if degraded else ""

    aggregate_sentiment = getattr(bc, 'aggregate_news_sentiment', 'neutral')
    sent_emoji = {
        "strongly_bullish": "🟢🟢", "bullish": "🟢", "neutral": "🟡",
        "bearish": "🔴", "strongly_bearish": "🔴🔴",
    }.get(aggregate_sentiment, "⚪")

    lines = [
        f"### Bull Case: {bc.ticker}{degraded_tag}",
        f"**Conviction:** {bc.conviction_score}/10 | **Horizon:** {bc.time_horizon} | **Sentiment:** {sent_emoji} {aggregate_sentiment.replace('_', ' ').title()}",
        "",
        f"**Thesis:** {bc.thesis}",
        "",
        "**Key Catalysts:**",
    ]
    for cat in bc.catalysts[:3]:
        lines.append(f"- {cat}")
    if len(bc.catalysts) > 3:
        lines.append(f"- *...and {len(bc.catalysts) - 3} more*")

    return "\n".join(lines)


def _format_bear_preview(state: dict) -> str:
    """Compact preview of bear case for HITL review."""
    bc = state.get("bear_case")
    if not bc:
        return "No bear case available."

    degraded = any(_PARSING_SENTINEL in str(r) for r in (bc.risks or []))
    degraded_tag = " *(parsing degraded)*" if degraded else ""

    lines = [
        f"### Bear Case: {bc.ticker}{degraded_tag}",
        f"**Bearish Conviction:** {bc.bearish_conviction}/10 | **Structure:** {bc.position_structure}",
        "",
        "**Top Risks:**",
    ]
    for risk in bc.risks[:3]:
        lines.append(f"- {risk}")
    if len(bc.risks) > 3:
        lines.append(f"- *...and {len(bc.risks) - 3} more*")

    if bc.worst_case_scenario:
        lines.extend(["", f"**Worst Case:** {bc.worst_case_scenario}"])

    return "\n".join(lines)


def _format_macro_preview(state: dict) -> str:
    """Compact preview of macro view for HITL review."""
    mv = state.get("macro_view")
    if not mv:
        return "No macro analysis available."

    degraded = _PARSING_SENTINEL in str(mv.economic_cycle_phase)
    degraded_tag = " *(parsing degraded)*" if degraded else ""

    fav = mv.macro_favorability
    fav_label = "Favorable" if fav >= 7 else "Neutral" if fav >= 4 else "Hostile"

    impact_text = mv.macro_impact_on_stock or ""
    if len(impact_text) > 200:
        impact_text = impact_text[:200] + "..."

    vol_regime = getattr(mv, 'annualized_vol_regime', '')
    directionality = getattr(mv, 'portfolio_directionality', '')
    dir_emoji = "🟢" if "long" in directionality.lower() else "🔴" if "short" in directionality.lower() else "🟡" if directionality else ""

    lines = [
        f"### Macro Environment + Portfolio Strategy: {mv.ticker}{degraded_tag}",
        f"**Favorability:** {fav}/10 ({fav_label})",
        f"**Cycle:** {mv.economic_cycle_phase} | **Rates:** {mv.rate_environment}",
    ]

    if vol_regime or directionality:
        lines.append(f"**Vol Regime:** {vol_regime} | **Net Exposure:** {dir_emoji} {directionality}")

    lines.extend([
        "",
        f"**Impact:** {impact_text}",
    ])

    return "\n".join(lines)


def _format_debate_preview(state: dict) -> str:
    """Compact preview of debate results for HITL review."""
    lr = state.get("long_rebuttal")
    sr = state.get("short_rebuttal")
    rr = state.get("risk_rebuttal")

    lines = ["### Debate Summary"]

    # Note convergence if bull/short scores were close
    bull = state.get("bull_case")
    short = state.get("short_case")
    if bull and short:
        spread = abs(bull.conviction_score - short.conviction_score)
        if spread < 2.0:
            lines.append(
                f"\n> **Convergence noted** — spread {spread:.1f} "
                f"(< 2.0). Long and Short analysts largely agree."
            )

    if lr:
        if lr.revised_conviction is not None:
            lines.append(f"\n**Long Analyst revised conviction:** {lr.revised_conviction}/10")
        if lr.points:
            lines.append(f"**Key challenges:** {', '.join(lr.points[:2])}")
        if lr.concessions:
            lines.append(f"**Concessions:** {', '.join(lr.concessions[:2])}")

    if sr:
        if sr.revised_conviction is not None:
            lines.append(f"\n**Short Analyst revised conviction:** {sr.revised_conviction}/10")
        if sr.points:
            lines.append(f"**Key challenges:** {', '.join(sr.points[:2])}")
        if sr.concessions:
            lines.append(f"**Concessions:** {', '.join(sr.concessions[:2])}")

    if rr:
        if rr.revised_conviction is not None:
            lines.append(f"\n**Risk Mgr revised risk:** {rr.revised_conviction}/10")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Full report builder (for copy + download)
# ---------------------------------------------------------------------------

def _build_session_section() -> str:
    """Build a session memory summary section for the exported report."""
    from orchestrator.memory import get_prior_analyses, get_session_summary

    summary = get_session_summary()
    if not summary:
        return ""

    lines = [
        "# Session History",
        "",
        "Prior analyses run during this session:",
        "",
        "| Ticker | Runs | Latest Recommendation | Conviction |",
        "|--------|------|-----------------------|------------|",
    ]

    for ticker, count in summary.items():
        prior = get_prior_analyses(ticker)
        if prior:
            latest = prior[-1]
            rec = latest.get("recommendation", "—")
            conv = latest.get("conviction", "—")
            lines.append(f"| {ticker} | {count} | {rec} | {conv} |")
        else:
            lines.append(f"| {ticker} | {count} | — | — |")

    return "\n".join(lines)


def _format_optimizer_section(optimization_result, memo=None) -> str:
    """Format the Black-Litterman optimizer output as markdown tables."""
    if not optimization_result or not getattr(optimization_result, 'success', False):
        error_msg = getattr(optimization_result, 'error_message', '') if optimization_result else ''
        if error_msg:
            return (
                "\n## Computed Portfolio Analytics (Black-Litterman)\n\n"
                f"*Optimizer did not produce results: {error_msg}*\n"
            )
        return ""

    opt = optimization_result
    lines = [
        "",
        "## Computed Portfolio Analytics (Black-Litterman)",
        "",
        "*Actual computed values from pypfopt Black-Litterman model — "
        "compare with LLM heuristics above.*",
        "",
    ]

    # Side-by-side comparison table (heuristic vs computed)
    if memo:
        idio_ret_h = getattr(memo, 'idio_return_estimate', '') or '—'
        sharpe_h = getattr(memo, 'sharpe_estimate', '') or '—'
        sortino_h = getattr(memo, 'sortino_estimate', '') or '—'

        # Extract numeric portion from heuristic strings for delta
        def _extract_pct(s):
            import re
            m = re.search(r'([+-]?\d+(?:\.\d+)?)\s*%', str(s))
            if m:
                return float(m.group(1))
            m = re.search(r'([+-]?\d+\.\d+)', str(s))
            if m:
                v = float(m.group(1))
                return v * 100 if abs(v) < 5 else v
            return None

        def _extract_ratio(s):
            import re
            m = re.search(r'([+-]?\d+\.\d+)', str(s))
            return float(m.group(1)) if m else None

        idio_num = _extract_pct(idio_ret_h)
        sharpe_num = _extract_ratio(sharpe_h)
        sortino_num = _extract_ratio(sortino_h)

        bl_ret_pct = opt.bl_expected_return * 100

        def _delta(heuristic, computed):
            if heuristic is not None and computed is not None:
                d = computed - heuristic
                return f"{d:+.1f}{'%' if abs(heuristic) > 1 else ''}"
            return "—"

        lines.extend([
            "### Heuristic vs Computed Comparison",
            "",
            "| Metric | LLM Heuristic | BL Computed | Delta |",
            "|--------|---------------|-------------|-------|",
            f"| **Expected Alpha** | {idio_ret_h.split('—')[0].strip() if '—' in str(idio_ret_h) else idio_ret_h} | "
            f"{bl_ret_pct:.1f}% (BL posterior) | {_delta(idio_num, bl_ret_pct)} |",
            f"| **Sharpe** | {sharpe_h.split('—')[0].strip() if '—' in str(sharpe_h) else sharpe_h} | "
            f"{opt.computed_sharpe:.2f} (computed) | {_delta(sharpe_num, opt.computed_sharpe)} |",
            f"| **Sortino** | {sortino_h.split('—')[0].strip() if '—' in str(sortino_h) else sortino_h} | "
            f"{opt.computed_sortino:.2f} (computed) | {_delta(sortino_num, opt.computed_sortino)} |",
            f"| **Vol** | — | {opt.annualized_vol * 100:.1f}% (realized) | — |",
            f"| **Downside Vol** | — | {opt.downside_vol * 100:.1f}% (realized) | — |",
            "",
        ])

    # BL model output
    lines.extend([
        "### Black-Litterman Model Output",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| **Optimal Weight ({opt.ticker})** | {opt.optimal_weight_pct} |",
        f"| **BL Expected Return** | {opt.bl_expected_return * 100:.2f}% |",
        f"| **Equilibrium Return (Prior)** | {opt.equilibrium_return * 100:.2f}% |",
        f"| **Portfolio Vol** | {opt.portfolio_vol * 100:.2f}% |",
        f"| **Covariance Method** | {opt.covariance_method} |",
        f"| **Lookback** | {opt.lookback_days} trading days |",
        f"| **Risk Aversion (delta)** | {opt.risk_aversion} |",
        f"| **Tau** | {opt.tau} |",
        "",
    ])

    # Factor exposures
    if opt.factor_exposures:
        lines.extend([
            "### Computed Factor Exposures (OLS Regression)",
            "",
            "| Factor | Beta | t-stat | p-value | Significant |",
            "|--------|------|--------|---------|-------------|",
        ])
        for fe in opt.factor_exposures:
            sig = "Yes" if fe.p_value < 0.05 else "No"
            lines.append(
                f"| **{fe.factor_name}** | {fe.beta:.4f} | {fe.t_stat:.2f} | "
                f"{fe.p_value:.4f} | {sig} |"
            )
        lines.append("")

    # Risk contribution (MCTR) — top 5
    if opt.risk_contributions:
        lines.extend([
            "### Risk Contribution (MCTR) — Top 5",
            "",
            "| Ticker | Weight | Marginal CTR | % of Portfolio Risk |",
            "|--------|--------|-------------|---------------------|",
        ])
        for rc in opt.risk_contributions[:5]:
            lines.append(
                f"| **{rc.ticker}** | {rc.weight:.1%} | {rc.marginal_ctr:.4f} | "
                f"{rc.pct_contribution:.1%} |"
            )
        lines.append("")

    # Universe weights (non-zero)
    if opt.universe_weights:
        lines.extend([
            "### BL Optimal Universe Weights",
            "",
            "| Ticker | Weight |",
            "|--------|--------|",
        ])
        for t, w in sorted(opt.universe_weights.items(), key=lambda x: -x[1]):
            lines.append(f"| {t} | {w:.1%} |")
        lines.append("")

    return "\n".join(lines)


def _build_full_report(
    result: CommitteeResult,
    memo_md: str,
    bull_md: str,
    short_md: str,
    bear_md: str,
    macro_md: str,
    debate_md: str,
    conviction_md: str,
    provider_name: str,
    status_md: str = "",
) -> str:
    """Build a single consolidated report from all sections."""
    divider = "\n\n---\n\n"
    sections = [
        f"# Investment Committee Report: {result.ticker}",
        f"*Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} | Provider: {provider_name} | "
        f"Duration: {result.total_duration_ms/1000:.1f}s*",
        divider,
        memo_md,
        divider,
        bull_md,
        divider,
        short_md,
        divider,
        bear_md,
        divider,
        macro_md,
        divider,
        debate_md,
        divider,
        conviction_md,
    ]

    # Session history (prior runs in this session)
    session_section = _build_session_section()
    if session_section:
        sections.extend([divider, session_section])

    # Session summary / execution log
    if status_md:
        sections.extend([divider, status_md])

    sections.extend([
        divider,
        "---\n*Disclaimer: This is AI-generated analysis for demonstration purposes only. "
        "Not financial advice.*",
    ])
    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Output formatters — tables + richer layout
# ---------------------------------------------------------------------------

def _format_committee_memo(result: CommitteeResult, provider_name: str = "") -> str:
    """Format the final committee memo as markdown with tables."""
    memo = result.committee_memo
    if not memo:
        return "No committee memo generated."

    rec = memo.recommendation.upper()

    # Color-code the recommendation
    rec_emoji = {
        "STRONG BUY": "🟢🟢", "BUY": "🟢", "HOLD": "🟡",
        "UNDERWEIGHT": "🟠", "SELL": "🔴", "ACTIVE SHORT": "🔴🔴", "AVOID": "⚫",
    }.get(rec, "⚪")

    # T signal display
    t_signal = getattr(memo, 't_signal', 0.0)
    t_direction = getattr(memo, 'position_direction', 0)
    t_confidence = getattr(memo, 'raw_confidence', 0.5)
    t_label = "LONG" if t_direction > 0 else "SHORT" if t_direction < 0 else "FLAT"
    t_emoji = "🟢" if t_signal > 0.3 else "🔴" if t_signal < -0.3 else "🟡"
    t_bar_width = 20
    t_abs = abs(t_signal)
    t_filled = int(t_abs * t_bar_width)
    t_bar = ("+" if t_signal >= 0 else "-") * t_filled + "·" * (t_bar_width - t_filled)

    lines = [
        f"# Investment Committee Memo: {memo.ticker}",
        "",
        "| | |",
        "|---|---|",
        f"| **Recommendation** | {rec_emoji} **{rec}** |",
        f"| **Position Size** | {memo.position_size} |",
        f"| **Conviction** | {memo.conviction}/10 |",
        f"| **Time Horizon** | {memo.time_horizon} |",
        f"| **T Signal (Trading)** | {t_emoji} **{t_signal:+.4f}** ({t_label}, certainty: {t_confidence:.0%}) |",
        f"| **Duration** | {result.total_duration_ms/1000:.1f}s |",
        f"| **Provider** | {provider_name} |",
        "",
        f"**T Signal (Trading Signal) Gauge:** `[{t_bar}]` {t_signal:+.4f}",
        "",
    ]

    # Parsing degradation warning
    parsing_failures = getattr(result, 'parsing_failures', [])
    if parsing_failures:
        failed_agents = ", ".join(
            f.replace("_", " ").title() for f in parsing_failures
        )
        degraded_features = (
            "quantitative sizing heuristics, return decomposition, "
            "sentiment analysis, event paths, and factor exposures"
        )
        lines.extend([
            f"> ⚠️ **Output quality degraded.** The following agent(s) could not produce "
            f"structured output: **{failed_agents}**. Scores and rationales are estimates. "
            f"Features unavailable: {degraded_features}. "
            f"Consider re-running with a more capable model (e.g., Claude Sonnet, GPT-4o).",
            "",
        ])

    # --- Executive Summary ---
    # Build a structured summary paragraph from the memo fields
    rec_word = rec.lower()
    conviction_adj = (
        "high" if memo.conviction >= 7.5 else
        "moderate" if memo.conviction >= 5.0 else
        "low"
    )
    top_factor = memo.key_factors[0] if memo.key_factors else "multiple considerations"
    top_risk = memo.risk_mitigants[0] if memo.risk_mitigants else "standard risk controls"
    event_path_early = getattr(memo, 'event_path', [])
    event_highlight = ""
    if event_path_early:
        event_highlight = f" The nearest catalyst is {event_path_early[0].split(':')[0].strip('.')}."

    lines.extend([
        "---",
        "",
        "## Executive Summary",
        "",
        f"The committee recommends **{rec}** on {memo.ticker} with **{conviction_adj} conviction "
        f"({memo.conviction}/10)** and a **{memo.position_size.lower()}** sizing. "
        f"The thesis is built on {top_factor.lower() if top_factor[0].isupper() else top_factor}. "
        f"Primary risk management requires {top_risk.lower() if top_risk[0].isupper() else top_risk}."
        f"{event_highlight} "
        f"T signal: **{t_signal:+.4f}** ({t_label}).",
        "",
    ])

    # --- PM Synthesis Rationale ---
    pm_rationale = getattr(memo, 'pm_synthesis_rationale', '')
    if pm_rationale:
        lines.extend([
            "## PM Synthesis",
            "",
            f"*{pm_rationale}*",
            "",
        ])

    kf_highlight = ""
    if memo.key_factors:
        kf_text = memo.key_factors[0][:80]
        kf_ellipsis = "..." if len(memo.key_factors[0]) > 80 else ""
        kf_highlight = f"***{len(memo.key_factors)} factors drove the final call — #1: {kf_text}{kf_ellipsis}***"

    lines.extend([
        "---",
        "",
        "## Thesis",
        "",
        memo.thesis_summary,
        "",
        "## Key Decision Factors",
        kf_highlight,
        "",
        "| # | Factor |",
        "|---|--------|",
    ])
    for i, factor in enumerate(memo.key_factors, 1):
        lines.append(f"| {i} | {factor} |")

    # Bull/Bear accepted in a side-by-side table
    bull_pts = memo.bull_points_accepted or ["—"]
    bear_pts = memo.bear_points_accepted or ["—"]
    max_rows = max(len(bull_pts), len(bear_pts))

    lines.extend([
        "",
        "## Evidence Weighed",
        f"***{len(bull_pts)} bull vs {len(bear_pts)} bear arguments evaluated***",
        "",
        "| Bull Points Accepted | Bear Points Accepted |",
        "|---------------------|---------------------|",
    ])
    for i in range(max_rows):
        bull = bull_pts[i] if i < len(bull_pts) else ""
        bear = bear_pts[i] if i < len(bear_pts) else ""
        lines.append(f"| {bull} | {bear} |")

    if memo.dissenting_points:
        lines.extend([
            "",
            "## Where PM Overruled",
            f"***{len(memo.dissenting_points)} point{'s' if len(memo.dissenting_points) != 1 else ''} "
            f"where PM exercised independent judgment***",
        ])
        for point in memo.dissenting_points:
            lines.append(f"> {point}")

    if memo.risk_mitigants:
        risk_text = memo.risk_mitigants[0][:80]
        risk_ellipsis = "..." if len(memo.risk_mitigants[0]) > 80 else ""
        risk_highlight = f"***Primary: {risk_text}{risk_ellipsis}***"
    else:
        risk_highlight = ""
    lines.extend([
        "",
        "## Risk Mitigants Required",
        risk_highlight,
    ])
    for i, mit in enumerate(memo.risk_mitigants, 1):
        lines.append(f"{i}. {mit}")

    # Head-Trader: Implied Vol Assessment
    iv_assessment = getattr(memo, 'implied_vol_assessment', '')
    if iv_assessment:
        # Extract IV vs HV highlight
        iv_snippet = iv_assessment[:100].split(".")[0] if iv_assessment else ""
        lines.extend([
            "",
            "## Volatility Assessment (Trading Desk)",
            f"***{iv_snippet}***" if iv_snippet else "",
            "",
            iv_assessment,
        ])

    # Head-Trader: Event Path
    event_path = getattr(memo, 'event_path', [])
    if event_path:
        # Extract the first event as the key focus
        first_event = event_path[0].split(":")[0].strip() if event_path else ""
        lines.extend([
            "",
            "## Event Path",
            f"***Focus: {first_event} — {len(event_path)} events mapped***" if first_event else "",
            "",
            "| # | Event & Expected Impact |",
            "|---|------------------------|",
        ])
        for i, event in enumerate(event_path, 1):
            lines.append(f"| {i} | {event} |")

    # Head-Trader: Conviction Change Triggers
    triggers = getattr(memo, 'conviction_change_triggers', {})
    if triggers:
        cut_trigger = triggers.get('cut_position', '')
        cut_snippet = cut_trigger[:80] if cut_trigger else "see below"
        lines.extend([
            "",
            "## Conviction Change Triggers",
            f"***Stop-loss: {cut_snippet}***",
            "",
            "| Action | Trigger |",
            "|--------|---------|",
        ])
        for action, trigger in triggers.items():
            label = action.replace("_", " ").title()
            lines.append(f"| **{label}** | {trigger} |")

    # Head-Trader: Factor Exposures
    factors = getattr(memo, 'factor_exposures', {})
    if factors:
        # Find the dominant factor tilt
        dominant = next(
            (f"{k}: {v.split('—')[0].strip()}" for k, v in factors.items()
             if any(w in v.lower() for w in ("high", "negative", "positive", "strong"))),
            "mixed factor profile"
        )
        lines.extend([
            "",
            "## Factor Exposures",
            f"***Dominant tilt — {dominant}***",
            "",
            "| Factor | Exposure |",
            "|--------|----------|",
        ])
        for factor, exposure in factors.items():
            lines.append(f"| **{factor.title()}** | {exposure} |")

    # Quantitative heuristic synthesis
    idio_ret = getattr(memo, 'idio_return_estimate', '')
    sharpe_est = getattr(memo, 'sharpe_estimate', '')
    sortino_est = getattr(memo, 'sortino_estimate', '')
    sizing_method = getattr(memo, 'sizing_method_used', '')
    nmv_rationale = getattr(memo, 'target_nmv_rationale', '')
    vol_target_r = getattr(memo, 'vol_target_rationale', '')

    if any([idio_ret, sharpe_est, sortino_est, sizing_method, nmv_rationale]):
        alpha_snippet = idio_ret.split("—")[0].strip() if idio_ret else "see below"
        lines.extend([
            "",
            "## Quantitative Sizing Heuristics",
            f"***Alpha estimate: {alpha_snippet} | Method: {sizing_method.split('—')[0].strip() if sizing_method else 'TBD'}***",
            "",
            "*LLM-reasoned estimates — heuristic framework, not optimizer output.*",
            "",
            "| Metric | Estimate |",
            "|--------|----------|",
        ])
        if idio_ret:
            lines.append(f"| **Idiosyncratic Return (Alpha)** | {idio_ret} |")
        if sharpe_est:
            lines.append(f"| **Est. Sharpe** | {sharpe_est} |")
        if sortino_est:
            lines.append(f"| **Est. Sortino** | {sortino_est} |")
        if sizing_method:
            lines.append(f"| **Sizing Method** | {sizing_method} |")

        if nmv_rationale:
            lines.extend(["", f"**NMV Rationale:** {nmv_rationale}"])
        if vol_target_r:
            lines.extend(["", f"**Vol Target Rationale:** {vol_target_r}"])

    # Black-Litterman optimizer output (computed, not heuristic)
    opt_result = getattr(result, 'optimization_result', None)
    if opt_result:
        opt_section = _format_optimizer_section(opt_result, memo)
        if opt_section:
            lines.append(opt_section)

    # T Signal Detail
    lines.extend([
        "",
        "## T Signal — Trading Signal (RL Input Feature)",
        "",
        "*The **T signal** is a single scalar trading indicator derived from the PM's conviction. "
        "**T** stands for **Trading signal**. It compresses direction and certainty into one number "
        "for downstream reinforcement-learning or systematic consumption.*",
        "",
        "*Formula: **T = direction × C**, where C = ε + (1 − ε)(1 − H). "
        "Direction is −1 (short), 0 (flat), or +1 (long). C is entropy-adjusted certainty "
        "(ε = 0.01 floor). T ∈ [−1, +1]: positive = long conviction, negative = short conviction, zero = no signal.*",
        "",
        "| Component | Value |",
        "|-----------|-------|",
        f"| **Direction** | {t_direction} ({t_label}) |",
        f"| **Raw Confidence** | {t_confidence:.4f} |",
        f"| **Entropy-Adjusted Certainty (C)** | {0.01 + 0.99 * t_confidence:.4f} |",
        f"| **T = direction * C** | **{t_signal:+.4f}** |",
        "",
        "*T signal interpretation: +1.0 = maximum long conviction, -1.0 = maximum short conviction, 0 = no signal*",
        "",
        "*Entropy-weighted confidence adapted from Darmanin & Vella, "
        "\"[Language Model Guided RL in Quantitative Trading](https://arxiv.org/abs/2508.02366)\" "
        "(arXiv:2508.02366v3, Oct 2025).*",
    ])

    lines.extend([
        "",
        "---",
        f"*Analysis completed in {result.total_duration_ms/1000:.1f}s using {provider_name}*",
    ])

    return "\n".join(lines)


def _format_bull_case(result: CommitteeResult) -> str:
    """Format the sector analyst's bull case with tables."""
    bc = result.bull_case
    if not bc:
        return "No bull case generated."

    lines = [
        f"# Bull Case: {bc.ticker}",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| **Conviction** | {bc.conviction_score}/10 |",
        f"| **Time Horizon** | {bc.time_horizon} |",
        "",
        "## Investment Thesis",
        "",
        bc.thesis,
        "",
    ]

    # Technical outlook
    if bc.technical_outlook:
        lines.extend([
            "## Technical Outlook",
            "",
            bc.technical_outlook,
            "",
        ])

    # Supporting evidence as numbered list
    lines.extend(["## Supporting Evidence", ""])
    for i, ev in enumerate(bc.supporting_evidence, 1):
        lines.append(f"{i}. {ev}")

    # Catalysts
    lines.extend(["", "## Near-Term Catalysts"])
    for i, cat in enumerate(bc.catalysts, 1):
        lines.append(f"{i}. {cat}")

    # Catalyst calendar as table
    if bc.catalyst_calendar:
        lines.extend([
            "",
            "## 12-Month Catalyst Calendar",
            "",
            "| Timeframe | Event | Expected Impact |",
            "|-----------|-------|-----------------|",
        ])
        for entry in bc.catalyst_calendar:
            tf = entry.get("timeframe", "TBD")
            ev = entry.get("event", "—")
            imp = entry.get("impact", "—")
            lines.append(f"| {tf} | {ev} | {imp} |")

    # Key metrics as table
    if bc.key_metrics:
        lines.extend([
            "",
            "## Key Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ])
        for k, v in bc.key_metrics.items():
            lines.append(f"| {k} | {v} |")

    # Sentiment factors
    sentiment_factors = getattr(bc, 'sentiment_factors', [])
    aggregate_sentiment = getattr(bc, 'aggregate_news_sentiment', 'neutral')
    sentiment_divergence = getattr(bc, 'sentiment_divergence', '')

    if sentiment_factors:
        # Sentiment color
        sent_emoji = {
            "strongly_bullish": "🟢🟢", "bullish": "🟢", "neutral": "🟡",
            "bearish": "🔴", "strongly_bearish": "🔴🔴",
        }.get(aggregate_sentiment, "⚪")

        lines.extend([
            "",
            f"## News Sentiment Analysis {sent_emoji}",
            "",
            f"**Aggregate Sentiment:** {sent_emoji} {aggregate_sentiment.replace('_', ' ').title()}",
            "",
            "| Headline | Sentiment | Strength | Catalyst Type |",
            "|----------|-----------|----------|---------------|",
        ])
        for sf in sentiment_factors:
            headline = sf.get("headline", "")[:60]
            if len(sf.get("headline", "")) > 60:
                headline += "..."
            sent = sf.get("sentiment", "neutral")
            strength = sf.get("signal_strength", "moderate")
            cat_type = sf.get("catalyst_type", "")
            sent_icon = "🟢" if sent == "bullish" else "🔴" if sent == "bearish" else "🟡"
            lines.append(f"| {headline} | {sent_icon} {sent} | {strength} | {cat_type} |")

    if sentiment_divergence:
        lines.extend([
            "",
            "## Sentiment-Price Divergence",
            "",
            f"> {sentiment_divergence}",
        ])

    # Quantitative heuristics (return decomposition)
    price_target = getattr(bc, 'price_target', '')
    total_ret = getattr(bc, 'forecasted_total_return', '')
    industry_ret = getattr(bc, 'estimated_industry_return', '')
    idio_ret = getattr(bc, 'idiosyncratic_return', '')
    sharpe = getattr(bc, 'estimated_sharpe', '')
    sortino = getattr(bc, 'estimated_sortino', '')

    if any([price_target, total_ret, idio_ret, sharpe]):
        lines.extend([
            "",
            "## Return Decomposition (Heuristic Estimates)",
            "",
            "*These are LLM-reasoned approximations, not precise calculations.*",
            "",
            "| Component | Estimate & Reasoning |",
            "|-----------|----------------------|",
        ])
        if price_target:
            lines.append(f"| **Price Target** | {price_target} |")
        if total_ret:
            lines.append(f"| **Forecasted Total Return** | {total_ret} |")
        if industry_ret:
            lines.append(f"| **Est. Industry Return** | {industry_ret} |")
        if idio_ret:
            lines.append(f"| **Idiosyncratic Return (Alpha)** | {idio_ret} |")
        if sharpe:
            lines.append(f"| **Est. Sharpe** | {sharpe} |")
        if sortino:
            lines.append(f"| **Est. Sortino** | {sortino} |")

    return "\n".join(lines)


def _format_bear_case(result: CommitteeResult) -> str:
    """Format the risk manager's bear case with sizing and structuring."""
    bc = result.bear_case
    if not bc:
        return "No bear case generated."

    lines = [
        f"# Risk Assessment: {bc.ticker}",
        "",
        "| | |",
        "|---|---|",
        f"| **Bearish Conviction** | {bc.bearish_conviction}/10 |",
    ]

    # Position structure
    pos_struct = getattr(bc, 'position_structure', '')
    if pos_struct:
        lines.append(f"| **Position Structure** | {pos_struct} |")
    stop_loss = getattr(bc, 'stop_loss_level', '')
    if stop_loss:
        lines.append(f"| **Stop-Loss** | {stop_loss} |")
    max_risk = getattr(bc, 'max_risk_allocation', '')
    if max_risk:
        lines.append(f"| **Max Risk Allocation** | {max_risk} |")
    lines.append("")

    # Risks as numbered list
    lines.extend(["## Primary Risks", ""])
    for i, risk in enumerate(bc.risks, 1):
        lines.append(f"{i}. {risk}")

    # Causal chain table
    lines.extend([
        "",
        "## Causal Chain Analysis",
        "",
        "| Order | Effect |",
        "|-------|--------|",
    ])
    for effect in bc.second_order_effects:
        lines.append(f"| 2nd Order | {effect} |")
    for effect in bc.third_order_effects:
        lines.append(f"| 3rd Order | {effect} |")

    lines.extend([
        "",
        "## Worst Case Scenario",
        "",
        bc.worst_case_scenario,
    ])

    # Stress scenarios table
    stress_scenarios = getattr(bc, 'stress_scenarios', [])
    if stress_scenarios:
        lines.extend([
            "",
            "## Stress Scenarios",
            "",
            "| Scenario | P&L Impact |",
            "|----------|------------|",
        ])
        for scenario in stress_scenarios:
            name = scenario.get("scenario", "—")
            impact = scenario.get("impact", "—")
            lines.append(f"| {name} | {impact} |")

    # Correlation flags
    correlation_flags = getattr(bc, 'correlation_flags', [])
    if correlation_flags:
        lines.extend(["", "## Correlation / Crowding Flags", ""])
        for i, flag in enumerate(correlation_flags, 1):
            lines.append(f"{i}. {flag}")

    # Key vulnerabilities
    if bc.key_vulnerabilities:
        lines.extend([
            "",
            "## Key Vulnerabilities",
            "",
            "| Area | Vulnerability |",
            "|------|--------------|",
        ])
        for k, v in bc.key_vulnerabilities.items():
            lines.append(f"| {k} | {v} |")

    return "\n".join(lines)


def _format_short_case(result: CommitteeResult) -> str:
    """Format the short analyst's short case."""
    sc = result.short_case
    if not sc:
        return "No short case generated."

    # Thesis type emoji
    type_emoji = {
        "alpha_short": "🎯", "hedge": "🛡️",
        "pair_leg": "⚖️", "no_position": "⚫",
    }.get(getattr(sc, 'thesis_type', 'no_position'), "⚫")

    lines = [
        f"# Short Case: {sc.ticker}",
        "",
        "| | |",
        "|---|---|",
        f"| **Conviction** | {sc.conviction_score}/10 |",
        f"| **Thesis Type** | {type_emoji} {getattr(sc, 'thesis_type', 'N/A')} |",
    ]

    alpha_beta = getattr(sc, 'alpha_vs_beta_assessment', '')
    if alpha_beta:
        lines.append(f"| **Alpha vs Beta** | {alpha_beta} |")
    borrow = getattr(sc, 'borrow_assessment', '')
    if borrow:
        lines.append(f"| **Borrow Assessment** | {borrow} |")
    est_ret = getattr(sc, 'estimated_short_return', '')
    if est_ret:
        lines.append(f"| **Est. Short Return** | {est_ret} |")
    idio_ret = getattr(sc, 'idiosyncratic_return', '')
    if idio_ret:
        lines.append(f"| **Idiosyncratic Return** | {idio_ret} |")
    est_sharpe = getattr(sc, 'estimated_sharpe', '')
    if est_sharpe:
        lines.append(f"| **Est. Sharpe** | {est_sharpe} |")
    lines.append("")

    # Short thesis
    thesis = getattr(sc, 'short_thesis', '')
    if thesis:
        lines.extend([
            "## Short Thesis",
            "",
            thesis,
            "",
        ])

    # Event path
    event_path = getattr(sc, 'event_path', [])
    if event_path:
        lines.extend(["## Event Path (ordered)", ""])
        for i, event in enumerate(event_path, 1):
            lines.append(f"{i}. {event}")
        lines.append("")

    # Supporting evidence
    evidence = getattr(sc, 'supporting_evidence', [])
    if evidence:
        lines.extend(["## Supporting Evidence", ""])
        for i, ev in enumerate(evidence, 1):
            lines.append(f"{i}. {ev}")
        lines.append("")

    # Key vulnerabilities
    vulns = getattr(sc, 'key_vulnerabilities', {})
    if vulns:
        lines.extend([
            "## Key Vulnerabilities",
            "",
            "| Area | Vulnerability |",
            "|------|--------------|",
        ])
        for k, v in vulns.items():
            lines.append(f"| {k} | {v} |")

    return "\n".join(lines)


def _format_macro_view(result: CommitteeResult) -> str:
    """Format the macro analyst's top-down view with tables."""
    mv = result.macro_view
    if not mv:
        return "No macro analysis generated."

    # Favorability color
    fav = mv.macro_favorability
    fav_emoji = "🟢" if fav >= 7 else "🟡" if fav >= 4 else "🔴"

    lines = [
        f"# Macro Environment: {mv.ticker}",
        "",
        "## Economic Backdrop",
        "",
        "| Dimension | Assessment |",
        "|-----------|-----------|",
        f"| **Cycle Phase** | {mv.economic_cycle_phase} |",
        f"| **Rate Environment** | {mv.rate_environment} |",
        f"| **Central Bank Outlook** | {mv.central_bank_outlook} |",
        f"| **Sector Positioning** | {mv.sector_positioning} |",
        f"| **Macro Favorability** | {fav_emoji} **{fav}/10** |",
        "",
    ]

    # Cycle evidence
    if mv.cycle_evidence:
        lines.extend(["## Cycle Evidence", ""])
        for i, ev in enumerate(mv.cycle_evidence, 1):
            lines.append(f"{i}. {ev}")
        lines.append("")

    # Sector rotation
    if mv.rotation_implications:
        lines.extend([
            "## Sector Rotation",
            "",
            mv.rotation_implications,
            "",
        ])

    # Cross-asset signals table
    if mv.cross_asset_signals:
        lines.extend([
            "## Cross-Asset Signals",
            "",
            "| Asset Class | Signal |",
            "|------------|--------|",
        ])
        for asset, signal in mv.cross_asset_signals.items():
            label = asset.replace("_", " ").title()
            lines.append(f"| **{label}** | {signal} |")
        lines.append("")

    # Geopolitical risks
    if mv.geopolitical_risks:
        lines.extend(["## Geopolitical Risks", ""])
        for i, risk in enumerate(mv.geopolitical_risks, 1):
            lines.append(f"{i}. {risk}")
        lines.append("")

    # Tailwinds / Headwinds side-by-side
    tw = mv.tailwinds or ["—"]
    hw = mv.headwinds or ["—"]
    max_rows = max(len(tw), len(hw))

    lines.extend([
        "## Macro Tailwinds vs. Headwinds",
        "",
        "| Tailwinds | Headwinds |",
        "|-------------|-------------|",
    ])
    for i in range(max_rows):
        t = tw[i] if i < len(tw) else ""
        h = hw[i] if i < len(hw) else ""
        lines.append(f"| {t} | {h} |")

    # Net impact
    lines.extend([
        "",
        "## Net Macro Impact",
        "",
        mv.macro_impact_on_stock if mv.macro_impact_on_stock else "No specific impact narrative provided.",
    ])

    # Portfolio Strategy section
    vol_regime = getattr(mv, 'annualized_vol_regime', '')
    vol_guidance = getattr(mv, 'vol_budget_guidance', '')
    directionality = getattr(mv, 'portfolio_directionality', '')
    style_assessment = getattr(mv, 'sector_style_assessment', '')
    correlation = getattr(mv, 'correlation_regime', '')

    if any([vol_regime, vol_guidance, directionality, style_assessment, correlation]):
        # Direction emoji
        dir_emoji = "🟢" if "long" in directionality.lower() else "🔴" if "short" in directionality.lower() else "🟡"

        lines.extend([
            "",
            "---",
            "",
            "## Portfolio Strategy Guidance",
            "",
        ])

        if vol_regime or directionality:
            lines.extend([
                "| Dimension | Assessment |",
                "|-----------|-----------|",
            ])
            if vol_regime:
                lines.append(f"| **Vol Regime** | {vol_regime} |")
            if directionality:
                lines.append(f"| **Net Exposure** | {dir_emoji} {directionality} |")
            if correlation:
                lines.append(f"| **Correlation Regime** | {correlation} |")
            lines.append("")

        if vol_guidance:
            lines.extend([
                "### Vol Budget Guidance",
                "",
                vol_guidance,
                "",
            ])

        if style_assessment:
            lines.extend([
                "### Sector & Style Assessment",
                "",
                style_assessment,
                "",
            ])

    # Quantitative portfolio construction heuristics
    sector_vol = getattr(mv, 'sector_avg_volatility', '')
    sizing_method = getattr(mv, 'recommended_sizing_method', '')
    vol_target = getattr(mv, 'portfolio_vol_target', '')

    if any([sector_vol, sizing_method, vol_target]):
        lines.extend([
            "",
            "## Position Sizing Framework (Heuristic)",
            "",
            "*LLM-reasoned estimates to guide PM sizing decisions, not optimizer outputs.*",
            "",
            "| Dimension | Guidance |",
            "|-----------|----------|",
        ])
        if sector_vol:
            lines.append(f"| **Sector Avg Volatility** | {sector_vol} |")
        if sizing_method:
            lines.append(f"| **Recommended Sizing Method** | {sizing_method} |")
        if vol_target:
            lines.append(f"| **Portfolio Vol Target** | {vol_target} |")

    return "\n".join(lines)


def _format_debate(result: CommitteeResult) -> str:
    """Format the adversarial debate transcript."""
    lines = [
        f"# Investment Committee Debate: {result.ticker}",
        "",
    ]

    # Show convergence note if bull/bear scores were close
    if result.bull_case and result.bear_case:
        spread = abs(result.bull_case.conviction_score - result.bear_case.bearish_conviction)
        if spread < 2.0:
            lines.extend([
                f"> **ℹ️ Convergence noted** — bull conviction "
                f"({result.bull_case.conviction_score}/10) and bearish conviction "
                f"({result.bear_case.bearish_conviction}/10) spread is {spread:.1f} "
                f"(within 2.0 threshold). Agents largely agree on this name.",
                "",
            ])

    if result.long_rebuttal:
        ar = result.long_rebuttal
        lines.extend([
            "## Long Analyst's Rebuttal (to Short Case)",
            "",
            "### Challenges to Short Analyst",
            "",
        ])
        for i, point in enumerate(ar.points, 1):
            lines.append(f"{i}. {point}")
        lines.extend(["", "### Concessions", ""])
        for con in ar.concessions:
            lines.append(f"> {con}")
        if ar.revised_conviction is not None:
            lines.append(f"\n**Revised Conviction:** {ar.revised_conviction}/10")
        lines.append("")

    if result.short_rebuttal:
        sr = result.short_rebuttal
        lines.extend([
            "## Short Analyst's Rebuttal (to Bull Case)",
            "",
            "### Challenges to Long Analyst",
            "",
        ])
        for i, point in enumerate(sr.points, 1):
            lines.append(f"{i}. {point}")
        lines.extend(["", "### Concessions", ""])
        for con in sr.concessions:
            lines.append(f"> {con}")
        if sr.revised_conviction is not None:
            lines.append(f"\n**Revised Short Conviction:** {sr.revised_conviction}/10")
        lines.append("")

    if result.risk_rebuttal:
        rr = result.risk_rebuttal
        lines.extend([
            "## Risk Manager's Commentary (Sizing)",
            "",
            "### Sizing Feedback",
            "",
        ])
        for i, point in enumerate(rr.points, 1):
            lines.append(f"{i}. {point}")
        lines.extend(["", "### Concessions", ""])
        for con in rr.concessions:
            lines.append(f"> {con}")
        if rr.revised_conviction is not None:
            lines.append(f"\n**Revised Risk Score:** {rr.revised_conviction}/10")

    return "\n".join(lines)


def _format_conviction_evolution(result: CommitteeResult) -> str:
    """Format the conviction evolution timeline with a visual chart."""
    timeline = result.conviction_timeline
    if not timeline:
        return "No conviction data captured."

    # Human-readable stance labels
    def _stance(snap) -> str:
        if snap.agent == "Portfolio Manager":
            return "Final Decision"
        if snap.agent == "Macro Analyst":
            return "Macro Backdrop"
        if snap.agent == "Short Analyst":
            return "Short Case"
        return "Bull Case" if snap.score_type == "conviction" else "Risk Assessment"

    lines = [
        f"# Conviction Evolution: {result.ticker}",
        "",
        "How each agent's conviction shifted across the analysis phases — and *why*.",
        "",
    ]

    # ── Data table ──
    lines.extend([
        "## Score Timeline",
        "",
        "| Phase | Agent | Stance | Score | Rationale |",
        "|-------|-------|--------|-------|-----------|",
    ])
    for snap in timeline:
        rationale = getattr(snap, "rationale", "") or ""
        # Truncate long rationales for table readability
        display_rationale = rationale[:200] + "..." if len(rationale) > 200 else rationale
        lines.append(
            f"| {snap.phase} | {snap.agent} | {_stance(snap)} "
            f"| **{snap.score}/10** | {display_rationale} |"
        )

    # ── Group scores by agent (used by map + interpretation) ──
    bull_scores = [s for s in timeline if s.agent == "Sector Analyst"]
    bear_scores = [s for s in timeline if s.agent == "Risk Manager"]
    macro_scores = [s for s in timeline if s.agent == "Macro Analyst"]
    pm_scores = [s for s in timeline if s.agent == "Portfolio Manager"]

    # ── Visual bar chart — 4 agent sections with initial/final dual bars ──
    lines.extend(["", "## Visual Conviction Map", ""])

    bar_width = 30

    def _draw_bar(score, char, width=bar_width):
        filled = int((score / 10) * width)
        return char * filled + "·" * (width - filled)

    # Sector Analyst (Bull)
    lines.append("```")
    lines.append("🟢 Sector Analyst (Bull)")
    if bull_scores:
        b_initial = bull_scores[0].score
        if len(bull_scores) >= 2:
            b_final = bull_scores[-1].score
            b_delta = b_final - b_initial
            b_sign = "+" if b_delta >= 0 else ""
            lines.append(f"   Initial  [{_draw_bar(b_initial, '▒')}] {b_initial}")
            lines.append(f"   Final    [{_draw_bar(b_final, '▓')}] {b_final}  Δ {b_sign}{b_delta:.1f}")
        else:
            lines.append(f"   Score    [{_draw_bar(b_initial, '▓')}] {b_initial}  (no debate data)")
    lines.append("```")
    lines.append("")

    # Risk Manager (Bear)
    lines.append("```")
    lines.append("🔴 Risk Manager (Bear)")
    if bear_scores:
        r_initial = bear_scores[0].score
        if len(bear_scores) >= 2:
            r_final = bear_scores[-1].score
            r_delta = r_final - r_initial
            r_sign = "+" if r_delta >= 0 else ""
            lines.append(f"   Initial  [{_draw_bar(r_initial, '▒')}] {r_initial}")
            lines.append(f"   Final    [{_draw_bar(r_final, '░')}] {r_final}  Δ {r_sign}{r_delta:.1f}")
        else:
            lines.append(f"   Score    [{_draw_bar(r_initial, '░')}] {r_initial}  (no debate data)")
    lines.append("```")
    lines.append("")

    # Macro Analyst
    lines.append("```")
    lines.append("🟣 Macro Analyst")
    if macro_scores:
        lines.append(f"   Backdrop [{_draw_bar(macro_scores[0].score, '▒')}] {macro_scores[0].score}  (no debate shift)")
    lines.append("```")
    lines.append("")

    # Portfolio Manager
    t_sig = getattr(result.committee_memo, 't_signal', 0.0) if result.committee_memo else 0.0
    lines.append("```")
    lines.append("🔵 Portfolio Manager")
    if pm_scores:
        lines.append(f"   Decision [{_draw_bar(pm_scores[0].score, '█')}] {pm_scores[0].score}  T: {t_sig:+.2f}")
    lines.append("```")

    lines.extend(["", "## How Scores Shifted", ""])

    if len(bull_scores) >= 2:
        initial, revised = bull_scores[0].score, bull_scores[-1].score
        delta = revised - initial
        direction = "more bullish" if delta > 0 else "less bullish" if delta < 0 else "unchanged"
        arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
        rationale = bull_scores[-1].rationale
        lines.append(
            f"- **Sector Analyst (Bull):** {initial}/10 → {revised}/10 "
            f"({arrow} {direction}, shifted {abs(delta):.1f} after debate)"
        )
        if rationale:
            lines.append(f"  - *{rationale}*")
    elif bull_scores:
        lines.append(f"- **Sector Analyst (Bull):** {bull_scores[0].score}/10")
        if bull_scores[0].rationale:
            lines.append(f"  - *{bull_scores[0].rationale}*")

    if len(bear_scores) >= 2:
        initial, revised = bear_scores[0].score, bear_scores[-1].score
        delta = revised - initial
        direction = "more bearish" if delta > 0 else "less bearish" if delta < 0 else "unchanged"
        arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
        rationale = bear_scores[-1].rationale
        lines.append(
            f"- **Risk Manager (Bear):** {initial}/10 → {revised}/10 "
            f"({arrow} {direction}, shifted {abs(delta):.1f} after debate)"
        )
        if rationale:
            lines.append(f"  - *{rationale}*")
    elif bear_scores:
        lines.append(f"- **Risk Manager (Bear):** {bear_scores[0].score}/10")
        if bear_scores[0].rationale:
            lines.append(f"  - *{bear_scores[0].rationale}*")

    if macro_scores:
        lines.append(f"- **Macro Analyst:** Favorability {macro_scores[0].score}/10")
        if macro_scores[0].rationale:
            lines.append(f"  - *{macro_scores[0].rationale}*")
    if pm_scores:
        lines.append(f"- **Portfolio Manager:** Final conviction {pm_scores[0].score}/10")
        if pm_scores[0].rationale:
            lines.append(f"  - *{pm_scores[0].rationale}*")

    # ── Interpretation ──
    lines.extend(["", "## Interpretation", ""])

    if bull_scores and bear_scores and pm_scores:
        bull_initial = bull_scores[0].score
        bull_final = bull_scores[-1].score
        bear_initial = bear_scores[0].score
        bear_final = bear_scores[-1].score
        macro_fav = macro_scores[0].score if macro_scores else 5.0
        pm_final = pm_scores[0].score

        # ── Debate Dynamics ──
        lines.append("### Debate Dynamics")
        lines.append("")

        bull_delta = bull_final - bull_initial
        bear_delta = bear_final - bear_initial
        initial_spread = abs(bull_initial - bear_initial)
        final_spread = abs(bull_final - bear_final)

        if len(bull_scores) >= 2 and len(bear_scores) >= 2:
            spread_verb = "narrowed" if final_spread < initial_spread else "widened" if final_spread > initial_spread else "held steady at"
            spread_val = f"from {initial_spread:.1f} to {final_spread:.1f} points" if final_spread != initial_spread else f"at {final_spread:.1f} points"
            lines.append(
                f"The adversarial debate {spread_verb} the bull-bear spread {spread_val}."
            )

            bull_verb = "softened" if bull_delta < 0 else "hardened" if bull_delta > 0 else "held"
            bear_verb = "softened" if bear_delta < 0 else "hardened" if bear_delta > 0 else "held"
            bull_sign = "+" if bull_delta >= 0 else ""
            bear_sign = "+" if bear_delta >= 0 else ""

            lines.append(
                f"The bull {bull_verb} from {bull_initial} to {bull_final} "
                f"(Δ {bull_sign}{bull_delta:.1f}), while the bear "
                f"{bear_verb} from {bear_initial} to {bear_final} "
                f"(Δ {bear_sign}{bear_delta:.1f})."
            )

            # Convergence / divergence assessment
            if bull_delta <= 0 and bear_delta <= 0:
                lines.append(
                    "Both sides moderated — convergence suggests the debate surfaced "
                    "genuine trade-offs rather than entrenching positions."
                )
            elif bull_delta >= 0 and bear_delta >= 0:
                lines.append(
                    "Both sides hardened — divergence suggests the debate reinforced "
                    "each agent's priors rather than finding common ground."
                )
            else:
                stronger = "bull" if bull_delta > 0 else "bear"
                weaker = "bear" if bull_delta > 0 else "bull"
                lines.append(
                    f"The {stronger} strengthened while the {weaker} conceded ground — "
                    f"an asymmetric shift favoring the {stronger} thesis."
                )
        else:
            lines.append("Insufficient debate data to assess dynamics.")

        # ── Macro Context ──
        lines.append("")
        lines.append("### Macro Context")
        lines.append("")

        if macro_fav >= 7:
            macro_label = "favorable"
        elif macro_fav >= 4:
            macro_label = "neutral"
        else:
            macro_label = "hostile"

        macro_detail = ""
        if result.macro_view:
            cycle = getattr(result.macro_view, 'economic_cycle_phase', '') or ''
            rates = getattr(result.macro_view, 'rate_environment', '') or ''
            if cycle and rates:
                macro_detail = f" {cycle.capitalize()} cycle dynamics with a {rates} rate environment"
                if macro_fav >= 7:
                    macro_detail += " provide tailwinds for the thesis."
                elif macro_fav < 4:
                    macro_detail += " create headwinds for the thesis."
                else:
                    macro_detail += " create a mixed setting — neither strongly supportive nor adverse."

        lines.append(
            f"The macro backdrop scored **{macro_fav}/10** (**{macro_label}**)."
            f"{macro_detail}"
        )

        # ── PM Synthesis ──
        lines.append("")
        lines.append("### PM Synthesis")
        lines.append("")

        memo = result.committee_memo
        rec = memo.recommendation if memo else ""
        direction = getattr(memo, 'position_direction', 0) if memo else 0
        pos_size = getattr(memo, 'position_size', '') if memo else ""
        horizon = getattr(memo, 'time_horizon', '') if memo else ""
        thesis = getattr(memo, 'thesis_summary', '') if memo else ""

        if direction < 0:
            side_label = "bear"
        elif direction > 0:
            side_label = "bull"
        else:
            side_label = "neither bull nor bear"

        if pm_final >= 7:
            conv_label = "high conviction"
        elif pm_final >= 4:
            conv_label = "moderate conviction"
        else:
            conv_label = "low conviction"

        lines.append(
            f"The PM issued **{rec}** with **{conv_label}** ({pm_final}/10), "
            f"siding with the **{side_label}** thesis."
        )

        # Add the overrule context — did PM side against the stronger arguer?
        if direction > 0 and bear_final > bull_final:
            lines.append(
                f"The PM overruled the bear's stronger conviction ({bear_final}/10) "
                f"in favor of the bull case ({bull_final}/10)."
            )
        elif direction < 0 and bull_final > bear_final:
            lines.append(
                f"The PM overruled the bull's stronger conviction ({bull_final}/10) "
                f"in favor of the bear case ({bear_final}/10)."
            )

        # First sentence of thesis as the "weighted X as decisive" anchor
        if thesis:
            first_sent = thesis.split(". ")[0].rstrip(".")
            if len(first_sent) > 20:
                lines.append(f"Core thesis: *{first_sent}.*")

        sizing_parts = []
        if pos_size:
            sizing_parts.append(f"Position: {pos_size}")
        if horizon:
            sizing_parts.append(f"Horizon: {horizon}")
        if sizing_parts:
            lines.append(f"{' · '.join(sizing_parts)}.")

        # ── Signal Strength ──
        lines.append("")
        lines.append("### Signal Strength")
        lines.append("")

        t_val = getattr(memo, 't_signal', 0.0) if memo else 0.0
        raw_conf = getattr(memo, 'raw_confidence', 0.5) if memo else 0.5

        t_abs = abs(t_val)
        if t_abs > 0.5:
            strength = "strong"
        elif t_abs > 0.2:
            strength = "moderate"
        else:
            strength = "weak"

        if t_val > 0:
            t_dir = "long"
        elif t_val < 0:
            t_dir = "short"
        else:
            t_dir = "flat"

        lines.append(
            f"T signal: **{t_val:+.4f}** — **{strength} {t_dir}** signal "
            f"with {raw_conf:.0%} certainty."
        )

        if t_abs > 0.5:
            lines.append(
                "T magnitude above 0.5 reflects meaningful conviction strength, "
                "suitable for downstream RL or systematic consumption."
            )
        elif t_abs > 0.2:
            lines.append(
                "T magnitude in the 0.2–0.5 range indicates a directional lean "
                "but with notable uncertainty — size accordingly."
            )
        else:
            lines.append(
                "T magnitude below 0.2 indicates insufficient conviction for "
                "a high-confidence directional signal."
            )

    lines.extend([
        "",
        "---",
        "*🟢 Sector Analyst (▒ initial, ▓ final) · "
        "🔴 Risk Manager (▒ initial, ░ final) · "
        "🟣 Macro Analyst (▒ backdrop) · "
        "🔵 Portfolio Manager (█ decision)*",
    ])

    return "\n".join(lines)


def _format_xai_analysis(result: CommitteeResult) -> str:
    """Format the XAI pre-screen analysis as markdown."""
    xai = result.xai_result
    if not xai:
        return "*XAI pre-screen not available for this run.*"

    # Handle both Pydantic model and raw dict
    if isinstance(xai, dict):
        ticker = xai.get("ticker", "")
        narrative = xai.get("narrative", "")
        comp_time = xai.get("computation_time_ms", 0)
        features = xai.get("features_used", {})
        ranking = xai.get("feature_importance_ranking", [])
        distress = xai.get("distress", {})
        returns = xai.get("returns", {})
    else:
        ticker = xai.ticker
        narrative = xai.narrative
        comp_time = xai.computation_time_ms
        features = xai.features_used
        ranking = xai.feature_importance_ranking
        distress = xai.distress if hasattr(xai, 'distress') else {}
        returns = xai.returns if hasattr(xai, 'returns') else {}

    # Distress data
    if isinstance(distress, dict):
        pfd = distress.get("pfd", 0)
        z_score = distress.get("z_score")
        zone = distress.get("distress_zone", "")
        model_used = distress.get("model_used", "")
        top_risk = distress.get("top_risk_factors", [])
    else:
        pfd = distress.pfd
        z_score = distress.z_score
        zone = distress.distress_zone
        model_used = distress.model_used
        top_risk = distress.top_risk_factors

    # Returns data
    if isinstance(returns, dict):
        is_distressed = returns.get("is_distressed", False)
        distress_flag = returns.get("distress_flag", "")
        er = returns.get("expected_return", 0)
        er_pct = returns.get("expected_return_pct", "")
        ey_proxy = returns.get("earnings_yield_proxy", 0)
        top_return = returns.get("top_return_factors", [])
    else:
        is_distressed = returns.is_distressed
        distress_flag = returns.distress_flag
        er = returns.expected_return
        er_pct = returns.expected_return_pct
        ey_proxy = returns.earnings_yield_proxy
        top_return = returns.top_return_factors

    # Zone styling
    zone_emoji = {"safe": "🟢", "grey": "🟡", "distress": "🔴"}.get(zone, "⚪")

    lines = [
        f"# XAI Pre-Screen: {ticker}",
        "",
        f"*Explainable AI analysis using Shapley values — computed in {comp_time:.0f}ms*",
        "",
        "---",
        "",
        "## Distress Assessment",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| **Probability of Financial Distress (PFD)** | {pfd:.1%} |",
    ]
    if z_score is not None:
        lines.append(f"| **Altman Z-Score** | {z_score:.2f} |")
    lines.extend([
        f"| **Distress Zone** | {zone_emoji} **{zone.upper()}** |",
        f"| **Model Used** | {model_used} |",
        f"| **Screening Result** | {distress_flag} |",
        "",
    ])

    # Top risk factors
    if top_risk:
        lines.extend([
            "### Top Risk Factors (SHAP)",
            "",
            "| Feature | SHAP Contribution |",
            "|---------|-------------------|",
        ])
        for factor in top_risk[:5]:
            if isinstance(factor, dict):
                for fname, fval in factor.items():
                    direction = "+" if fval > 0 else ""
                    lines.append(f"| {fname} | {direction}{fval:.4f} |")
            else:
                lines.append(f"| {factor} | — |")
        lines.append("")

    # Expected return
    lines.extend([
        "## Expected Return (Risk-Adjusted)",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| **Expected Return** | **{er_pct}** |",
        f"| **Earnings Yield Proxy** | {ey_proxy:.4f} |",
        f"| **Formula** | ER = (1 - PFD) x Earnings Yield |",
        f"| **Distressed?** | {'Yes' if is_distressed else 'No'} |",
        "",
    ])

    # Top return drivers
    if top_return:
        lines.extend([
            "### Top Return Drivers (SHAP)",
            "",
            "| Feature | SHAP Contribution |",
            "|---------|-------------------|",
        ])
        for factor in top_return[:5]:
            if isinstance(factor, dict):
                for fname, fval in factor.items():
                    direction = "+" if fval > 0 else ""
                    lines.append(f"| {fname} | {direction}{fval:.4f} |")
            else:
                lines.append(f"| {factor} | — |")
        lines.append("")

    # Feature importance ranking
    if ranking:
        lines.extend([
            "## Feature Importance Ranking",
            "",
            "| Rank | Feature |",
            "|------|---------|",
        ])
        for i, feat in enumerate(ranking[:10], 1):
            lines.append(f"| {i} | {feat} |")
        lines.append("")

    # Features used
    if features:
        lines.extend([
            "## Features Extracted",
            "",
            "| Feature | Value |",
            "|---------|-------|",
        ])
        for fname, fval in features.items():
            lines.append(f"| {fname} | {fval:.4f} |")
        lines.append("")

    # Narrative summary
    if narrative:
        lines.extend([
            "## Narrative Summary",
            "",
            narrative,
            "",
        ])

    lines.extend([
        "---",
        "*Based on: Sotic & Radovanovic (2024), \"Explainable AI in Finance\" "
        "(doi:10.20935/AcadAI8017)*",
    ])

    return "\n".join(lines)


def _format_status(result: CommitteeResult, messages: list[str], provider_name: str = "") -> str:
    """Format the status/summary view with tables."""
    stats = TraceRenderer.summary_stats(result.traces)

    model_map = {
        LLMProvider.ANTHROPIC: settings.anthropic_model,
        LLMProvider.GOOGLE: settings.google_model,
        LLMProvider.OPENAI: settings.openai_model,
        LLMProvider.HUGGINGFACE: settings.hf_model,
        LLMProvider.OLLAMA: settings.ollama_model,
    }
    provider = PROVIDER_DISPLAY.get(provider_name, settings.llm_provider)
    model_name = model_map.get(provider, "unknown")

    lines = [
        f"# Session Summary: {result.ticker}",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| **LLM Provider** | {provider_name} |",
        f"| **Model** | `{model_name}` |",
        f"| **Total Duration** | {stats['total_duration_s']}s |",
        f"| **Total Agents** | {stats['total_agents']} |",
        f"| **Total Reasoning Steps** | {stats['total_steps']} |",
        f"| **Total Tokens** | {stats['total_tokens']} |",
        f"| **Debate Rounds** | {settings.max_debate_rounds} |",
        "",
        "## Agent Breakdown",
        "",
        "| Agent | Steps | Duration |",
        "|-------|-------|----------|",
    ]

    for agent, info in stats["per_agent"].items():
        lines.append(f"| {agent} | {info['steps']} | {info['duration_s']}s |")

    # Parsing degradation summary
    parsing_failures = getattr(result, 'parsing_failures', [])
    if parsing_failures:
        failed_agents = ", ".join(
            f.replace("_", " ").title() for f in parsing_failures
        )
        lines.extend([
            "",
            "## ⚠️ Parsing Degradation",
            "",
            f"The following agent(s) could not produce structured JSON output: **{failed_agents}**.",
            "",
            "This typically occurs with smaller language models that cannot reliably produce "
            "complex structured output. The following features are unavailable in this run:",
            "",
            "- Quantitative sizing heuristics (Sharpe, Sortino, sizing method)",
            "- Return decomposition (price target, idiosyncratic return)",
            "- News sentiment extraction and divergence analysis",
            "- Trading-fluent PM fields (implied vol, event paths, factor exposures)",
            "- Conviction timeline rationales may be incomplete",
            "",
            "**Recommendation:** Re-run with a more capable model (Claude Sonnet, GPT-4o, "
            "or Gemini 2.5 Pro) for full-featured output.",
        ])

    lines.extend(["", "## Execution Log", "```"])
    lines.extend(messages)
    lines.append("```")

    return "\n".join(lines)


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

<span class="agent-chips">Sector Analyst &middot; Risk Manager &middot; Macro Strategist &middot; Portfolio Manager</span>
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
                     run_btn, phase1_btn, phase2_btn]

        def _lock_inputs():
            updates = [gr.update(interactive=False) for _ in range(8)]  # input fields
            updates.extend([
                gr.update(interactive=False, value="..."),  # run_btn
                gr.update(interactive=False, value="..."),  # phase1_btn
                gr.update(interactive=False, value="..."),  # phase2_btn
            ])
            return updates

        def _unlock_inputs():
            updates = [gr.update(interactive=True) for _ in range(8)]  # input fields
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
                     report_path_state],
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
                     report_path_state, review_accordion],
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
