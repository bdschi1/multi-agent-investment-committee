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

Architecture:
    User Input → Data Gathering → Parallel Analysis → Debate → Synthesis → Memo
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gradio as gr

from config.settings import settings, LLMProvider
from tools.data_aggregator import DataAggregator
from orchestrator.committee import InvestmentCommittee, CommitteeResult
from orchestrator.reasoning_trace import TraceRenderer

logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)

# Ensure local-only directories exist
RUNS_DIR = Path("runs")
EXPORTS_DIR = Path("exports")
RUNS_DIR.mkdir(exist_ok=True)
EXPORTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Provider-specific model factories
# ---------------------------------------------------------------------------

def _create_anthropic_model(model_name: str | None = None) -> callable:
    """Create a Claude (Anthropic) callable."""
    from anthropic import Anthropic

    client = Anthropic(api_key=settings.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY"))
    model = model_name or settings.anthropic_model

    def call(prompt: str) -> str:
        response = client.messages.create(
            model=model,
            max_tokens=settings.max_tokens_per_agent,
            temperature=settings.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    return call


def _create_google_model(model_name: str | None = None) -> callable:
    """Create a Gemini (Google) callable."""
    from google import genai

    client = genai.Client(api_key=settings.google_api_key or os.environ.get("GOOGLE_API_KEY"))
    model = model_name or settings.google_model

    def call(prompt: str) -> str:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config={
                "temperature": settings.temperature,
                "max_output_tokens": settings.max_tokens_per_agent,
            },
        )
        return response.text

    return call


def _create_openai_model(model_name: str | None = None) -> callable:
    """Create an OpenAI callable."""
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError(
            "OpenAI package not installed. Run: pip install -e '.[openai]'"
        )

    client = OpenAI(api_key=settings.openai_api_key or os.environ.get("OPENAI_API_KEY"))
    model = model_name or settings.openai_model

    def call(prompt: str) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=settings.temperature,
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

    def call(prompt: str) -> str:
        response = client.text_generation(
            prompt,
            max_new_tokens=settings.max_tokens_per_agent,
            temperature=max(settings.temperature, 0.01),  # HF needs >0
            do_sample=True,
        )
        return response

    return call


def _create_ollama_model(model_name: str | None = None) -> callable:
    """Create an Ollama (local) callable for open-source models."""
    try:
        import ollama as ollama_lib
    except ImportError:
        raise RuntimeError(
            "Ollama package not installed. Run: pip install -e '.[ollama]'\n"
            "Also ensure Ollama is running: ollama serve"
        )

    model = model_name or settings.ollama_model
    client = ollama_lib.Client(host=settings.ollama_base_url)

    def call(prompt: str) -> str:
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": settings.temperature,
                "num_predict": settings.max_tokens_per_agent,
            },
        )
        return response["message"]["content"]

    return call


# ---------------------------------------------------------------------------
# Unified model factory
# ---------------------------------------------------------------------------

PROVIDER_FACTORIES = {
    LLMProvider.ANTHROPIC: _create_anthropic_model,
    LLMProvider.GOOGLE: _create_google_model,
    LLMProvider.OPENAI: _create_openai_model,
    LLMProvider.HUGGINGFACE: _create_huggingface_model,
    LLMProvider.OLLAMA: _create_ollama_model,
}

# Display names for the UI dropdown
PROVIDER_DISPLAY = {
    "Claude (Anthropic)": LLMProvider.ANTHROPIC,
    "Gemini (Google)": LLMProvider.GOOGLE,
    "GPT (OpenAI)": LLMProvider.OPENAI,
    "HuggingFace API": LLMProvider.HUGGINGFACE,
    "Ollama (Local)": LLMProvider.OLLAMA,
}

# Reverse map for default selection
PROVIDER_TO_DISPLAY = {v: k for k, v in PROVIDER_DISPLAY.items()}


def create_model(provider: LLMProvider | None = None) -> callable:
    """
    Create the LLM callable for the given provider.

    Architecture note: Every provider returns a simple `callable(str) -> str`.
    This uniform interface means agents don't know or care which LLM they're
    talking to. Swapping providers is a one-line change.
    """
    provider = provider or settings.llm_provider
    factory = PROVIDER_FACTORIES.get(provider)
    if not factory:
        raise ValueError(f"Unknown provider: {provider}")
    return factory()


# ---------------------------------------------------------------------------
# JSON run logger (local only — never pushed to git/HF)
# ---------------------------------------------------------------------------

def _log_run(result: CommitteeResult, provider_name: str, model_name: str,
             debate_rounds: int, user_context: str) -> None:
    """Append run results to a local JSON log file."""
    try:
        log_file = RUNS_DIR / "run_log.jsonl"
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ticker": result.ticker,
            "provider": provider_name,
            "model": model_name,
            "debate_rounds": debate_rounds,
            "user_context": user_context,
            "duration_s": round(result.total_duration_ms / 1000, 1),
            "total_tokens": result.total_tokens,
            "recommendation": result.committee_memo.recommendation if result.committee_memo else None,
            "conviction": result.committee_memo.conviction if result.committee_memo else None,
            "bull_conviction": result.bull_case.conviction_score if result.bull_case else None,
            "bear_risk_score": result.bear_case.risk_score if result.bear_case else None,
            "macro_favorability": result.macro_view.macro_favorability if result.macro_view else None,
            "full_result": result.to_dict(),
        }
        with open(log_file, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        logger.info(f"Run logged to {log_file}")
    except Exception as e:
        logger.warning(f"Failed to log run: {e}")


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def run_committee_analysis(
    ticker: str,
    user_context: str,
    provider_name: str,
    debate_rounds: int,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str, str, str, str, str, str, str, str, str]:
    """
    Run the full investment committee analysis.

    Returns formatted outputs for each Gradio tab (8 markdown tabs + 1 file).
    """
    if not ticker or not ticker.strip():
        return (
            "Please enter a ticker symbol.",
            "", "", "", "", "", "", "", None
        )

    ticker = ticker.strip().upper()
    status_messages = []

    # Resolve provider from UI dropdown
    provider = PROVIDER_DISPLAY.get(provider_name, settings.llm_provider)

    # Override debate rounds from UI
    original_rounds = settings.max_debate_rounds
    settings.max_debate_rounds = debate_rounds

    def on_status(msg: str):
        status_messages.append(msg)
        if "Phase 1" in msg:
            progress(0.15, desc="⏳ Phase 1/3 — Analysts + Macro analyzing in parallel...")
        elif "Bull case:" in msg:
            progress(0.35, desc="✅ Phase 1 complete — initial scores in")
        elif "Phase 2" in msg:
            progress(0.40, desc="⚔️ Phase 2/3 — Adversarial debate starting...")
        elif "Debate round" in msg:
            progress(0.50, desc=f"⚔️ {msg.strip()}")
        elif "Debate complete" in msg:
            progress(0.65, desc="✅ Debate complete — scores revised")
        elif "Phase 3" in msg:
            progress(0.70, desc="🧠 Phase 3/3 — Portfolio Manager synthesizing...")
        elif "Decision:" in msg:
            progress(0.90, desc="📋 Decision reached — formatting report...")
        elif "Committee complete" in msg:
            progress(1.0, desc="✅ Committee complete!")

    try:
        # Initialize model and committee
        progress(0.05, desc=f"Initializing {provider_name}...")
        model = create_model(provider)
        committee = InvestmentCommittee(model=model)

        # Gather data
        progress(0.1, desc=f"Gathering market data for {ticker}...")
        context = DataAggregator.gather_context(ticker, user_context)

        # Run the committee (synchronous — parallel via ThreadPoolExecutor internally)
        result: CommitteeResult = committee.run(ticker, context, on_status=on_status)

        # Determine model used
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
        bear_md = _format_bear_case(result)
        macro_md = _format_macro_view(result)
        debate_md = _format_debate(result)
        conviction_md = _format_conviction_evolution(result)
        trace_md = TraceRenderer.to_gradio_accordion(result.traces)
        status_md = _format_status(result, status_messages, provider_name)

        # Generate full report text for copy/download
        full_report = _build_full_report(
            result, memo_md, bull_md, bear_md, macro_md, debate_md, conviction_md, provider_name
        )

        # Write PDF-ready markdown to exports
        export_path = EXPORTS_DIR / f"{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(export_path, "w") as f:
            f.write(full_report)

        return memo_md, bull_md, bear_md, macro_md, debate_md, conviction_md, trace_md, status_md, str(export_path)

    except Exception as e:
        logger.exception(f"Committee analysis failed for {ticker}")
        error_msg = (
            f"## Error\n\n"
            f"**Provider:** {provider_name}\n\n"
            f"**Error:** {str(e)}\n\n"
            f"Check that your API key is set in `.env` and the provider is available."
        )
        return error_msg, "", "", "", "", "", "", f"Error: {str(e)}", None
    finally:
        # Restore original debate rounds
        settings.max_debate_rounds = original_rounds


# ---------------------------------------------------------------------------
# Full report builder (for copy + download)
# ---------------------------------------------------------------------------

def _build_full_report(
    result: CommitteeResult,
    memo_md: str,
    bull_md: str,
    bear_md: str,
    macro_md: str,
    debate_md: str,
    conviction_md: str,
    provider_name: str,
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
        bear_md,
        divider,
        macro_md,
        divider,
        debate_md,
        divider,
        conviction_md,
        divider,
        "---\n*Disclaimer: This is AI-generated analysis for demonstration purposes only. "
        "Not financial advice.*",
    ]
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

    lines = [
        f"# Investment Committee Memo: {memo.ticker}",
        "",
        "| | |",
        "|---|---|",
        f"| **Recommendation** | {rec_emoji} **{rec}** |",
        f"| **Position Size** | {memo.position_size} |",
        f"| **Conviction** | {memo.conviction}/10 |",
        f"| **Time Horizon** | {memo.time_horizon} |",
        f"| **Duration** | {result.total_duration_ms/1000:.1f}s |",
        f"| **Provider** | {provider_name} |",
        "",
        "---",
        "",
        "## Thesis",
        "",
        memo.thesis_summary,
        "",
        "## Key Decision Factors",
        "",
        "| # | Factor |",
        "|---|--------|",
    ]
    for i, factor in enumerate(memo.key_factors, 1):
        lines.append(f"| {i} | {factor} |")

    # Bull/Bear accepted in a side-by-side table
    bull_pts = memo.bull_points_accepted or ["—"]
    bear_pts = memo.bear_points_accepted or ["—"]
    max_rows = max(len(bull_pts), len(bear_pts))

    lines.extend([
        "",
        "## Evidence Weighed",
        "",
        "| Bull Points Accepted | Bear Points Accepted |",
        "|---------------------|---------------------|",
    ])
    for i in range(max_rows):
        bull = bull_pts[i] if i < len(bull_pts) else ""
        bear = bear_pts[i] if i < len(bear_pts) else ""
        lines.append(f"| {bull} | {bear} |")

    if memo.dissenting_points:
        lines.extend(["", "## Where PM Overruled"])
        for point in memo.dissenting_points:
            lines.append(f"> {point}")

    lines.extend(["", "## Risk Mitigants Required"])
    for i, mit in enumerate(memo.risk_mitigants, 1):
        lines.append(f"{i}. {mit}")

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

    return "\n".join(lines)


def _format_bear_case(result: CommitteeResult) -> str:
    """Format the risk manager's bear case with tables and short pitch."""
    bc = result.bear_case
    if not bc:
        return "No bear case generated."

    # Header with actionable recommendation
    rec_emoji = {
        "ACTIVE SHORT": "🔴🔴", "SELL": "🔴",
        "UNDERWEIGHT": "🟠", "HEDGE": "🟡", "AVOID": "⚫",
    }.get(bc.actionable_recommendation.upper(), "⚫")

    lines = [
        f"# Bear Case: {bc.ticker}",
        "",
        "| | |",
        "|---|---|",
        f"| **Risk Score** | {bc.risk_score}/10 |",
        f"| **Recommendation** | {rec_emoji} {bc.actionable_recommendation} |",
        "",
    ]

    # Short thesis callout
    if bc.short_thesis:
        lines.extend([
            "## Active Short Thesis",
            "",
            f"> **{bc.short_thesis}**",
            "",
        ])

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

    # Technical levels table
    if bc.technical_levels:
        lines.extend([
            "",
            "## Technical Levels",
            "",
            "| Level | Value |",
            "|-------|-------|",
        ])
        for k, v in bc.technical_levels.items():
            label = k.replace("_", " ").title()
            lines.append(f"| {label} | {v} |")

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
        "| 🟢 Tailwinds | 🔴 Headwinds |",
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

    return "\n".join(lines)


def _format_debate(result: CommitteeResult) -> str:
    """Format the adversarial debate transcript."""
    lines = [
        f"# Investment Committee Debate: {result.ticker}",
        "",
    ]

    if result.analyst_rebuttal:
        ar = result.analyst_rebuttal
        lines.extend([
            "## Sector Analyst's Rebuttal (to Bear Case)",
            "",
            "### Challenges to Risk Manager",
            "",
        ])
        for i, point in enumerate(ar.points, 1):
            lines.append(f"{i}. {point}")
        lines.extend(["", "### Concessions", ""])
        for con in ar.concessions:
            lines.append(f"> ✓ {con}")
        if ar.revised_conviction is not None:
            lines.append(f"\n**Revised Conviction:** {ar.revised_conviction}/10")
        lines.append("")

    if result.risk_rebuttal:
        rr = result.risk_rebuttal
        lines.extend([
            "## Risk Manager's Rebuttal (to Bull Case)",
            "",
            "### Challenges to Analyst",
            "",
        ])
        for i, point in enumerate(rr.points, 1):
            lines.append(f"{i}. {point}")
        lines.extend(["", "### Concessions", ""])
        for con in rr.concessions:
            lines.append(f"> ✓ {con}")
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
        return "Bull Case" if snap.score_type == "conviction" else "Bear Case"

    def _what_score_means(snap) -> str:
        """Explain what the score means in context."""
        if snap.agent == "Portfolio Manager":
            return f"PM overall confidence in recommendation"
        if snap.agent == "Macro Analyst":
            return f"How favorable the macro environment is for {result.ticker}"
        if snap.score_type == "conviction":
            return f"How bullish the analyst is on {result.ticker}"
        return f"How much risk the bear sees in {result.ticker}"

    lines = [
        f"# Conviction Evolution: {result.ticker}",
        "",
        "How each agent's confidence shifted across the analysis phases.",
        "",
    ]

    # ── Data table — no "Type" column, uses Stance instead ──
    lines.extend([
        "## Score Timeline",
        "",
        "| Phase | Agent | Stance | Score | Meaning |",
        "|-------|-------|--------|-------|---------|",
    ])
    for snap in timeline:
        lines.append(
            f"| {snap.phase} | {snap.agent} | {_stance(snap)} "
            f"| **{snap.score}/10** | {_what_score_means(snap)} |"
        )

    # ── Visual bar chart — unified bars, color by agent role ──
    lines.extend(["", "## Visual Conviction Map", ""])

    bar_width = 30
    for snap in timeline:
        filled = int((snap.score / 10) * bar_width)
        empty = bar_width - filled
        # Use different chars per agent for visual distinction
        if snap.agent == "Sector Analyst":
            bar_char = "▓"  # Bull
            prefix = "🟢"
        elif snap.agent == "Risk Manager":
            bar_char = "░"  # Bear
            prefix = "🔴"
        elif snap.agent == "Macro Analyst":
            bar_char = "▒"  # Macro
            prefix = "🟣"
        else:
            bar_char = "█"  # PM
            prefix = "🔵"
        bar = bar_char * filled + "·" * empty
        label = f"{snap.agent} ({snap.phase})"
        lines.append(f"```")
        lines.append(f"{prefix} {label:<38} [{bar}] {snap.score}/10")
        lines.append(f"```")

    # ── Narrative ──
    analyst_scores = [s for s in timeline if s.agent == "Sector Analyst"]
    risk_scores = [s for s in timeline if s.agent == "Risk Manager"]
    macro_scores = [s for s in timeline if s.agent == "Macro Analyst"]
    pm_scores = [s for s in timeline if s.agent == "Portfolio Manager"]

    lines.extend(["", "## How Scores Shifted", ""])

    if len(analyst_scores) >= 2:
        initial = analyst_scores[0].score
        revised = analyst_scores[-1].score
        delta = revised - initial
        direction = "more bullish" if delta > 0 else "less bullish" if delta < 0 else "unchanged"
        arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
        lines.append(
            f"- 🟢 **Sector Analyst (Bull):** {initial}/10 → {revised}/10 "
            f"({arrow} {direction}, shifted {abs(delta):.1f} after debate)"
        )
    elif analyst_scores:
        lines.append(f"- 🟢 **Sector Analyst (Bull):** {analyst_scores[0].score}/10")

    if len(risk_scores) >= 2:
        initial = risk_scores[0].score
        revised = risk_scores[-1].score
        delta = revised - initial
        direction = "sees more risk" if delta > 0 else "sees less risk" if delta < 0 else "unchanged"
        arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
        lines.append(
            f"- 🔴 **Risk Manager (Bear):** {initial}/10 → {revised}/10 "
            f"({arrow} {direction}, shifted {abs(delta):.1f} after debate)"
        )
    elif risk_scores:
        lines.append(f"- 🔴 **Risk Manager (Bear):** {risk_scores[0].score}/10")

    if macro_scores:
        lines.append(f"- 🟣 **Macro Analyst:** Favorability {macro_scores[0].score}/10")

    if pm_scores:
        lines.append(f"- 🔵 **Portfolio Manager:** Final conviction {pm_scores[0].score}/10")

    # ── Interpretation ──
    lines.extend(["", "## Interpretation", ""])

    if analyst_scores and risk_scores and pm_scores:
        bull_final = analyst_scores[-1].score
        bear_final = risk_scores[-1].score
        macro_fav = macro_scores[0].score if macro_scores else 5.0
        pm_final = pm_scores[0].score

        # Bull-bear spread
        lines.append(
            f"The bull scored **{bull_final}/10** confidence while the bear scored "
            f"**{bear_final}/10** risk severity."
        )

        if macro_scores:
            if macro_fav >= 7:
                lines.append(
                    f"\nThe macro backdrop is **favorable** ({macro_fav}/10) — "
                    f"the economic environment supports the bull case."
                )
            elif macro_fav >= 4:
                lines.append(
                    f"\nThe macro backdrop is **neutral** ({macro_fav}/10) — "
                    f"neither a strong tailwind nor headwind."
                )
            else:
                lines.append(
                    f"\nThe macro backdrop is **hostile** ({macro_fav}/10) — "
                    f"economic conditions are working against this name."
                )

        if bull_final > (10 - bear_final):
            lines.append(
                "\nBull conviction outweighs the bear's risk assessment — "
                "the analyst sees more upside potential than the risk manager sees downside."
            )
        elif bull_final < (10 - bear_final):
            lines.append(
                "\nBear risk assessment dominates — the risk manager's concerns "
                "outweigh the analyst's bullish conviction."
            )
        else:
            lines.append(
                "\nBull and bear are evenly matched — a true toss-up for the PM to resolve."
            )

        if pm_final >= 7:
            lines.append(
                f"\nPM sided bullish with **{pm_final}/10** conviction — "
                f"high confidence in the positive thesis."
            )
        elif pm_final >= 4:
            lines.append(
                f"\nPM landed at **{pm_final}/10** — moderate conviction, "
                f"acknowledging valid points on both sides."
            )
        else:
            lines.append(
                f"\nPM conviction is low at **{pm_final}/10** — "
                f"the bear case materially impacted the final decision."
            )

    # Legend
    lines.extend([
        "",
        "---",
        "*🟢 = Sector Analyst (higher = more bullish) · "
        "🔴 = Risk Manager (higher = more risk) · "
        "🟣 = Macro Analyst (higher = more favorable macro) · "
        "🔵 = Portfolio Manager (final synthesis)*",
    ])

    return "\n".join(lines)


def _format_status(result: CommitteeResult, messages: list[str], provider_name: str = "") -> str:
    """Format the status/summary view with tables."""
    stats = TraceRenderer.summary_stats(result.traces)

    # Determine model used
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

    lines.extend(["", "## Execution Log", "```"])
    lines.extend(messages)
    lines.append("```")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    """Build the Gradio interface."""

    # Determine default provider display name
    default_display = PROVIDER_TO_DISPLAY.get(
        settings.llm_provider, "Claude (Anthropic)"
    )

    with gr.Blocks(
        title="Multi-Agent Investment Committee",
    ) as app:
        gr.Markdown(
            """
            <div class="header">

            # Multi-Agent Investment Committee

            **Four AI agents** reason, debate, and synthesize investment theses in real time.

            `Sector Analyst (Bull)` ⚔️ `Risk Manager (Bear)` · `Macro Analyst (Top-Down)` → `Portfolio Manager (Decision)`

            </div>
            """,
        )

        # ── Committee Controls: Ticker + Provider + Debate Rounds ──
        with gr.Group():
            with gr.Row():
                ticker_input = gr.Textbox(
                    label="Ticker",
                    placeholder="e.g. NVDA",
                    max_lines=1,
                    scale=1,
                )
                provider_dropdown = gr.Dropdown(
                    choices=list(PROVIDER_DISPLAY.keys()),
                    value=default_display,
                    label="LLM Provider",
                    interactive=True,
                    scale=2,
                )
                debate_rounds_input = gr.Slider(
                    minimum=1,
                    maximum=20,
                    step=1,
                    value=settings.max_debate_rounds,
                    label="Debate Rounds",
                    scale=1,
                )

        # Expert Guidance + Start button
        with gr.Row():
            with gr.Column(scale=5):
                context_input = gr.Textbox(
                    label="Expert Guidance (optional)",
                    placeholder=(
                        "Steer the analysis: e.g. 'Consider pharma sector rotation, "
                        "geopolitical tariff exposure, compare valuation vs. sector median'"
                    ),
                    max_lines=2,
                    info="Injected as domain expertise into all four agents' reasoning",
                )
            with gr.Column(scale=1, min_width=160):
                run_btn = gr.Button(
                    "Start Committee Meeting",
                    variant="primary",
                    size="lg",
                    elem_classes=["start-btn"],
                )

        # Hidden state for report export path
        report_path_state = gr.State(value=None)

        with gr.Tabs():
            with gr.TabItem("Committee Memo"):
                memo_output = gr.Markdown(label="Investment Committee Memo")

            with gr.TabItem("Bull Case"):
                bull_output = gr.Markdown(label="Sector Analyst — Bull Case")

            with gr.TabItem("Bear Case"):
                bear_output = gr.Markdown(label="Risk Manager — Bear Case")

            with gr.TabItem("Macro View"):
                macro_output = gr.Markdown(label="Macro Analyst — Top-Down Environment")

            with gr.TabItem("Debate"):
                debate_output = gr.Markdown(label="Adversarial Debate Transcript")

            with gr.TabItem("Conviction Tracker"):
                conviction_output = gr.Markdown(label="Conviction Evolution")

            with gr.TabItem("Reasoning Trace"):
                trace_output = gr.Markdown(label="Agent Reasoning Traces")

            with gr.TabItem("Session Info"):
                status_output = gr.Markdown(label="Session Summary")

        # Export section
        with gr.Row():
            with gr.Column(scale=1):
                copy_btn = gr.Button("📋 Copy Full Report to Clipboard", size="sm")
                copy_output = gr.Textbox(
                    label="Full Report (select all + copy)",
                    lines=5,
                    max_lines=10,
                    visible=False,
                )
            with gr.Column(scale=1):
                download_btn = gr.Button("📥 Download Report (.md)", size="sm")
                download_output = gr.File(label="Download", visible=False)

        # Wire up the main analysis (show_progress ensures visible progress bar on HF Spaces)
        run_btn.click(
            fn=run_committee_analysis,
            inputs=[ticker_input, context_input, provider_dropdown, debate_rounds_input],
            outputs=[memo_output, bull_output, bear_output, macro_output, debate_output,
                     conviction_output, trace_output, status_output, report_path_state],
            show_progress="full",
        )

        # Copy button — show textbox with full report
        def show_report_for_copy(*tab_outputs):
            """Combine all tab outputs into one copyable text."""
            memo, bull, bear, macro, debate, conviction, trace, status = tab_outputs
            full = "\n\n---\n\n".join([
                s for s in [memo, bull, bear, macro, debate, conviction, status] if s
            ])
            return gr.update(value=full, visible=True)

        copy_btn.click(
            fn=show_report_for_copy,
            inputs=[memo_output, bull_output, bear_output, macro_output, debate_output,
                    conviction_output, trace_output, status_output],
            outputs=[copy_output],
        )

        # Download button — serve the exported file
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
            """
            <div class="disclaimer">

            **Disclaimer:** This is a demonstration of multi-agent AI reasoning architecture.
            It is NOT financial advice. All analyses are AI-generated and should not be used
            for actual investment decisions. Always consult qualified financial professionals.

            **Architecture:** Agents follow a structured think → plan → act → reflect loop
            with adversarial debate. See the Reasoning Trace tab to inspect how each agent reasons.

            **Providers:** Claude (Anthropic) | Gemini (Google) | GPT (OpenAI) | HuggingFace | Ollama (local)

            </div>
            """,
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
        inbrowser=True,  # Auto-open browser
        theme=gr.themes.Soft(),
        css="""
        .header { text-align: center; margin-bottom: 20px; }
        .disclaimer { font-size: 0.85em; color: #888; margin-top: 20px; padding: 10px;
                       border: 1px solid #ddd; border-radius: 5px; }
        .copy-btn { margin-top: 10px; }
        .start-btn button {
            border-radius: 2rem !important;
            padding: 0.75rem 1.5rem !important;
            font-weight: 600 !important;
            min-height: 3rem !important;
            font-size: 0.95rem !important;
        }
        """,
    )
