"""Formatting functions for investment committee results."""
from __future__ import annotations

from datetime import datetime

from app_lib.model_factory import PROVIDER_DISPLAY
from config.settings import LLMProvider, settings
from optimizer.strategies import STRATEGY_DISPLAY_NAMES
from orchestrator.committee import CommitteeResult
from orchestrator.reasoning_trace import TraceRenderer



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
    """Format portfolio optimizer output as markdown tables."""
    if not optimization_result or not getattr(optimization_result, 'success', False):
        error_msg = getattr(optimization_result, 'error_message', '') if optimization_result else ''
        if error_msg:
            return (
                "\n## Computed Portfolio Analytics\n\n"
                f"*Optimizer did not produce results: {error_msg}*\n"
            )
        return ""

    opt = optimization_result
    method = getattr(opt, 'optimizer_method', 'black_litterman')
    display_name = getattr(opt, 'optimizer_display_name', 'Black-Litterman')
    is_bl = method == "black_litterman"

    lines = [
        "",
        f"## Computed Portfolio Analytics ({display_name})",
        "",
    ]

    if is_bl:
        lines.append(
            "*Actual computed values from pypfopt Black-Litterman model — "
            "compare with LLM heuristics above.*"
        )
    else:
        lines.append(
            f"*Computed portfolio weights and risk metrics using {display_name}.*"
        )
    lines.append("")

    # Side-by-side comparison table (BL only — compares heuristic vs BL posterior)
    if is_bl and memo and opt.bl_expected_return is not None:
        idio_ret_h = getattr(memo, 'idio_return_estimate', '') or '—'
        sharpe_h = getattr(memo, 'sharpe_estimate', '') or '—'
        sortino_h = getattr(memo, 'sortino_estimate', '') or '—'

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

    # Model output table — strategy-specific
    if is_bl and opt.bl_expected_return is not None:
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
    else:
        lines.extend([
            f"### {display_name} Output",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| **Optimal Weight ({opt.ticker})** | {opt.optimal_weight_pct} |",
            f"| **Sharpe (computed)** | {opt.computed_sharpe:.2f} |",
            f"| **Sortino (computed)** | {opt.computed_sortino:.2f} |",
            f"| **Portfolio Vol** | {opt.portfolio_vol * 100:.2f}% |",
            f"| **Annualized Vol** | {opt.annualized_vol * 100:.2f}% |",
            f"| **Downside Vol** | {opt.downside_vol * 100:.2f}% |",
            f"| **Covariance Method** | {opt.covariance_method} |",
            f"| **Lookback** | {opt.lookback_days} trading days |",
            "",
        ])

    # Factor exposures (all strategies)
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

    # Risk contribution (MCTR) — top 5 (all strategies)
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

    # Universe weights (non-zero, all strategies)
    if opt.universe_weights:
        lines.extend([
            f"### Optimal Universe Weights ({display_name})",
            "",
            "| Ticker | Weight |",
            "|--------|--------|",
        ])
        for t, w in sorted(opt.universe_weights.items(), key=lambda x: -x[1]):
            lines.append(f"| {t} | {w:.1%} |")
        lines.append("")

    return "\n".join(lines)


def _format_ensemble_section(ensemble_result) -> str:
    """Format ensemble optimizer output as markdown tables."""
    if not ensemble_result or not getattr(ensemble_result, 'success', False):
        error_msg = getattr(ensemble_result, 'error_message', '') if ensemble_result else ''
        if error_msg:
            return (
                "\n## Ensemble Portfolio Analytics\n\n"
                f"*Ensemble optimizer did not produce results: {error_msg}*\n"
            )
        return ""

    ens = ensemble_result
    blend_desc = ", ".join(
        f"{STRATEGY_DISPLAY_NAMES.get(k, k)} {v:.0%}"
        for k, v in ens.ensemble_weights_used.items()
        if v > 0
    )

    lines = [
        "",
        "## Ensemble Portfolio Analytics (All Strategies)",
        "",
        f"*Runs all strategies on shared universe and covariance matrix. "
        f"Blend: {blend_desc}.*",
        "",
    ]

    # Strategy comparison table
    if ens.strategy_comparisons:
        lines.extend([
            "### Strategy Comparison",
            "",
            "| Strategy | Role | Target Wt | Port Vol | Sharpe | Sortino | Max Wt | HHI |",
            "|----------|------|-----------|----------|--------|---------|--------|-----|",
        ])
        for sc in ens.strategy_comparisons:
            lines.append(
                f"| **{sc.strategy_name}** | {sc.role} | "
                f"{sc.target_weight:.1%} | {sc.portfolio_vol * 100:.1f}% | "
                f"{sc.sharpe:.2f} | {sc.sortino:.2f} | "
                f"{sc.max_single_weight:.1%} | {sc.hhi:.3f} |"
            )
        lines.append("")

    # Blended allocation
    lines.extend([
        "### Blended Allocation",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| **Blended Target Weight ({ens.ticker})** | {ens.blended_target_weight:.1%} |",
        f"| **Blended Portfolio Vol** | {ens.blended_portfolio_vol * 100:.1f}% |",
        f"| **Blended Sharpe** | {ens.blended_sharpe:.2f} |",
        f"| **Blended Sortino** | {ens.blended_sortino:.2f} |",
        f"| **Blended HHI** | {ens.blended_hhi:.3f} |",
        "",
    ])

    # Weight consensus matrix
    if ens.consensus:
        # Build header from available strategy keys
        strat_keys = []
        strat_short = {
            "black_litterman": "BL",
            "hrp": "HRP",
            "mean_variance": "MV",
            "min_variance": "MinVar",
            "risk_parity": "RP",
            "equal_weight": "EW",
        }
        for tc in ens.consensus:
            for k in tc.weights_by_strategy:
                if k not in strat_keys:
                    strat_keys.append(k)

        header_names = [strat_short.get(k, k) for k in strat_keys]
        header = "| Ticker | " + " | ".join(header_names) + " | Mean | Std |"
        sep = "|--------" + "|------" * len(strat_keys) + "|------|-----|"

        lines.extend([
            "### Weight Consensus Matrix",
            "",
            header,
            sep,
        ])
        for tc in ens.consensus:
            wt_cells = " | ".join(
                f"{tc.weights_by_strategy.get(k, 0.0):.1%}" for k in strat_keys
            )
            marker = " *" if tc.is_target else ""
            lines.append(
                f"| **{tc.ticker}**{marker} | {wt_cells} | "
                f"{tc.mean_weight:.1%} | {tc.std_weight * 100:.1f}pp |"
            )
        lines.append("")

    # Divergence flags
    if ens.divergence_flags:
        lines.extend(["### Divergence Flags", ""])
        for df in ens.divergence_flags:
            icon = "+" if df.flag_type == "high_agreement" else "!"
            lines.append(f"- **[{icon}] {df.ticker}:** {df.description}")
        lines.append("")

    # Layered interpretation
    if ens.layered_narrative:
        lines.extend([
            "### Layered Interpretation",
            "",
            ens.layered_narrative,
            "",
        ])

    # Blended MCTR — top 5
    if ens.blended_risk_contributions:
        lines.extend([
            "### Blended Risk Contribution (MCTR) — Top 5",
            "",
            "| Ticker | Weight | Marginal CTR | % of Portfolio Risk |",
            "|--------|--------|-------------|---------------------|",
        ])
        for rc in ens.blended_risk_contributions[:5]:
            lines.append(
                f"| **{rc.ticker}** | {rc.weight:.1%} | {rc.marginal_ctr:.4f} | "
                f"{rc.pct_contribution:.1%} |"
            )
        lines.append("")

    # Failed strategies
    if ens.failed_strategies:
        lines.extend([
            "### Failed Strategies",
            "",
            f"*The following strategies did not produce results: "
            f"{', '.join(ens.failed_strategies)}.*",
            "",
        ])

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

    # Optimizer output (computed, not heuristic)
    opt_result = getattr(result, 'optimization_result', None)
    if opt_result:
        from optimizer.models import EnsembleResult
        if isinstance(opt_result, EnsembleResult):
            opt_section = _format_ensemble_section(opt_result)
        else:
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


def _format_status(
    result: CommitteeResult,
    messages: list[str],
    provider_name: str = "",
    mode: str = "Full Auto",
    num_source_files: int = 0,
    expert_guidance: bool = False,
    pm_guidance: bool = False,
) -> str:
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

    # Format duration as Xm Ys
    total_s = stats['total_duration_s']
    mins = int(total_s // 60)
    secs = round(total_s % 60, 1)
    duration_str = f"{mins}m {secs}s" if mins else f"{secs}s"

    # Injection flags
    injections = []
    if expert_guidance:
        injections.append("Expert")
    if pm_guidance:
        injections.append("PM")
    injection_str = ", ".join(injections) if injections else "None"

    lines = [
        f"# Session Summary: {result.ticker}",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| **LLM Provider** | {provider_name} |",
        f"| **Model** | `{model_name}` |",
        f"| **Total Duration** | {duration_str} |",
        f"| **Mode** | {mode} |",
        f"| **Source Files** | {num_source_files} |",
        f"| **Injected Guidance** | {injection_str} |",
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

