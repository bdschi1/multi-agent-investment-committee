"""
Plotly conviction-trajectory charts for the Gradio UI.

Two public functions:
    build_conviction_trajectory  — Likert 0-10 view
    build_conviction_probability — Probability 0-1 view (Risk Manager inverted)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go

if TYPE_CHECKING:
    from orchestrator.committee import CommitteeMemo, ConvictionSnapshot

# ── Agent colours ──────────────────────────────────────────────────────────
AGENT_COLORS = {
    "Sector Analyst": "#2ecc71",
    "Risk Manager": "#e74c3c",
    "Macro Analyst": "#9b59b6",
    "Portfolio Manager": "#3498db",
}

_PHASE_ORDER = [
    "Initial Analysis",
    "Post-Debate",
    "Debate Round 1",
    "Debate Round 2",
    "Debate Round 3",
    "PM Decision",
]


def _phase_sort_key(phase: str) -> int:
    """Return an integer sort key so phases appear in chronological order."""
    try:
        return _PHASE_ORDER.index(phase)
    except ValueError:
        # Unknown phase — put it between debate and PM
        return 4


def _group_by_agent(
    timeline: list[ConvictionSnapshot],
) -> dict[str, list[ConvictionSnapshot]]:
    """Group snapshots by agent name, preserving chronological order."""
    groups: dict[str, list[ConvictionSnapshot]] = {}
    for snap in timeline:
        groups.setdefault(snap.agent, []).append(snap)
    # Sort each agent's snapshots by phase order
    for agent in groups:
        groups[agent].sort(key=lambda s: _phase_sort_key(s.phase))
    return groups


def _find_inflection(snapshots: list[ConvictionSnapshot]) -> int | None:
    """Return the index of the largest absolute delta between consecutive snapshots."""
    if len(snapshots) < 2:
        return None
    max_delta = 0.0
    max_idx = None
    for i in range(1, len(snapshots)):
        delta = abs(snapshots[i].score - snapshots[i - 1].score)
        if delta > max_delta:
            max_delta = delta
            max_idx = i
    # Only annotate if there's a meaningful shift (> 0.5)
    if max_delta > 0.5 and max_idx is not None:
        return max_idx
    return None


def _empty_figure(title: str) -> go.Figure:
    """Return a placeholder figure when there's no data."""
    fig = go.Figure()
    fig.add_annotation(
        text="No conviction data captured",
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16, color="#888"),
    )
    fig.update_layout(
        title=title,
        height=450,
        template="plotly_dark",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


# ── Public API ─────────────────────────────────────────────────────────────

def build_conviction_trajectory(
    timeline: list[ConvictionSnapshot],
    ticker: str,
    memo: CommitteeMemo | None = None,
) -> go.Figure:
    """
    Build an interactive Plotly line chart of conviction scores (0-10 Likert).

    Parameters
    ----------
    timeline : list[ConvictionSnapshot]
        The conviction_timeline from CommitteeResult.
    ticker : str
        Ticker symbol for the chart title.
    memo : CommitteeMemo | None
        If provided, the T signal is annotated in the top-right corner.

    Returns
    -------
    go.Figure
    """
    if not timeline:
        return _empty_figure(f"{ticker} — Conviction Trajectory")

    groups = _group_by_agent(timeline)

    fig = go.Figure()

    # Determine the global x-axis phase order from all snapshots
    seen_phases: list[str] = []
    for snap in sorted(timeline, key=lambda s: _phase_sort_key(s.phase)):
        if snap.phase not in seen_phases:
            seen_phases.append(snap.phase)

    # Track agents with fill-between for spread visualisation
    bull_trace_added = False
    bear_trace_added = False

    for agent, snapshots in groups.items():
        color = AGENT_COLORS.get(agent, "#95a5a6")
        phases = [s.phase for s in snapshots]
        scores = [s.score for s in snapshots]

        show_fill = None
        # Fill between Sector Analyst (bull) and Risk Manager (bear)
        if agent == "Risk Manager" and bull_trace_added:
            show_fill = "tonexty"

        fig.add_trace(go.Scatter(
            x=phases,
            y=scores,
            mode="lines+markers",
            name=agent,
            line=dict(color=color, width=2.5),
            marker=dict(size=8, color=color),
            fill=show_fill,
            fillcolor="rgba(200, 200, 200, 0.08)" if show_fill else None,
            hovertemplate=(
                "<b>%{x}</b><br>"
                f"{agent}: " + "%{y:.1f}/10<br>"
                "<extra></extra>"
            ),
        ))

        if agent == "Sector Analyst":
            bull_trace_added = True
        if agent == "Risk Manager":
            bear_trace_added = True

        # Annotate the biggest inflection point for this agent
        inflection_idx = _find_inflection(snapshots)
        if inflection_idx is not None:
            snap = snapshots[inflection_idx]
            rationale = (snap.rationale or "")[:60]
            if len(snap.rationale or "") > 60:
                rationale += "..."
            delta = snap.score - snapshots[inflection_idx - 1].score
            sign = "+" if delta >= 0 else ""
            fig.add_annotation(
                x=snap.phase,
                y=snap.score,
                text=f"{sign}{delta:.1f}: {rationale}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowcolor=color,
                font=dict(size=9, color=color),
                bgcolor="rgba(0,0,0,0.7)",
                bordercolor=color,
                borderwidth=1,
                borderpad=3,
                ax=0,
                ay=-40,
            )

    # Neutral reference line
    fig.add_hline(
        y=5.0,
        line_dash="dot",
        line_color="rgba(255,255,255,0.3)",
        annotation_text="Neutral (5)",
        annotation_position="bottom left",
        annotation_font_color="rgba(255,255,255,0.4)",
    )

    # T signal annotation
    if memo:
        t_signal = getattr(memo, "t_signal", None)
        if t_signal is not None:
            fig.add_annotation(
                text=f"T signal: {t_signal:+.2f}",
                xref="paper", yref="paper",
                x=0.98, y=0.98,
                showarrow=False,
                font=dict(size=12, color="#3498db"),
                bgcolor="rgba(0,0,0,0.6)",
                bordercolor="#3498db",
                borderwidth=1,
                borderpad=5,
                xanchor="right",
                yanchor="top",
            )

    fig.update_layout(
        title=f"{ticker} — Conviction Trajectory",
        xaxis_title="Phase",
        yaxis_title="Conviction Score (0-10)",
        yaxis=dict(range=[-0.5, 10.5], dtick=2),
        height=450,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(t=80, b=40),
    )

    return fig


def build_conviction_probability(
    timeline: list[ConvictionSnapshot],
    ticker: str,
    memo: CommitteeMemo | None = None,
) -> go.Figure:
    """
    Build an interactive Plotly chart with scores normalised to 0-1 probability.

    Risk Manager scores are inverted: (10 - bearish_conviction) / 10
    so that higher = more favorable across all agents.

    Parameters
    ----------
    timeline : list[ConvictionSnapshot]
        The conviction_timeline from CommitteeResult.
    ticker : str
        Ticker symbol for the chart title.
    memo : CommitteeMemo | None
        If provided, the T signal is annotated in the top-right corner.

    Returns
    -------
    go.Figure
    """
    if not timeline:
        return _empty_figure(f"{ticker} — Conviction Probability")

    groups = _group_by_agent(timeline)

    fig = go.Figure()

    bull_trace_added = False

    for agent, snapshots in groups.items():
        color = AGENT_COLORS.get(agent, "#95a5a6")
        phases = [s.phase for s in snapshots]

        # Normalise: Risk Manager bearish inverted so higher = more favorable
        if agent == "Risk Manager":
            scores = [(10 - s.score) / 10 for s in snapshots]
        else:
            scores = [s.score / 10 for s in snapshots]

        show_fill = None
        if agent == "Risk Manager" and bull_trace_added:
            show_fill = "tonexty"

        label = f"{agent} (inv)" if agent == "Risk Manager" else agent

        fig.add_trace(go.Scatter(
            x=phases,
            y=scores,
            mode="lines+markers",
            name=label,
            line=dict(color=color, width=2.5),
            marker=dict(size=8, color=color),
            fill=show_fill,
            fillcolor="rgba(200, 200, 200, 0.08)" if show_fill else None,
            hovertemplate=(
                "<b>%{x}</b><br>"
                f"{label}: " + "%{y:.2f}<br>"
                "<extra></extra>"
            ),
        ))

        if agent == "Sector Analyst":
            bull_trace_added = True

        # Annotate inflection (using raw scores for delta detection)
        inflection_idx = _find_inflection(snapshots)
        if inflection_idx is not None:
            snap = snapshots[inflection_idx]
            rationale = (snap.rationale or "")[:60]
            if len(snap.rationale or "") > 60:
                rationale += "..."
            prob = scores[inflection_idx]
            prev_prob = scores[inflection_idx - 1]
            delta = prob - prev_prob
            sign = "+" if delta >= 0 else ""
            fig.add_annotation(
                x=snap.phase,
                y=prob,
                text=f"{sign}{delta:.2f}: {rationale}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowcolor=color,
                font=dict(size=9, color=color),
                bgcolor="rgba(0,0,0,0.7)",
                bordercolor=color,
                borderwidth=1,
                borderpad=3,
                ax=0,
                ay=-40,
            )

    # 0.5 neutral line
    fig.add_hline(
        y=0.5,
        line_dash="dot",
        line_color="rgba(255,255,255,0.3)",
        annotation_text="Neutral (0.5)",
        annotation_position="bottom left",
        annotation_font_color="rgba(255,255,255,0.4)",
    )

    # T signal
    if memo:
        t_signal = getattr(memo, "t_signal", None)
        if t_signal is not None:
            fig.add_annotation(
                text=f"T signal: {t_signal:+.2f}",
                xref="paper", yref="paper",
                x=0.98, y=0.98,
                showarrow=False,
                font=dict(size=12, color="#3498db"),
                bgcolor="rgba(0,0,0,0.6)",
                bordercolor="#3498db",
                borderwidth=1,
                borderpad=5,
                xanchor="right",
                yanchor="top",
            )

    fig.update_layout(
        title=f"{ticker} — Conviction Probability",
        xaxis_title="Phase",
        yaxis_title="Probability",
        yaxis=dict(range=[-0.05, 1.05], dtick=0.2),
        height=450,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(t=80, b=40),
    )

    return fig
