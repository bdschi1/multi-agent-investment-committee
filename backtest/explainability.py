"""
Explainability / attribution — decomposes IC signals into agent contributions.

For each signal, attributes the final T signal to the contributions of:
    - Bull case (sector analyst): conviction and thesis strength
    - Bear case (risk manager): bearish conviction and risk factors
    - Macro view (macro analyst): macro favorability
    - Debate dynamics: how much the debate shifted conviction

This helps identify which agents are driving performance and whether
the committee dynamics are adding value.
"""

from __future__ import annotations

import logging
import statistics
from typing import Optional

from backtest.database import SignalDatabase
from backtest.models import SignalRecord, AttributionResult

logger = logging.getLogger(__name__)


class ExplainabilityAnalyzer:
    """Decomposes IC signals into agent-level contributions."""

    def __init__(self, db: SignalDatabase | None = None):
        self.db = db or SignalDatabase()

    def attribute_signal(self, signal: SignalRecord) -> AttributionResult:
        """
        Decompose a single signal into agent contributions.

        Attribution model:
            T_signal = direction * f(bull_conviction, bear_conviction, macro, debate)

        We estimate relative influence by computing each agent's
        contribution to the net conviction score.
        """
        bull_c = signal.bull_conviction or 5.0
        bear_c = signal.bear_conviction or 5.0
        macro_f = signal.macro_favorability or 5.0

        # Use stored influence values if available (from optimizer or pipeline)
        if signal.bull_influence is not None:
            return AttributionResult(
                ticker=signal.ticker,
                t_signal=signal.t_signal,
                bull_contribution=signal.bull_influence,
                bear_contribution=signal.bear_influence or 0.0,
                macro_contribution=signal.macro_influence or 0.0,
                debate_contribution=signal.debate_shift or 0.0,
                dominant_agent=_dominant_agent(
                    signal.bull_influence,
                    signal.bear_influence or 0.0,
                    signal.macro_influence or 0.0,
                ),
            )

        # Heuristic attribution: decompose conviction into agent contributions
        # Bull pulls toward long (+), Bear pulls toward short (-), Macro is a modifier
        bull_pull = (bull_c - 5.0) / 5.0   # [-1, +1], positive = long pressure
        bear_pull = -(bear_c - 5.0) / 5.0  # [-1, +1], positive = short pressure (high bear = negative)
        macro_pull = (macro_f - 5.0) / 5.0  # [-1, +1], positive = favorable

        # Normalize to sum to ~1 in absolute terms
        total_abs = abs(bull_pull) + abs(bear_pull) + abs(macro_pull)
        if total_abs > 0:
            bull_contrib = bull_pull / total_abs
            bear_contrib = bear_pull / total_abs
            macro_contrib = macro_pull / total_abs
        else:
            bull_contrib = bear_contrib = macro_contrib = 0.0

        # Debate shift: difference between raw conviction inputs and final output
        expected_net = (bull_pull + bear_pull + macro_pull) / 3
        actual_net = signal.t_signal
        debate_shift = actual_net - expected_net if abs(expected_net) > 0.01 else 0.0

        return AttributionResult(
            ticker=signal.ticker,
            t_signal=signal.t_signal,
            bull_contribution=bull_contrib,
            bear_contribution=bear_contrib,
            macro_contribution=macro_contrib,
            debate_contribution=debate_shift,
            dominant_agent=_dominant_agent(bull_contrib, bear_contrib, macro_contrib),
        )

    def attribute_all(
        self,
        ticker: str | None = None,
        limit: int = 100,
    ) -> list[AttributionResult]:
        """Attribute all stored signals."""
        signals = self.db.get_signals(ticker=ticker, limit=limit)
        return [self.attribute_signal(s) for s in signals]

    def compute_agent_statistics(
        self,
        attributions: list[AttributionResult],
    ) -> dict[str, dict[str, float]]:
        """
        Compute aggregate statistics for each agent's contributions.

        Returns dict of agent_name → {avg_contribution, dominance_rate, avg_magnitude}.
        """
        if not attributions:
            return {}

        agents = {
            "Bull (Sector Analyst)": [a.bull_contribution for a in attributions],
            "Bear (Risk Manager)": [a.bear_contribution for a in attributions],
            "Macro Analyst": [a.macro_contribution for a in attributions],
        }

        results = {}
        for name, contribs in agents.items():
            dominance_count = sum(
                1 for a in attributions
                if a.dominant_agent == name.split(" (")[0]
            )
            results[name] = {
                "avg_contribution": statistics.mean(contribs) if contribs else 0,
                "avg_magnitude": statistics.mean(abs(c) for c in contribs) if contribs else 0,
                "dominance_rate": dominance_count / len(attributions) if attributions else 0,
            }

        # Debate statistics
        debate_shifts = [a.debate_contribution for a in attributions]
        results["Debate Dynamics"] = {
            "avg_contribution": statistics.mean(debate_shifts) if debate_shifts else 0,
            "avg_magnitude": statistics.mean(abs(d) for d in debate_shifts) if debate_shifts else 0,
            "dominance_rate": 0.0,
        }

        return results

    def format_report(
        self,
        attributions: list[AttributionResult],
        agent_stats: dict[str, dict[str, float]] | None = None,
    ) -> str:
        """Format explainability analysis as a markdown report."""
        if not attributions:
            return "## Explainability / Attribution\n\nNo signals to analyze."

        if agent_stats is None:
            agent_stats = self.compute_agent_statistics(attributions)

        lines = [
            "## Explainability / Agent Attribution",
            "",
            "### Agent Influence Summary",
            "",
            "| Agent | Avg Contribution | Avg Magnitude | Dominance Rate |",
            "|-------|-----------------|---------------|----------------|",
        ]

        for name, stats in agent_stats.items():
            lines.append(
                f"| {name} | {stats['avg_contribution']:+.3f} | "
                f"{stats['avg_magnitude']:.3f} | "
                f"{stats['dominance_rate']:.0%} |"
            )

        lines.extend(["", "### Per-Signal Attribution (Recent)", ""])
        lines.append(
            "| Ticker | T Signal | Bull | Bear | Macro | Debate | Dominant |"
        )
        lines.append(
            "|--------|----------|------|------|-------|--------|----------|"
        )

        for a in attributions[:20]:  # Show top 20
            lines.append(
                f"| {a.ticker} | {a.t_signal:+.3f} | "
                f"{a.bull_contribution:+.2f} | "
                f"{a.bear_contribution:+.2f} | "
                f"{a.macro_contribution:+.2f} | "
                f"{a.debate_contribution:+.2f} | "
                f"{a.dominant_agent} |"
            )

        return "\n".join(lines)


def _dominant_agent(bull: float, bear: float, macro: float) -> str:
    """Identify which agent had the most influence."""
    agents = {
        "Bull": abs(bull),
        "Bear": abs(bear),
        "Macro": abs(macro),
    }
    return max(agents, key=agents.get)
