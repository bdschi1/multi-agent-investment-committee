"""
Reasoning trace rendering utilities.

Converts structured reasoning traces into human-readable formats
for the Gradio UI, markdown export, and debugging.
"""

from __future__ import annotations

from agents.base import ReasoningTrace, StepType

STEP_ICONS = {
    StepType.THINK: "🧠",
    StepType.PLAN: "📋",
    StepType.ACT: "⚡",
    StepType.REFLECT: "🔍",
    StepType.TOOL_CALL: "🔧",
    StepType.REBUTTAL: "⚔️",
}

ROLE_LABELS = {
    "sector_analyst": "Long Analyst (Bull)",
    "short_analyst": "Short Analyst",
    "risk_manager": "Risk Manager (Sizing)",
    "macro_analyst": "Macro Analyst (Top-Down)",
    "portfolio_manager": "Portfolio Manager (PM)",
}


class TraceRenderer:
    """Renders reasoning traces into various display formats."""

    @staticmethod
    def to_markdown(traces: dict[str, ReasoningTrace]) -> str:
        """Render all traces as a markdown document."""
        lines = ["# Reasoning Trace\n"]

        for agent_key, trace in traces.items():
            label = ROLE_LABELS.get(agent_key, agent_key)
            lines.append(f"## {label}")
            lines.append(f"**Ticker:** {trace.ticker}")
            lines.append(f"**Duration:** {trace.total_duration_ms/1000:.1f}s")
            lines.append(f"**Tokens:** {trace.total_tokens}")
            lines.append("")

            for i, step in enumerate(trace.steps, 1):
                icon = STEP_ICONS.get(step.step_type, "•")
                lines.append(f"### {icon} Step {i}: {step.step_type.value.upper()}")
                if step.duration_ms:
                    lines.append(f"*({step.duration_ms/1000:.1f}s)*\n")
                # Truncate very long content for readability
                content = step.content
                if len(content) > 2000:
                    content = content[:2000] + "\n\n*[truncated for display]*"
                lines.append(content)
                lines.append("")

            lines.append("---\n")

        return "\n".join(lines)

    @staticmethod
    def to_gradio_accordion(traces: dict[str, ReasoningTrace]) -> str:
        """
        Render traces as markdown with headers suitable for Gradio's
        markdown rendering (which supports collapsible sections via <details>).
        """
        lines = []

        for agent_key, trace in traces.items():
            label = ROLE_LABELS.get(agent_key, agent_key)
            lines.append(f"<details><summary><strong>{label}</strong> — "
                         f"{trace.total_duration_ms/1000:.1f}s</summary>\n")

            for i, step in enumerate(trace.steps, 1):
                icon = STEP_ICONS.get(step.step_type, "•")
                step_label = step.step_type.value.upper()
                duration = f" ({step.duration_ms/1000:.1f}s)" if step.duration_ms else ""

                lines.append(f"<details><summary>{icon} Step {i}: "
                             f"{step_label}{duration}</summary>\n")
                content = step.content
                if len(content) > 1500:
                    content = content[:1500] + "\n\n*[truncated]*"
                lines.append(f"```\n{content}\n```\n")
                lines.append("</details>\n")

            lines.append("</details>\n")
            lines.append("---\n")

        return "\n".join(lines)

    @staticmethod
    def summary_stats(traces: dict[str, ReasoningTrace]) -> dict:
        """Return summary statistics for display."""
        total_steps = sum(len(t.steps) for t in traces.values())
        total_tokens = sum(t.total_tokens for t in traces.values())
        total_duration = sum(t.total_duration_ms for t in traces.values())

        return {
            "total_agents": len(traces),
            "total_steps": total_steps,
            "total_tokens": total_tokens,
            "total_duration_s": round(total_duration / 1000, 1),
            "per_agent": {
                ROLE_LABELS.get(k, k): {
                    "steps": len(t.steps),
                    "tokens": t.total_tokens,
                    "duration_s": round(t.total_duration_ms / 1000, 1),
                }
                for k, t in traces.items()
            },
        }
