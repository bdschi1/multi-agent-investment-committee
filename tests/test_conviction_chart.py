"""Tests for orchestrator.conviction_chart — Plotly conviction visualizations."""

from __future__ import annotations

from dataclasses import dataclass

import plotly.graph_objects as go
import pytest

from orchestrator.conviction_chart import (
    _find_inflection,
    _group_by_agent,
    build_conviction_probability,
    build_conviction_trajectory,
)

# ── Lightweight stand-in for ConvictionSnapshot (avoids importing full stack) ──

@dataclass
class _Snap:
    phase: str
    agent: str
    score: float
    score_type: str
    rationale: str = ""


def _make_full_timeline() -> list[_Snap]:
    """Standard 4 initial + 2 debate + 1 PM timeline."""
    return [
        # Initial Analysis (4 agents)
        _Snap("Initial Analysis", "Long Analyst", 7.5, "conviction",
              "Strong revenue growth, expanding TAM"),
        _Snap("Initial Analysis", "Short Analyst", 4.5, "bearish",
              "Overvalued relative to growth rate"),
        _Snap("Initial Analysis", "Risk Manager", 6.0, "bearish",
              "Elevated valuation, margin risk"),
        _Snap("Initial Analysis", "Macro Analyst", 6.5, "favorability",
              "Neutral macro, benign rates"),
        # Debate Round 1
        _Snap("Debate Round 1", "Long Analyst", 8.0, "conviction",
              "Rebutted margin concern with operating leverage thesis"),
        _Snap("Debate Round 1", "Short Analyst", 5.0, "bearish",
              "Conceded operating leverage but valuation still stretched"),
        # PM Decision
        _Snap("PM Decision", "Portfolio Manager", 7.8, "conviction",
              "Overweight — asymmetric upside from product cycle"),
    ]


# ── Tests ──────────────────────────────────────────────────────────────────

class TestEmptyTimeline:
    def test_trajectory_empty(self):
        fig = build_conviction_trajectory([], "AAPL")
        assert isinstance(fig, go.Figure)
        # No real traces — just the annotation
        assert len(fig.data) == 0
        assert any("No conviction data" in a.text for a in fig.layout.annotations)

    def test_probability_empty(self):
        fig = build_conviction_probability([], "AAPL")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0


class TestTrajectoryChart:
    def test_trace_count(self):
        timeline = _make_full_timeline()
        fig = build_conviction_trajectory(timeline, "NVDA")
        # 5 agents → 5 traces
        assert len(fig.data) == 5

    def test_agent_names_in_traces(self):
        timeline = _make_full_timeline()
        fig = build_conviction_trajectory(timeline, "NVDA")
        trace_names = {t.name for t in fig.data}
        assert "Long Analyst" in trace_names
        assert "Short Analyst" in trace_names
        assert "Risk Manager" in trace_names
        assert "Macro Analyst" in trace_names
        assert "Portfolio Manager" in trace_names

    def test_neutral_hline(self):
        timeline = _make_full_timeline()
        fig = build_conviction_trajectory(timeline, "NVDA")
        # plotly_dark template + hline adds a shape
        shapes = fig.layout.shapes
        assert any(s.y0 == 5.0 for s in shapes)

    def test_t_signal_annotation(self):
        """When a memo with t_signal is provided, it should appear in annotations."""
        timeline = _make_full_timeline()

        @dataclass
        class _MockMemo:
            t_signal: float = 0.42

        fig = build_conviction_trajectory(timeline, "NVDA", memo=_MockMemo())
        texts = [a.text for a in fig.layout.annotations]
        assert any("T signal" in t and "+0.42" in t for t in texts)

    def test_no_t_signal_without_memo(self):
        timeline = _make_full_timeline()
        fig = build_conviction_trajectory(timeline, "NVDA", memo=None)
        texts = [a.text for a in fig.layout.annotations]
        assert not any("T signal" in t for t in texts)


class TestProbabilityChart:
    def test_risk_manager_inverted(self):
        """Risk Manager bearish score should be inverted: (10 - score) / 10."""
        timeline = [
            _Snap("Initial Analysis", "Risk Manager", 6.0, "bearish"),
        ]
        fig = build_conviction_probability(timeline, "TEST")
        # Single trace, single y value
        rm_trace = fig.data[0]
        # Expected: (10 - 6.0) / 10 = 0.4
        assert rm_trace.y[0] == pytest.approx(0.4)

    def test_bull_not_inverted(self):
        """Sector Analyst score should be straight division: score / 10."""
        timeline = [
            _Snap("Initial Analysis", "Sector Analyst", 7.5, "conviction"),
        ]
        fig = build_conviction_probability(timeline, "TEST")
        assert fig.data[0].y[0] == pytest.approx(0.75)

    def test_label_indicates_inversion(self):
        timeline = [
            _Snap("Initial Analysis", "Risk Manager", 6.0, "bearish"),
        ]
        fig = build_conviction_probability(timeline, "TEST")
        assert "(inv)" in fig.data[0].name


class TestInflectionDetection:
    def test_largest_delta_selected(self):
        snaps = [
            _Snap("Initial Analysis", "Sector Analyst", 5.0, "conviction"),
            _Snap("Debate Round 1", "Sector Analyst", 5.5, "conviction"),
            _Snap("Debate Round 2", "Sector Analyst", 8.0, "conviction"),
        ]
        idx = _find_inflection(snaps)
        # Largest jump: 5.5 → 8.0 (delta 2.5) at index 2
        assert idx == 2

    def test_no_inflection_when_flat(self):
        snaps = [
            _Snap("Initial Analysis", "Sector Analyst", 5.0, "conviction"),
            _Snap("Debate Round 1", "Sector Analyst", 5.2, "conviction"),
        ]
        idx = _find_inflection(snaps)
        # Delta 0.2 < 0.5 threshold → None
        assert idx is None

    def test_single_snapshot_returns_none(self):
        snaps = [_Snap("Initial Analysis", "Sector Analyst", 7.0, "conviction")]
        assert _find_inflection(snaps) is None

    def test_empty_returns_none(self):
        assert _find_inflection([]) is None


class TestGroupByAgent:
    def test_groups_correctly(self):
        timeline = _make_full_timeline()
        groups = _group_by_agent(timeline)
        assert len(groups["Long Analyst"]) == 2  # Initial + Debate
        assert len(groups["Short Analyst"]) == 2  # Initial + Debate
        assert len(groups["Risk Manager"]) == 1
        assert len(groups["Macro Analyst"]) == 1
        assert len(groups["Portfolio Manager"]) == 1

    def test_phases_sorted_chronologically(self):
        timeline = _make_full_timeline()
        groups = _group_by_agent(timeline)
        sa = groups["Long Analyst"]
        assert sa[0].phase == "Initial Analysis"
        assert sa[1].phase == "Debate Round 1"
