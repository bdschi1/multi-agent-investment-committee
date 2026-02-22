"""Tests for backtest.persist — shared signal persistence utility."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import pytest

from backtest.database import SignalDatabase
from backtest.persist import persist_signal
from orchestrator.committee import CommitteeResult


# ---------------------------------------------------------------------------
# Minimal mock objects for CommitteeMemo, BullCase, BearCase, MacroView
# ---------------------------------------------------------------------------

@dataclass
class _MockMemo:
    ticker: str = "AAPL"
    recommendation: str = "BUY"
    t_signal: float = 0.75
    conviction: float = 8.0
    position_direction: int = 1
    raw_confidence: float = 0.8


@dataclass
class _MockBull:
    conviction_score: float = 8.0


@dataclass
class _MockBear:
    bearish_conviction: float = 4.0


@dataclass
class _MockMacro:
    macro_favorability: float = 7.0


@dataclass
class _MockOptResult:
    success: bool = True
    optimal_weight: float = 0.12
    computed_sharpe: float = 1.5
    computed_sortino: float = 2.1


def _make_result(**overrides) -> CommitteeResult:
    defaults = dict(
        ticker="AAPL",
        committee_memo=_MockMemo(),
        bull_case=_MockBull(),
        bear_case=_MockBear(),
        macro_view=_MockMacro(),
        total_duration_ms=5000.0,
        total_tokens=1200,
    )
    defaults.update(overrides)
    return CommitteeResult(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPersistSignal:
    def test_persists_valid_result(self, tmp_path):
        db = SignalDatabase(db_path=tmp_path / "test.db")
        result = _make_result()
        signal_id = persist_signal(result, "anthropic", "claude-sonnet", db=db)
        assert signal_id is not None
        assert signal_id >= 1

        signals = db.get_signals(ticker="AAPL")
        assert len(signals) == 1
        assert signals[0].recommendation == "BUY"
        assert signals[0].t_signal == 0.75
        assert signals[0].conviction == 8.0
        db.close()

    def test_returns_signal_id(self, tmp_path):
        db = SignalDatabase(db_path=tmp_path / "test.db")
        result = _make_result()
        signal_id = persist_signal(result, "ollama", "llama3.1:8b", db=db)
        assert isinstance(signal_id, int)
        assert signal_id >= 1
        db.close()

    def test_handles_missing_memo(self, tmp_path):
        db = SignalDatabase(db_path=tmp_path / "test.db")
        result = _make_result(committee_memo=None)
        signal_id = persist_signal(result, "anthropic", "claude-sonnet", db=db)
        assert signal_id is None
        assert db.get_signals() == []
        db.close()

    def test_includes_optimizer_results(self, tmp_path):
        db = SignalDatabase(db_path=tmp_path / "test.db")
        result = _make_result(optimization_result=_MockOptResult())
        signal_id = persist_signal(result, "anthropic", "claude-sonnet", db=db)

        signals = db.get_signals(ticker="AAPL")
        assert len(signals) == 1
        assert signals[0].bl_optimal_weight == 0.12
        assert signals[0].bl_sharpe == 1.5
        assert signals[0].bl_sortino == 2.1
        db.close()

    def test_reuses_db_connection(self, tmp_path):
        db = SignalDatabase(db_path=tmp_path / "test.db")

        r1 = _make_result(ticker="AAPL")
        r2 = _make_result(ticker="NVDA")
        r2.committee_memo.ticker = "NVDA"

        id1 = persist_signal(r1, "anthropic", "claude", db=db)
        id2 = persist_signal(r2, "anthropic", "claude", db=db)

        assert id1 is not None and id2 is not None
        assert id1 != id2
        assert len(db.get_signals()) == 2
        db.close()

    def test_creates_own_db_if_none_provided(self, tmp_path, monkeypatch):
        """When db=None, persist_signal creates and closes its own connection."""
        result = _make_result()
        # Monkeypatch SignalDatabase default path to use tmp_path
        monkeypatch.setattr(
            "backtest.persist.SignalDatabase",
            lambda db_path=None: SignalDatabase(db_path=tmp_path / "auto.db"),
        )
        signal_id = persist_signal(result, "ollama", "llama3.1:8b")
        assert signal_id is not None
