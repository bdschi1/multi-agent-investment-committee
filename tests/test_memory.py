"""Tests for agent memory (BM25 retrieval) and post-trade reflection engine."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from backtest.database import SignalDatabase
from backtest.models import ReflectionRecord, SignalRecord
from backtest.reflection import ReflectionEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signal(
    ticker: str = "AAPL",
    position_direction: int = 1,
    bull_conviction: float = 8.0,
    bear_conviction: float = 4.0,
    macro_favorability: float = 7.0,
    conviction: float = 7.5,
    return_20d: float | None = None,
    **kwargs,
) -> SignalRecord:
    return SignalRecord(
        ticker=ticker,
        signal_date=datetime.now(UTC),
        position_direction=position_direction,
        bull_conviction=bull_conviction,
        bear_conviction=bear_conviction,
        macro_favorability=macro_favorability,
        conviction=conviction,
        return_20d=return_20d,
        **kwargs,
    )


def _make_reflection(
    signal_id: int = 1,
    agent_role: str = "sector_analyst",
    ticker: str = "AAPL",
    was_correct: int = 1,
    lesson: str = "Bull thesis was well-calibrated for AAPL earnings beat.",
    what_worked: str = "Catalyst timing was accurate.",
    what_failed: str = "",
    confidence_calibration: str = "High conviction correctly placed.",
) -> ReflectionRecord:
    return ReflectionRecord(
        signal_id=signal_id,
        agent_role=agent_role,
        ticker=ticker,
        was_correct=was_correct,
        lesson=lesson,
        what_worked=what_worked,
        what_failed=what_failed,
        confidence_calibration=confidence_calibration,
    )


# ---------------------------------------------------------------------------
# Reflection Database Tests
# ---------------------------------------------------------------------------

class TestReflectionDatabase:
    def test_store_and_retrieve(self, tmp_path):
        db = SignalDatabase(db_path=tmp_path / "test.db")
        r = _make_reflection()
        rid = db.store_reflection(r)
        assert rid >= 1

        results = db.get_reflections()
        assert len(results) == 1
        assert results[0].agent_role == "sector_analyst"
        assert results[0].ticker == "AAPL"
        db.close()

    def test_filter_by_role(self, tmp_path):
        db = SignalDatabase(db_path=tmp_path / "test.db")
        db.store_reflection(_make_reflection(agent_role="sector_analyst"))
        db.store_reflection(_make_reflection(agent_role="risk_manager"))
        db.store_reflection(_make_reflection(agent_role="sector_analyst"))

        sector = db.get_reflections(agent_role="sector_analyst")
        assert len(sector) == 2
        risk = db.get_reflections(agent_role="risk_manager")
        assert len(risk) == 1
        db.close()

    def test_filter_by_ticker(self, tmp_path):
        db = SignalDatabase(db_path=tmp_path / "test.db")
        db.store_reflection(_make_reflection(ticker="AAPL"))
        db.store_reflection(_make_reflection(ticker="NVDA"))
        db.store_reflection(_make_reflection(ticker="AAPL"))

        aapl = db.get_reflections(ticker="AAPL")
        assert len(aapl) == 2
        nvda = db.get_reflections(ticker="NVDA")
        assert len(nvda) == 1
        db.close()

    def test_filter_by_role_and_ticker(self, tmp_path):
        db = SignalDatabase(db_path=tmp_path / "test.db")
        db.store_reflection(_make_reflection(agent_role="sector_analyst", ticker="AAPL"))
        db.store_reflection(_make_reflection(agent_role="sector_analyst", ticker="NVDA"))
        db.store_reflection(_make_reflection(agent_role="risk_manager", ticker="AAPL"))

        results = db.get_reflections(agent_role="sector_analyst", ticker="AAPL")
        assert len(results) == 1
        db.close()

    def test_get_unreflected_signals(self, tmp_path):
        db = SignalDatabase(db_path=tmp_path / "test.db")

        # Signal with return but no reflection
        sig1 = _make_signal(ticker="AAPL", return_20d=0.05)
        id1 = db.store_signal(sig1)

        # Signal with return AND reflection
        sig2 = _make_signal(ticker="NVDA", return_20d=-0.03)
        id2 = db.store_signal(sig2)
        db.store_reflection(_make_reflection(signal_id=id2, ticker="NVDA"))

        # Signal without return
        sig3 = _make_signal(ticker="MSFT", return_20d=None)
        db.store_signal(sig3)

        unreflected = db.get_unreflected_signals()
        assert len(unreflected) == 1
        assert unreflected[0].ticker == "AAPL"
        db.close()


# ---------------------------------------------------------------------------
# Reflection Engine Tests
# ---------------------------------------------------------------------------

class TestReflectionEngine:
    def test_rule_based_correct(self, tmp_path):
        db = SignalDatabase(db_path=tmp_path / "test.db")
        sig = _make_signal(ticker="AAPL", return_20d=0.08, bull_conviction=8.0, position_direction=1)
        db.store_signal(sig)

        engine = ReflectionEngine(db)
        count = engine.run_reflections()

        # 5 agent roles × 1 signal = 5 reflections
        assert count == 5

        # Check sector_analyst was correct (bull, positive return)
        sector_refs = db.get_reflections(agent_role="sector_analyst")
        assert len(sector_refs) == 1
        assert sector_refs[0].was_correct == 1
        assert "correct" in sector_refs[0].lesson.lower()
        db.close()

    def test_rule_based_incorrect(self, tmp_path):
        db = SignalDatabase(db_path=tmp_path / "test.db")
        sig = _make_signal(ticker="AAPL", return_20d=-0.05, bull_conviction=8.0, position_direction=1)
        db.store_signal(sig)

        engine = ReflectionEngine(db)
        count = engine.run_reflections()
        assert count == 5

        # sector_analyst predicted bull but return was negative
        sector_refs = db.get_reflections(agent_role="sector_analyst")
        assert sector_refs[0].was_correct == 0
        assert "wrong" in sector_refs[0].lesson.lower()
        db.close()

    def test_skips_signals_without_returns(self, tmp_path):
        db = SignalDatabase(db_path=tmp_path / "test.db")
        sig = _make_signal(ticker="AAPL", return_20d=None)
        db.store_signal(sig)

        engine = ReflectionEngine(db)
        count = engine.run_reflections()
        assert count == 0
        db.close()

    def test_skips_already_reflected(self, tmp_path):
        db = SignalDatabase(db_path=tmp_path / "test.db")
        sig = _make_signal(ticker="AAPL", return_20d=0.05)
        sig_id = db.store_signal(sig)

        # First run generates reflections
        engine = ReflectionEngine(db)
        count1 = engine.run_reflections()
        assert count1 == 5

        # Second run skips (already reflected)
        count2 = engine.run_reflections()
        assert count2 == 0
        db.close()

    def test_short_analyst_correct_on_negative_return(self, tmp_path):
        db = SignalDatabase(db_path=tmp_path / "test.db")
        sig = _make_signal(ticker="AAPL", return_20d=-0.10, bear_conviction=9.0)
        db.store_signal(sig)

        engine = ReflectionEngine(db)
        engine.run_reflections()

        # short_analyst predicted bearish (direction=-1), return was -10% → correct
        short_refs = db.get_reflections(agent_role="short_analyst")
        assert short_refs[0].was_correct == 1
        db.close()

    def test_portfolio_manager_uses_position_direction(self, tmp_path):
        db = SignalDatabase(db_path=tmp_path / "test.db")
        sig = _make_signal(ticker="AAPL", return_20d=0.05, position_direction=-1)
        db.store_signal(sig)

        engine = ReflectionEngine(db)
        engine.run_reflections()

        # PM predicted direction=-1 (short), but return was +5% → incorrect
        pm_refs = db.get_reflections(agent_role="portfolio_manager")
        assert pm_refs[0].was_correct == 0
        assert pm_refs[0].predicted_direction == -1
        db.close()

    def test_high_conviction_calibration(self, tmp_path):
        db = SignalDatabase(db_path=tmp_path / "test.db")
        sig = _make_signal(ticker="AAPL", return_20d=-0.05, bull_conviction=9.0)
        db.store_signal(sig)

        engine = ReflectionEngine(db)
        engine.run_reflections()

        sector_refs = db.get_reflections(agent_role="sector_analyst")
        assert "misplaced" in sector_refs[0].confidence_calibration.lower()
        db.close()

    def test_llm_reflection_with_mock(self, tmp_path):
        db = SignalDatabase(db_path=tmp_path / "test.db")
        sig = _make_signal(ticker="AAPL", return_20d=0.05)
        db.store_signal(sig)

        def mock_model(prompt):
            return (
                "LESSON: The earnings catalyst was the key driver.\n"
                "WHAT_WORKED: Timing the catalyst window correctly.\n"
                "WHAT_FAILED: Underestimated volatility around the event.\n"
                "CALIBRATION: Conviction was appropriate for the outcome."
            )

        engine = ReflectionEngine(db, model=mock_model)
        count = engine.run_reflections()
        assert count == 5

        sector_refs = db.get_reflections(agent_role="sector_analyst")
        assert "earnings catalyst" in sector_refs[0].lesson.lower()
        db.close()


# ---------------------------------------------------------------------------
# Agent Memory (BM25 retrieval) Tests
# ---------------------------------------------------------------------------

class TestBuildAgentMemory:
    def test_empty_when_no_reflections(self, tmp_path):
        """Returns empty list when no reflections exist."""
        from orchestrator.memory import build_agent_memory
        db = SignalDatabase(db_path=tmp_path / "empty.db")
        result = build_agent_memory("sector_analyst", "AAPL", db=db)
        assert result == []
        db.close()

    def test_returns_relevant_results(self, tmp_path):
        """Returns reflections ranked by BM25 relevance."""
        db = SignalDatabase(db_path=tmp_path / "mem.db")
        db.store_reflection(_make_reflection(
            ticker="AAPL", agent_role="sector_analyst",
            lesson="AAPL earnings beat expectations, tech sector momentum strong.",
        ))
        db.store_reflection(_make_reflection(
            ticker="XOM", agent_role="sector_analyst",
            lesson="XOM oil price decline hurt energy sector broadly.",
        ))
        db.store_reflection(_make_reflection(
            ticker="MSFT", agent_role="sector_analyst",
            lesson="MSFT cloud growth drove tech sector rally alongside AAPL.",
        ))

        from orchestrator.memory import build_agent_memory
        results = build_agent_memory("sector_analyst", "AAPL", "tech earnings", db=db)
        assert len(results) > 0
        tickers_returned = [r["ticker"] for r in results]
        assert "AAPL" in tickers_returned
        db.close()

    def test_filters_by_agent_role(self, tmp_path):
        """Only returns reflections for the requested agent role."""
        db = SignalDatabase(db_path=tmp_path / "mem.db")
        db.store_reflection(_make_reflection(
            ticker="AAPL", agent_role="sector_analyst",
            lesson="AAPL bull thesis correct on earnings beat.",
        ))
        db.store_reflection(_make_reflection(
            ticker="AAPL", agent_role="risk_manager",
            lesson="AAPL risk assessment missed downside scenario.",
        ))

        from orchestrator.memory import build_agent_memory
        results = build_agent_memory("sector_analyst", "AAPL", db=db)
        assert len(results) == 1
        assert results[0]["ticker"] == "AAPL"
        assert "bull thesis" in results[0]["lesson"].lower()
        db.close()

    def test_respects_top_n(self, tmp_path):
        """Respects the top_n parameter."""
        db = SignalDatabase(db_path=tmp_path / "mem.db")
        for i in range(10):
            db.store_reflection(_make_reflection(
                ticker="AAPL", agent_role="sector_analyst",
                lesson=f"AAPL lesson number {i} about tech earnings and catalysts.",
            ))

        from orchestrator.memory import build_agent_memory
        results = build_agent_memory("sector_analyst", "AAPL", top_n=2, db=db)
        assert len(results) <= 2
        db.close()


# ---------------------------------------------------------------------------
# Session Memory Backward Compatibility
# ---------------------------------------------------------------------------

class TestSessionMemoryBackwardCompat:
    def test_store_and_retrieve(self):
        from orchestrator.memory import clear_session, get_prior_analyses, store_analysis
        clear_session()
        store_analysis("AAPL", {"recommendation": "BUY", "conviction": 8.0})
        priors = get_prior_analyses("AAPL")
        assert len(priors) == 1
        assert priors[0]["recommendation"] == "BUY"
        clear_session()

    def test_clear_session(self):
        from orchestrator.memory import clear_session, get_session_tickers, store_analysis
        clear_session()
        store_analysis("AAPL", {"recommendation": "BUY"})
        assert "AAPL" in get_session_tickers()
        clear_session()
        assert get_session_tickers() == []

    def test_session_summary(self):
        from orchestrator.memory import clear_session, get_session_summary, store_analysis
        clear_session()
        store_analysis("AAPL", {"recommendation": "BUY"})
        store_analysis("AAPL", {"recommendation": "HOLD"})
        store_analysis("NVDA", {"recommendation": "BUY"})
        summary = get_session_summary()
        assert summary["AAPL"] == 2
        assert summary["NVDA"] == 1
        clear_session()
