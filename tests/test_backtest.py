"""
Tests for the backtest, calibration, and analytics modules.

All tests use in-memory databases and mock data — no API keys needed.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from backtest.alpha_decay import AlphaDecayAnalyzer
from backtest.benchmark import BenchmarkAnalyzer
from backtest.calibration import CalibrationAnalyzer
from backtest.database import SignalDatabase
from backtest.explainability import ExplainabilityAnalyzer
from backtest.models import (
    AlphaDecayPoint,
    AttributionResult,
    BacktestResult,
    BenchmarkComparison,
    CalibrationBucket,
    PortfolioSnapshot,
    SignalRecord,
)
from backtest.portfolio import MultiAssetPortfolio
from backtest.runner import BacktestRunner, _rank_ic

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path):
    """Create a temporary SignalDatabase."""
    return SignalDatabase(db_path=tmp_path / "test_signals.db")


@pytest.fixture
def populated_db(db):
    """Database pre-populated with sample signals."""
    signals = [
        # NVDA: strong long, correct
        SignalRecord(
            ticker="NVDA", signal_date=datetime(2025, 1, 15, tzinfo=UTC),
            provider="anthropic", model_name="claude-sonnet",
            recommendation="BUY", t_signal=0.72, conviction=7.2,
            position_direction=1, raw_confidence=0.72,
            bull_conviction=8.2, bear_conviction=5.5, macro_favorability=6.5,
            return_1d=0.02, return_5d=0.05, return_10d=0.08,
            return_20d=0.12, return_60d=0.25,
            price_at_signal=140.0,
        ),
        # NVDA: weak long, wrong
        SignalRecord(
            ticker="NVDA", signal_date=datetime(2025, 2, 15, tzinfo=UTC),
            provider="ollama", model_name="llama3.1:8b",
            recommendation="HOLD", t_signal=0.1, conviction=5.2,
            position_direction=1, raw_confidence=0.52,
            bull_conviction=5.5, bear_conviction=5.0, macro_favorability=5.0,
            return_1d=-0.01, return_5d=-0.03, return_10d=-0.04,
            return_20d=-0.06, return_60d=-0.08,
            price_at_signal=155.0,
        ),
        # COST: short, correct
        SignalRecord(
            ticker="COST", signal_date=datetime(2025, 1, 20, tzinfo=UTC),
            provider="anthropic", model_name="claude-sonnet",
            recommendation="SELL", t_signal=-0.65, conviction=8.0,
            position_direction=-1, raw_confidence=0.65,
            bull_conviction=4.0, bear_conviction=8.5, macro_favorability=4.2,
            return_1d=-0.015, return_5d=-0.04, return_10d=-0.06,
            return_20d=-0.09, return_60d=-0.15,
            price_at_signal=920.0,
        ),
        # AAPL: neutral (no direction), should be excluded from directional analysis
        SignalRecord(
            ticker="AAPL", signal_date=datetime(2025, 1, 25, tzinfo=UTC),
            provider="anthropic", model_name="claude-sonnet",
            recommendation="HOLD", t_signal=0.0, conviction=5.0,
            position_direction=0, raw_confidence=0.5,
            bull_conviction=5.0, bear_conviction=5.0, macro_favorability=5.0,
            return_1d=0.005, return_5d=0.01, return_10d=0.02,
            return_20d=0.03, return_60d=0.05,
            price_at_signal=185.0,
        ),
        # META: strong long, correct (high conviction)
        SignalRecord(
            ticker="META", signal_date=datetime(2025, 2, 1, tzinfo=UTC),
            provider="anthropic", model_name="claude-sonnet",
            recommendation="STRONG BUY", t_signal=0.85, conviction=9.0,
            position_direction=1, raw_confidence=0.85,
            bull_conviction=9.0, bear_conviction=3.0, macro_favorability=7.5,
            return_1d=0.03, return_5d=0.07, return_10d=0.11,
            return_20d=0.18, return_60d=0.30,
            price_at_signal=480.0,
        ),
    ]

    for sig in signals:
        db.store_signal(sig)

    return db


# ---------------------------------------------------------------------------
# Database tests
# ---------------------------------------------------------------------------

class TestSignalDatabase:
    def test_store_and_retrieve(self, db):
        sig = SignalRecord(
            ticker="TSLA", signal_date=datetime(2025, 1, 1, tzinfo=UTC),
            t_signal=0.5, conviction=7.0, position_direction=1,
        )
        sig_id = db.store_signal(sig)
        assert sig_id is not None
        assert sig_id > 0

        retrieved = db.get_signal_by_id(sig_id)
        assert retrieved is not None
        assert retrieved.ticker == "TSLA"
        assert retrieved.t_signal == 0.5

    def test_count_signals(self, populated_db):
        assert populated_db.count_signals() == 5

    def test_get_all_tickers(self, populated_db):
        tickers = populated_db.get_all_tickers()
        assert set(tickers) == {"AAPL", "COST", "META", "NVDA"}

    def test_update_returns(self, db):
        sig = SignalRecord(
            ticker="MSFT", signal_date=datetime(2025, 1, 1, tzinfo=UTC),
        )
        sig_id = db.store_signal(sig)
        db.update_returns(sig_id, return_1d=0.02, return_5d=0.05)

        updated = db.get_signal_by_id(sig_id)
        assert updated.return_1d == 0.02
        assert updated.return_5d == 0.05

    def test_get_signals_by_ticker(self, populated_db):
        nvda_signals = populated_db.get_signals(ticker="NVDA")
        assert len(nvda_signals) == 2
        for s in nvda_signals:
            assert s.ticker == "NVDA"

    def test_store_snapshot(self, db):
        snap = PortfolioSnapshot(
            tickers=["NVDA", "META"],
            weights={"NVDA": 0.6, "META": 0.4},
            gross_exposure=1.0,
            net_exposure=1.0,
        )
        snap_id = db.store_snapshot(snap)
        assert snap_id > 0

        snapshots = db.get_snapshots()
        assert len(snapshots) == 1
        assert set(snapshots[0].tickers) == {"NVDA", "META"}


# ---------------------------------------------------------------------------
# Backtest runner tests
# ---------------------------------------------------------------------------

class TestBacktestRunner:
    def test_run_backtest_basic(self, populated_db):
        runner = BacktestRunner(populated_db)
        result = runner.run_backtest(horizon="return_20d")

        # 4 signals have direction != 0 and return_20d != None
        assert result.num_signals >= 3
        assert isinstance(result.total_return, float)
        assert isinstance(result.sharpe_ratio, float)
        assert 0 <= result.win_rate <= 1
        assert 0 <= result.direction_accuracy <= 1

    def test_run_backtest_by_ticker(self, populated_db):
        runner = BacktestRunner(populated_db)
        result = runner.run_backtest(ticker="NVDA", horizon="return_20d")
        assert result.num_signals >= 1
        assert all(t == "NVDA" for t in result.tickers)

    def test_run_backtest_by_provider(self, populated_db):
        runner = BacktestRunner(populated_db)
        result = runner.run_backtest(provider="anthropic", horizon="return_20d")
        assert result.num_signals >= 2

    def test_run_backtest_empty_db(self, db):
        runner = BacktestRunner(db)
        result = runner.run_backtest()
        assert result.num_signals == 0


class TestRankIC:
    def test_perfect_correlation(self):
        ic = _rank_ic([1, 2, 3, 4, 5], [10, 20, 30, 40, 50])
        assert abs(ic - 1.0) < 0.01

    def test_perfect_inverse(self):
        ic = _rank_ic([1, 2, 3, 4, 5], [50, 40, 30, 20, 10])
        assert abs(ic - (-1.0)) < 0.01

    def test_no_correlation(self):
        # Random-ish data
        ic = _rank_ic([1, 2, 3, 4, 5], [3, 1, 5, 2, 4])
        assert -1 <= ic <= 1

    def test_insufficient_data(self):
        assert _rank_ic([1], [2]) == 0.0
        assert _rank_ic([1, 2], [3, 4]) == 0.0


# ---------------------------------------------------------------------------
# Calibration tests
# ---------------------------------------------------------------------------

class TestCalibrationAnalyzer:
    def test_compute_calibration(self, populated_db):
        analyzer = CalibrationAnalyzer(populated_db)
        buckets = analyzer.compute_calibration(horizon="return_20d")

        assert len(buckets) > 0
        total_signals = sum(b.num_signals for b in buckets)
        # Only directional signals counted (position_direction != 0)
        assert total_signals >= 3

        for b in buckets:
            if b.num_signals > 0:
                assert 0 <= b.hit_rate <= 1

    def test_conviction_return_correlation(self, populated_db):
        analyzer = CalibrationAnalyzer(populated_db)
        corr = analyzer.compute_conviction_return_correlation(horizon="return_20d")
        assert -1 <= corr <= 1

    def test_format_report(self, populated_db):
        analyzer = CalibrationAnalyzer(populated_db)
        buckets = analyzer.compute_calibration()
        corr = analyzer.compute_conviction_return_correlation()
        report = analyzer.format_report(buckets, corr)
        assert "Calibration Analysis" in report


# ---------------------------------------------------------------------------
# Alpha decay tests
# ---------------------------------------------------------------------------

class TestAlphaDecayAnalyzer:
    def test_compute_decay_curve(self, populated_db):
        analyzer = AlphaDecayAnalyzer(populated_db)
        curve = analyzer.compute_decay_curve()

        assert len(curve) == 5  # 5 horizons
        for point in curve:
            assert point.horizon_days in (1, 5, 10, 20, 60)

    def test_find_optimal_horizon(self, populated_db):
        analyzer = AlphaDecayAnalyzer(populated_db)
        curve = analyzer.compute_decay_curve()
        optimal = analyzer.find_optimal_horizon(curve)
        # With 5 signals, at least some horizons should have enough data
        if optimal is not None:
            assert optimal in (1, 5, 10, 20, 60)

    def test_format_report(self, populated_db):
        analyzer = AlphaDecayAnalyzer(populated_db)
        curve = analyzer.compute_decay_curve()
        optimal = analyzer.find_optimal_horizon(curve)
        report = analyzer.format_report(curve, optimal)
        assert "Alpha Decay" in report


# ---------------------------------------------------------------------------
# Benchmark tests
# ---------------------------------------------------------------------------

class TestBenchmarkAnalyzer:
    @patch("backtest.benchmark.yf.download")
    def test_run_comparison(self, mock_download, populated_db):
        # Mock SPY download
        mock_download.return_value = MagicMock(
            empty=True,
        )

        analyzer = BenchmarkAnalyzer(populated_db)
        comps = analyzer.run_comparison(horizon="return_20d")

        # Should have IC Signal, Always Long, Momentum
        assert len(comps) >= 3
        names = [c.strategy_name for c in comps]
        assert "IC Signal" in names
        assert "Always Long" in names

    def test_format_report_empty(self, db):
        analyzer = BenchmarkAnalyzer(db)
        comps = analyzer.run_comparison()
        report = analyzer.format_report(comps)
        assert "Benchmark Comparison" in report


# ---------------------------------------------------------------------------
# Portfolio tests
# ---------------------------------------------------------------------------

class TestMultiAssetPortfolio:
    def test_build_snapshot(self, populated_db):
        portfolio = MultiAssetPortfolio(populated_db)
        snapshot = portfolio.build_snapshot()

        assert len(snapshot.tickers) >= 1
        assert snapshot.gross_exposure > 0
        assert snapshot.num_longs + snapshot.num_shorts == len(snapshot.tickers)

    def test_weight_methods(self, populated_db):
        portfolio = MultiAssetPortfolio(populated_db)

        snap_t = portfolio.build_snapshot(weight_method="t_signal")
        snap_eq = portfolio.build_snapshot(weight_method="equal")
        snap_conv = portfolio.build_snapshot(weight_method="conviction")

        # All should produce weights that sum to something reasonable
        for snap in [snap_t, snap_eq, snap_conv]:
            assert len(snap.weights) >= 1

    def test_min_conviction_filter(self, populated_db):
        portfolio = MultiAssetPortfolio(populated_db)

        # High threshold should exclude low-conviction signals
        snap_high = portfolio.build_snapshot(min_conviction=8.0)
        snap_low = portfolio.build_snapshot(min_conviction=1.0)

        assert len(snap_high.tickers) <= len(snap_low.tickers)

    def test_format_report(self, populated_db):
        portfolio = MultiAssetPortfolio(populated_db)
        snapshot = portfolio.build_snapshot()
        report = portfolio.format_report(snapshot)
        assert "Multi-Asset Portfolio" in report


# ---------------------------------------------------------------------------
# Explainability tests
# ---------------------------------------------------------------------------

class TestExplainabilityAnalyzer:
    def test_attribute_signal(self, populated_db):
        analyzer = ExplainabilityAnalyzer(populated_db)
        signals = populated_db.get_signals(limit=1)
        attr = analyzer.attribute_signal(signals[0])

        assert isinstance(attr, AttributionResult)
        assert attr.dominant_agent in ("Bull", "Bear", "Macro")

    def test_attribute_all(self, populated_db):
        analyzer = ExplainabilityAnalyzer(populated_db)
        attrs = analyzer.attribute_all()
        assert len(attrs) == 5  # 5 signals in populated_db

    def test_compute_agent_statistics(self, populated_db):
        analyzer = ExplainabilityAnalyzer(populated_db)
        attrs = analyzer.attribute_all()
        stats = analyzer.compute_agent_statistics(attrs)

        assert "Bull (Sector Analyst)" in stats
        assert "Bear (Risk Manager)" in stats
        assert "Macro Analyst" in stats
        assert "Debate Dynamics" in stats

        for name, agent_stats in stats.items():
            assert "avg_contribution" in agent_stats
            assert "avg_magnitude" in agent_stats
            assert "dominance_rate" in agent_stats

    def test_format_report(self, populated_db):
        analyzer = ExplainabilityAnalyzer(populated_db)
        attrs = analyzer.attribute_all()
        stats = analyzer.compute_agent_statistics(attrs)
        report = analyzer.format_report(attrs, stats)
        assert "Agent Attribution" in report


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestModels:
    def test_signal_record_defaults(self):
        sig = SignalRecord(ticker="TEST")
        assert sig.t_signal == 0.0
        assert sig.conviction == 5.0
        assert sig.position_direction == 0
        assert sig.return_1d is None

    def test_portfolio_snapshot_defaults(self):
        snap = PortfolioSnapshot()
        assert snap.tickers == []
        assert snap.weights == {}
        assert snap.gross_exposure == 0.0

    def test_backtest_result_defaults(self):
        result = BacktestResult()
        assert result.num_signals == 0
        assert result.total_return == 0.0

    def test_calibration_bucket_defaults(self):
        bucket = CalibrationBucket()
        assert bucket.num_signals == 0

    def test_alpha_decay_point_defaults(self):
        point = AlphaDecayPoint()
        assert point.horizon_days == 0
        assert point.information_coefficient == 0.0

    def test_benchmark_comparison_defaults(self):
        comp = BenchmarkComparison()
        assert comp.strategy_name == ""

    def test_attribution_result_defaults(self):
        attr = AttributionResult()
        assert attr.ticker == ""
        assert attr.dominant_agent == ""


# ---------------------------------------------------------------------------
# JSON retry tests
# ---------------------------------------------------------------------------

class TestRetryExtractJson:
    def test_retry_succeeds_on_first_try(self):
        from agents.base import retry_extract_json
        model = MagicMock()  # shouldn't be called
        result, retried = retry_extract_json(
            model, "prompt", '{"ticker": "NVDA", "thesis": "good"}',
        )
        assert result["ticker"] == "NVDA"
        assert not retried
        model.assert_not_called()

    def test_retry_recovers_from_bad_json(self):
        from agents.base import retry_extract_json
        model = MagicMock(return_value='{"ticker": "NVDA", "thesis": "good"}')
        result, retried = retry_extract_json(
            model, "prompt", "This is not JSON at all",
        )
        assert result["ticker"] == "NVDA"
        assert retried
        model.assert_called_once()

    def test_retry_fails_completely(self):
        from agents.base import retry_extract_json
        model = MagicMock(return_value="Still not JSON")
        with pytest.raises(ValueError, match="retries"):
            retry_extract_json(model, "prompt", "Not JSON", max_retries=1)
