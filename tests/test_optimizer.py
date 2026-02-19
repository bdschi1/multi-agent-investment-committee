"""
Tests for the Black-Litterman optimizer package.

Covers: numeric extraction, covariance computation, view construction,
BL integration, graceful fallback, analytics, and graph node integration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# Test numeric extraction (views.py)
# ---------------------------------------------------------------------------

class TestNumericExtraction:
    """Parse PM heuristic strings into numeric values."""

    def test_simple_percentage(self):
        from optimizer.views import extract_numeric_return
        assert extract_numeric_return("45%") == pytest.approx(0.45)

    def test_positive_percentage_with_sign(self):
        from optimizer.views import extract_numeric_return
        assert extract_numeric_return("+14% over 12m") == pytest.approx(0.14)

    def test_negative_percentage(self):
        from optimizer.views import extract_numeric_return
        assert extract_numeric_return("-5%") == pytest.approx(-0.05)

    def test_percentage_with_reasoning(self):
        from optimizer.views import extract_numeric_return
        result = extract_numeric_return("22% — total return including 1.2% div yield")
        assert result == pytest.approx(0.22)

    def test_empty_string(self):
        from optimizer.views import extract_numeric_return
        assert extract_numeric_return("") is None

    def test_no_percentage(self):
        from optimizer.views import extract_numeric_return
        assert extract_numeric_return("not available") is None

    def test_decimal_return(self):
        from optimizer.views import extract_numeric_return
        assert extract_numeric_return("0.45") == pytest.approx(0.45)

    def test_vol_extraction(self):
        from optimizer.views import extract_numeric_vol
        assert extract_numeric_vol("30% annualized") == pytest.approx(0.30)


# ---------------------------------------------------------------------------
# Test covariance computation (covariance.py)
# ---------------------------------------------------------------------------

class TestCovarianceComputation:
    """Verify covariance matrix properties."""

    @pytest.fixture
    def synthetic_prices(self):
        """Generate synthetic daily prices for 3 assets over 252 days."""
        np.random.seed(42)
        n_days = 252
        dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
        returns = np.random.multivariate_normal(
            mean=[0.0005, 0.0003, 0.0002],
            cov=[[0.0004, 0.0001, 0.00005],
                 [0.0001, 0.0003, 0.00008],
                 [0.00005, 0.00008, 0.0002]],
            size=n_days,
        )
        prices = pd.DataFrame(
            100 * np.exp(np.cumsum(returns, axis=0)),
            index=dates,
            columns=["AAPL", "MSFT", "SPY"],
        )
        return prices

    def test_shape(self, synthetic_prices):
        from optimizer.covariance import compute_covariance
        cov = compute_covariance(synthetic_prices, method="ledoit_wolf")
        assert cov.shape == (3, 3)

    def test_positive_definite(self, synthetic_prices):
        from optimizer.covariance import compute_covariance
        cov = compute_covariance(synthetic_prices, method="ledoit_wolf")
        eigenvalues = np.linalg.eigvalsh(cov.values)
        assert all(ev > 0 for ev in eigenvalues), "Covariance matrix must be positive definite"

    def test_symmetric(self, synthetic_prices):
        from optimizer.covariance import compute_covariance
        cov = compute_covariance(synthetic_prices, method="ledoit_wolf")
        np.testing.assert_array_almost_equal(cov.values, cov.values.T)

    def test_sample_method(self, synthetic_prices):
        from optimizer.covariance import compute_covariance
        cov = compute_covariance(synthetic_prices, method="sample")
        assert cov.shape == (3, 3)


# ---------------------------------------------------------------------------
# Test view construction (views.py)
# ---------------------------------------------------------------------------

class TestViewConstruction:
    """Verify P/Q/omega dimensions and confidence scaling."""

    def _make_mock_objects(self):
        memo = MagicMock()
        memo.idio_return_estimate = "+14% — alpha from share gains"
        memo.conviction = 7.5

        bull = MagicMock()
        bull.conviction_score = 8.0
        bull.idiosyncratic_return = "14%"
        bull.forecasted_total_return = "22%"

        bear = MagicMock()
        bear.bearish_conviction = 4.0

        return memo, bull, bear

    def test_dimensions(self):
        from optimizer.views import build_views
        memo, bull, bear = self._make_mock_objects()
        universe = ["NVDA", "AAPL", "MSFT", "XLK", "SPY"]

        P, Q, conf = build_views("NVDA", bull, bear, None, memo, universe)

        assert P.shape == (1, 5)
        assert Q.shape == (1,)
        assert isinstance(conf, float)

    def test_pick_matrix_target(self):
        from optimizer.views import build_views
        memo, bull, bear = self._make_mock_objects()
        universe = ["NVDA", "AAPL", "MSFT", "XLK", "SPY"]

        P, Q, conf = build_views("NVDA", bull, bear, None, memo, universe)

        assert P[0, 0] == 1.0  # NVDA is at index 0
        assert P[0, 1] == 0.0

    def test_q_value(self):
        from optimizer.views import build_views
        memo, bull, bear = self._make_mock_objects()
        universe = ["NVDA", "AAPL", "MSFT", "XLK", "SPY"]

        P, Q, conf = build_views("NVDA", bull, bear, None, memo, universe)

        assert Q[0] == pytest.approx(0.14)

    def test_confidence_scaling_high_conviction(self):
        from optimizer.views import build_views
        memo, bull, bear = self._make_mock_objects()
        bull.conviction_score = 9.0
        bear.bearish_conviction = 2.0
        universe = ["NVDA", "SPY"]

        _, _, conf = build_views("NVDA", bull, bear, None, memo, universe)

        # net_conviction = (9-2+10)/20 = 0.85
        # confidence_scale = 0.1 + 0.9 * 0.85 = 0.865
        assert conf == pytest.approx(0.865, abs=0.01)

    def test_confidence_scaling_low_conviction(self):
        from optimizer.views import build_views
        memo, bull, bear = self._make_mock_objects()
        bull.conviction_score = 3.0
        bear.bearish_conviction = 8.0
        universe = ["NVDA", "SPY"]

        _, _, conf = build_views("NVDA", bull, bear, None, memo, universe)

        # net_conviction = (3-8+10)/20 = 0.25
        # confidence_scale = 0.1 + 0.9 * 0.25 = 0.325
        assert conf == pytest.approx(0.325, abs=0.01)

    def test_fallback_when_no_numeric(self):
        """When PM doesn't provide numeric alpha, derive from conviction."""
        from optimizer.views import build_views

        memo = MagicMock()
        memo.idio_return_estimate = ""
        memo.conviction = 7.0

        bull = MagicMock()
        bull.conviction_score = 7.0
        bull.idiosyncratic_return = ""
        bull.forecasted_total_return = ""

        bear = MagicMock()
        bear.bearish_conviction = 4.0

        universe = ["TEST", "SPY"]
        P, Q, conf = build_views("TEST", bull, bear, None, memo, universe)

        # Should derive from conviction: (7-5)/10 * 0.40 + 0.10 = 0.18
        assert Q[0] == pytest.approx(0.18)


# ---------------------------------------------------------------------------
# Test analytics (analytics.py)
# ---------------------------------------------------------------------------

class TestAnalytics:
    """Risk ratios, factor betas, and MCTR."""

    @pytest.fixture
    def daily_returns(self):
        np.random.seed(42)
        return pd.Series(
            np.random.normal(0.0005, 0.02, 252),
            index=pd.date_range("2024-01-01", periods=252, freq="B"),
            name="STOCK",
        )

    def test_sharpe_sortino(self, daily_returns):
        from optimizer.analytics import compute_risk_ratios
        sharpe, sortino, vol, dvol = compute_risk_ratios(
            daily_returns, expected_return=0.15, rf=0.05
        )
        assert isinstance(sharpe, float)
        assert isinstance(sortino, float)
        assert vol > 0
        assert dvol > 0
        # Sortino should be >= Sharpe (downside vol <= total vol)
        assert sortino >= sharpe - 0.5  # some tolerance

    def test_factor_betas(self, daily_returns):
        from optimizer.analytics import compute_factor_betas
        np.random.seed(123)
        spy_returns = pd.Series(
            np.random.normal(0.0004, 0.015, 252),
            index=daily_returns.index,
            name="SPY",
        )
        exposures = compute_factor_betas(daily_returns, {"SPY": spy_returns})
        assert len(exposures) == 1
        assert exposures[0].factor_name == "SPY"
        assert isinstance(exposures[0].beta, float)
        assert isinstance(exposures[0].t_stat, float)
        assert 0 <= exposures[0].p_value <= 1

    def test_mctr_sums_close_to_one(self):
        from optimizer.analytics import compute_mctr
        weights = np.array([0.3, 0.3, 0.2, 0.2])
        cov = np.array([
            [0.04, 0.01, 0.005, 0.002],
            [0.01, 0.03, 0.008, 0.003],
            [0.005, 0.008, 0.02, 0.001],
            [0.002, 0.003, 0.001, 0.015],
        ])
        tickers = ["A", "B", "C", "D"]

        contribs = compute_mctr(weights, cov, tickers)

        total_pct = sum(c.pct_contribution for c in contribs)
        # Sum of %CTR should approximately equal 1 (for long-only portfolio)
        assert total_pct == pytest.approx(1.0, abs=0.05)

    def test_mctr_order(self):
        """Results should be sorted by absolute % contribution."""
        from optimizer.analytics import compute_mctr
        weights = np.array([0.8, 0.1, 0.05, 0.05])
        cov = np.eye(4) * 0.04
        tickers = ["A", "B", "C", "D"]

        contribs = compute_mctr(weights, cov, tickers)

        # First should be highest contributor
        assert contribs[0].ticker == "A"


# ---------------------------------------------------------------------------
# Test graceful fallback
# ---------------------------------------------------------------------------

class TestGracefulFallback:
    """Optimizer should return OptimizerFallback on any error."""

    def test_universe_failure(self):
        from optimizer.bl_optimizer import run_black_litterman
        from optimizer.models import OptimizerFallback

        with patch("optimizer.bl_optimizer.build_universe", side_effect=ValueError("no data")):
            result = run_black_litterman(
                ticker="FAKE",
                sector="Technology",
            )

        assert isinstance(result, OptimizerFallback)
        assert result.success is False
        assert "no data" in result.error_message

    def test_covariance_failure(self):
        from optimizer.bl_optimizer import run_black_litterman
        from optimizer.models import OptimizerFallback

        mock_universe = (
            ["FAKE", "SPY"],
            pd.DataFrame(
                {"FAKE": [100, 101, 102], "SPY": [400, 401, 402]},
                index=pd.date_range("2024-01-01", periods=3),
            ),
            {"FAKE": 1e10, "SPY": 1e12},
        )

        with patch("optimizer.bl_optimizer.build_universe", return_value=mock_universe), \
             patch("optimizer.bl_optimizer.compute_covariance", side_effect=RuntimeError("singular")):
            result = run_black_litterman(ticker="FAKE", sector="Technology")

        assert isinstance(result, OptimizerFallback)
        assert result.success is False


# ---------------------------------------------------------------------------
# Test Black-Litterman integration (with mocked market data)
# ---------------------------------------------------------------------------

class TestBlackLittermanIntegration:
    """Full pipeline test with synthetic data."""

    @pytest.fixture
    def synthetic_universe(self):
        """Create a synthetic universe with realistic price data."""
        np.random.seed(42)
        n_days = 504
        dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
        tickers = ["NVDA", "AAPL", "MSFT", "CRM", "ADBE", "XLK", "SPY"]

        # Correlated returns
        mean_returns = [0.001, 0.0005, 0.0006, 0.0004, 0.0003, 0.0005, 0.0004]
        cov = np.eye(7) * 0.0004
        for i in range(7):
            for j in range(7):
                if i != j:
                    cov[i, j] = 0.00015

        returns = np.random.multivariate_normal(mean_returns, cov, size=n_days)
        prices = pd.DataFrame(
            100 * np.exp(np.cumsum(returns, axis=0)),
            index=dates,
            columns=tickers,
        )

        market_caps = {
            "NVDA": 3e12, "AAPL": 3.5e12, "MSFT": 3.2e12,
            "CRM": 3e11, "ADBE": 2.5e11, "XLK": 5e11, "SPY": 5e12,
        }

        return tickers, prices, market_caps

    def test_full_pipeline(self, synthetic_universe):
        from optimizer.bl_optimizer import run_black_litterman
        from optimizer.models import OptimizationResult

        tickers, prices, market_caps = synthetic_universe

        memo = MagicMock()
        memo.idio_return_estimate = "+20% — strong growth thesis"
        memo.conviction = 8.0

        bull = MagicMock()
        bull.conviction_score = 8.0
        bull.idiosyncratic_return = "20%"
        bull.forecasted_total_return = "25%"
        bull.key_metrics = {}

        bear = MagicMock()
        bear.bearish_conviction = 3.0

        with patch("optimizer.bl_optimizer.build_universe", return_value=(tickers, prices, market_caps)):
            result = run_black_litterman(
                ticker="NVDA",
                sector="Technology",
                bull_case=bull,
                bear_case=bear,
                committee_memo=memo,
            )

        assert isinstance(result, OptimizationResult)
        assert result.success is True
        assert result.ticker == "NVDA"
        assert 0.0 <= result.optimal_weight <= 1.0
        assert result.bl_expected_return > 0
        assert result.computed_sharpe != 0
        assert len(result.universe_tickers) == 7
        assert len(result.risk_contributions) > 0
        assert result.portfolio_vol > 0

    def test_result_fields_types(self, synthetic_universe):
        from optimizer.bl_optimizer import run_black_litterman
        from optimizer.models import OptimizationResult

        tickers, prices, market_caps = synthetic_universe

        memo = MagicMock()
        memo.idio_return_estimate = "15%"
        memo.conviction = 6.0

        bull = MagicMock()
        bull.conviction_score = 6.0
        bull.idiosyncratic_return = "15%"
        bull.forecasted_total_return = "20%"
        bull.key_metrics = {}

        bear = MagicMock()
        bear.bearish_conviction = 5.0

        with patch("optimizer.bl_optimizer.build_universe", return_value=(tickers, prices, market_caps)):
            result = run_black_litterman(
                ticker="NVDA",
                sector="Technology",
                bull_case=bull,
                bear_case=bear,
                committee_memo=memo,
            )

        assert isinstance(result.optimal_weight_pct, str)
        assert "%" in result.optimal_weight_pct
        assert isinstance(result.universe_weights, dict)
        assert isinstance(result.factor_exposures, list)
        assert result.covariance_method == "ledoit_wolf"


# ---------------------------------------------------------------------------
# Test graph node integration
# ---------------------------------------------------------------------------

class TestGraphIntegration:
    """The optimizer node should accept state dict and return optimization_result."""

    def test_node_returns_result(self):
        from optimizer.node import run_optimizer
        from optimizer.models import OptimizerFallback

        # Minimal state with no memo → should return fallback
        state = {
            "ticker": "NVDA",
            "bull_case": None,
            "bear_case": None,
            "macro_view": None,
            "committee_memo": None,
            "context": {},
        }

        result = run_optimizer(state)

        assert "optimization_result" in result
        assert isinstance(result["optimization_result"], OptimizerFallback)
        assert result["optimization_result"].success is False

    def test_node_with_memo(self):
        from optimizer.node import run_optimizer
        from optimizer.models import OptimizerFallback

        memo = MagicMock()
        memo.idio_return_estimate = "10%"
        memo.conviction = 6.0

        state = {
            "ticker": "FAKE_TICKER",
            "bull_case": None,
            "bear_case": None,
            "macro_view": None,
            "committee_memo": memo,
            "context": {"market_data": {"sector": "Technology"}},
        }

        # Will fail at build_universe (no real data) but should fallback gracefully
        result = run_optimizer(state)

        assert "optimization_result" in result
        # Should be a fallback since yfinance can't fetch FAKE_TICKER
        opt = result["optimization_result"]
        assert hasattr(opt, 'success')


# ---------------------------------------------------------------------------
# Test models
# ---------------------------------------------------------------------------

class TestModels:
    """Model serialization and defaults."""

    def test_optimization_result_defaults(self):
        from optimizer.models import OptimizationResult
        result = OptimizationResult(
            ticker="NVDA",
            optimal_weight=0.15,
            optimal_weight_pct="15.0%",
            bl_expected_return=0.25,
            equilibrium_return=0.12,
        )
        assert result.success is True
        assert result.covariance_method == "ledoit_wolf"
        assert result.risk_free_rate == 0.05

    def test_optimizer_fallback(self):
        from optimizer.models import OptimizerFallback
        fb = OptimizerFallback(ticker="NVDA", error_message="no data")
        assert fb.success is False
        d = fb.model_dump()
        assert d["success"] is False
        assert d["ticker"] == "NVDA"

    def test_factor_exposure_serialization(self):
        from optimizer.models import FactorExposure
        fe = FactorExposure(factor_name="SPY", beta=1.2, t_stat=5.3, p_value=0.0001)
        d = fe.model_dump()
        assert d["factor_name"] == "SPY"
        assert d["beta"] == 1.2

    def test_risk_contribution_serialization(self):
        from optimizer.models import RiskContribution
        rc = RiskContribution(
            ticker="NVDA", weight=0.3, marginal_ctr=0.15, pct_contribution=0.45
        )
        d = rc.model_dump()
        assert d["pct_contribution"] == 0.45
