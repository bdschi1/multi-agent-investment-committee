"""
Tests for multi-strategy portfolio optimizer.

Covers: strategy registry, all 6 strategies (weights sum to 1, positive weights,
target ticker present), run_optimization() dispatcher, backward compatibility,
and OptimizationResult model changes.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_prices():
    """Create synthetic price data for 7 tickers over 504 days."""
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
    return tickers, prices


@pytest.fixture
def synthetic_universe(synthetic_prices):
    """Full universe fixture with prices and market caps."""
    tickers, prices = synthetic_prices
    market_caps = {
        "NVDA": 3e12, "AAPL": 3.5e12, "MSFT": 3.2e12,
        "CRM": 3e11, "ADBE": 2.5e11, "XLK": 5e11, "SPY": 5e12,
    }
    return tickers, prices, market_caps


@pytest.fixture
def cov_matrix(synthetic_prices):
    """Compute covariance matrix from synthetic prices."""
    from optimizer.covariance import compute_covariance
    _, prices = synthetic_prices
    return compute_covariance(prices)


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

class TestStrategyRegistry:

    def test_all_strategies_registered(self):
        from optimizer.strategies import STRATEGY_REGISTRY
        expected = {
            "black_litterman", "hrp", "mean_variance",
            "min_variance", "risk_parity", "equal_weight",
        }
        assert set(STRATEGY_REGISTRY.keys()) == expected

    def test_display_names_cover_registry(self):
        from optimizer.strategies import STRATEGY_DISPLAY_NAMES, STRATEGY_REGISTRY
        # All registry keys must be in display names (ensemble is extra — meta-strategy)
        assert set(STRATEGY_REGISTRY.keys()).issubset(set(STRATEGY_DISPLAY_NAMES.keys()))

    def test_reverse_mapping(self):
        from optimizer.strategies import (
            DISPLAY_TO_STRATEGY_KEY,
            STRATEGY_DISPLAY_NAMES,
        )
        for key, display in STRATEGY_DISPLAY_NAMES.items():
            assert DISPLAY_TO_STRATEGY_KEY[display] == key

    def test_get_strategy_valid(self):
        from optimizer.strategies import get_strategy
        for key in ["black_litterman", "hrp", "mean_variance",
                     "min_variance", "risk_parity", "equal_weight"]:
            strategy = get_strategy(key)
            assert hasattr(strategy, "optimize")
            assert hasattr(strategy, "name")
            assert hasattr(strategy, "key")

    def test_get_strategy_invalid(self):
        from optimizer.strategies import get_strategy
        with pytest.raises(ValueError, match="Unknown optimizer strategy"):
            get_strategy("nonexistent")


# ---------------------------------------------------------------------------
# Equal Weight
# ---------------------------------------------------------------------------

class TestEqualWeight:

    def test_weights_sum_to_one(self, synthetic_universe, cov_matrix):
        from optimizer.strategies import EqualWeightStrategy
        tickers, prices, mcaps = synthetic_universe
        strategy = EqualWeightStrategy()
        result = strategy.optimize(tickers, prices, cov_matrix, mcaps, "NVDA")
        total = sum(result["weights"].values())
        assert total == pytest.approx(1.0)

    def test_all_weights_equal(self, synthetic_universe, cov_matrix):
        from optimizer.strategies import EqualWeightStrategy
        tickers, prices, mcaps = synthetic_universe
        strategy = EqualWeightStrategy()
        result = strategy.optimize(tickers, prices, cov_matrix, mcaps, "NVDA")
        expected_w = 1.0 / len(tickers)
        for w in result["weights"].values():
            assert w == pytest.approx(expected_w)

    def test_all_tickers_present(self, synthetic_universe, cov_matrix):
        from optimizer.strategies import EqualWeightStrategy
        tickers, prices, mcaps = synthetic_universe
        strategy = EqualWeightStrategy()
        result = strategy.optimize(tickers, prices, cov_matrix, mcaps, "NVDA")
        assert set(result["weights"].keys()) == set(tickers)

    def test_no_expected_returns(self, synthetic_universe, cov_matrix):
        from optimizer.strategies import EqualWeightStrategy
        tickers, prices, mcaps = synthetic_universe
        strategy = EqualWeightStrategy()
        result = strategy.optimize(tickers, prices, cov_matrix, mcaps, "NVDA")
        assert result["expected_returns"] is None


# ---------------------------------------------------------------------------
# HRP
# ---------------------------------------------------------------------------

class TestHRP:

    def test_weights_sum_to_one(self, synthetic_universe, cov_matrix):
        from optimizer.strategies import HRPStrategy
        tickers, prices, mcaps = synthetic_universe
        strategy = HRPStrategy()
        result = strategy.optimize(tickers, prices, cov_matrix, mcaps, "NVDA")
        total = sum(result["weights"].values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_positive_weights(self, synthetic_universe, cov_matrix):
        from optimizer.strategies import HRPStrategy
        tickers, prices, mcaps = synthetic_universe
        strategy = HRPStrategy()
        result = strategy.optimize(tickers, prices, cov_matrix, mcaps, "NVDA")
        for w in result["weights"].values():
            assert w >= 0.0

    def test_target_ticker_present(self, synthetic_universe, cov_matrix):
        from optimizer.strategies import HRPStrategy
        tickers, prices, mcaps = synthetic_universe
        strategy = HRPStrategy()
        result = strategy.optimize(tickers, prices, cov_matrix, mcaps, "NVDA")
        assert "NVDA" in result["weights"]


# ---------------------------------------------------------------------------
# Mean-Variance
# ---------------------------------------------------------------------------

class TestMeanVariance:

    def test_weights_sum_to_one(self, synthetic_universe, cov_matrix):
        from optimizer.strategies import MeanVarianceStrategy
        tickers, prices, mcaps = synthetic_universe
        strategy = MeanVarianceStrategy()
        result = strategy.optimize(tickers, prices, cov_matrix, mcaps, "NVDA")
        total = sum(result["weights"].values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_returns_expected_returns(self, synthetic_universe, cov_matrix):
        from optimizer.strategies import MeanVarianceStrategy
        tickers, prices, mcaps = synthetic_universe
        strategy = MeanVarianceStrategy()
        result = strategy.optimize(tickers, prices, cov_matrix, mcaps, "NVDA")
        assert result["expected_returns"] is not None
        assert "NVDA" in result["expected_returns"]

    def test_positive_weights(self, synthetic_universe, cov_matrix):
        from optimizer.strategies import MeanVarianceStrategy
        tickers, prices, mcaps = synthetic_universe
        strategy = MeanVarianceStrategy()
        result = strategy.optimize(tickers, prices, cov_matrix, mcaps, "NVDA")
        for w in result["weights"].values():
            assert w >= -0.01  # allow tiny floating-point negatives


# ---------------------------------------------------------------------------
# Minimum Variance
# ---------------------------------------------------------------------------

class TestMinVariance:

    def test_weights_sum_to_one(self, synthetic_universe, cov_matrix):
        from optimizer.strategies import MinVarianceStrategy
        tickers, prices, mcaps = synthetic_universe
        strategy = MinVarianceStrategy()
        result = strategy.optimize(tickers, prices, cov_matrix, mcaps, "NVDA")
        total = sum(result["weights"].values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_lower_vol_than_equal_weight(self, synthetic_universe, cov_matrix):
        """Min-variance should produce lower portfolio vol than 1/N."""
        from optimizer.strategies import EqualWeightStrategy, MinVarianceStrategy
        tickers, prices, mcaps = synthetic_universe

        cov_arr = cov_matrix.values

        # Min-var weights
        mv = MinVarianceStrategy()
        mv_result = mv.optimize(tickers, prices, cov_matrix, mcaps, "NVDA")
        w_mv = np.array([mv_result["weights"].get(t, 0.0) for t in tickers])
        vol_mv = np.sqrt(w_mv @ cov_arr @ w_mv)

        # Equal weights
        ew = EqualWeightStrategy()
        ew_result = ew.optimize(tickers, prices, cov_matrix, mcaps, "NVDA")
        w_ew = np.array([ew_result["weights"].get(t, 0.0) for t in tickers])
        vol_ew = np.sqrt(w_ew @ cov_arr @ w_ew)

        assert vol_mv <= vol_ew + 1e-6


# ---------------------------------------------------------------------------
# Risk Parity (ERC)
# ---------------------------------------------------------------------------

class TestRiskParity:

    def test_weights_sum_to_one(self, synthetic_universe, cov_matrix):
        from optimizer.strategies import RiskParityStrategy
        tickers, prices, mcaps = synthetic_universe
        strategy = RiskParityStrategy()
        result = strategy.optimize(tickers, prices, cov_matrix, mcaps, "NVDA")
        total = sum(result["weights"].values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_positive_weights(self, synthetic_universe, cov_matrix):
        from optimizer.strategies import RiskParityStrategy
        tickers, prices, mcaps = synthetic_universe
        strategy = RiskParityStrategy()
        result = strategy.optimize(tickers, prices, cov_matrix, mcaps, "NVDA")
        for w in result["weights"].values():
            assert w >= 0.0

    def test_risk_contributions_approximately_equal(self, synthetic_universe, cov_matrix):
        """Core ERC property: each asset contributes ~equal risk."""
        from optimizer.strategies import RiskParityStrategy
        tickers, prices, mcaps = synthetic_universe
        strategy = RiskParityStrategy()
        result = strategy.optimize(tickers, prices, cov_matrix, mcaps, "NVDA")

        cov_arr = cov_matrix.values
        w = np.array([result["weights"].get(t, 0.0) for t in tickers])
        port_vol = np.sqrt(w @ cov_arr @ w)

        if port_vol > 1e-8:
            mctr = (cov_arr @ w) / port_vol
            rc = w * mctr / port_vol  # percentage risk contribution

            target = 1.0 / len(tickers)
            for i, contrib in enumerate(rc):
                assert contrib == pytest.approx(target, abs=0.03), (
                    f"Risk contribution for {tickers[i]}: {contrib:.4f} vs target {target:.4f}"
                )


# ---------------------------------------------------------------------------
# run_optimization() dispatcher
# ---------------------------------------------------------------------------

class TestRunOptimizationDispatcher:

    @pytest.fixture
    def _mock_universe(self, synthetic_universe):
        return synthetic_universe

    def _make_mock_inputs(self):
        memo = MagicMock()
        memo.idio_return_estimate = "+15%"
        memo.conviction = 7.0
        bull = MagicMock()
        bull.conviction_score = 7.0
        bull.idiosyncratic_return = "15%"
        bull.forecasted_total_return = "20%"
        bull.key_metrics = {}
        bear = MagicMock()
        bear.bearish_conviction = 4.0
        return memo, bull, bear

    @pytest.mark.parametrize("method", [
        "black_litterman", "hrp", "mean_variance",
        "min_variance", "risk_parity", "equal_weight",
    ])
    def test_all_methods_produce_result(self, method, _mock_universe):
        from optimizer.bl_optimizer import run_optimization
        from optimizer.models import OptimizationResult
        tickers, prices, market_caps = _mock_universe
        memo, bull, bear = self._make_mock_inputs()

        with patch("optimizer.bl_optimizer.build_universe",
                   return_value=(tickers, prices, market_caps)):
            result = run_optimization(
                ticker="NVDA", sector="Technology",
                optimizer_method=method,
                bull_case=bull, bear_case=bear, committee_memo=memo,
            )

        assert isinstance(result, OptimizationResult)
        assert result.success is True
        assert result.optimizer_method == method
        assert 0 <= result.optimal_weight <= 1.0
        assert result.universe_weights  # non-empty
        assert result.computed_sharpe is not None

    @pytest.mark.parametrize("method", [
        "black_litterman", "hrp", "mean_variance",
        "min_variance", "risk_parity", "equal_weight",
    ])
    def test_display_name_set(self, method, _mock_universe):
        from optimizer.bl_optimizer import run_optimization
        from optimizer.strategies import STRATEGY_DISPLAY_NAMES
        tickers, prices, market_caps = _mock_universe
        memo, bull, bear = self._make_mock_inputs()

        with patch("optimizer.bl_optimizer.build_universe",
                   return_value=(tickers, prices, market_caps)):
            result = run_optimization(
                ticker="NVDA", sector="Technology",
                optimizer_method=method,
                bull_case=bull, bear_case=bear, committee_memo=memo,
            )

        assert result.optimizer_display_name == STRATEGY_DISPLAY_NAMES[method]

    def test_invalid_method_returns_fallback(self, _mock_universe):
        from optimizer.bl_optimizer import run_optimization
        from optimizer.models import OptimizerFallback
        tickers, prices, market_caps = _mock_universe

        with patch("optimizer.bl_optimizer.build_universe",
                   return_value=(tickers, prices, market_caps)):
            result = run_optimization(
                ticker="NVDA", sector="Technology",
                optimizer_method="nonexistent",
            )

        assert isinstance(result, OptimizerFallback)
        assert result.success is False


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompat:

    def test_run_black_litterman_still_works(self, synthetic_universe):
        """The legacy function name should still work."""
        from optimizer.bl_optimizer import run_black_litterman
        from optimizer.models import OptimizationResult
        tickers, prices, market_caps = synthetic_universe

        memo = MagicMock()
        memo.idio_return_estimate = "+10%"
        memo.conviction = 6.0
        bull = MagicMock()
        bull.conviction_score = 6.0
        bull.idiosyncratic_return = "10%"
        bull.forecasted_total_return = "15%"
        bull.key_metrics = {}
        bear = MagicMock()
        bear.bearish_conviction = 4.0

        with patch("optimizer.bl_optimizer.build_universe",
                   return_value=(tickers, prices, market_caps)):
            result = run_black_litterman(
                ticker="NVDA", sector="Technology",
                bull_case=bull, bear_case=bear, committee_memo=memo,
            )

        assert isinstance(result, OptimizationResult)
        assert result.success is True
        assert result.optimizer_method == "black_litterman"
        assert result.bl_expected_return is not None

    def test_bl_specific_fields_optional(self):
        """Non-BL results should not require bl_expected_return."""
        from optimizer.models import OptimizationResult
        result = OptimizationResult(
            ticker="NVDA",
            optimizer_method="hrp",
            optimizer_display_name="HRP",
            optimal_weight=0.15,
            optimal_weight_pct="15.0%",
        )
        assert result.bl_expected_return is None
        assert result.equilibrium_return is None

    def test_bl_result_still_accepts_floats(self):
        """Existing code that passes bl_expected_return as float should still work."""
        from optimizer.models import OptimizationResult
        result = OptimizationResult(
            ticker="NVDA",
            optimal_weight=0.15,
            optimal_weight_pct="15.0%",
            bl_expected_return=0.25,
            equilibrium_return=0.12,
        )
        assert result.bl_expected_return == 0.25
        assert result.optimizer_method == "black_litterman"  # default


# ---------------------------------------------------------------------------
# Node method propagation
# ---------------------------------------------------------------------------

class TestNodePropagation:

    @pytest.mark.parametrize("method", ["hrp", "equal_weight", "min_variance"])
    def test_node_reads_method_from_config(self, method, synthetic_universe):
        from optimizer.models import OptimizationResult
        from optimizer.node import run_optimizer
        tickers, prices, market_caps = synthetic_universe

        memo = MagicMock()
        memo.idio_return_estimate = "10%"
        memo.conviction = 6.0

        state = {
            "ticker": "NVDA",
            "bull_case": None,
            "bear_case": None,
            "macro_view": None,
            "committee_memo": memo,
            "context": {"market_data": {"sector": "Technology"}},
        }

        with patch("optimizer.bl_optimizer.build_universe",
                   return_value=(tickers, prices, market_caps)):
            result = run_optimizer(
                state,
                config={"configurable": {"optimizer_method": method}},
            )

        opt = result["optimization_result"]
        assert isinstance(opt, OptimizationResult)
        assert opt.optimizer_method == method

    def test_node_defaults_to_bl(self, synthetic_universe):
        """When no optimizer_method in config, default to black_litterman."""
        from optimizer.node import run_optimizer
        tickers, prices, market_caps = synthetic_universe

        memo = MagicMock()
        memo.idio_return_estimate = "10%"
        memo.conviction = 6.0
        bull = MagicMock()
        bull.conviction_score = 6.0
        bull.idiosyncratic_return = "10%"
        bull.forecasted_total_return = "15%"
        bull.key_metrics = {}
        bear = MagicMock()
        bear.bearish_conviction = 4.0

        state = {
            "ticker": "NVDA",
            "bull_case": bull,
            "bear_case": bear,
            "macro_view": None,
            "committee_memo": memo,
            "context": {"market_data": {"sector": "Technology"}},
        }

        with patch("optimizer.bl_optimizer.build_universe",
                   return_value=(tickers, prices, market_caps)):
            result = run_optimizer(state, config={"configurable": {}})

        assert result["optimization_result"].optimizer_method == "black_litterman"

    def test_non_bl_runs_without_memo(self, synthetic_universe):
        """Non-BL strategies should work even without a committee memo."""
        from optimizer.models import OptimizationResult
        from optimizer.node import run_optimizer
        tickers, prices, market_caps = synthetic_universe

        state = {
            "ticker": "NVDA",
            "bull_case": None,
            "bear_case": None,
            "macro_view": None,
            "committee_memo": None,  # No memo
            "context": {"market_data": {"sector": "Technology"}},
        }

        with patch("optimizer.bl_optimizer.build_universe",
                   return_value=(tickers, prices, market_caps)):
            result = run_optimizer(
                state,
                config={"configurable": {"optimizer_method": "equal_weight"}},
            )

        opt = result["optimization_result"]
        assert isinstance(opt, OptimizationResult)
        assert opt.success is True
        assert opt.optimizer_method == "equal_weight"


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class TestSettings:

    def test_optimizer_method_in_settings(self):
        from config.settings import Settings
        s = Settings(optimizer_method="hrp")
        assert s.optimizer_method == "hrp"

    def test_default_is_black_litterman(self):
        from config.settings import Settings
        s = Settings()
        assert s.optimizer_method == "black_litterman"
