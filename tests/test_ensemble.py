"""
Tests for the ensemble multi-strategy portfolio optimizer.

Covers: compute_hhi, blended weights, consensus matrix, divergence flags,
run_ensemble dispatcher, node dispatch, ensemble formatter, backward compat.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from optimizer.ensemble import (
    DEFAULT_ENSEMBLE_WEIGHTS,
    STRATEGY_ROLES,
    _build_comparisons,
    _build_consensus,
    _build_layered_narrative,
    _compute_blended_weights,
    _compute_divergence,
    compute_hhi,
)
from optimizer.models import (
    DivergenceFlag,
    EnsembleResult,
    OptimizationResult,
    OptimizerFallback,
    RiskContribution,
    StrategyComparison,
    TickerConsensus,
)


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
def universe(synthetic_prices):
    tickers, prices = synthetic_prices
    return tickers


@pytest.fixture
def sample_raw_weights():
    """Sample per-strategy raw weights for 7 tickers."""
    return {
        "black_litterman": {
            "NVDA": 0.20, "AAPL": 0.30, "MSFT": 0.15,
            "CRM": 0.10, "ADBE": 0.05, "XLK": 0.10, "SPY": 0.10,
        },
        "risk_parity": {
            "NVDA": 0.14, "AAPL": 0.14, "MSFT": 0.15,
            "CRM": 0.14, "ADBE": 0.15, "XLK": 0.14, "SPY": 0.14,
        },
        "min_variance": {
            "NVDA": 0.05, "AAPL": 0.20, "MSFT": 0.25,
            "CRM": 0.10, "ADBE": 0.10, "XLK": 0.15, "SPY": 0.15,
        },
        "hrp": {
            "NVDA": 0.12, "AAPL": 0.18, "MSFT": 0.16,
            "CRM": 0.13, "ADBE": 0.14, "XLK": 0.13, "SPY": 0.14,
        },
        "mean_variance": {
            "NVDA": 0.25, "AAPL": 0.35, "MSFT": 0.10,
            "CRM": 0.05, "ADBE": 0.05, "XLK": 0.10, "SPY": 0.10,
        },
        "equal_weight": {
            t: 1.0 / 7 for t in ["NVDA", "AAPL", "MSFT", "CRM", "ADBE", "XLK", "SPY"]
        },
    }


@pytest.fixture
def sample_opt_results():
    """Sample OptimizationResult objects for each strategy."""
    results = {}
    data = {
        "black_litterman": (0.20, 0.14, 1.42, 1.85),
        "risk_parity": (0.14, 0.12, 0.89, 1.12),
        "min_variance": (0.05, 0.10, 0.72, 0.91),
        "hrp": (0.12, 0.12, 0.95, 1.18),
        "mean_variance": (0.25, 0.16, 1.51, 1.92),
        "equal_weight": (1.0 / 7, 0.14, 0.78, 0.98),
    }
    for key, (wt, vol, sharpe, sortino) in data.items():
        from optimizer.strategies import STRATEGY_DISPLAY_NAMES
        results[key] = OptimizationResult(
            ticker="NVDA",
            optimizer_method=key,
            optimizer_display_name=STRATEGY_DISPLAY_NAMES.get(key, key),
            optimal_weight=wt,
            optimal_weight_pct=f"{wt * 100:.1f}%",
            portfolio_vol=vol,
            computed_sharpe=sharpe,
            computed_sortino=sortino,
            universe_tickers=["NVDA", "AAPL", "MSFT", "CRM", "ADBE", "XLK", "SPY"],
        )
    return results


# ---------------------------------------------------------------------------
# compute_hhi
# ---------------------------------------------------------------------------

class TestComputeHHI:

    def test_equal_weights(self):
        n = 7
        weights = {f"T{i}": 1.0 / n for i in range(n)}
        hhi = compute_hhi(weights)
        assert abs(hhi - 1.0 / n) < 1e-8

    def test_concentrated(self):
        weights = {"A": 1.0, "B": 0.0, "C": 0.0}
        assert compute_hhi(weights) == pytest.approx(1.0)

    def test_empty(self):
        assert compute_hhi({}) == 0.0

    def test_two_equal(self):
        weights = {"A": 0.5, "B": 0.5}
        assert compute_hhi(weights) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Blended weights
# ---------------------------------------------------------------------------

class TestBlendedWeights:

    def test_equal_blend(self, sample_raw_weights):
        universe = ["NVDA", "AAPL", "MSFT", "CRM", "ADBE", "XLK", "SPY"]
        equal_coeffs = {k: 1.0 / 6 for k in sample_raw_weights}
        blended = _compute_blended_weights(universe, sample_raw_weights, equal_coeffs)
        total = sum(blended.values())
        assert abs(total - 1.0) < 1e-6

    def test_default_blend_sums_to_one(self, sample_raw_weights):
        universe = ["NVDA", "AAPL", "MSFT", "CRM", "ADBE", "XLK", "SPY"]
        blended = _compute_blended_weights(
            universe, sample_raw_weights, DEFAULT_ENSEMBLE_WEIGHTS
        )
        total = sum(blended.values())
        assert abs(total - 1.0) < 1e-6

    def test_missing_strategy_normalizes(self, sample_raw_weights):
        """If one strategy is missing from raw_weights, blend still sums to 1."""
        universe = ["NVDA", "AAPL", "MSFT", "CRM", "ADBE", "XLK", "SPY"]
        partial = {k: v for k, v in sample_raw_weights.items() if k != "hrp"}
        blended = _compute_blended_weights(
            universe, partial, DEFAULT_ENSEMBLE_WEIGHTS
        )
        total = sum(blended.values())
        assert abs(total - 1.0) < 1e-6

    def test_single_strategy_blend(self):
        universe = ["A", "B"]
        raw = {"black_litterman": {"A": 0.6, "B": 0.4}}
        coeffs = {"black_litterman": 1.0}
        blended = _compute_blended_weights(universe, raw, coeffs)
        assert blended["A"] == pytest.approx(0.6)
        assert blended["B"] == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Consensus matrix
# ---------------------------------------------------------------------------

class TestConsensusMatrix:

    def test_all_tickers_present(self, sample_raw_weights):
        universe = ["NVDA", "AAPL", "MSFT", "CRM", "ADBE", "XLK", "SPY"]
        consensus = _build_consensus(universe, sample_raw_weights, "NVDA")
        tickers = {tc.ticker for tc in consensus}
        assert tickers == set(universe)

    def test_target_flagged(self, sample_raw_weights):
        universe = ["NVDA", "AAPL", "MSFT", "CRM", "ADBE", "XLK", "SPY"]
        consensus = _build_consensus(universe, sample_raw_weights, "NVDA")
        target = next(tc for tc in consensus if tc.ticker == "NVDA")
        assert target.is_target is True
        non_targets = [tc for tc in consensus if tc.ticker != "NVDA"]
        assert all(tc.is_target is False for tc in non_targets)

    def test_mean_computed_correctly(self, sample_raw_weights):
        universe = ["NVDA", "AAPL", "MSFT", "CRM", "ADBE", "XLK", "SPY"]
        consensus = _build_consensus(universe, sample_raw_weights, "NVDA")
        nvda = next(tc for tc in consensus if tc.ticker == "NVDA")
        expected_mean = np.mean([
            sample_raw_weights[m].get("NVDA", 0.0)
            for m in sample_raw_weights
        ])
        assert nvda.mean_weight == pytest.approx(expected_mean, abs=1e-5)

    def test_sorted_by_std_descending(self, sample_raw_weights):
        universe = ["NVDA", "AAPL", "MSFT", "CRM", "ADBE", "XLK", "SPY"]
        consensus = _build_consensus(universe, sample_raw_weights, "NVDA")
        stds = [tc.std_weight for tc in consensus]
        assert stds == sorted(stds, reverse=True)


# ---------------------------------------------------------------------------
# Divergence flags
# ---------------------------------------------------------------------------

class TestDivergenceFlags:

    def test_high_agreement(self):
        consensus = [
            TickerConsensus(ticker="SPY", std_weight=0.01),
            TickerConsensus(ticker="NVDA", std_weight=0.08),
        ]
        flags = _compute_divergence(consensus)
        agree = [f for f in flags if f.flag_type == "high_agreement"]
        assert len(agree) == 1
        assert agree[0].ticker == "SPY"

    def test_high_disagreement(self):
        consensus = [
            TickerConsensus(ticker="AAPL", std_weight=0.15),
            TickerConsensus(ticker="MSFT", std_weight=0.05),
        ]
        flags = _compute_divergence(consensus)
        disagree = [f for f in flags if f.flag_type == "high_disagreement"]
        assert len(disagree) == 1
        assert disagree[0].ticker == "AAPL"

    def test_moderate_no_flag(self):
        consensus = [
            TickerConsensus(ticker="CRM", std_weight=0.05),
        ]
        flags = _compute_divergence(consensus)
        assert len(flags) == 0


# ---------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------

class TestBuildComparisons:

    def test_one_row_per_strategy(self, sample_opt_results, sample_raw_weights):
        comparisons = _build_comparisons(sample_opt_results, sample_raw_weights)
        assert len(comparisons) == len(sample_opt_results)

    def test_roles_assigned(self, sample_opt_results, sample_raw_weights):
        comparisons = _build_comparisons(sample_opt_results, sample_raw_weights)
        by_key = {c.strategy_key: c for c in comparisons}
        assert by_key["black_litterman"].role == "Core Allocation"
        assert by_key["risk_parity"].role == "Sanity Check"
        assert by_key["min_variance"].role == "Defensive Overlay"


# ---------------------------------------------------------------------------
# Layered narrative
# ---------------------------------------------------------------------------

class TestLayeredNarrative:

    def test_contains_all_sections(self, sample_opt_results, sample_raw_weights):
        consensus = [
            TickerConsensus(
                ticker="NVDA", is_target=True,
                mean_weight=0.15, std_weight=0.06,
            ),
        ]
        narrative = _build_layered_narrative(
            "NVDA", sample_opt_results, sample_raw_weights, consensus
        )
        assert "Core Allocation" in narrative
        assert "Sanity Check" in narrative
        assert "Defensive Overlay" in narrative
        assert "Consensus" in narrative

    def test_delta_analysis_when_divergent(self, sample_opt_results, sample_raw_weights):
        consensus = [
            TickerConsensus(ticker="NVDA", is_target=True, mean_weight=0.15, std_weight=0.06),
        ]
        narrative = _build_layered_narrative(
            "NVDA", sample_opt_results, sample_raw_weights, consensus
        )
        # BL weight (0.20) - RP weight (0.14) = 0.06 > 0.05 threshold
        assert "overweights" in narrative or "underweights" in narrative


# ---------------------------------------------------------------------------
# run_ensemble integration (mocked universe)
# ---------------------------------------------------------------------------

class TestRunEnsemble:

    @pytest.fixture
    def mock_universe(self, synthetic_prices):
        """Patch build_universe to return synthetic data."""
        tickers, prices = synthetic_prices
        market_caps = {
            "NVDA": 3e12, "AAPL": 3.5e12, "MSFT": 3.2e12,
            "CRM": 3e11, "ADBE": 2.5e11, "XLK": 5e11, "SPY": 5e12,
        }
        return patch(
            "optimizer.ensemble.build_universe",
            return_value=(tickers, prices, market_caps),
        )

    def test_all_strategies_run(self, mock_universe):
        from optimizer.ensemble import run_ensemble
        with mock_universe:
            result = run_ensemble("NVDA", "Technology")
        assert isinstance(result, EnsembleResult)
        assert result.success is True
        # BL may fail without memo, but others should succeed
        assert len(result.strategy_results) >= 5

    def test_blended_weights_sum_to_one(self, mock_universe):
        from optimizer.ensemble import run_ensemble
        with mock_universe:
            result = run_ensemble("NVDA", "Technology")
        total = sum(result.blended_weights.values())
        assert abs(total - 1.0) < 1e-4

    def test_consensus_has_all_tickers(self, mock_universe):
        from optimizer.ensemble import run_ensemble
        with mock_universe:
            result = run_ensemble("NVDA", "Technology")
        consensus_tickers = {tc.ticker for tc in result.consensus}
        assert "NVDA" in consensus_tickers

    def test_comparisons_match_results(self, mock_universe):
        from optimizer.ensemble import run_ensemble
        with mock_universe:
            result = run_ensemble("NVDA", "Technology")
        assert len(result.strategy_comparisons) == len(result.strategy_results)

    def test_narrative_not_empty(self, mock_universe):
        from optimizer.ensemble import run_ensemble
        with mock_universe:
            result = run_ensemble("NVDA", "Technology")
        assert len(result.layered_narrative) > 0

    def test_ensemble_weights_recorded(self, mock_universe):
        from optimizer.ensemble import run_ensemble
        with mock_universe:
            result = run_ensemble("NVDA", "Technology")
        assert result.ensemble_weights_used == DEFAULT_ENSEMBLE_WEIGHTS

    def test_custom_ensemble_weights(self, mock_universe):
        from optimizer.ensemble import run_ensemble
        custom = {"equal_weight": 0.5, "risk_parity": 0.5}
        with mock_universe:
            result = run_ensemble(
                "NVDA", "Technology", ensemble_weights=custom
            )
        assert result.ensemble_weights_used == custom


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------

class TestRunEnsembleGraceful:

    def test_one_strategy_fails(self, synthetic_prices):
        """If one strategy raises, ensemble continues with the rest."""
        from optimizer.ensemble import run_ensemble

        tickers, prices = synthetic_prices
        market_caps = {t: 1e12 for t in tickers}

        with patch("optimizer.ensemble.build_universe",
                   return_value=(tickers, prices, market_caps)):
            with patch("optimizer.ensemble.get_strategy") as mock_get:
                real_get = __import__(
                    "optimizer.strategies", fromlist=["get_strategy"]
                ).get_strategy

                def side_effect(key):
                    if key == "hrp":
                        raise RuntimeError("HRP broke")
                    return real_get(key)

                mock_get.side_effect = side_effect
                result = run_ensemble("NVDA", "Technology")

        assert result.success is True
        assert "hrp" in result.failed_strategies
        assert "hrp" not in result.strategy_results


# ---------------------------------------------------------------------------
# Node dispatch
# ---------------------------------------------------------------------------

class TestEnsembleNode:

    def test_ensemble_dispatches_correctly(self):
        from optimizer.node import run_optimizer

        mock_result = EnsembleResult(
            ticker="NVDA",
            strategy_results={},
            blended_target_weight=0.15,
        )

        with patch("optimizer.ensemble.run_ensemble", return_value=mock_result):
            state = {"ticker": "NVDA", "context": {}}
            config = {"configurable": {"optimizer_method": "ensemble"}}
            output = run_optimizer(state, config)

        assert output["optimization_result"] is mock_result

    def test_single_strategy_still_works(self):
        from optimizer.node import run_optimizer

        mock_result = OptimizationResult(ticker="NVDA")

        with patch("optimizer.bl_optimizer.run_optimization", return_value=mock_result):
            state = {"ticker": "NVDA", "context": {}}
            config = {"configurable": {"optimizer_method": "equal_weight"}}
            output = run_optimizer(state, config)

        assert output["optimization_result"] is mock_result

    def test_ensemble_status_callback(self):
        from optimizer.node import run_optimizer

        mock_result = EnsembleResult(
            ticker="NVDA",
            strategy_results={"eq": OptimizationResult(ticker="NVDA")},
            blended_target_weight=0.15,
        )
        status_calls = []

        with patch("optimizer.ensemble.run_ensemble", return_value=mock_result):
            state = {"ticker": "NVDA", "context": {}}
            config = {
                "configurable": {
                    "optimizer_method": "ensemble",
                    "on_status": lambda msg: status_calls.append(msg),
                }
            }
            run_optimizer(state, config)

        assert any("Ensemble" in s for s in status_calls)


# ---------------------------------------------------------------------------
# Ensemble formatter
# ---------------------------------------------------------------------------

class TestEnsembleFormatter:

    @pytest.fixture
    def mock_ensemble_result(self, sample_opt_results):
        return EnsembleResult(
            ticker="NVDA",
            strategy_results=sample_opt_results,
            strategy_comparisons=[
                StrategyComparison(
                    strategy_key="black_litterman",
                    strategy_name="Black-Litterman",
                    role="Core Allocation",
                    target_weight=0.20,
                    portfolio_vol=0.14,
                    sharpe=1.42,
                    sortino=1.85,
                    max_single_weight=0.30,
                    hhi=0.18,
                ),
                StrategyComparison(
                    strategy_key="risk_parity",
                    strategy_name="Risk Parity (ERC)",
                    role="Sanity Check",
                    target_weight=0.14,
                    portfolio_vol=0.12,
                    sharpe=0.89,
                    sortino=1.12,
                    max_single_weight=0.15,
                    hhi=0.145,
                ),
            ],
            consensus=[
                TickerConsensus(
                    ticker="NVDA",
                    is_target=True,
                    weights_by_strategy={
                        "black_litterman": 0.20,
                        "risk_parity": 0.14,
                    },
                    mean_weight=0.17,
                    std_weight=0.03,
                ),
            ],
            blended_weights={"NVDA": 0.17, "AAPL": 0.25},
            blended_target_weight=0.17,
            blended_portfolio_vol=0.13,
            blended_sharpe=1.10,
            blended_sortino=1.40,
            blended_hhi=0.16,
            blended_risk_contributions=[
                RiskContribution(
                    ticker="NVDA", weight=0.17,
                    marginal_ctr=0.034, pct_contribution=0.21,
                ),
            ],
            ensemble_weights_used=DEFAULT_ENSEMBLE_WEIGHTS,
            divergence_flags=[
                DivergenceFlag(
                    ticker="SPY",
                    flag_type="high_agreement",
                    description="Low weight dispersion (std=0.010)",
                    std_weight=0.01,
                ),
            ],
            layered_narrative="**Core Allocation:** BL assigns NVDA 20.0% weight.",
            universe_tickers=["NVDA", "AAPL"],
        )

    def test_returns_markdown(self, mock_ensemble_result):
        from app_lib.formatters import _format_ensemble_section
        md = _format_ensemble_section(mock_ensemble_result)
        assert "## Ensemble Portfolio Analytics" in md

    def test_strategy_comparison_table(self, mock_ensemble_result):
        from app_lib.formatters import _format_ensemble_section
        md = _format_ensemble_section(mock_ensemble_result)
        assert "### Strategy Comparison" in md
        assert "Core Allocation" in md
        assert "Sanity Check" in md

    def test_blended_allocation_table(self, mock_ensemble_result):
        from app_lib.formatters import _format_ensemble_section
        md = _format_ensemble_section(mock_ensemble_result)
        assert "### Blended Allocation" in md
        assert "17.0%" in md

    def test_consensus_matrix(self, mock_ensemble_result):
        from app_lib.formatters import _format_ensemble_section
        md = _format_ensemble_section(mock_ensemble_result)
        assert "### Weight Consensus Matrix" in md

    def test_divergence_flags(self, mock_ensemble_result):
        from app_lib.formatters import _format_ensemble_section
        md = _format_ensemble_section(mock_ensemble_result)
        assert "### Divergence Flags" in md
        assert "SPY" in md

    def test_layered_interpretation(self, mock_ensemble_result):
        from app_lib.formatters import _format_ensemble_section
        md = _format_ensemble_section(mock_ensemble_result)
        assert "### Layered Interpretation" in md

    def test_blended_mctr(self, mock_ensemble_result):
        from app_lib.formatters import _format_ensemble_section
        md = _format_ensemble_section(mock_ensemble_result)
        assert "### Blended Risk Contribution" in md

    def test_failed_result_shows_error(self):
        from app_lib.formatters import _format_ensemble_section
        failed = EnsembleResult(
            ticker="NVDA", success=False,
            error_message="something broke",
        )
        md = _format_ensemble_section(failed)
        assert "something broke" in md


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompat:

    def test_ensemble_not_in_strategy_registry(self):
        from optimizer.strategies import STRATEGY_REGISTRY
        assert "ensemble" not in STRATEGY_REGISTRY

    def test_ensemble_in_display_names(self):
        from optimizer.strategies import STRATEGY_DISPLAY_NAMES
        assert "ensemble" in STRATEGY_DISPLAY_NAMES
        assert STRATEGY_DISPLAY_NAMES["ensemble"] == "Ensemble (All Strategies)"

    def test_display_to_key_mapping(self):
        from optimizer.strategies import DISPLAY_TO_STRATEGY_KEY
        assert DISPLAY_TO_STRATEGY_KEY["Ensemble (All Strategies)"] == "ensemble"

    def test_get_strategy_raises_for_ensemble(self):
        from optimizer.strategies import get_strategy
        with pytest.raises(ValueError, match="Unknown optimizer strategy"):
            get_strategy("ensemble")


# ---------------------------------------------------------------------------
# Default ensemble weights
# ---------------------------------------------------------------------------

class TestDefaultEnsembleWeights:

    def test_sums_to_one(self):
        total = sum(DEFAULT_ENSEMBLE_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-8

    def test_all_strategies_present(self):
        from optimizer.strategies import STRATEGY_REGISTRY
        for key in STRATEGY_REGISTRY:
            assert key in DEFAULT_ENSEMBLE_WEIGHTS

    def test_roles_cover_all_strategies(self):
        from optimizer.strategies import STRATEGY_REGISTRY
        for key in STRATEGY_REGISTRY:
            assert key in STRATEGY_ROLES
