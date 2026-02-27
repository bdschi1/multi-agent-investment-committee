"""
Portfolio optimization strategies.

Each strategy takes shared inputs (universe, prices, covariance, market_caps)
and returns a dict with cleaned weights and strategy-specific metadata.

Strategies:
    - BlackLittermanStrategy: PM views → BL posterior → max-Sharpe
    - HRPStrategy: Hierarchical Risk Parity (tree-based, no cov inversion)
    - MeanVarianceStrategy: Historical returns → max-Sharpe (classic Markowitz)
    - MinVarianceStrategy: Minimize portfolio variance (no return estimates)
    - RiskParityStrategy: Equal risk contribution via scipy SLSQP
    - EqualWeightStrategy: Simple 1/N allocation
"""

from __future__ import annotations

import abc
import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class OptimizerStrategy(abc.ABC):
    """Abstract base class for portfolio optimization strategies."""

    name: str  # Human-readable, e.g. "Black-Litterman"
    key: str   # Machine key, e.g. "black_litterman"

    @abc.abstractmethod
    def optimize(
        self,
        universe: list[str],
        prices: pd.DataFrame,
        cov_matrix: pd.DataFrame,
        market_caps: dict[str, float],
        ticker: str,
        risk_free_rate: float = 0.05,
        # BL-specific kwargs (ignored by non-BL strategies)
        bull_case: Any = None,
        bear_case: Any = None,
        macro_view: Any = None,
        committee_memo: Any = None,
        risk_aversion: float = 2.5,
        tau: float = 0.05,
    ) -> dict:
        """
        Run optimization.

        Returns dict with:
            - "weights": dict[str, float]  (ticker → weight, sums to ~1.0)
            - "expected_returns": dict[str, float] | None
            - "metadata": dict  (strategy-specific, e.g. bl_expected_return)
        """
        ...


# ---------------------------------------------------------------------------
# Black-Litterman
# ---------------------------------------------------------------------------

class BlackLittermanStrategy(OptimizerStrategy):
    """PM views → BL posterior returns → max-Sharpe optimization."""

    name = "Black-Litterman"
    key = "black_litterman"

    def optimize(self, universe, prices, cov_matrix, market_caps, ticker,
                 risk_free_rate=0.05, bull_case=None, bear_case=None,
                 macro_view=None, committee_memo=None,
                 risk_aversion=2.5, tau=0.05) -> dict:
        from pypfopt import BlackLittermanModel, EfficientFrontier
        from pypfopt.black_litterman import market_implied_prior_returns

        from optimizer.views import build_views

        # Market-cap weights for equilibrium prior
        total_mcap = sum(market_caps.get(t, 1e10) for t in universe)
        w_mkt = np.array([market_caps.get(t, 1e10) / total_mcap for t in universe])  # noqa: F841

        # Build views from PM output
        P, Q, confidence_scale = build_views(
            ticker=ticker,
            bull_case=bull_case,
            bear_case=bear_case,
            macro_view=macro_view,
            memo=committee_memo,
            universe_tickers=universe,
        )

        # BL model
        bl = BlackLittermanModel(
            cov_matrix,
            pi="market",
            market_caps=market_caps,
            risk_aversion=risk_aversion,
            Q=Q,
            P=P,
            tau=tau,
        )

        # Adjust omega based on confidence
        sigma_arr = cov_matrix.values if hasattr(cov_matrix, 'values') else np.array(cov_matrix)
        omega_raw = P @ (tau * sigma_arr) @ P.T
        omega = omega_raw / confidence_scale
        bl.omega = omega

        # Posterior returns
        bl_returns = bl.bl_returns()
        bl_cov = bl.bl_cov()

        # Equilibrium returns for comparison
        eq_returns = market_implied_prior_returns(
            market_caps, risk_aversion, cov_matrix
        )

        # Max-Sharpe optimization
        ef = EfficientFrontier(bl_returns, bl_cov)
        ef.max_sharpe(risk_free_rate=risk_free_rate)
        cleaned_weights = ef.clean_weights()

        # BL-specific expected returns
        bl_ret_target = float(bl_returns[ticker]) if ticker in bl_returns.index else float(Q[0])
        eq_ret_target = float(eq_returns[ticker]) if ticker in eq_returns.index else 0.0

        return {
            "weights": dict(cleaned_weights),
            "expected_returns": {t: float(bl_returns[t]) for t in universe if t in bl_returns.index},
            "metadata": {
                "bl_expected_return": round(bl_ret_target, 4),
                "equilibrium_return": round(eq_ret_target, 4),
            },
        }


# ---------------------------------------------------------------------------
# Hierarchical Risk Parity
# ---------------------------------------------------------------------------

class HRPStrategy(OptimizerStrategy):
    """Hierarchical Risk Parity — tree-based, no covariance inversion."""

    name = "Hierarchical Risk Parity (HRP)"
    key = "hrp"

    def optimize(self, universe, prices, cov_matrix, market_caps, ticker,
                 risk_free_rate=0.05, **kwargs) -> dict:
        from pypfopt import HRPOpt

        returns = prices[universe].pct_change().dropna()
        hrp = HRPOpt(returns=returns, cov_matrix=cov_matrix)
        hrp.optimize()
        cleaned = hrp.clean_weights()

        return {
            "weights": dict(cleaned),
            "expected_returns": None,
            "metadata": {"linkage_method": "single"},
        }


# ---------------------------------------------------------------------------
# Mean-Variance (Max Sharpe)
# ---------------------------------------------------------------------------

class MeanVarianceStrategy(OptimizerStrategy):
    """Classic Markowitz max-Sharpe using historical expected returns."""

    name = "Mean-Variance (Max Sharpe)"
    key = "mean_variance"

    def optimize(self, universe, prices, cov_matrix, market_caps, ticker,
                 risk_free_rate=0.05, **kwargs) -> dict:
        from pypfopt import EfficientFrontier
        from pypfopt.expected_returns import mean_historical_return

        mu = mean_historical_return(prices[universe])
        ef = EfficientFrontier(mu, cov_matrix)

        try:
            ef.max_sharpe(risk_free_rate=risk_free_rate)
        except ValueError:
            # No asset exceeds risk-free rate — fall back to min volatility
            logger.warning(
                "No asset expected return exceeds risk-free rate "
                f"({risk_free_rate:.1%}) — falling back to min_volatility"
            )
            ef = EfficientFrontier(mu, cov_matrix)
            ef.min_volatility()

        cleaned = ef.clean_weights()

        return {
            "weights": dict(cleaned),
            "expected_returns": {t: float(mu[t]) for t in universe if t in mu.index},
            "metadata": {},
        }


# ---------------------------------------------------------------------------
# Minimum Variance
# ---------------------------------------------------------------------------

class MinVarianceStrategy(OptimizerStrategy):
    """Minimize portfolio variance — no return estimates needed."""

    name = "Minimum Variance"
    key = "min_variance"

    def optimize(self, universe, prices, cov_matrix, market_caps, ticker,
                 risk_free_rate=0.05, **kwargs) -> dict:
        from pypfopt import EfficientFrontier
        from pypfopt.expected_returns import mean_historical_return

        # pypfopt requires expected_returns for init, but min_volatility ignores them
        mu = mean_historical_return(prices[universe])
        ef = EfficientFrontier(mu, cov_matrix)
        ef.min_volatility()
        cleaned = ef.clean_weights()

        return {
            "weights": dict(cleaned),
            "expected_returns": None,
            "metadata": {},
        }


# ---------------------------------------------------------------------------
# Risk Parity (Equal Risk Contribution)
# ---------------------------------------------------------------------------

class RiskParityStrategy(OptimizerStrategy):
    """Equal risk contribution via scipy SLSQP optimization."""

    name = "Risk Parity (ERC)"
    key = "risk_parity"

    def optimize(self, universe, prices, cov_matrix, market_caps, ticker,
                 risk_free_rate=0.05, **kwargs) -> dict:
        from scipy.optimize import minimize

        n = len(universe)
        cov_arr = cov_matrix.values if hasattr(cov_matrix, 'values') else np.array(cov_matrix)
        target_rc = 1.0 / n

        def objective(w):
            w = np.array(w)
            port_vol = np.sqrt(w @ cov_arr @ w)
            if port_vol < 1e-12:
                return 0.0
            mctr = (cov_arr @ w) / port_vol
            rc = w * mctr / port_vol  # percentage risk contribution
            return np.sum((rc - target_rc) ** 2)

        w0 = np.ones(n) / n
        bounds = [(0.0, 1.0)] * n
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        result = minimize(
            objective, w0, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )

        method_used = "ERC"
        if result.success:
            weights = {universe[i]: float(result.x[i]) for i in range(n)}
        else:
            # Fallback: inverse-volatility weighting
            logger.warning("ERC solver did not converge — falling back to inverse-vol")
            vols = np.sqrt(np.diag(cov_arr))
            inv_vol = 1.0 / np.maximum(vols, 1e-8)
            w_iv = inv_vol / inv_vol.sum()
            weights = {universe[i]: float(w_iv[i]) for i in range(n)}
            method_used = "inverse_vol_fallback"

        return {
            "weights": weights,
            "expected_returns": None,
            "metadata": {"method": method_used},
        }


# ---------------------------------------------------------------------------
# Equal Weight
# ---------------------------------------------------------------------------

class EqualWeightStrategy(OptimizerStrategy):
    """Simple 1/N allocation."""

    name = "Equal Weight (1/N)"
    key = "equal_weight"

    def optimize(self, universe, prices, cov_matrix, market_caps, ticker,
                 risk_free_rate=0.05, **kwargs) -> dict:
        n = len(universe)
        weights = {t: 1.0 / n for t in universe}
        return {
            "weights": weights,
            "expected_returns": None,
            "metadata": {},
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

STRATEGY_REGISTRY: dict[str, type[OptimizerStrategy]] = {
    "black_litterman": BlackLittermanStrategy,
    "hrp": HRPStrategy,
    "mean_variance": MeanVarianceStrategy,
    "min_variance": MinVarianceStrategy,
    "risk_parity": RiskParityStrategy,
    "equal_weight": EqualWeightStrategy,
}

STRATEGY_DISPLAY_NAMES: dict[str, str] = {
    cls.key: cls.name for cls in STRATEGY_REGISTRY.values()
}
# Ensemble is a meta-strategy (not in STRATEGY_REGISTRY) but needs display mapping
STRATEGY_DISPLAY_NAMES["ensemble"] = "Ensemble (All Strategies)"

# Reverse mapping: display name → key
DISPLAY_TO_STRATEGY_KEY: dict[str, str] = {
    v: k for k, v in STRATEGY_DISPLAY_NAMES.items()
}


def get_strategy(key: str) -> OptimizerStrategy:
    """Instantiate a strategy by key."""
    cls = STRATEGY_REGISTRY.get(key)
    if cls is None:
        raise ValueError(
            f"Unknown optimizer strategy: {key!r}. "
            f"Available: {list(STRATEGY_REGISTRY.keys())}"
        )
    return cls()
