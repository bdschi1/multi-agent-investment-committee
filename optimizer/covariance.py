"""Covariance estimation for Black-Litterman optimization."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def compute_covariance(
    prices: pd.DataFrame,
    method: str = "ledoit_wolf",
) -> pd.DataFrame:
    """
    Compute the covariance matrix from daily prices.

    Uses pypfopt's CovarianceShrinkage for robust estimation.

    Args:
        prices: Daily adjusted close prices (DatetimeIndex, columns=tickers)
        method: Shrinkage method — "ledoit_wolf" (default), "oracle_approximating",
                "exponential", or "sample"

    Returns:
        Covariance matrix as a DataFrame (annualized, tickers × tickers)
    """
    from pypfopt.risk_models import CovarianceShrinkage

    if method == "sample":
        # Simple sample covariance (no shrinkage)
        returns = prices.pct_change().dropna()
        cov = returns.cov() * 252  # annualize
        return cov

    cs = CovarianceShrinkage(prices)

    if method == "ledoit_wolf":
        cov = cs.ledoit_wolf()
    elif method == "oracle_approximating":
        cov = cs.oracle_approximating()
    elif method == "exponential":
        # Exponential covariance via pypfopt
        from pypfopt.risk_models import exp_cov
        cov = exp_cov(prices)
    else:
        logger.warning(f"Unknown covariance method '{method}', falling back to ledoit_wolf")
        cov = cs.ledoit_wolf()

    return cov
