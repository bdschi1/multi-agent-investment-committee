"""Post-optimization analytics: risk ratios, factor betas, MCTR."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from optimizer.models import FactorExposure, RiskContribution

logger = logging.getLogger(__name__)


def compute_risk_ratios(
    returns: pd.Series,
    expected_return: float,
    rf: float = 0.05,
) -> tuple[float, float, float, float]:
    """
    Compute risk-adjusted performance metrics from historical returns.

    Args:
        returns: Daily return series for the asset
        expected_return: Annualized expected return (from BL posterior)
        rf: Risk-free rate (annualized)

    Returns:
        (sharpe, sortino, annualized_vol, downside_vol)
    """
    ann_vol = float(returns.std() * np.sqrt(252))

    # Downside vol: std of returns below zero (annualized)
    downside = returns[returns < 0]
    downside_vol = float(downside.std() * np.sqrt(252)) if len(downside) > 1 else ann_vol

    excess = expected_return - rf
    sharpe = excess / ann_vol if ann_vol > 1e-8 else 0.0
    sortino = excess / downside_vol if downside_vol > 1e-8 else 0.0

    return sharpe, sortino, ann_vol, downside_vol


def compute_factor_betas(
    stock_returns: pd.Series,
    factor_returns: dict[str, pd.Series],
) -> list[FactorExposure]:
    """
    OLS regression of stock returns on factor returns.

    R_stock = alpha + beta_1 * F_1 + beta_2 * F_2 + ... + epsilon

    Args:
        stock_returns: Daily returns for the target stock
        factor_returns: {factor_name: daily_returns_series}

    Returns:
        List of FactorExposure with betas, t-stats, p-values.
    """
    from scipy import stats as sp_stats

    factor_names = list(factor_returns.keys())
    if not factor_names:
        return []

    # Build factor matrix (align indices)
    combined = pd.DataFrame({"stock": stock_returns})
    for name, series in factor_returns.items():
        combined[name] = series
    combined = combined.dropna()

    if len(combined) < 30:
        logger.warning(f"Insufficient data for factor regression ({len(combined)} obs)")
        return []

    y = combined["stock"].values
    X = combined[factor_names].values
    # Add intercept
    X_with_intercept = np.column_stack([np.ones(len(X)), X])

    # OLS via numpy lstsq
    coeffs, residuals, rank, sv = np.linalg.lstsq(X_with_intercept, y, rcond=None)

    # Compute t-statistics and p-values
    n = len(y)
    k = X_with_intercept.shape[1]
    y_hat = X_with_intercept @ coeffs
    resid = y - y_hat
    mse = np.sum(resid**2) / max(n - k, 1)
    var_coeffs = mse * np.linalg.pinv(X_with_intercept.T @ X_with_intercept)

    exposures = []
    for i, name in enumerate(factor_names):
        beta = float(coeffs[i + 1])  # skip intercept
        se = float(np.sqrt(max(var_coeffs[i + 1, i + 1], 1e-16)))
        t_stat = beta / se if se > 1e-12 else 0.0
        p_value = float(2 * (1 - sp_stats.t.cdf(abs(t_stat), df=max(n - k, 1))))

        exposures.append(FactorExposure(
            factor_name=name,
            beta=round(beta, 4),
            t_stat=round(t_stat, 2),
            p_value=round(p_value, 4),
        ))

    return exposures


def compute_mctr(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
    tickers: list[str],
) -> list[RiskContribution]:
    """
    Compute marginal contribution to risk (MCTR) for each asset.

    MCTR_i = (Sigma @ w)_i / sigma_p
    %CTR_i = w_i * MCTR_i / sigma_p

    Args:
        weights: (n,) weight vector
        cov_matrix: (n, n) covariance matrix
        tickers: asset names

    Returns:
        List of RiskContribution sorted by absolute % contribution.
    """
    w = np.array(weights).flatten()
    cov = np.array(cov_matrix)

    # Portfolio variance and vol
    port_var = float(w.T @ cov @ w)
    port_vol = np.sqrt(max(port_var, 1e-16))

    # Marginal contribution to risk
    mctr = (cov @ w) / port_vol

    # Percentage contribution
    pct_ctr = (w * mctr) / port_vol

    contributions = []
    for i, t in enumerate(tickers):
        contributions.append(RiskContribution(
            ticker=t,
            weight=round(float(w[i]), 6),
            marginal_ctr=round(float(mctr[i]), 6),
            pct_contribution=round(float(pct_ctr[i]), 6),
        ))

    # Sort by absolute percentage contribution (descending)
    contributions.sort(key=lambda x: abs(x.pct_contribution), reverse=True)

    return contributions
