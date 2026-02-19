"""
Main Black-Litterman optimization pipeline.

Takes IC output (PM conviction, bull/bear cases) and computes actual
portfolio weights, risk metrics, and factor exposures using pypfopt.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from optimizer.models import OptimizationResult, OptimizerFallback
from optimizer.views import build_views
from optimizer.universe import build_universe, SECTOR_ETF_MAP
from optimizer.covariance import compute_covariance
from optimizer.analytics import compute_risk_ratios, compute_factor_betas, compute_mctr

logger = logging.getLogger(__name__)


def run_black_litterman(
    ticker: str,
    sector: str,
    bull_case=None,
    bear_case=None,
    macro_view=None,
    committee_memo=None,
    risk_free_rate: float = 0.05,
    risk_aversion: float = 2.5,
    tau: float = 0.05,
    max_peers: int = 5,
    lookback: str = "2y",
    cov_method: str = "ledoit_wolf",
) -> OptimizationResult | OptimizerFallback:
    """
    Run the full Black-Litterman optimization pipeline.

    Pipeline:
        1. Build universe (target + peers + sector ETF + SPY)
        2. Compute shrunk covariance matrix
        3. Derive market-cap equilibrium prior
        4. Extract views from IC output
        5. Run BL model → posterior returns
        6. Max-Sharpe optimization → optimal weights
        7. Compute analytics (Sharpe, Sortino, factor betas, MCTR)

    Returns:
        OptimizationResult on success, OptimizerFallback on any failure.
    """
    try:
        # 1. Build universe
        universe, prices, market_caps = build_universe(
            ticker, sector, max_peers=max_peers, lookback=lookback
        )

        # 2. Covariance estimation
        cov_matrix = compute_covariance(prices, method=cov_method)

        # 3. Market-cap weights for equilibrium prior
        total_mcap = sum(market_caps.get(t, 1e10) for t in universe)
        w_mkt = np.array([market_caps.get(t, 1e10) / total_mcap for t in universe])

        # 4. Build views from IC output
        P, Q, confidence_scale = build_views(
            ticker=ticker,
            bull_case=bull_case,
            bear_case=bear_case,
            macro_view=macro_view,
            memo=committee_memo,
            universe_tickers=universe,
        )

        # 5. Run pypfopt Black-Litterman model
        from pypfopt import BlackLittermanModel, EfficientFrontier
        from pypfopt.black_litterman import market_implied_prior_returns

        # Equilibrium returns (pi = delta * Sigma * w_mkt)
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
        # omega = P @ (tau * Sigma) @ P.T / confidence_scale
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

        # 6. Max-Sharpe optimization
        ef = EfficientFrontier(bl_returns, bl_cov)
        ef.max_sharpe(risk_free_rate=risk_free_rate)
        cleaned_weights = ef.clean_weights()

        # Extract target weight
        target_weight = cleaned_weights.get(ticker, 0.0)

        # Get all weights as array (aligned with universe order)
        weight_arr = np.array([cleaned_weights.get(t, 0.0) for t in universe])

        # 7. Analytics
        # Daily returns for the target stock
        daily_returns = prices[ticker].pct_change().dropna()

        # BL posterior expected return for target
        bl_ret_target = float(bl_returns[ticker]) if ticker in bl_returns.index else float(Q[0])
        eq_ret_target = float(eq_returns[ticker]) if ticker in eq_returns.index else 0.0

        # Risk ratios
        sharpe, sortino, ann_vol, down_vol = compute_risk_ratios(
            daily_returns, bl_ret_target, rf=risk_free_rate
        )

        # Factor betas: regress target on SPY + sector ETF
        factor_rets = {}
        spy_col = "SPY" if "SPY" in prices.columns else None
        sector_etf = SECTOR_ETF_MAP.get(sector, None)
        sect_col = sector_etf if sector_etf and sector_etf in prices.columns else None

        if spy_col:
            factor_rets["SPY"] = prices[spy_col].pct_change().dropna()
        if sect_col and sect_col != spy_col:
            factor_rets[sect_col] = prices[sect_col].pct_change().dropna()

        factor_exposures = compute_factor_betas(daily_returns, factor_rets)

        # MCTR
        cov_arr = cov_matrix.values if hasattr(cov_matrix, 'values') else np.array(cov_matrix)
        risk_contribs = compute_mctr(weight_arr, cov_arr, universe)

        # Portfolio vol
        port_var = float(weight_arr.T @ cov_arr @ weight_arr)
        port_vol = float(np.sqrt(max(port_var, 1e-16)))

        result = OptimizationResult(
            success=True,
            ticker=ticker,
            optimal_weight=round(target_weight, 6),
            optimal_weight_pct=f"{target_weight * 100:.1f}%",
            bl_expected_return=round(bl_ret_target, 4),
            equilibrium_return=round(eq_ret_target, 4),
            computed_sharpe=round(sharpe, 2),
            computed_sortino=round(sortino, 2),
            annualized_vol=round(ann_vol, 4),
            downside_vol=round(down_vol, 4),
            factor_exposures=factor_exposures,
            portfolio_vol=round(port_vol, 4),
            risk_contributions=risk_contribs,
            universe_tickers=universe,
            universe_weights={t: round(w, 6) for t, w in cleaned_weights.items() if w > 1e-6},
            covariance_method=cov_method,
            lookback_days=len(prices),
            risk_free_rate=risk_free_rate,
            tau=tau,
            risk_aversion=risk_aversion,
        )

        logger.info(
            f"BL optimizer: {ticker} weight={target_weight:.1%}, "
            f"BL return={bl_ret_target:.2%}, Sharpe={sharpe:.2f}"
        )

        return result

    except Exception as e:
        logger.error(f"Black-Litterman optimizer failed for {ticker}: {e}", exc_info=True)
        return OptimizerFallback(
            success=False,
            error_message=f"Optimizer failed: {str(e)}",
            ticker=ticker,
        )
