"""
Portfolio optimization pipeline.

Shared infrastructure (universe, covariance, analytics) with pluggable
optimization strategies. Default: Black-Litterman.

run_optimization() is the main entry point — dispatches to the selected
strategy and computes shared analytics.

run_black_litterman() is a backward-compatible wrapper.
"""

from __future__ import annotations

import logging

import numpy as np

from optimizer.analytics import compute_factor_betas, compute_mctr, compute_risk_ratios
from optimizer.covariance import compute_covariance
from optimizer.models import OptimizationResult, OptimizerFallback
from optimizer.universe import SECTOR_ETF_MAP, build_universe

logger = logging.getLogger(__name__)


def run_optimization(
    ticker: str,
    sector: str,
    optimizer_method: str = "black_litterman",
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
    Run portfolio optimization with the selected strategy.

    Shared pipeline:
        1. Build universe (target + peers + sector ETF + SPY)
        2. Compute shrunk covariance matrix
        3. Dispatch to selected strategy
        4. Compute analytics (Sharpe, Sortino, factor betas, MCTR)

    Returns:
        OptimizationResult on success, OptimizerFallback on any failure.
    """
    from optimizer.strategies import STRATEGY_DISPLAY_NAMES, get_strategy

    try:
        # 1. Build universe (shared)
        universe, prices, market_caps = build_universe(
            ticker, sector, max_peers=max_peers, lookback=lookback
        )

        # 2. Covariance estimation (shared)
        cov_matrix = compute_covariance(prices, method=cov_method)

        # 3. Dispatch to strategy
        strategy = get_strategy(optimizer_method)

        strategy_result = strategy.optimize(
            universe=universe,
            prices=prices,
            cov_matrix=cov_matrix,
            market_caps=market_caps,
            ticker=ticker,
            risk_free_rate=risk_free_rate,
            bull_case=bull_case,
            bear_case=bear_case,
            macro_view=macro_view,
            committee_memo=committee_memo,
            risk_aversion=risk_aversion,
            tau=tau,
        )

        cleaned_weights = strategy_result["weights"]
        strategy_metadata = strategy_result.get("metadata", {})
        strategy_expected_returns = strategy_result.get("expected_returns")

        # Extract target weight
        target_weight = cleaned_weights.get(ticker, 0.0)

        # Get all weights as array (aligned with universe order)
        weight_arr = np.array([cleaned_weights.get(t, 0.0) for t in universe])

        # 4. Analytics (shared across all strategies)
        daily_returns = prices[ticker].pct_change().dropna()

        # Expected return for Sharpe/Sortino computation
        if strategy_expected_returns and ticker in strategy_expected_returns:
            exp_ret_target = strategy_expected_returns[ticker]
        else:
            exp_ret_target = float(daily_returns.mean() * 252)

        # Risk ratios
        sharpe, sortino, ann_vol, down_vol = compute_risk_ratios(
            daily_returns, exp_ret_target, rf=risk_free_rate
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

        display_name = STRATEGY_DISPLAY_NAMES.get(optimizer_method, optimizer_method)

        result = OptimizationResult(
            success=True,
            ticker=ticker,
            optimizer_method=optimizer_method,
            optimizer_display_name=display_name,
            optimal_weight=round(target_weight, 6),
            optimal_weight_pct=f"{target_weight * 100:.1f}%",
            bl_expected_return=strategy_metadata.get("bl_expected_return"),
            equilibrium_return=strategy_metadata.get("equilibrium_return"),
            expected_return=round(exp_ret_target, 4),
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
            tau=tau if optimizer_method == "black_litterman" else 0.0,
            risk_aversion=risk_aversion if optimizer_method == "black_litterman" else 0.0,
        )

        logger.info(
            f"{display_name} optimizer: {ticker} weight={target_weight:.1%}, "
            f"Sharpe={sharpe:.2f}"
        )

        return result

    except Exception as e:
        logger.error(
            f"Optimizer ({optimizer_method}) failed for {ticker}: {e}",
            exc_info=True,
        )
        return OptimizerFallback(
            success=False,
            error_message=f"Optimizer failed: {str(e)}",
            ticker=ticker,
        )


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
    """Backward-compatible wrapper: delegates to run_optimization with method='black_litterman'."""
    return run_optimization(
        ticker=ticker,
        sector=sector,
        optimizer_method="black_litterman",
        bull_case=bull_case,
        bear_case=bear_case,
        macro_view=macro_view,
        committee_memo=committee_memo,
        risk_free_rate=risk_free_rate,
        risk_aversion=risk_aversion,
        tau=tau,
        max_peers=max_peers,
        lookback=lookback,
        cov_method=cov_method,
    )
