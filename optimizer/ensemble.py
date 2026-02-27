"""
Ensemble portfolio optimizer.

Runs all 6 strategies on shared universe/covariance, computes cross-strategy
analytics, blended allocation, weight consensus, and layered interpretation.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from optimizer.analytics import compute_factor_betas, compute_mctr, compute_risk_ratios
from optimizer.covariance import compute_covariance
from optimizer.models import (
    DivergenceFlag,
    EnsembleResult,
    OptimizationResult,
    OptimizerFallback,
    StrategyComparison,
    TickerConsensus,
)
from optimizer.strategies import STRATEGY_DISPLAY_NAMES, STRATEGY_REGISTRY, get_strategy
from optimizer.universe import SECTOR_ETF_MAP, build_universe

logger = logging.getLogger(__name__)

# Role-based ensemble weights (sum to 1.0)
DEFAULT_ENSEMBLE_WEIGHTS: dict[str, float] = {
    "black_litterman": 0.35,  # Core allocation (PM views-driven)
    "risk_parity": 0.25,      # Sanity check (risk-balanced)
    "min_variance": 0.20,     # Defensive overlay (lowest-vol)
    "hrp": 0.10,              # Reference (tree-based, robust)
    "mean_variance": 0.05,    # Reference (classic Markowitz)
    "equal_weight": 0.05,     # Reference (naive baseline)
}

STRATEGY_ROLES: dict[str, str] = {
    "black_litterman": "Core Allocation",
    "risk_parity": "Sanity Check",
    "min_variance": "Defensive Overlay",
    "hrp": "Reference (HRP)",
    "mean_variance": "Reference (MV)",
    "equal_weight": "Reference (1/N)",
}


def compute_hhi(weights: dict[str, float]) -> float:
    """Herfindahl-Hirschman Index — measures portfolio concentration."""
    vals = np.array([w for w in weights.values() if w > 1e-8])
    if len(vals) == 0:
        return 0.0
    return float(np.sum(vals ** 2))


def run_ensemble(
    ticker: str,
    sector: str,
    bull_case: Any = None,
    bear_case: Any = None,
    macro_view: Any = None,
    committee_memo: Any = None,
    risk_free_rate: float = 0.05,
    risk_aversion: float = 2.5,
    tau: float = 0.05,
    max_peers: int = 5,
    lookback: str = "2y",
    cov_method: str = "ledoit_wolf",
    ensemble_weights: dict[str, float] | None = None,
) -> EnsembleResult | OptimizerFallback:
    """
    Run all 6 strategies on shared universe/covariance, return EnsembleResult.

    Pipeline:
        1. Build universe (once)
        2. Compute covariance (once)
        3. Run each strategy (loop — cheap in-memory)
        4. Compute per-strategy analytics
        5. Build weight consensus matrix
        6. Compute blended allocation
        7. Identify divergence flags
        8. Generate layered interpretation narrative
    """
    blend_coeffs = ensemble_weights or DEFAULT_ENSEMBLE_WEIGHTS

    try:
        # 1. Build universe (once — expensive, yfinance API calls)
        universe, prices, market_caps = build_universe(
            ticker, sector, max_peers=max_peers, lookback=lookback
        )

        # 2. Covariance (once — matrix computation)
        cov_matrix = compute_covariance(prices, method=cov_method)
        cov_arr = (
            cov_matrix.values
            if hasattr(cov_matrix, "values")
            else np.array(cov_matrix)
        )

        # Shared analytics inputs
        daily_returns = prices[ticker].pct_change().dropna()
        factor_rets: dict[str, pd.Series] = {}
        spy_col = "SPY" if "SPY" in prices.columns else None
        sector_etf = SECTOR_ETF_MAP.get(sector)
        sect_col = (
            sector_etf
            if sector_etf and sector_etf in prices.columns
            else None
        )
        if spy_col:
            factor_rets["SPY"] = prices[spy_col].pct_change().dropna()
        if sect_col and sect_col != spy_col:
            factor_rets[sect_col] = prices[sect_col].pct_change().dropna()

        # 3. Run each strategy
        strategy_results: dict[str, OptimizationResult] = {}
        strategy_raw_weights: dict[str, dict[str, float]] = {}
        failed_strategies: list[str] = []

        for method_key in STRATEGY_REGISTRY:
            try:
                strategy = get_strategy(method_key)
                raw = strategy.optimize(
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

                cleaned_weights = raw["weights"]
                strategy_raw_weights[method_key] = cleaned_weights
                strategy_metadata = raw.get("metadata", {})
                strategy_expected_returns = raw.get("expected_returns")

                # Per-strategy analytics
                target_weight = cleaned_weights.get(ticker, 0.0)
                weight_arr = np.array(
                    [cleaned_weights.get(t, 0.0) for t in universe]
                )

                if strategy_expected_returns and ticker in strategy_expected_returns:
                    exp_ret = strategy_expected_returns[ticker]
                else:
                    exp_ret = float(daily_returns.mean() * 252)

                sharpe, sortino, ann_vol, down_vol = compute_risk_ratios(
                    daily_returns, exp_ret, rf=risk_free_rate
                )
                risk_contribs = compute_mctr(weight_arr, cov_arr, universe)
                port_var = float(weight_arr.T @ cov_arr @ weight_arr)
                port_vol = float(np.sqrt(max(port_var, 1e-16)))

                display_name = STRATEGY_DISPLAY_NAMES.get(method_key, method_key)

                result = OptimizationResult(
                    success=True,
                    ticker=ticker,
                    optimizer_method=method_key,
                    optimizer_display_name=display_name,
                    optimal_weight=round(target_weight, 6),
                    optimal_weight_pct=f"{target_weight * 100:.1f}%",
                    bl_expected_return=strategy_metadata.get("bl_expected_return"),
                    equilibrium_return=strategy_metadata.get("equilibrium_return"),
                    expected_return=round(exp_ret, 4),
                    computed_sharpe=round(sharpe, 2),
                    computed_sortino=round(sortino, 2),
                    annualized_vol=round(ann_vol, 4),
                    downside_vol=round(down_vol, 4),
                    factor_exposures=compute_factor_betas(daily_returns, factor_rets),
                    portfolio_vol=round(port_vol, 4),
                    risk_contributions=risk_contribs,
                    universe_tickers=universe,
                    universe_weights={
                        t: round(w, 6)
                        for t, w in cleaned_weights.items()
                        if w > 1e-6
                    },
                    covariance_method=cov_method,
                    lookback_days=len(prices),
                    risk_free_rate=risk_free_rate,
                    tau=tau if method_key == "black_litterman" else 0.0,
                    risk_aversion=(
                        risk_aversion if method_key == "black_litterman" else 0.0
                    ),
                )
                strategy_results[method_key] = result

            except Exception as e:
                logger.warning(f"Ensemble: {method_key} failed: {e}")
                failed_strategies.append(method_key)

        if not strategy_results:
            return OptimizerFallback(
                ticker=ticker,
                error_message="All strategies failed in ensemble mode",
            )

        # 5. Weight consensus matrix
        consensus = _build_consensus(universe, strategy_raw_weights, ticker)

        # 6. Blended allocation
        blended_weights = _compute_blended_weights(
            universe, strategy_raw_weights, blend_coeffs
        )

        # Blended analytics
        blend_arr = np.array([blended_weights.get(t, 0.0) for t in universe])
        blend_port_var = float(blend_arr.T @ cov_arr @ blend_arr)
        blend_port_vol = float(np.sqrt(max(blend_port_var, 1e-16)))
        blend_exp_ret = float(daily_returns.mean() * 252)
        blend_sharpe, blend_sortino, _, _ = compute_risk_ratios(
            daily_returns, blend_exp_ret, rf=risk_free_rate
        )
        blend_mctr = compute_mctr(blend_arr, cov_arr, universe)
        blend_hhi = compute_hhi(blended_weights)

        # 7. Divergence flags
        divergence = _compute_divergence(consensus)

        # 8. Strategy comparison table
        comparisons = _build_comparisons(strategy_results, strategy_raw_weights)

        # 9. Layered interpretation
        narrative = _build_layered_narrative(
            ticker, strategy_results, strategy_raw_weights, consensus
        )

        return EnsembleResult(
            success=True,
            ticker=ticker,
            strategy_results=strategy_results,
            strategy_comparisons=comparisons,
            consensus=consensus,
            blended_weights=blended_weights,
            blended_target_weight=round(blended_weights.get(ticker, 0.0), 6),
            blended_portfolio_vol=round(blend_port_vol, 4),
            blended_sharpe=round(blend_sharpe, 2),
            blended_sortino=round(blend_sortino, 2),
            blended_hhi=round(blend_hhi, 4),
            blended_risk_contributions=blend_mctr,
            ensemble_weights_used=blend_coeffs,
            divergence_flags=divergence,
            layered_narrative=narrative,
            universe_tickers=universe,
            failed_strategies=failed_strategies,
            covariance_method=cov_method,
            lookback_days=len(prices),
        )

    except Exception as e:
        logger.error(f"Ensemble optimizer failed: {e}", exc_info=True)
        return OptimizerFallback(
            success=False,
            error_message=f"Ensemble optimizer failed: {str(e)}",
            ticker=ticker,
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_consensus(
    universe: list[str],
    raw_weights: dict[str, dict[str, float]],
    target_ticker: str,
) -> list[TickerConsensus]:
    """Build per-ticker consensus across all strategies."""
    results = []
    methods = list(raw_weights.keys())
    for t in universe:
        weights_per_strategy = {m: raw_weights[m].get(t, 0.0) for m in methods}
        vals = list(weights_per_strategy.values())
        results.append(
            TickerConsensus(
                ticker=t,
                is_target=t == target_ticker,
                weights_by_strategy=weights_per_strategy,
                mean_weight=round(float(np.mean(vals)), 6),
                std_weight=round(float(np.std(vals)), 6),
                min_weight=round(float(np.min(vals)), 6),
                max_weight=round(float(np.max(vals)), 6),
            )
        )
    # Sort by std descending (most divergent first)
    results.sort(key=lambda x: x.std_weight, reverse=True)
    return results


def _compute_blended_weights(
    universe: list[str],
    raw_weights: dict[str, dict[str, float]],
    blend_coefficients: dict[str, float],
) -> dict[str, float]:
    """Weighted average of strategy weights using blend coefficients."""
    blended = {t: 0.0 for t in universe}
    total_coeff = 0.0

    for method_key, coeff in blend_coefficients.items():
        if method_key in raw_weights:
            for t in universe:
                blended[t] += coeff * raw_weights[method_key].get(t, 0.0)
            total_coeff += coeff

    # Normalize to sum to 1.0
    if total_coeff > 1e-8:
        for t in universe:
            blended[t] /= total_coeff

    return blended


def _compute_divergence(
    consensus: list[TickerConsensus],
) -> list[DivergenceFlag]:
    """Identify highest-agreement and highest-disagreement tickers."""
    flags = []
    for tc in consensus:
        if tc.std_weight < 0.02:
            flags.append(
                DivergenceFlag(
                    ticker=tc.ticker,
                    flag_type="high_agreement",
                    description=(
                        f"Low weight dispersion (std={tc.std_weight:.3f})"
                    ),
                    std_weight=tc.std_weight,
                )
            )
        elif tc.std_weight > 0.10:
            flags.append(
                DivergenceFlag(
                    ticker=tc.ticker,
                    flag_type="high_disagreement",
                    description=(
                        f"High weight dispersion (std={tc.std_weight:.3f})"
                    ),
                    std_weight=tc.std_weight,
                )
            )
    return flags


def _build_comparisons(
    results: dict[str, OptimizationResult],
    raw_weights: dict[str, dict[str, float]],
) -> list[StrategyComparison]:
    """Build comparison rows for the strategy table."""
    comparisons = []
    for key, opt in results.items():
        max_w = max(raw_weights[key].values()) if raw_weights.get(key) else 0.0
        comparisons.append(
            StrategyComparison(
                strategy_key=key,
                strategy_name=opt.optimizer_display_name,
                role=STRATEGY_ROLES.get(key, ""),
                target_weight=opt.optimal_weight,
                portfolio_vol=opt.portfolio_vol,
                sharpe=opt.computed_sharpe,
                sortino=opt.computed_sortino,
                max_single_weight=round(max_w, 4),
                hhi=compute_hhi(raw_weights.get(key, {})),
            )
        )
    return comparisons


def _build_layered_narrative(
    ticker: str,
    results: dict[str, OptimizationResult],
    raw_weights: dict[str, dict[str, float]],
    consensus: list[TickerConsensus],
) -> str:
    """Generate role-based narrative comparing BL, RP, MinVar."""
    lines = []

    bl = results.get("black_litterman")
    rp = results.get("risk_parity")
    mv = results.get("min_variance")

    if bl:
        lines.append(
            f"**Core Allocation (Black-Litterman):** "
            f"BL assigns {ticker} a {bl.optimal_weight_pct} weight based on PM views. "
            f"Portfolio vol: {bl.portfolio_vol * 100:.1f}%, "
            f"Sharpe: {bl.computed_sharpe:.2f}."
        )

    if rp:
        lines.append(
            f"**Sanity Check (Risk Parity):** "
            f"With no return assumptions, pure risk-balanced allocation gives {ticker} "
            f"{rp.optimal_weight_pct}. "
            f"Portfolio vol: {rp.portfolio_vol * 100:.1f}%."
        )

    if bl and rp:
        delta = bl.optimal_weight - rp.optimal_weight
        if abs(delta) > 0.05:
            direction = "overweights" if delta > 0 else "underweights"
            lines.append(
                f"BL {direction} {ticker} by {abs(delta) * 100:.1f}pp vs risk parity, "
                f"indicating the PM's views meaningfully tilt the allocation."
            )
        else:
            lines.append(
                f"BL and risk parity broadly agree on {ticker} "
                f"(delta {abs(delta) * 100:.1f}pp), "
                f"suggesting the allocation is risk-balanced."
            )

    if mv:
        lines.append(
            f"**Defensive Overlay (Min Variance):** "
            f"The lowest-vol portfolio assigns {ticker} {mv.optimal_weight_pct}. "
            f"Portfolio vol: {mv.portfolio_vol * 100:.1f}%."
        )

    # Target ticker consensus
    target_consensus = next((c for c in consensus if c.is_target), None)
    if target_consensus:
        lines.append(
            f"**Consensus:** Across all strategies, {ticker} averages "
            f"{target_consensus.mean_weight * 100:.1f}% weight "
            f"(std: {target_consensus.std_weight * 100:.1f}pp)."
        )

    return "\n\n".join(lines)
