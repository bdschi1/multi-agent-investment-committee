"""
Portfolio optimizer package.

Supports multiple optimization strategies: Black-Litterman (default),
Hierarchical Risk Parity, Mean-Variance, Minimum Variance, Risk Parity,
Equal Weight, and Ensemble (all strategies). All share universe construction,
covariance estimation, and post-optimization analytics.

Runs as a LangGraph node between run_portfolio_manager and finalize.
"""

from optimizer.bl_optimizer import run_black_litterman, run_optimization
from optimizer.ensemble import DEFAULT_ENSEMBLE_WEIGHTS, run_ensemble
from optimizer.models import (
    DivergenceFlag,
    EnsembleResult,
    OptimizationResult,
    OptimizerFallback,
    StrategyComparison,
    TickerConsensus,
)
from optimizer.strategies import (
    DISPLAY_TO_STRATEGY_KEY,
    STRATEGY_DISPLAY_NAMES,
    STRATEGY_REGISTRY,
    OptimizerStrategy,
    get_strategy,
)

__all__ = [
    "run_black_litterman",
    "run_optimization",
    "run_ensemble",
    "DEFAULT_ENSEMBLE_WEIGHTS",
    "OptimizationResult",
    "OptimizerFallback",
    "EnsembleResult",
    "StrategyComparison",
    "TickerConsensus",
    "DivergenceFlag",
    "OptimizerStrategy",
    "STRATEGY_REGISTRY",
    "STRATEGY_DISPLAY_NAMES",
    "DISPLAY_TO_STRATEGY_KEY",
    "get_strategy",
]
