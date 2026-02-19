"""
Black-Litterman portfolio optimizer.

Computes actual BL weights, risk metrics, and factor exposures from
the PM's conviction output. Runs as a LangGraph node between
run_portfolio_manager and finalize.
"""

from optimizer.models import OptimizationResult, OptimizerFallback
from optimizer.bl_optimizer import run_black_litterman

__all__ = ["run_black_litterman", "OptimizationResult", "OptimizerFallback"]
