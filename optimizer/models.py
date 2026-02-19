"""Pydantic output models for the Black-Litterman optimizer."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FactorExposure(BaseModel):
    """Single factor regression result."""

    factor_name: str
    beta: float
    t_stat: float
    p_value: float


class RiskContribution(BaseModel):
    """Marginal contribution to risk for one asset."""

    ticker: str
    weight: float
    marginal_ctr: float  # dSigma_p / dw_i
    pct_contribution: float  # w_i * MCTR_i / sigma_p


class OptimizationResult(BaseModel):
    """Full output from the Black-Litterman optimizer."""

    success: bool = True
    error_message: str = ""
    ticker: str

    # BL outputs
    optimal_weight: float = Field(description="BL weight for target stock")
    optimal_weight_pct: str = Field(description="e.g. '12.3%'")
    bl_expected_return: float = Field(description="Posterior expected excess return")
    equilibrium_return: float = Field(description="Prior (CAPM) return before views")

    # Computed risk metrics
    computed_sharpe: float = 0.0
    computed_sortino: float = 0.0
    annualized_vol: float = 0.0
    downside_vol: float = 0.0

    # Factor betas (from regression)
    factor_exposures: list[FactorExposure] = Field(default_factory=list)

    # Risk decomposition
    portfolio_vol: float = 0.0
    risk_contributions: list[RiskContribution] = Field(default_factory=list)

    # Universe
    universe_tickers: list[str] = Field(default_factory=list)
    universe_weights: dict[str, float] = Field(default_factory=dict)

    # Metadata
    covariance_method: str = "ledoit_wolf"
    lookback_days: int = 504
    risk_free_rate: float = 0.05
    tau: float = 0.05
    risk_aversion: float = 2.5


class OptimizerFallback(BaseModel):
    """Returned when the optimizer fails gracefully."""

    success: bool = False
    error_message: str = "Optimizer did not produce results"
    ticker: str = ""
