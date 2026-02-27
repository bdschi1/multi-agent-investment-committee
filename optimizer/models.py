"""Pydantic output models for the portfolio optimizer."""

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
    """Full output from a portfolio optimizer."""

    success: bool = True
    error_message: str = ""
    ticker: str

    # Strategy identification
    optimizer_method: str = Field(default="black_litterman", description="Strategy key")
    optimizer_display_name: str = Field(
        default="Black-Litterman", description="Human-readable strategy name"
    )

    # Target weight (all strategies)
    optimal_weight: float = Field(default=0.0, description="Weight for target stock")
    optimal_weight_pct: str = Field(default="0.0%", description="e.g. '12.3%'")

    # BL-specific outputs (None for non-BL strategies)
    bl_expected_return: float | None = Field(
        default=None, description="BL posterior expected excess return"
    )
    equilibrium_return: float | None = Field(
        default=None, description="Prior (CAPM) return before views"
    )

    # Generic expected return (for any strategy that produces one)
    expected_return: float | None = Field(
        default=None, description="Strategy's expected return for target"
    )

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


# ---------------------------------------------------------------------------
# Ensemble models
# ---------------------------------------------------------------------------

class StrategyComparison(BaseModel):
    """One row in the cross-strategy comparison table."""

    strategy_key: str
    strategy_name: str
    role: str = ""
    target_weight: float = 0.0
    portfolio_vol: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_single_weight: float = 0.0
    hhi: float = 0.0


class TickerConsensus(BaseModel):
    """Weight consensus for a single ticker across all strategies."""

    ticker: str
    is_target: bool = False
    weights_by_strategy: dict[str, float] = Field(default_factory=dict)
    mean_weight: float = 0.0
    std_weight: float = 0.0
    min_weight: float = 0.0
    max_weight: float = 0.0


class DivergenceFlag(BaseModel):
    """Flags high agreement or disagreement on a ticker across strategies."""

    ticker: str
    flag_type: str  # "high_agreement" or "high_disagreement"
    description: str = ""
    std_weight: float = 0.0


class EnsembleResult(BaseModel):
    """Full output from the ensemble optimizer (all strategies)."""

    success: bool = True
    error_message: str = ""
    ticker: str

    # Per-strategy full results
    strategy_results: dict[str, OptimizationResult] = Field(default_factory=dict)

    # Comparison table data
    strategy_comparisons: list[StrategyComparison] = Field(default_factory=list)

    # Weight consensus (per ticker)
    consensus: list[TickerConsensus] = Field(default_factory=list)

    # Blended allocation
    blended_weights: dict[str, float] = Field(default_factory=dict)
    blended_target_weight: float = 0.0
    blended_portfolio_vol: float = 0.0
    blended_sharpe: float = 0.0
    blended_sortino: float = 0.0
    blended_hhi: float = 0.0
    blended_risk_contributions: list[RiskContribution] = Field(default_factory=list)
    ensemble_weights_used: dict[str, float] = Field(default_factory=dict)

    # Divergence
    divergence_flags: list[DivergenceFlag] = Field(default_factory=list)

    # Narrative
    layered_narrative: str = ""

    # Metadata
    universe_tickers: list[str] = Field(default_factory=list)
    failed_strategies: list[str] = Field(default_factory=list)
    covariance_method: str = "ledoit_wolf"
    lookback_days: int = 504
