"""
Data models for the backtest and analytics system.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class SignalRecord(BaseModel):
    """A single IC signal stored in the database."""

    id: Optional[int] = None
    ticker: str
    signal_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    provider: str = ""
    model_name: str = ""

    # IC outputs
    recommendation: str = ""
    t_signal: float = 0.0
    conviction: float = 5.0
    position_direction: int = 0
    raw_confidence: float = 0.5
    bull_conviction: float = 5.0
    bear_conviction: float = 5.0
    macro_favorability: float = 5.0

    # BL optimizer outputs
    bl_optimal_weight: Optional[float] = None
    bl_sharpe: Optional[float] = None
    bl_sortino: Optional[float] = None

    # Heuristic outputs
    sharpe_heuristic: Optional[float] = None
    sortino_heuristic: Optional[float] = None

    # Realized returns (filled later by backtest)
    price_at_signal: Optional[float] = None
    return_1d: Optional[float] = None
    return_5d: Optional[float] = None
    return_10d: Optional[float] = None
    return_20d: Optional[float] = None
    return_60d: Optional[float] = None

    # Metadata
    duration_s: float = 0.0
    total_tokens: int = 0

    # Explainability
    bull_influence: Optional[float] = None
    bear_influence: Optional[float] = None
    macro_influence: Optional[float] = None
    debate_shift: Optional[float] = None


class PortfolioSnapshot(BaseModel):
    """Point-in-time snapshot of the paper portfolio."""

    id: Optional[int] = None
    snapshot_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tickers: list[str] = Field(default_factory=list)
    weights: dict[str, float] = Field(default_factory=dict)
    t_signals: dict[str, float] = Field(default_factory=dict)
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    portfolio_return: float = 0.0
    cumulative_return: float = 0.0
    drawdown: float = 0.0
    portfolio_sharpe: Optional[float] = None
    num_longs: int = 0
    num_shorts: int = 0


class BacktestResult(BaseModel):
    """Summary result of a backtest run."""

    id: Optional[int] = None
    run_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    start_date: str = ""
    end_date: str = ""
    tickers: list[str] = Field(default_factory=list)
    provider: str = ""
    num_signals: int = 0

    # Performance
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0

    # Calibration
    direction_accuracy: float = 0.0
    avg_conviction_when_right: float = 0.0
    avg_conviction_when_wrong: float = 0.0

    # Benchmark comparison
    spy_return: float = 0.0
    momentum_return: float = 0.0
    excess_return_vs_spy: float = 0.0

    # Alpha decay
    ic_1d: Optional[float] = None
    ic_5d: Optional[float] = None
    ic_20d: Optional[float] = None
    ic_60d: Optional[float] = None
    optimal_holding_period: Optional[int] = None


class CalibrationBucket(BaseModel):
    """One bucket in a calibration analysis."""

    conviction_range: str = ""
    min_conviction: float = 0.0
    max_conviction: float = 10.0
    num_signals: int = 0
    avg_conviction: float = 0.0
    avg_realized_return: float = 0.0
    hit_rate: float = 0.0
    avg_return_when_right: float = 0.0
    avg_return_when_wrong: float = 0.0


class AlphaDecayPoint(BaseModel):
    """Alpha at a specific holding period."""

    horizon_days: int = 0
    information_coefficient: float = 0.0
    avg_return: float = 0.0
    hit_rate: float = 0.0
    t_statistic: float = 0.0
    num_signals: int = 0


class BenchmarkComparison(BaseModel):
    """Comparison against a benchmark strategy."""

    strategy_name: str = ""
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    excess_return_vs_ic: float = 0.0


class AttributionResult(BaseModel):
    """Attribution of a single signal to agent contributions."""

    ticker: str = ""
    t_signal: float = 0.0
    bull_contribution: float = 0.0
    bear_contribution: float = 0.0
    macro_contribution: float = 0.0
    debate_contribution: float = 0.0
    dominant_agent: str = ""
    dominant_evidence: list[str] = Field(default_factory=list)
