"""
API request/response models for the FastAPI endpoint.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    """Request body for running an IC analysis."""

    ticker: str = Field(..., description="Stock ticker symbol (e.g. NVDA, COST)")
    provider: str = Field(
        default="anthropic",
        description="LLM provider: anthropic, google, openai, deepseek, huggingface, ollama",
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Specific model name (overrides provider default)",
    )
    user_context: str = Field(
        default="",
        description="Additional context or investment thesis to consider",
    )
    debate_rounds: int = Field(default=2, ge=1, le=20)


class SignalResponse(BaseModel):
    """Summarized signal response from an IC run."""

    ticker: str
    recommendation: str
    conviction: float
    t_signal: float
    position_direction: int
    bull_conviction: float
    bear_conviction: float
    macro_favorability: float
    thesis_summary: str = ""
    duration_s: float = 0.0
    provider: str = ""
    model_name: str = ""


class AnalysisResponse(BaseModel):
    """Full response from an IC analysis run."""

    success: bool = True
    error: str = ""
    signal: Optional[SignalResponse] = None
    signal_id: Optional[int] = None
    bull_case: Optional[dict[str, Any]] = None
    bear_case: Optional[dict[str, Any]] = None
    macro_view: Optional[dict[str, Any]] = None
    committee_memo: Optional[dict[str, Any]] = None
    optimization_result: Optional[dict[str, Any]] = None
    duration_s: float = 0.0


class BacktestResponse(BaseModel):
    """Response from running a backtest."""

    num_signals: int = 0
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    win_rate: float = 0.0
    direction_accuracy: float = 0.0
    max_drawdown: float = 0.0
    spy_return: float = 0.0
    excess_return_vs_spy: float = 0.0


class PortfolioResponse(BaseModel):
    """Response from building a portfolio snapshot."""

    tickers: list[str] = Field(default_factory=list)
    weights: dict[str, float] = Field(default_factory=dict)
    t_signals: dict[str, float] = Field(default_factory=dict)
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    num_longs: int = 0
    num_shorts: int = 0


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    version: str = ""
    num_signals: int = 0
    num_tickers: int = 0
