"""Pydantic output models for the XAI (Explainable AI) module."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DistressAssessment(BaseModel):
    """Step 1-2: Probability of Financial Distress + SHAP risk factors."""

    pfd: float = Field(description="Probability of Financial Distress [0, 1]")
    z_score: float | None = Field(default=None, description="Altman Z-Score (if Z-Score model)")
    distress_zone: str = Field(description="'safe' / 'grey' / 'distress'")
    model_used: str = Field(description="'altman_z_score' or 'xgboost'")
    top_risk_factors: list[dict[str, float]] = Field(
        default_factory=list, description="[{feature: shap_value}, ...]"
    )
    shap_base_value: float = 0.0
    distress_plot_base64: str = Field(default="", description="SHAP waterfall PNG (base64)")


class ReturnDecomposition(BaseModel):
    """Steps 3-5: Distress screening + expected return + SHAP profitability factors."""

    is_distressed: bool = False
    distress_flag: str = Field(default="", description="Screening result text")
    earnings_yield_proxy: float = Field(default=0.0, description="1/PE as proxy for P")
    expected_return: float = Field(default=0.0, description="ER = (1-PFD) * P")
    expected_return_pct: str = Field(default="0.0%", description="e.g. '8.5%'")
    top_return_factors: list[dict[str, float]] = Field(default_factory=list)
    shap_base_value: float = 0.0
    return_plot_base64: str = Field(default="", description="SHAP waterfall PNG (base64)")


class XAIResult(BaseModel):
    """Complete XAI analysis output for one ticker."""

    ticker: str
    distress: DistressAssessment
    returns: ReturnDecomposition
    features_used: dict[str, float] = Field(default_factory=dict)
    feature_importance_ranking: list[str] = Field(default_factory=list)
    narrative: str = Field(default="", description="Human-readable summary for agents")
    computation_time_ms: float = 0.0


class XAIFallback(BaseModel):
    """Returned when XAI analysis fails gracefully."""

    success: bool = False
    error_message: str = "XAI analysis did not produce results"
    ticker: str = ""
