"""XAI Pipeline — orchestrates all 5 steps of the XAI procedure.

Step 1: Estimate PFD (Altman Z-Score or XGBoost)
Step 2: SHAP explain distress risk factors
Step 3: Screen for financial distress
Step 4: Compute expected return = (1-PFD) * earnings_yield
Step 5: SHAP explain return/profitability factors
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from xai.distress import AltmanZScoreModel, get_distress_model
from xai.explainer import SHAPExplainer
from xai.features import extract_features, features_to_array
from xai.models import DistressAssessment, ReturnDecomposition, XAIResult
from xai.returns import compute_expected_return, screen_distress

logger = logging.getLogger(__name__)


class XAIPipeline:
    """Orchestrates the 5-step XAI analysis for a single ticker."""

    def __init__(
        self,
        distress_threshold: float = 0.5,
        artifact_path: str | None = None,
    ):
        self.distress_threshold = distress_threshold
        self.artifact_path = artifact_path

    def analyze(self, ticker: str, fundamentals: dict[str, Any]) -> XAIResult:
        """Run the full 5-step XAI analysis.

        Args:
            ticker: Stock ticker symbol.
            fundamentals: Dict from context["financial_metrics"].

        Returns:
            XAIResult with distress assessment, return decomposition, and narrative.
        """
        t0 = time.perf_counter()

        # Step 1: Extract features
        features = extract_features(fundamentals)
        x_arr, feature_names = features_to_array(features)

        # Step 2: Get distress model + estimate PFD
        model = get_distress_model(self.artifact_path)
        proba = model.predict_proba(x_arr)
        pfd = float(proba[0, 1])

        # Z-Score specific metadata
        z_score = None
        distress_zone = "unknown"
        if isinstance(model, AltmanZScoreModel):
            z_score = float(model.compute_z_score(x_arr))
            distress_zone = model.classify_zone(z_score)
        else:
            # For XGBoost, classify based on PFD
            if pfd < 0.2:
                distress_zone = "safe"
            elif pfd < 0.5:
                distress_zone = "grey"
            else:
                distress_zone = "distress"

        # Step 2b: SHAP explain distress
        explainer = SHAPExplainer(top_k=5)
        distress_shap = explainer.explain(model, x_arr, feature_names)

        # Build top risk factors from SHAP (positive SHAP = increases distress risk)
        top_risk_factors = distress_shap.top_positive

        distress = DistressAssessment(
            pfd=pfd,
            z_score=z_score,
            distress_zone=distress_zone,
            model_used=model.name,
            top_risk_factors=top_risk_factors,
            shap_base_value=distress_shap.base_value,
            distress_plot_base64=distress_shap.waterfall_plot_base64,
        )

        # Step 3: Distress screening
        screen = screen_distress(pfd, self.distress_threshold)

        # Step 4: Expected return
        ret_estimate = compute_expected_return(pfd, fundamentals)

        # Step 5: SHAP explain return drivers
        # Use the same model but focus on profitability-related explanations
        return_shap = distress_shap  # reuse — same model, same features
        # For returns, negative SHAP (reduces distress) = positive return driver
        top_return_factors = distress_shap.top_negative

        returns = ReturnDecomposition(
            is_distressed=screen.is_distressed,
            distress_flag=screen.flag,
            earnings_yield_proxy=ret_estimate.earnings_yield_proxy,
            expected_return=ret_estimate.expected_return,
            expected_return_pct=ret_estimate.expected_return_pct,
            top_return_factors=top_return_factors,
            shap_base_value=return_shap.base_value,
            return_plot_base64=return_shap.waterfall_plot_base64,
        )

        # Feature importance ranking (by absolute SHAP value)
        abs_shap = np.abs(distress_shap.shap_values)
        if abs_shap.size == len(feature_names):
            ranking_idx = np.argsort(-abs_shap)
            importance_ranking = [feature_names[i] for i in ranking_idx]
        else:
            importance_ranking = list(feature_names)

        # Generate narrative
        narrative = self._generate_narrative(ticker, distress, returns, features)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return XAIResult(
            ticker=ticker,
            distress=distress,
            returns=returns,
            features_used=features,
            feature_importance_ranking=importance_ranking,
            narrative=narrative,
            computation_time_ms=elapsed_ms,
        )

    def _generate_narrative(
        self,
        ticker: str,
        distress: DistressAssessment,
        returns: ReturnDecomposition,
        features: dict[str, float],
    ) -> str:
        """Generate human-readable summary for LLM agents."""
        parts = []

        # Distress summary
        z_str = f", Z={distress.z_score:.1f}" if distress.z_score is not None else ""
        parts.append(
            f"{ticker} XAI pre-screen: PFD={distress.pfd:.1%}{z_str}, "
            f"zone={distress.distress_zone} ({distress.model_used})."
        )

        # Top risk factor
        if distress.top_risk_factors:
            first = distress.top_risk_factors[0]
            factor_name = next(iter(first))
            factor_val = first[factor_name]
            parts.append(
                f"Top risk factor: {factor_name} (SHAP={factor_val:+.4f})."
            )

        # Expected return
        parts.append(
            f"Expected risk-adjusted return: {returns.expected_return_pct}."
        )

        # Top return driver
        if returns.top_return_factors:
            first = returns.top_return_factors[0]
            factor_name = next(iter(first))
            factor_val = first[factor_name]
            parts.append(
                f"Key return driver: {factor_name} (SHAP={factor_val:+.4f})."
            )

        # Screening flag
        if returns.is_distressed:
            parts.append("WARNING: Company flagged as financially distressed.")

        return " ".join(parts)
