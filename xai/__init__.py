"""
Explainable AI (XAI) module for investment analysis.

Implements a five-step XAI procedure combining distress estimation
(Altman Z-Score or XGBoost) with Shapley value explanations for
investment decision-making. Runs as a LangGraph node between
gather_data and the analyst fan-out, injecting quantitative context
for agents.

Shapley value computation is always available:
    - With `shap` library: uses shap.Explainer (auto-selects backend)
    - Without `shap`: uses built-in calculators (xai/shapley.py)
      - Exact analytical Shapley for the Altman Z-Score (linear model)
      - Permutation-based approximate Shapley for XGBoost or custom models

Based on:
    Šotić, A. & Radovanović, R. (2024). "Explainable AI in Finance:
    A Five-Step XAI Procedure." Academia AI & Applications, 1(2).
    doi:10.20935/AcadAI8017
"""

from xai.models import DistressAssessment, ReturnDecomposition, XAIFallback, XAIResult
from xai.pipeline import XAIPipeline
from xai.shapley import ExactLinearShapley, PermutationShapley, compute_shapley_values

__all__ = [
    "XAIPipeline",
    "XAIResult",
    "XAIFallback",
    "DistressAssessment",
    "ReturnDecomposition",
    "ExactLinearShapley",
    "PermutationShapley",
    "compute_shapley_values",
]
