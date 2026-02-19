"""Steps 2 & 5: SHAP-based explanations for distress and return models.

Lazy-imports shap library when available. When shap is not installed,
falls back to built-in Shapley value calculators (xai/shapley.py):
    - Exact analytical Shapley for the Altman Z-Score linear model
    - Permutation-based approximate Shapley for any other model

This ensures XAI explanations always produce meaningful feature
attributions, even in a minimal install without optional dependencies.
"""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

_SHAP_AVAILABLE = None


def _check_shap() -> bool:
    """Lazy check for shap availability."""
    global _SHAP_AVAILABLE
    if _SHAP_AVAILABLE is None:
        try:
            import shap  # noqa: F401
            _SHAP_AVAILABLE = True
        except ImportError:
            _SHAP_AVAILABLE = False
            logger.info("shap not installed — SHAP explanations will be empty")
    return _SHAP_AVAILABLE


@dataclass
class SHAPExplanation:
    """Container for SHAP explanation results."""

    shap_values: np.ndarray = field(default_factory=lambda: np.array([]))
    base_value: float = 0.0
    feature_names: list[str] = field(default_factory=list)
    top_positive: list[dict[str, float]] = field(default_factory=list)
    top_negative: list[dict[str, float]] = field(default_factory=list)
    waterfall_plot_base64: str = ""


class SHAPExplainer:
    """Compute SHAP values for any model with predict/predict_proba interface."""

    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    def explain(
        self,
        model,
        features: np.ndarray,
        feature_names: list[str],
    ) -> SHAPExplanation:
        """Compute SHAP values for a single prediction.

        Args:
            model: Object with predict_proba(X) method.
            features: 1D array of feature values.
            feature_names: List of feature names matching features array.

        Returns:
            SHAPExplanation with values and top drivers.
        """
        if not _check_shap():
            return self._builtin_shapley_fallback(model, features, feature_names)

        import shap

        x_arr = features.reshape(1, -1)

        try:
            # Use Explainer which auto-selects backend
            # For tree models it uses TreeExplainer, for others KernelExplainer
            explainer = shap.Explainer(model.predict_proba, x_arr)
            shap_result = explainer(x_arr)

            # For binary classification, use the distress class (index 1)
            if shap_result.values.ndim == 3:
                # Shape: (1, n_features, n_classes) → take class 1
                sv = shap_result.values[0, :, 1]
                bv = float(shap_result.base_values[0, 1])
            elif shap_result.values.ndim == 2:
                sv = shap_result.values[0]
                bv = float(np.atleast_1d(shap_result.base_values)[0])
            else:
                sv = shap_result.values
                bv = float(np.atleast_1d(shap_result.base_values)[0])

            top_pos, top_neg = self._extract_top_drivers(sv, feature_names)
            plot_b64 = self._generate_waterfall_plot(sv, bv, feature_names)

            return SHAPExplanation(
                shap_values=sv,
                base_value=bv,
                feature_names=feature_names,
                top_positive=top_pos,
                top_negative=top_neg,
                waterfall_plot_base64=plot_b64,
            )

        except Exception:
            logger.warning("SHAP explanation failed, returning empty", exc_info=True)
            return self._empty_explanation(feature_names)

    def _extract_top_drivers(
        self,
        shap_values: np.ndarray,
        feature_names: list[str],
    ) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
        """Extract top K positive and negative SHAP drivers."""
        pairs = list(zip(feature_names, shap_values, strict=True))

        positive = sorted(
            [(n, float(v)) for n, v in pairs if v > 0],
            key=lambda x: x[1],
            reverse=True,
        )[:self.top_k]

        negative = sorted(
            [(n, float(v)) for n, v in pairs if v < 0],
            key=lambda x: x[1],
        )[:self.top_k]

        return (
            [{n: v} for n, v in positive],
            [{n: v} for n, v in negative],
        )

    def _generate_waterfall_plot(
        self,
        shap_values: np.ndarray,
        base_value: float,
        feature_names: list[str],
    ) -> str:
        """Generate SHAP waterfall plot as base64-encoded PNG."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import shap

            explanation = shap.Explanation(
                values=shap_values,
                base_values=base_value,
                feature_names=feature_names,
            )

            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(explanation, max_display=10, show=False)

            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
            plt.close(fig)
            buf.seek(0)
            return base64.b64encode(buf.read()).decode("utf-8")

        except Exception:
            logger.debug("Waterfall plot generation failed", exc_info=True)
            return ""

    def _builtin_shapley_fallback(
        self,
        model,
        features: np.ndarray,
        feature_names: list[str],
    ) -> SHAPExplanation:
        """Compute Shapley values using built-in calculators (no shap library).

        Uses exact analytical Shapley for AltmanZScoreModel (linear),
        permutation-based approximation for any other model.
        """
        try:
            from xai.shapley import compute_shapley_values

            result = compute_shapley_values(model, features, feature_names)
            top_pos, top_neg = self._extract_top_drivers(result.values, feature_names)

            return SHAPExplanation(
                shap_values=result.values,
                base_value=result.base_value,
                feature_names=feature_names,
                top_positive=top_pos,
                top_negative=top_neg,
                waterfall_plot_base64="",  # no plot without shap/matplotlib
            )
        except Exception:
            logger.warning("Built-in Shapley fallback failed", exc_info=True)
            return self._empty_explanation(feature_names)

    def _empty_explanation(self, feature_names: list[str]) -> SHAPExplanation:
        """Return empty explanation when all methods fail."""
        return SHAPExplanation(
            shap_values=np.zeros(len(feature_names)),
            base_value=0.0,
            feature_names=feature_names,
            top_positive=[],
            top_negative=[],
            waterfall_plot_base64="",
        )
