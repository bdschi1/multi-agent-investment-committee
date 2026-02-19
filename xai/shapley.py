"""Built-in Shapley value calculators — no external dependencies.

Provides exact and approximate Shapley value computation so XAI
explanations work even when the `shap` library is not installed.

Two strategies:
    1. ExactLinearShapley: Analytical solution for the Altman Z-Score
       (linear model). φ_i = w_i * (x_i - E[x_i]) with chain-rule
       correction through the sigmoid for PFD.
    2. PermutationShapley: Sampling-based approximation for any model
       with a predict_proba(x) interface. Uses ~200 random permutations
       (sufficient for 12 features).

Reference:
    Šotić, A. & Radovanović, R. (2024). "Explainable AI in Finance:
    A Five-Step XAI Procedure." Academia AI & Applications, 1(2).
    doi:10.20935/AcadAI8017
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ShapleyResult:
    """Container for Shapley value computation results."""

    values: np.ndarray = field(default_factory=lambda: np.array([]))
    base_value: float = 0.0
    feature_names: list[str] = field(default_factory=list)
    method: str = ""


class ExactLinearShapley:
    """Exact Shapley values for the Altman Z-Score linear model.

    For a linear model f(x) = w'g(x) where g() applies per-feature
    transforms, the Shapley value of feature i is exactly:

        φ_i = w_i * (g_i(x_i) - E[g_i(x_i)])

    For PFD = sigmoid(-(Z - c)), we apply the chain rule:

        φ_i^PFD = φ_i^Z * sigmoid'(Z) * (-1)

    where sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z)).
    """

    # Default feature means (baseline expectations for public equities)
    _DEFAULT_MEANS: dict[str, float] = {
        "current_ratio": 1.5,    # transform: clip(x-1, -0.5, 2.0)  → ~0.5
        "roe": 0.12,             # transform: clip(x, -1, 1)        → 0.12
        "operating_margin": 0.15,  # transform: clip(x, -1, 1)      → 0.15
        "debt_to_equity": 80.0,  # transform: clip(100/x, 0.01, 10) → 1.25
        "roa": 0.06,             # transform: clip(x, -0.5, 0.5)    → 0.06
    }

    # Altman weights
    _WEIGHTS = np.array([1.2, 1.4, 3.3, 0.6, 1.0])
    _SIGMOID_CENTER = 2.675

    # Map from Z-component index to FEATURE_NAMES index
    _Z_FEATURE_MAP = {0: 0, 1: 1, 2: 2, 3: 4, 4: 11}
    # Reverse: FEATURE_NAMES index → Z-component index
    _FEATURE_Z_MAP = {v: k for k, v in _Z_FEATURE_MAP.items()}

    def _transform_features(self, x_arr: np.ndarray) -> np.ndarray:
        """Apply Z-Score transforms to raw features → 5 Z-inputs."""
        z = np.zeros(5)
        z[0] = np.clip(x_arr[0] - 1.0, -0.5, 2.0)    # current_ratio
        z[1] = np.clip(x_arr[1], -1.0, 1.0)            # roe
        z[2] = np.clip(x_arr[2], -1.0, 1.0)            # operating_margin
        dte = np.clip(x_arr[4], 1.0, 1000.0)
        z[3] = np.clip(100.0 / dte, 0.01, 10.0)        # debt_to_equity
        z[4] = np.clip(x_arr[11], -0.5, 0.5)           # roa
        return z

    def _transform_baseline(self) -> np.ndarray:
        """Transform default means to Z-Score space."""
        baseline = np.zeros(5)
        baseline[0] = np.clip(self._DEFAULT_MEANS["current_ratio"] - 1.0, -0.5, 2.0)
        baseline[1] = np.clip(self._DEFAULT_MEANS["roe"], -1.0, 1.0)
        baseline[2] = np.clip(self._DEFAULT_MEANS["operating_margin"], -1.0, 1.0)
        dte = np.clip(self._DEFAULT_MEANS["debt_to_equity"], 1.0, 1000.0)
        baseline[3] = np.clip(100.0 / dte, 0.01, 10.0)
        baseline[4] = np.clip(self._DEFAULT_MEANS["roa"], -0.5, 0.5)
        return baseline

    def compute(
        self,
        features: np.ndarray,
        feature_names: list[str],
        target: str = "pfd",
    ) -> ShapleyResult:
        """Compute exact Shapley values for the Z-Score model.

        Args:
            features: 1D array of 12 features in FEATURE_NAMES order.
            feature_names: List of feature names.
            target: "z_score" for raw Z, "pfd" for probability of distress.

        Returns:
            ShapleyResult with Shapley values for all 12 features.
        """
        z_inputs = self._transform_features(features)
        z_baseline = self._transform_baseline()

        # Shapley values for Z-Score (exact for linear model)
        z_shapley = self._WEIGHTS * (z_inputs - z_baseline)

        if target == "z_score":
            # Map 5 Z-components back to 12 features
            full_shapley = np.zeros(len(feature_names))
            for z_idx, feat_idx in self._Z_FEATURE_MAP.items():
                full_shapley[feat_idx] = z_shapley[z_idx]

            z_score = float(np.dot(self._WEIGHTS, z_inputs))
            z_base = float(np.dot(self._WEIGHTS, z_baseline))

            return ShapleyResult(
                values=full_shapley,
                base_value=z_base,
                feature_names=feature_names,
                method="exact_linear_z_score",
            )

        # PFD target: apply chain rule through sigmoid
        z_score = float(np.dot(self._WEIGHTS, z_inputs))
        pfd = 1.0 / (1.0 + np.exp(z_score - self._SIGMOID_CENTER))
        sigmoid_deriv = pfd * (1.0 - pfd)

        # φ_i^PFD = φ_i^Z * (-sigmoid'(Z))
        # Negative because higher Z → lower PFD
        pfd_shapley_5 = z_shapley * (-sigmoid_deriv)

        # Map to full 12-feature vector
        full_shapley = np.zeros(len(feature_names))
        for z_idx, feat_idx in self._Z_FEATURE_MAP.items():
            full_shapley[feat_idx] = pfd_shapley_5[z_idx]

        # Base value: PFD at baseline features
        z_base = float(np.dot(self._WEIGHTS, z_baseline))
        pfd_base = 1.0 / (1.0 + np.exp(z_base - self._SIGMOID_CENTER))

        return ShapleyResult(
            values=full_shapley,
            base_value=pfd_base,
            feature_names=feature_names,
            method="exact_linear_pfd",
        )


class PermutationShapley:
    """Approximate Shapley values via random permutation sampling.

    Works with any model that has a predict_proba(x) → (n, 2) interface.
    Uses the marginal contribution definition of Shapley values:

        φ_i = E_π[ f(x_{S∪{i}}) - f(x_S) ]

    where π is a random permutation and S is the set of features
    preceding i in π. Features not in the coalition use baseline values.

    With 200 permutations over 12 features, approximation error is
    typically <5% of the exact value.
    """

    def __init__(self, n_permutations: int = 200, seed: int = 42):
        self.n_permutations = n_permutations
        self.rng = np.random.RandomState(seed)

    def compute(
        self,
        model,
        features: np.ndarray,
        feature_names: list[str],
        baseline: np.ndarray | None = None,
    ) -> ShapleyResult:
        """Compute approximate Shapley values via permutation sampling.

        Args:
            model: Object with predict_proba(x) → (n_samples, 2).
            features: 1D array of feature values.
            feature_names: Feature name list.
            baseline: 1D array of baseline/reference values.
                      If None, uses FEATURE_DEFAULTS.

        Returns:
            ShapleyResult with approximate Shapley values.
        """
        n_features = len(features)

        if baseline is None:
            from xai.features import FEATURE_DEFAULTS, FEATURE_NAMES
            baseline = np.array(
                [FEATURE_DEFAULTS[name] for name in FEATURE_NAMES],
                dtype=np.float64,
            )

        # Get base prediction (all baseline features)
        base_pred = float(model.predict_proba(baseline.reshape(1, -1))[0, 1])

        # Accumulate marginal contributions
        shapley_accum = np.zeros(n_features)

        for _ in range(self.n_permutations):
            perm = self.rng.permutation(n_features)
            # Build coalition incrementally
            x_current = baseline.copy()

            prev_pred = base_pred
            for feat_idx in perm:
                x_current[feat_idx] = features[feat_idx]
                curr_pred = float(model.predict_proba(x_current.reshape(1, -1))[0, 1])
                shapley_accum[feat_idx] += curr_pred - prev_pred
                prev_pred = curr_pred

        shapley_values = shapley_accum / self.n_permutations

        return ShapleyResult(
            values=shapley_values,
            base_value=base_pred,
            feature_names=feature_names,
            method=f"permutation_{self.n_permutations}",
        )


def compute_shapley_values(
    model,
    features: np.ndarray,
    feature_names: list[str],
    method: str = "auto",
) -> ShapleyResult:
    """Compute Shapley values using the best available method.

    Strategy:
        1. If method="exact" or model is AltmanZScoreModel → ExactLinearShapley
        2. If method="permutation" or any other model → PermutationShapley
        3. If method="auto" → exact for linear, permutation for others

    Args:
        model: Model with predict_proba interface.
        features: 1D feature array.
        feature_names: Feature name list.
        method: "auto", "exact", or "permutation".

    Returns:
        ShapleyResult with computed values.
    """
    from xai.distress import AltmanZScoreModel

    is_linear = isinstance(model, AltmanZScoreModel)

    if method == "exact" or (method == "auto" and is_linear):
        if not is_linear:
            logger.warning("Exact Shapley requested but model is not linear — falling back to permutation")
            calculator = PermutationShapley()
            return calculator.compute(model, features, feature_names)

        calculator = ExactLinearShapley()
        return calculator.compute(features, feature_names, target="pfd")

    # Permutation-based for any model
    calculator = PermutationShapley()
    return calculator.compute(model, features, feature_names)
