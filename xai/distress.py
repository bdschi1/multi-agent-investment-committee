"""Step 1: Probability of Financial Distress (PFD) estimation.

Two-tier approach:
    - AltmanZScoreModel: Always available, uses proxy variables from fundamentals.
    - XGBoostDistressModel: Optional, loaded from trained artifact.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_ARTIFACT_PATH = Path("xai/artifacts/distress_model.joblib")


class AltmanZScoreModel:
    """Altman Z-Score model using fundamental feature proxies.

    Implements predict/predict_proba for SHAP compatibility.

    Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
    where:
        X1: working capital / total assets proxy → (current_ratio - 1) clipped
        X2: retained earnings proxy → ROE
        X3: EBIT / total assets proxy → operating_margin
        X4: market equity / total liabilities proxy → 1/debt_to_equity (scaled)
        X5: sales / total assets proxy → ROA
    """

    # Altman coefficients
    _WEIGHTS = np.array([1.2, 1.4, 3.3, 0.6, 1.0])

    # Indices into FEATURE_NAMES for the 5 Z-Score components
    # current_ratio(0), roe(1), operating_margin(2), debt_to_equity(4), roa(11)
    _FEATURE_INDICES = [0, 1, 2, 4, 11]

    # Zone thresholds
    SAFE_THRESHOLD = 2.99
    GREY_THRESHOLD = 1.81

    # Sigmoid midpoint for calibration (center of grey zone)
    _SIGMOID_CENTER = 2.675

    @property
    def name(self) -> str:
        return "altman_z_score"

    def _extract_z_inputs(self, x_arr: np.ndarray) -> np.ndarray:
        """Extract and transform the 5 Z-Score components from feature array.

        Args:
            x_arr: Shape (n_features,) or (n_samples, n_features)

        Returns:
            Shape (5,) or (n_samples, 5) array of transformed Z-Score inputs.
        """
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)

        z_inputs = np.zeros((x_arr.shape[0], 5))

        # X1: working capital proxy = max(current_ratio - 1, -0.5)
        z_inputs[:, 0] = np.clip(x_arr[:, 0] - 1.0, -0.5, 2.0)

        # X2: retained earnings proxy = ROE (already decimal)
        z_inputs[:, 1] = np.clip(x_arr[:, 1], -1.0, 1.0)

        # X3: EBIT/assets proxy = operating_margin
        z_inputs[:, 2] = np.clip(x_arr[:, 2], -1.0, 1.0)

        # X4: equity/liabilities proxy = 1/(debt_to_equity/100) with floor
        # debt_to_equity is in percentage points (e.g., 80 = 80%)
        dte = np.clip(x_arr[:, 4], 1.0, 1000.0)  # avoid div by zero
        z_inputs[:, 3] = np.clip(100.0 / dte, 0.01, 10.0)

        # X5: sales/assets proxy = ROA
        z_inputs[:, 4] = np.clip(x_arr[:, 11], -0.5, 0.5)

        return z_inputs.squeeze() if z_inputs.shape[0] == 1 else z_inputs

    def compute_z_score(self, x_arr: np.ndarray) -> float | np.ndarray:
        """Compute Altman Z-Score from feature array."""
        z_inputs = self._extract_z_inputs(x_arr)
        if z_inputs.ndim == 1:
            return float(np.dot(self._WEIGHTS, z_inputs))
        return z_inputs @ self._WEIGHTS

    def classify_zone(self, z_score: float) -> str:
        """Classify Z-Score into distress zone."""
        if z_score > self.SAFE_THRESHOLD:
            return "safe"
        if z_score >= self.GREY_THRESHOLD:
            return "grey"
        return "distress"

    def predict_proba(self, x_arr: np.ndarray) -> np.ndarray:
        """Predict PFD probability via sigmoid calibration.

        Args:
            x_arr: Shape (n_features,) or (n_samples, n_features)

        Returns:
            Shape (n_samples, 2) array of [P(not distressed), P(distressed)].
        """
        z = self.compute_z_score(x_arr)
        z_arr = np.atleast_1d(z)
        # Sigmoid: PFD = 1 / (1 + exp(Z - center))
        pfd = 1.0 / (1.0 + np.exp(z_arr - self._SIGMOID_CENTER))
        return np.column_stack([1.0 - pfd, pfd])

    def predict(self, x_arr: np.ndarray) -> np.ndarray:
        """Binary prediction (0=safe, 1=distressed) at threshold 0.5."""
        proba = self.predict_proba(x_arr)
        return (proba[:, 1] >= 0.5).astype(int)


class XGBoostDistressModel:
    """Optional XGBoost-based distress classifier loaded from artifact."""

    def __init__(self):
        self._model = None

    @property
    def name(self) -> str:
        return "xgboost"

    def load(self, path: str | Path) -> None:
        """Load a trained model from joblib file."""
        import joblib
        self._model = joblib.load(str(path))
        logger.info("Loaded XGBoost distress model from %s", path)

    def save(self, path: str | Path) -> None:
        """Save trained model to joblib file."""
        if self._model is None:
            raise ValueError("No model to save — call train() first")
        import joblib
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, str(path))
        logger.info("Saved XGBoost distress model to %s", path)

    def train(self, x_arr: np.ndarray, y: np.ndarray) -> dict:
        """Train XGBoost classifier on labeled data.

        Args:
            x_arr: Shape (n_samples, n_features) training features
            y: Shape (n_samples,) binary labels (0=healthy, 1=distressed)

        Returns:
            Dict with training metrics.
        """
        from xgboost import XGBClassifier

        self._model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
        )
        self._model.fit(x_arr, y)
        preds = self._model.predict(x_arr)
        accuracy = float(np.mean(preds == y))
        return {"accuracy": accuracy, "n_samples": len(y)}

    def predict_proba(self, x_arr: np.ndarray) -> np.ndarray:
        """Predict distress probability."""
        if self._model is None:
            raise ValueError("Model not loaded — call load() or train() first")
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)
        return self._model.predict_proba(x_arr)

    def predict(self, x_arr: np.ndarray) -> np.ndarray:
        """Binary prediction."""
        if self._model is None:
            raise ValueError("Model not loaded — call load() or train() first")
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)
        return self._model.predict(x_arr)


def get_distress_model(
    artifact_path: str | Path | None = None,
) -> AltmanZScoreModel | XGBoostDistressModel:
    """Factory: return XGBoost model if artifact exists, else Altman Z-Score.

    Args:
        artifact_path: Path to XGBoost model artifact. Defaults to xai/artifacts/distress_model.joblib.

    Returns:
        Model instance with predict/predict_proba interface.
    """
    path = Path(artifact_path) if artifact_path else _DEFAULT_ARTIFACT_PATH

    if path.exists():
        try:
            model = XGBoostDistressModel()
            model.load(path)
            return model
        except Exception:
            logger.warning("Failed to load XGBoost model from %s, falling back to Altman Z-Score", path)

    return AltmanZScoreModel()
