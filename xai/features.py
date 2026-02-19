"""Feature extraction from fundamentals dict for XAI models.

Extracts 12 numeric features from the financial_metrics context dict,
handling percentage string parsing and missing value imputation.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Canonical feature ordering — must match model training
FEATURE_NAMES: list[str] = [
    "current_ratio",
    "roe",
    "operating_margin",
    "price_to_book",
    "debt_to_equity",
    "profit_margin",
    "gross_margin",
    "revenue_growth",
    "earnings_growth",
    "pe_trailing",
    "ev_to_ebitda",
    "roa",
]

# Sensible defaults for imputation (median-like for healthy public companies)
FEATURE_DEFAULTS: dict[str, float] = {
    "current_ratio": 1.5,
    "roe": 0.12,
    "operating_margin": 0.15,
    "price_to_book": 3.0,
    "debt_to_equity": 80.0,
    "profit_margin": 0.10,
    "gross_margin": 0.40,
    "revenue_growth": 0.05,
    "earnings_growth": 0.05,
    "pe_trailing": 20.0,
    "ev_to_ebitda": 15.0,
    "roa": 0.06,
}

# Features stored as percentage strings in fundamentals
_PCT_FIELDS = {
    "roe", "operating_margin", "profit_margin", "gross_margin",
    "revenue_growth", "earnings_growth", "roa",
}


def _parse_pct(value: Any) -> float | None:
    """Parse percentage string like '18.5%' to decimal 0.185, or pass through float."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        # If it's already a small decimal, return as-is
        # If it looks like a percentage (>1 and not a ratio field), convert
        return float(value)
    try:
        s = str(value).strip().replace("%", "").replace(",", "")
        v = float(s)
        # Values like "18.5" from "18.5%" should become 0.185
        return v / 100.0
    except (ValueError, TypeError):
        return None


def _parse_numeric(value: Any) -> float | None:
    """Parse a numeric value that may be string or None."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip().replace(",", ""))
    except (ValueError, TypeError):
        return None


def extract_features(fundamentals: dict[str, Any]) -> dict[str, float]:
    """Extract 12 numeric features from a fundamentals dict.

    Handles percentage string parsing for margin/growth/return fields
    and imputes missing values with sensible defaults.

    Args:
        fundamentals: Dict from context["financial_metrics"] (get_fundamentals output)

    Returns:
        Dict of feature_name -> float value for all 12 features.
    """
    features: dict[str, float] = {}

    for name in FEATURE_NAMES:
        raw = fundamentals.get(name)
        parsed = _parse_pct(raw) if name in _PCT_FIELDS else _parse_numeric(raw)

        if parsed is not None:
            features[name] = parsed
        else:
            features[name] = FEATURE_DEFAULTS[name]
            logger.debug("Feature '%s' missing, using default %.4f", name, FEATURE_DEFAULTS[name])

    return features


def features_to_array(features: dict[str, float]) -> tuple[np.ndarray, list[str]]:
    """Convert features dict to ordered numpy array.

    Args:
        features: Dict from extract_features()

    Returns:
        Tuple of (1D numpy array, list of feature names in same order).
    """
    values = [features.get(name, FEATURE_DEFAULTS[name]) for name in FEATURE_NAMES]
    return np.array(values, dtype=np.float64), list(FEATURE_NAMES)
