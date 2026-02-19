"""Extract Black-Litterman views from IC output (PM conviction → P, Q, omega)."""

from __future__ import annotations

import re
import logging

import numpy as np

logger = logging.getLogger(__name__)


def extract_numeric_return(text: str) -> float | None:
    """
    Parse a return/alpha percentage from PM heuristic text.

    Examples:
        "45% — PM's validated alpha" → 0.45
        "+14% over 12m" → 0.14
        "-5%" → -0.05
        "N/A" → None
    """
    if not text:
        return None

    # Match patterns like +45%, -5%, 14%, 0.45
    match = re.search(r'([+-]?\d+(?:\.\d+)?)\s*%', text)
    if match:
        return float(match.group(1)) / 100.0

    # Try matching a raw decimal like 0.45
    match = re.search(r'([+-]?\d+\.\d+)', text)
    if match:
        val = float(match.group(1))
        if abs(val) < 5.0:  # likely already a decimal return
            return val

    return None


def extract_numeric_vol(text: str) -> float | None:
    """Parse a volatility percentage from text."""
    return extract_numeric_return(text)  # Same regex logic


def build_views(
    ticker: str,
    bull_case,
    bear_case,
    macro_view,
    memo,
    universe_tickers: list[str],
    current_price: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Build Black-Litterman view matrices from IC output.

    Returns:
        (P, Q, omega_scale) where:
        - P: (1, n) pick matrix — absolute view on target stock
        - Q: (1,) view vector — PM's expected excess return
        - omega_scale: confidence scaling factor for omega computation
          (actual omega = P @ (tau * Sigma) @ P.T / confidence_scale)
    """
    n = len(universe_tickers)
    target_idx = universe_tickers.index(ticker) if ticker in universe_tickers else 0

    # P matrix: absolute view on target stock
    P = np.zeros((1, n))
    P[0, target_idx] = 1.0

    # Q: extract PM's idiosyncratic return estimate
    q_val = None

    # Try PM memo first (most refined estimate)
    if memo:
        q_val = extract_numeric_return(getattr(memo, 'idio_return_estimate', ''))

    # Fall back to bull case idiosyncratic return
    if q_val is None and bull_case:
        q_val = extract_numeric_return(getattr(bull_case, 'idiosyncratic_return', ''))

    # Fall back to bull case forecasted total return
    if q_val is None and bull_case:
        q_val = extract_numeric_return(getattr(bull_case, 'forecasted_total_return', ''))

    # Last resort: derive from conviction score
    if q_val is None:
        conviction = getattr(memo, 'conviction', 5.0) if memo else 5.0
        # Map conviction 0-10 to return -0.20 to +0.40
        q_val = (conviction - 5.0) / 10.0 * 0.40 + 0.10

    Q = np.array([q_val])

    # Confidence scaling: maps bull/bear conviction spread to omega uncertainty
    bull_conviction = getattr(bull_case, 'conviction_score', 5.0) if bull_case else 5.0
    bear_conviction = getattr(bear_case, 'bearish_conviction', 5.0) if bear_case else 5.0

    # net_conviction: 0 to 1 (higher = more confident in the view)
    net_conviction = (bull_conviction - bear_conviction + 10.0) / 20.0
    net_conviction = max(0.0, min(1.0, net_conviction))

    # confidence_scale: higher = more confident = lower omega = stronger BL tilt
    confidence_scale = 0.1 + 0.9 * net_conviction

    logger.info(
        f"BL views: Q={q_val:.4f}, bull={bull_conviction:.1f}, "
        f"bear={bear_conviction:.1f}, confidence_scale={confidence_scale:.3f}"
    )

    return P, Q, confidence_scale
