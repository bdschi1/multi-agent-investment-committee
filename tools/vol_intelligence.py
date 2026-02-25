"""
Volatility intelligence — quantitative vol analysis for agent context injection.

Combines realized vol (from price history), implied vol surface (from Heston/CEV
models), and derived signals (IV vs HV, skew flags, regime classification) into
a single structured dict that agents consume as part of their context.

This module is the bridge between raw numerical methods (volatility_surface.py,
financial_metrics.py) and the agent reasoning loop. It pre-computes everything
agents need so they can reason quantitatively instead of guessing.

Signals produced:
    - Realized vol metrics (multi-window HV, downside vol, vol ratio, regime)
    - Implied vol surface snapshot (ATM term structure, skew by maturity)
    - IV vs HV premium/discount with sizing implication
    - Skew-based tail risk and squeeze flags
    - Vol regime sizing multiplier for BL confidence scaling
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def compute_vol_intelligence(
    ticker: str,
    prices: np.ndarray | list[float],
    spot: float | None = None,
    r: float = 0.04,
) -> dict[str, Any]:
    """Compute comprehensive vol intelligence for agent context injection.

    Args:
        ticker: Stock ticker symbol
        prices: Daily close prices (most recent last), at least 30 data points
        spot: Current spot price (default: last price in series)
        r: Risk-free rate (default 0.04)

    Returns:
        Dict with realized_vol, implied_vol, iv_vs_hv, skew_flags,
        vol_regime_sizing, and agent_summary.
    """
    prices = np.asarray(prices, dtype=float)
    if spot is None:
        spot = float(prices[-1])

    result: dict[str, Any] = {"ticker": ticker, "spot": spot}

    # --- 1. Realized vol ---
    from tools.financial_metrics import FinancialMetricsTool
    realized = FinancialMetricsTool.compute_realized_vol(prices)
    result["realized_vol"] = realized

    # --- 2. Implied vol surface (Heston model with calibrated params) ---
    implied = _compute_implied_surface(spot, r, realized)
    result["implied_vol"] = implied

    # --- 3. IV vs HV signal ---
    result["iv_vs_hv"] = _compute_iv_hv_signal(realized, implied)

    # --- 4. Skew-based flags (squeeze risk, tail risk) ---
    result["skew_flags"] = _compute_skew_flags(implied)

    # --- 5. Vol regime sizing multiplier ---
    result["vol_regime_sizing"] = _compute_regime_multiplier(realized)

    # --- 6. Agent-ready summary ---
    result["agent_summary"] = _build_agent_summary(result)

    return result


def _compute_implied_surface(
    spot: float,
    r: float,
    realized: dict[str, Any],
) -> dict[str, Any]:
    """Generate implied vol surface with Heston params calibrated to realized vol."""
    from tools.volatility_surface import VolatilitySurfaceTool

    # Use realized vol to set sensible Heston starting params
    vol_30 = realized.get("vol_30d")
    if vol_30 is None:
        return {"error": "No realized vol available for Heston calibration"}

    # Convert percentage to variance
    sigma_decimal = vol_30 / 100.0
    v0 = sigma_decimal ** 2
    vbar = v0  # long-run variance = current variance (no mean-reversion assumption)

    # Standard Heston params with negative rho (leverage effect) and moderate vol-of-vol
    surface = VolatilitySurfaceTool.get_vol_surface(
        spot=spot,
        r=r,
        model="heston",
        kappa=1.5,
        gamma=0.5,
        vbar=vbar,
        v0=v0,
        rho=-0.7,
    )

    # Also get a 3-month smile for quick reference
    smile = VolatilitySurfaceTool.get_vol_smile(
        spot=spot,
        maturity=0.25,
        r=r,
        model="heston",
        kappa=1.5,
        gamma=0.5,
        vbar=vbar,
        v0=v0,
        rho=-0.7,
    )

    return {"surface": surface, "smile_3m": smile}


def _iv_hv_thresholds(realized: dict[str, Any]) -> tuple[float, float]:
    """Return (high_threshold, low_threshold) scaled by vol percentile.

    Low-vol names (pctile <20) get tighter bands (±3pp) because a 3pp
    premium on a 10% vol stock is 30% relative.  High-vol names (pctile >80)
    get wider bands (±8pp) because a 5pp premium on a 40% vol stock is
    only 12.5% relative and routine.

    Linearly interpolates between extremes so signals are comparable
    across the vol spectrum.
    """
    pctile = realized.get("vol_percentile_rank")
    if pctile is None:
        return 5.0, 2.0  # original defaults

    # Clamp to [0, 100]
    pctile = max(0.0, min(100.0, float(pctile)))

    # Linear interpolation: pctile 0 → tight (3, 1.2), pctile 100 → wide (8, 3.5)
    t = pctile / 100.0
    high = 3.0 + t * 5.0   # 3pp .. 8pp
    low = 1.2 + t * 2.3    # 1.2pp .. 3.5pp
    return round(high, 2), round(low, 2)


def _compute_iv_hv_signal(
    realized: dict[str, Any],
    implied: dict[str, Any],
) -> dict[str, Any]:
    """Compare implied vol to realized vol — the vol risk premium signal.

    Thresholds are scaled by the asset's own vol percentile rank so that
    a 5pp premium on a low-vol name is flagged as elevated, while the same
    5pp on a high-vol name registers as a slight premium.
    """
    hv_30 = realized.get("vol_30d")

    # Get ATM IV from surface
    surface = implied.get("surface", {})
    atm_ts = surface.get("atm_term_structure", {})

    # Use 3-month ATM IV as the benchmark
    iv_3m = atm_ts.get("0.250y") or atm_ts.get("0.083y")

    if hv_30 is None or iv_3m is None:
        return {"error": "Insufficient data for IV vs HV comparison"}

    premium = iv_3m - hv_30
    premium_pct = (premium / hv_30 * 100) if hv_30 > 0 else 0

    # Adaptive thresholds based on vol percentile
    high_th, low_th = _iv_hv_thresholds(realized)

    # Classification
    if premium > high_th:
        signal = "iv_elevated"
        description = (
            f"IV ({iv_3m:.1f}%) is {premium:.1f}pp above HV ({hv_30:.1f}%). "
            f"Market pricing more risk than realized. Options are expensive. "
            f"For longs: consider selling covered calls or waiting for IV crush. "
            f"For shorts: elevated IV makes put protection costly — size smaller."
        )
    elif premium > low_th:
        signal = "iv_slight_premium"
        description = (
            f"IV ({iv_3m:.1f}%) is modestly above HV ({hv_30:.1f}%). "
            f"Normal vol risk premium — no strong signal for positioning."
        )
    elif premium > -low_th:
        signal = "iv_fair"
        description = (
            f"IV ({iv_3m:.1f}%) is roughly in line with HV ({hv_30:.1f}%). "
            f"Vol risk premium is compressed — neutral signal."
        )
    elif premium > -high_th:
        signal = "iv_cheap"
        description = (
            f"IV ({iv_3m:.1f}%) is below HV ({hv_30:.1f}%). "
            f"Options are cheap relative to realized. "
            f"Good entry for hedges or protective structures."
        )
    else:
        signal = "iv_very_cheap"
        description = (
            f"IV ({iv_3m:.1f}%) is significantly below HV ({hv_30:.1f}%). "
            f"Unusual complacency — protection is underpriced. "
            f"Strong case for buying options as portfolio insurance."
        )

    return {
        "iv_3m_pct": iv_3m,
        "hv_30d_pct": hv_30,
        "premium_pp": round(premium, 2),
        "premium_pct": round(premium_pct, 1),
        "signal": signal,
        "description": description,
        "thresholds": {"high_pp": high_th, "low_pp": low_th},
    }


def _compute_skew_flags(implied: dict[str, Any]) -> dict[str, Any]:
    """Compute tail risk and squeeze flags from vol surface skew."""
    surface = implied.get("surface", {})
    skew_data = surface.get("skew_by_maturity", {})
    summary = surface.get("summary", {})

    avg_skew = summary.get("avg_skew_25d")
    if avg_skew is None and not skew_data:
        return {"error": "No skew data available"}

    flags = []

    # Analyze skew across tenors
    skew_values = []
    for tenor, metrics in skew_data.items():
        skew_25d = metrics.get("skew_25d", 0)
        skew_values.append(skew_25d)

        if skew_25d > 8:
            flags.append({
                "tenor": tenor,
                "type": "extreme_put_skew",
                "severity": "high",
                "message": (
                    f"{tenor} skew is {skew_25d:.1f}pp — extreme demand for downside "
                    f"protection. Tail risk heavily priced. Crash insurance is expensive."
                ),
            })
        elif skew_25d < -2:
            flags.append({
                "tenor": tenor,
                "type": "call_skew",
                "severity": "high",
                "message": (
                    f"{tenor} skew is inverted at {skew_25d:.1f}pp — call wing above "
                    f"put wing. Unusual: suggests demand for upside exposure. "
                    f"Elevated short squeeze risk."
                ),
            })

    # Aggregate squeeze risk assessment
    if avg_skew is not None:
        if avg_skew < 0:
            squeeze_risk = "elevated"
            squeeze_msg = (
                f"Avg skew is inverted ({avg_skew:.1f}pp) — market paying more for "
                f"upside than downside. Short positions face elevated squeeze risk."
            )
        elif avg_skew < 2:
            squeeze_risk = "moderate"
            squeeze_msg = (
                f"Avg skew is flat ({avg_skew:.1f}pp) — balanced demand for puts/calls. "
                f"No strong directional signal from options market."
            )
        elif avg_skew < 5:
            squeeze_risk = "low"
            squeeze_msg = (
                f"Avg skew is {avg_skew:.1f}pp — normal put demand. "
                f"Market pricing standard downside risk."
            )
        else:
            squeeze_risk = "very_low"
            squeeze_msg = (
                f"Avg skew is steep ({avg_skew:.1f}pp) — heavy put skew. "
                f"Market is fearful of downside. Short squeeze risk is low "
                f"but crash protection is expensive."
            )
    else:
        squeeze_risk = "unknown"
        squeeze_msg = "Insufficient skew data for squeeze assessment."

    # Term structure signal
    term_structure = summary.get("term_structure", "unknown")
    if term_structure == "backwardation":
        flags.append({
            "type": "vol_backwardation",
            "severity": "medium",
            "message": (
                "Vol term structure in backwardation — market pricing near-term "
                "event risk. Consider timing of any new positions around the event."
            ),
        })

    return {
        "avg_skew_25d": avg_skew,
        "squeeze_risk": squeeze_risk,
        "squeeze_assessment": squeeze_msg,
        "flags": flags,
        "term_structure": term_structure,
    }


def _compute_regime_multiplier(realized: dict[str, Any]) -> dict[str, Any]:
    """Compute vol regime sizing multiplier for BL confidence scaling.

    Returns a multiplier in [0.4, 1.2] that scales position sizes
    based on the current vol environment.
    """
    regime = realized.get("vol_regime", "unknown")
    vol_30 = realized.get("vol_30d")
    pctile = realized.get("vol_percentile_rank")

    multiplier_map = {
        "low": 1.2,       # Low vol: can take larger positions
        "normal": 1.0,    # Normal: standard sizing
        "elevated": 0.7,  # Elevated: reduce gross
        "crisis": 0.4,    # Crisis: defensive mode
    }
    multiplier = multiplier_map.get(regime, 1.0)

    # Adjust further by percentile
    if pctile is not None:
        if pctile > 90:
            multiplier *= 0.8  # Historically extreme vol: extra caution
        elif pctile < 10:
            multiplier *= 1.1  # Historically suppressed: can lean in slightly

    multiplier = round(max(0.3, min(1.3, multiplier)), 2)

    rationale_parts = [f"Vol regime: {regime}"]
    if vol_30 is not None:
        rationale_parts.append(f"(30d HV: {vol_30:.1f}%)")
    if pctile is not None:
        rationale_parts.append(f"at {pctile:.0f}th percentile")
    rationale_parts.append(f"=> sizing multiplier: {multiplier}x")

    return {
        "regime": regime,
        "multiplier": multiplier,
        "rationale": " ".join(rationale_parts),
    }


def _build_agent_summary(data: dict[str, Any]) -> str:
    """Build a concise agent-ready text summary of all vol intelligence."""
    parts = []
    ticker = data.get("ticker", "")

    # Realized vol headline
    rv = data.get("realized_vol", {})
    interp = rv.get("interpretation", "")
    if interp:
        parts.append(f"REALIZED VOL ({ticker}): {interp}")

    # IV vs HV headline
    ivhv = data.get("iv_vs_hv", {})
    desc = ivhv.get("description", "")
    if desc:
        parts.append(f"IV vs HV: {desc}")

    # Skew / squeeze headline
    skew = data.get("skew_flags", {})
    squeeze = skew.get("squeeze_assessment", "")
    if squeeze:
        parts.append(f"SKEW / SQUEEZE: {squeeze}")

    for flag in skew.get("flags", []):
        if flag.get("severity") == "high":
            parts.append(f"  FLAG: {flag['message']}")

    # Regime sizing
    regime = data.get("vol_regime_sizing", {})
    rationale = regime.get("rationale", "")
    if rationale:
        parts.append(f"SIZING: {rationale}")

    return "\n".join(parts) if parts else "Vol intelligence unavailable."
