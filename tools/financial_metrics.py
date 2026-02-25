"""
Financial metrics computation tool.

Computes derived financial metrics and peer comparisons
that go beyond raw data — showing the agents can do
quantitative reasoning, not just pass through API data.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class FinancialMetricsTool:
    """Computes derived financial metrics for investment analysis."""

    @staticmethod
    def compute_valuation_assessment(fundamentals: dict[str, Any]) -> dict[str, Any]:
        """
        Assess valuation based on multiple metrics.

        Returns a structured valuation view with relative assessments.
        """
        assessment = {
            "ticker": fundamentals.get("ticker", ""),
            "metrics": {},
            "flags": [],
            "overall_valuation": "unknown",
        }

        # P/E Assessment
        pe = fundamentals.get("pe_trailing")
        pe_fwd = fundamentals.get("pe_forward")
        if pe is not None:
            assessment["metrics"]["pe_trailing"] = pe
            if pe > 40:
                assessment["flags"].append("Very high trailing P/E (>40x)")
            elif pe > 25:
                assessment["flags"].append("Elevated trailing P/E (>25x)")
            elif pe < 10:
                assessment["flags"].append("Low trailing P/E (<10x) — value or distressed?")

        if pe_fwd is not None and pe is not None and pe_fwd > 0:
            assessment["metrics"]["pe_forward"] = pe_fwd
            if pe_fwd < pe:
                assessment["flags"].append(
                    f"Forward P/E ({pe_fwd:.1f}x) < trailing ({pe:.1f}x) — "
                    "earnings growth expected"
                )
            else:
                assessment["flags"].append(
                    f"Forward P/E ({pe_fwd:.1f}x) > trailing ({pe:.1f}x) — "
                    "earnings decline expected"
                )

        # PEG Assessment
        peg = fundamentals.get("peg_ratio")
        if peg is not None:
            assessment["metrics"]["peg_ratio"] = peg
            if peg < 1.0:
                assessment["flags"].append(f"PEG < 1.0 ({peg:.2f}) — potentially undervalued for growth")
            elif peg > 2.0:
                assessment["flags"].append(f"PEG > 2.0 ({peg:.2f}) — growth premium looks stretched")

        # EV/EBITDA Assessment
        ev_ebitda = fundamentals.get("ev_to_ebitda")
        if ev_ebitda is not None:
            assessment["metrics"]["ev_to_ebitda"] = ev_ebitda
            if ev_ebitda > 25:
                assessment["flags"].append(f"High EV/EBITDA ({ev_ebitda:.1f}x)")
            elif ev_ebitda < 8:
                assessment["flags"].append(f"Low EV/EBITDA ({ev_ebitda:.1f}x) — cheap or challenged?")

        # Profitability Assessment
        margin = fundamentals.get("profit_margin")
        if margin is not None:
            assessment["metrics"]["profit_margin"] = margin
            try:
                margin_val = float(margin.replace("%", ""))
                if margin_val > 20:
                    assessment["flags"].append(f"Strong profit margin ({margin})")
                elif margin_val < 5:
                    assessment["flags"].append(f"Thin profit margin ({margin})")
                elif margin_val < 0:
                    assessment["flags"].append(f"NEGATIVE profit margin ({margin}) — burning cash")
            except (ValueError, AttributeError):
                pass

        # Balance Sheet Assessment
        debt_equity = fundamentals.get("debt_to_equity")
        if debt_equity is not None:
            assessment["metrics"]["debt_to_equity"] = debt_equity
            if debt_equity > 200:
                assessment["flags"].append(f"High leverage: D/E ratio = {debt_equity:.0f}%")
            elif debt_equity < 30:
                assessment["flags"].append(f"Conservative balance sheet: D/E = {debt_equity:.0f}%")

        current = fundamentals.get("current_ratio")
        if current is not None:
            assessment["metrics"]["current_ratio"] = current
            if current < 1.0:
                assessment["flags"].append(
                    f"Current ratio < 1.0 ({current:.2f}) — potential liquidity risk"
                )

        # Growth Assessment
        rev_growth = fundamentals.get("revenue_growth")
        if rev_growth is not None:
            assessment["metrics"]["revenue_growth"] = rev_growth
            try:
                growth_val = float(rev_growth.replace("%", ""))
                if growth_val > 25:
                    assessment["flags"].append(f"Strong revenue growth ({rev_growth})")
                elif growth_val < 0:
                    assessment["flags"].append(f"Revenue DECLINING ({rev_growth})")
            except (ValueError, AttributeError):
                pass

        # Overall assessment
        bullish_flags = sum(
            1 for f in assessment["flags"]
            if any(w in f.lower() for w in ["strong", "low peg", "undervalued", "conservative"])
        )
        bearish_flags = sum(
            1 for f in assessment["flags"]
            if any(w in f.lower() for w in ["high", "negative", "declining", "burning", "risk", "stretched"])
        )

        if bullish_flags > bearish_flags + 1:
            assessment["overall_valuation"] = "attractive"
        elif bearish_flags > bullish_flags + 1:
            assessment["overall_valuation"] = "expensive_or_risky"
        else:
            assessment["overall_valuation"] = "mixed"

        return assessment

    @staticmethod
    def compute_quality_score(fundamentals: dict[str, Any]) -> dict[str, Any]:
        """
        Compute a simple quality score based on profitability and balance sheet.

        This is a heuristic — real funds use more sophisticated models —
        but it shows structured quantitative reasoning.
        """
        score = 0
        max_score = 0
        details = []

        def _parse_pct(val: str | None) -> float | None:
            if val is None:
                return None
            try:
                return float(str(val).replace("%", ""))
            except (ValueError, TypeError):
                return None

        # Profitability (up to 4 points)
        roe = _parse_pct(fundamentals.get("roe"))
        if roe is not None:
            max_score += 2
            if roe > 15:
                score += 2
                details.append(f"ROE {roe:.1f}% > 15%: +2")
            elif roe > 8:
                score += 1
                details.append(f"ROE {roe:.1f}% > 8%: +1")
            else:
                details.append(f"ROE {roe:.1f}% < 8%: +0")

        margin = _parse_pct(fundamentals.get("profit_margin"))
        if margin is not None:
            max_score += 2
            if margin > 15:
                score += 2
                details.append(f"Margin {margin:.1f}% > 15%: +2")
            elif margin > 5:
                score += 1
                details.append(f"Margin {margin:.1f}% > 5%: +1")
            else:
                details.append(f"Margin {margin:.1f}% < 5%: +0")

        # Growth (up to 2 points)
        rev_growth = _parse_pct(fundamentals.get("revenue_growth"))
        if rev_growth is not None:
            max_score += 2
            if rev_growth > 15:
                score += 2
                details.append(f"Revenue growth {rev_growth:.1f}% > 15%: +2")
            elif rev_growth > 5:
                score += 1
                details.append(f"Revenue growth {rev_growth:.1f}% > 5%: +1")
            else:
                details.append(f"Revenue growth {rev_growth:.1f}%: +0")

        # Balance sheet (up to 2 points)
        debt_eq = fundamentals.get("debt_to_equity")
        if debt_eq is not None:
            max_score += 2
            if debt_eq < 50:
                score += 2
                details.append(f"D/E {debt_eq:.0f}% < 50%: +2")
            elif debt_eq < 100:
                score += 1
                details.append(f"D/E {debt_eq:.0f}% < 100%: +1")
            else:
                details.append(f"D/E {debt_eq:.0f}% > 100%: +0")

        return {
            "quality_score": score,
            "max_score": max_score,
            "quality_pct": round(score / max_score * 100, 1) if max_score > 0 else 0,
            "quality_label": (
                "High Quality" if max_score > 0 and score / max_score > 0.7
                else "Medium Quality" if max_score > 0 and score / max_score > 0.4
                else "Low Quality"
            ),
            "details": details,
        }

    @staticmethod
    def compute_realized_vol(prices: np.ndarray | list[float]) -> dict[str, Any]:
        """Compute realized volatility metrics from a daily close price series.

        Returns multi-window annualized vol, downside vol, vol ratio,
        percentile rank, and vol regime classification.

        Args:
            prices: Daily close prices (most recent last), at least 30 data points.

        Returns:
            Dict with vol_10d, vol_30d, vol_60d, vol_90d (annualized %),
            downside_vol_30d, vol_ratio_10d_60d, vol_percentile_rank,
            vol_regime, and interpretation.
        """
        prices = np.asarray(prices, dtype=float)
        if len(prices) < 30:
            return {"error": "Need at least 30 daily prices for vol computation"}

        # Log returns
        log_returns = np.diff(np.log(prices))
        ann_factor = np.sqrt(252)

        result: dict[str, Any] = {}

        # Multi-window realized vol (annualized)
        for window, label in [(10, "vol_10d"), (30, "vol_30d"), (60, "vol_60d"), (90, "vol_90d")]:
            if len(log_returns) >= window:
                result[label] = round(float(np.std(log_returns[-window:], ddof=1) * ann_factor * 100), 2)
            else:
                result[label] = None

        # Downside vol (annualized) — std of negative returns only
        neg_returns_30 = log_returns[-30:][log_returns[-30:] < 0]
        if len(neg_returns_30) >= 3:
            result["downside_vol_30d"] = round(float(np.std(neg_returns_30, ddof=1) * ann_factor * 100), 2)
        else:
            result["downside_vol_30d"] = None

        # Vol ratio: 10d / 60d — mean-reversion signal
        if result.get("vol_10d") and result.get("vol_60d") and result["vol_60d"] > 0:
            result["vol_ratio_10d_60d"] = round(result["vol_10d"] / result["vol_60d"], 2)
        else:
            result["vol_ratio_10d_60d"] = None

        # Vol percentile rank: where 30d vol sits vs rolling 30d windows over trailing 1Y
        if len(log_returns) >= 60:
            rolling_vols = []
            for i in range(30, len(log_returns) + 1):
                window_returns = log_returns[i - 30:i]
                rolling_vols.append(float(np.std(window_returns, ddof=1) * ann_factor * 100))
            current_vol = rolling_vols[-1]
            rank = sum(1 for v in rolling_vols if v <= current_vol) / len(rolling_vols)
            result["vol_percentile_rank"] = round(rank * 100, 1)
        else:
            result["vol_percentile_rank"] = None

        # Vol regime classification (based on 30d annualized vol)
        vol_30 = result.get("vol_30d")
        if vol_30 is not None:
            if vol_30 < 12:
                result["vol_regime"] = "low"
            elif vol_30 < 18:
                result["vol_regime"] = "normal"
            elif vol_30 < 25:
                result["vol_regime"] = "elevated"
            else:
                result["vol_regime"] = "crisis"
        else:
            result["vol_regime"] = "unknown"

        # Interpretation
        result["interpretation"] = _interpret_realized_vol(result)

        return result


def _interpret_realized_vol(vol_data: dict[str, Any]) -> str:
    """Generate plain-English interpretation of realized vol for agents."""
    parts = []

    regime = vol_data.get("vol_regime", "unknown")
    vol_30 = vol_data.get("vol_30d")
    if vol_30 is not None:
        parts.append(f"30-day realized vol is {vol_30:.1f}% ({regime} regime).")

    ratio = vol_data.get("vol_ratio_10d_60d")
    if ratio is not None:
        if ratio > 1.3:
            parts.append(
                f"Vol ratio (10d/60d) is {ratio:.2f} — short-term vol spiking "
                f"vs longer-term average. Recent stress or event-driven move."
            )
        elif ratio < 0.7:
            parts.append(
                f"Vol ratio (10d/60d) is {ratio:.2f} — vol compression. "
                f"Mean reversion risk: compressed vol tends to expand."
            )
        else:
            parts.append(f"Vol ratio (10d/60d) is {ratio:.2f} — stable vol environment.")

    pctile = vol_data.get("vol_percentile_rank")
    if pctile is not None:
        if pctile > 80:
            parts.append(f"Vol at {pctile:.0f}th percentile — historically elevated.")
        elif pctile < 20:
            parts.append(f"Vol at {pctile:.0f}th percentile — historically suppressed.")
        else:
            parts.append(f"Vol at {pctile:.0f}th percentile vs trailing year.")

    downside = vol_data.get("downside_vol_30d")
    vol_30_val = vol_data.get("vol_30d")
    if downside is not None and vol_30_val is not None and vol_30_val > 0:
        asym = downside / vol_30_val
        if asym > 1.3:
            parts.append(
                f"Downside vol ({downside:.1f}%) significantly exceeds total vol "
                f"({vol_30_val:.1f}%) — left-tail risk is elevated. "
                f"Sortino will be worse than Sharpe suggests."
            )
        elif asym < 0.8:
            parts.append(
                f"Downside vol ({downside:.1f}%) is below total vol "
                f"({vol_30_val:.1f}%) — losses have been contained. "
                f"Sortino will be better than Sharpe suggests."
            )

    return " ".join(parts) if parts else "Insufficient data for vol interpretation."
