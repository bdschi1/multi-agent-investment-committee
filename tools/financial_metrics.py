"""
Financial metrics computation tool.

Computes derived financial metrics and peer comparisons
that go beyond raw data — showing the agents can do
quantitative reasoning, not just pass through API data.
"""

from __future__ import annotations

import logging
from typing import Any

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
