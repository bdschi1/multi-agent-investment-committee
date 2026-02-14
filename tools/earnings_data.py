"""
Earnings data tool — provider-agnostic.

Retrieves earnings history with surprise data — how often and
by how much a company beats or misses estimates. Consistent
beat patterns signal execution quality; misses signal risk.

Uses the active data provider via the shared abstraction layer.
Note: Earnings surprise data is primarily available via Yahoo Finance.
Bloomberg and IB providers will return limited results gracefully.
"""

from __future__ import annotations

import logging
from typing import Any

from tools.market_data import _get_default_provider

logger = logging.getLogger(__name__)


class EarningsDataTool:
    """Retrieves earnings history and surprise analysis."""

    @staticmethod
    def get_earnings_history(ticker: str) -> dict[str, Any]:
        """
        Get earnings history with beat/miss data.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dict with quarterly earnings, surprise stats, and trend assessment.
        """
        provider = _get_default_provider()

        try:
            earnings_df = provider.get_earnings_history(ticker)
            if earnings_df is None or (hasattr(earnings_df, "empty") and earnings_df.empty):
                # Fallback: try quarterly earnings
                return EarningsDataTool._fallback_earnings(ticker)

            quarters = []
            beats = 0
            misses = 0
            total_surprise_pct = 0.0

            for _, row in earnings_df.iterrows():
                estimate = row.get("epsEstimate")
                actual = row.get("epsActual")
                surprise = row.get("epsDifference")
                surprise_pct = row.get("surprisePercent")

                # Compute surprise if not provided
                if surprise_pct is None and estimate and actual and estimate != 0:
                    surprise_pct = ((actual - estimate) / abs(estimate)) * 100
                if surprise is None and estimate is not None and actual is not None:
                    surprise = actual - estimate

                # Track beats/misses
                if surprise is not None:
                    if surprise > 0:
                        beats += 1
                    elif surprise < 0:
                        misses += 1

                if surprise_pct is not None:
                    total_surprise_pct += float(surprise_pct)

                quarter_data = {
                    "date": str(row.get("quarter", row.name if hasattr(row, "name") else "")),
                    "eps_estimate": round(float(estimate), 3) if estimate is not None else None,
                    "eps_actual": round(float(actual), 3) if actual is not None else None,
                    "eps_difference": round(float(surprise), 3) if surprise is not None else None,
                    "surprise_pct": round(float(surprise_pct), 2) if surprise_pct is not None else None,
                    "result": "BEAT" if (surprise and surprise > 0) else "MISS" if (surprise and surprise < 0) else "INLINE",
                }
                quarters.append(quarter_data)

            total_reported = beats + misses
            avg_surprise = total_surprise_pct / len(quarters) if quarters else 0

            # Trend assessment
            if total_reported == 0:
                trend = "insufficient_data"
            elif beats >= total_reported * 0.75:
                trend = "consistent_beater"
            elif misses >= total_reported * 0.75:
                trend = "consistent_misser"
            elif beats > misses:
                trend = "mostly_beats"
            elif misses > beats:
                trend = "mostly_misses"
            else:
                trend = "mixed"

            return {
                "ticker": ticker,
                "quarters": quarters,
                "summary": {
                    "total_quarters": len(quarters),
                    "beats": beats,
                    "misses": misses,
                    "beat_rate_pct": round(beats / total_reported * 100, 1) if total_reported else 0,
                    "avg_surprise_pct": round(avg_surprise, 2),
                },
                "trend": trend,
            }

        except Exception as e:
            logger.error("Failed to get earnings data for %s: %s", ticker, e)
            return {
                "ticker": ticker,
                "quarters": [],
                "summary": {"total_quarters": 0},
                "trend": "unknown",
                "error": str(e),
            }

    @staticmethod
    def _fallback_earnings(ticker: str) -> dict[str, Any]:
        """Fallback: use quarterly earnings (revenue/earnings) if history unavailable."""
        provider = _get_default_provider()

        try:
            qe = provider.get_quarterly_earnings(ticker)
            if qe is None or (hasattr(qe, "empty") and qe.empty):
                return {
                    "ticker": ticker,
                    "quarters": [],
                    "summary": {"total_quarters": 0},
                    "trend": "no_data",
                    "note": f"No earnings history available (provider: {provider.name})",
                }

            quarters = []
            for idx, row in qe.iterrows():
                quarters.append({
                    "date": str(idx),
                    "revenue": float(row.get("Revenue", 0)),
                    "earnings": float(row.get("Earnings", 0)),
                })

            return {
                "ticker": ticker,
                "quarters": quarters,
                "summary": {"total_quarters": len(quarters)},
                "trend": "basic_data_only",
                "note": "Only revenue/earnings available, no surprise data",
            }
        except Exception:
            return {
                "ticker": ticker,
                "quarters": [],
                "summary": {"total_quarters": 0},
                "trend": "no_data",
            }
