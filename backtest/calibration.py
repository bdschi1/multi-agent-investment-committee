"""
Calibration analysis — maps IC conviction scores to realized returns.

Answers the question: "When the IC says conviction 8/10, how does
the stock actually perform?" Bins signals by conviction level and
computes hit rates and average returns per bucket.
"""

from __future__ import annotations

import logging
import statistics

from backtest.database import SignalDatabase
from backtest.models import CalibrationBucket

logger = logging.getLogger(__name__)

# Default conviction bucket boundaries
_DEFAULT_BUCKETS = [
    (0, 3, "Low (0-3)"),
    (3, 5, "Below Avg (3-5)"),
    (5, 7, "Average (5-7)"),
    (7, 8.5, "High (7-8.5)"),
    (8.5, 10, "Very High (8.5-10)"),
]


class CalibrationAnalyzer:
    """Analyzes calibration between IC conviction and realized returns."""

    def __init__(self, db: SignalDatabase | None = None):
        self.db = db or SignalDatabase()

    def compute_calibration(
        self,
        horizon: str = "return_20d",
        ticker: str | None = None,
        buckets: list[tuple[float, float, str]] | None = None,
    ) -> list[CalibrationBucket]:
        """
        Compute calibration buckets mapping conviction to realized returns.

        Args:
            horizon: Which return horizon to use (e.g. "return_5d", "return_20d")
            ticker: Filter to a specific ticker (None = all)
            buckets: Custom bucket definitions [(min, max, label), ...]

        Returns:
            List of CalibrationBucket with stats per conviction range.
        """
        signals = self.db.get_signals(ticker=ticker, limit=10000)
        evaluated = [
            s for s in signals
            if getattr(s, horizon, None) is not None
            and s.position_direction != 0
        ]

        if not evaluated:
            return []

        bucket_defs = buckets or _DEFAULT_BUCKETS
        results = []

        for low, high, label in bucket_defs:
            bucket_signals = [
                s for s in evaluated
                if low <= s.conviction < high
                or (high == 10 and s.conviction == 10)
            ]

            if not bucket_signals:
                results.append(CalibrationBucket(
                    conviction_range=label,
                    min_conviction=low,
                    max_conviction=high,
                    num_signals=0,
                ))
                continue

            returns = [getattr(s, horizon) for s in bucket_signals]
            convictions = [s.conviction for s in bucket_signals]

            # Directional returns: long → positive return is right, short → negative is right
            directional_pnls = [
                s.position_direction * getattr(s, horizon)
                for s in bucket_signals
            ]

            right_returns = [p for p in directional_pnls if p > 0]
            wrong_returns = [p for p in directional_pnls if p <= 0]
            hit_rate = len(right_returns) / len(directional_pnls) if directional_pnls else 0

            results.append(CalibrationBucket(
                conviction_range=label,
                min_conviction=low,
                max_conviction=high,
                num_signals=len(bucket_signals),
                avg_conviction=statistics.mean(convictions),
                avg_realized_return=statistics.mean(directional_pnls),
                hit_rate=hit_rate,
                avg_return_when_right=(
                    statistics.mean(right_returns) if right_returns else 0
                ),
                avg_return_when_wrong=(
                    statistics.mean(wrong_returns) if wrong_returns else 0
                ),
            ))

        return results

    def compute_conviction_return_correlation(
        self,
        horizon: str = "return_20d",
        ticker: str | None = None,
    ) -> float:
        """
        Compute rank correlation between conviction and realized directional returns.

        Measures whether higher conviction actually predicts larger returns.
        """
        signals = self.db.get_signals(ticker=ticker, limit=10000)
        evaluated = [
            s for s in signals
            if getattr(s, horizon, None) is not None
            and s.position_direction != 0
        ]

        if len(evaluated) < 5:
            return 0.0

        convictions = [s.conviction for s in evaluated]
        pnls = [s.position_direction * getattr(s, horizon) for s in evaluated]

        return _spearman(convictions, pnls)

    def format_report(
        self,
        buckets: list[CalibrationBucket],
        correlation: float = 0.0,
    ) -> str:
        """Format calibration results as a markdown report."""
        lines = [
            "## Calibration Analysis",
            "",
            "| Conviction Range | Signals | Hit Rate | Avg Return | Avg Win | Avg Loss |",
            "|------------------|---------|----------|------------|---------|----------|",
        ]

        for b in buckets:
            if b.num_signals == 0:
                lines.append(f"| {b.conviction_range} | 0 | — | — | — | — |")
                continue
            lines.append(
                f"| {b.conviction_range} | {b.num_signals} | "
                f"{b.hit_rate:.0%} | {b.avg_realized_return:+.2%} | "
                f"{b.avg_return_when_right:+.2%} | {b.avg_return_when_wrong:+.2%} |"
            )

        lines.append("")
        lines.append(f"**Conviction-Return Rank Correlation:** {correlation:+.3f}")
        lines.append("")

        if correlation > 0.3:
            lines.append("*Calibration is GOOD — higher conviction predicts larger returns.*")
        elif correlation > 0.1:
            lines.append("*Calibration is MODERATE — some predictive power in conviction scores.*")
        elif correlation > -0.1:
            lines.append("*Calibration is WEAK — conviction scores have limited predictive value.*")
        else:
            lines.append("*Calibration is INVERTED — higher conviction is associated with worse returns. Review agent prompts.*")

        return "\n".join(lines)


def _spearman(x: list[float], y: list[float]) -> float:
    """Compute Spearman rank correlation."""
    n = len(x)
    if n < 3:
        return 0.0

    def _rank(vals):
        sorted_idx = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        for rank, idx in enumerate(sorted_idx):
            ranks[idx] = rank + 1
        return ranks

    rx = _rank(x)
    ry = _rank(y)

    mx = sum(rx) / n
    my = sum(ry) / n

    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    sx = sum((r - mx) ** 2 for r in rx) ** 0.5
    sy = sum((r - my) ** 2 for r in ry) ** 0.5

    if sx == 0 or sy == 0:
        return 0.0
    return cov / (sx * sy)
