"""
Alpha decay analysis — measures how IC signal value decays over time.

Computes information coefficient (IC) at multiple forward horizons
(1d, 5d, 10d, 20d, 60d) to identify the optimal holding period
and quantify how quickly the signal's edge dissipates.
"""

from __future__ import annotations

import logging
import math
import statistics

from backtest.database import SignalDatabase
from backtest.models import AlphaDecayPoint

logger = logging.getLogger(__name__)

_HORIZONS = {
    1: "return_1d",
    5: "return_5d",
    10: "return_10d",
    20: "return_20d",
    60: "return_60d",
}


class AlphaDecayAnalyzer:
    """Computes alpha decay curves from stored IC signals."""

    def __init__(self, db: SignalDatabase | None = None):
        self.db = db or SignalDatabase()

    def compute_decay_curve(
        self,
        ticker: str | None = None,
        provider: str | None = None,
    ) -> list[AlphaDecayPoint]:
        """
        Compute IC at each horizon to build the alpha decay curve.

        Returns list of AlphaDecayPoint sorted by horizon.
        """
        signals = self.db.get_signals(ticker=ticker, limit=10000)
        if provider:
            signals = [s for s in signals if s.provider == provider]

        points = []

        for days, col_name in sorted(_HORIZONS.items()):
            evaluated = [
                s for s in signals
                if getattr(s, col_name, None) is not None
            ]

            if len(evaluated) < 5:
                points.append(AlphaDecayPoint(
                    horizon_days=days,
                    num_signals=len(evaluated),
                ))
                continue

            t_signals = [s.t_signal for s in evaluated]
            returns = [getattr(s, col_name) for s in evaluated]

            ic = _rank_ic(t_signals, returns)

            # Directional P&L
            directional_pnls = []
            for s in evaluated:
                ret = getattr(s, col_name)
                direction = s.position_direction or (1 if s.t_signal > 0 else (-1 if s.t_signal < 0 else 0))
                directional_pnls.append(direction * ret if direction != 0 else 0.0)

            avg_return = statistics.mean(directional_pnls) if directional_pnls else 0
            hit_rate = (
                sum(1 for p in directional_pnls if p > 0) / len(directional_pnls)
                if directional_pnls else 0
            )

            # T-statistic for IC
            n = len(evaluated)
            t_stat = ic * math.sqrt(n - 2) / math.sqrt(1 - ic ** 2) if abs(ic) < 1 and n > 2 else 0

            points.append(AlphaDecayPoint(
                horizon_days=days,
                information_coefficient=ic,
                avg_return=avg_return,
                hit_rate=hit_rate,
                t_statistic=t_stat,
                num_signals=n,
            ))

        return points

    def find_optimal_horizon(
        self,
        decay_curve: list[AlphaDecayPoint],
    ) -> int | None:
        """Find the horizon with the highest IC (optimal holding period)."""
        valid = [p for p in decay_curve if p.num_signals >= 5]
        if not valid:
            return None
        best = max(valid, key=lambda p: abs(p.information_coefficient))
        return best.horizon_days

    def format_report(
        self,
        decay_curve: list[AlphaDecayPoint],
        optimal_horizon: int | None = None,
    ) -> str:
        """Format alpha decay curve as a markdown report."""
        lines = [
            "## Alpha Decay Curve",
            "",
            "| Horizon | IC | Avg Return | Hit Rate | T-Stat | Signals |",
            "|---------|-----|------------|----------|--------|---------|",
        ]

        for p in decay_curve:
            if p.num_signals < 5:
                lines.append(
                    f"| {p.horizon_days}d | — | — | — | — | {p.num_signals} |"
                )
                continue

            sig = "**" if p.horizon_days == optimal_horizon else ""
            lines.append(
                f"| {sig}{p.horizon_days}d{sig} | "
                f"{p.information_coefficient:+.3f} | "
                f"{p.avg_return:+.2%} | "
                f"{p.hit_rate:.0%} | "
                f"{p.t_statistic:.2f} | "
                f"{p.num_signals} |"
            )

        lines.append("")
        if optimal_horizon:
            lines.append(f"**Optimal holding period: {optimal_horizon} days**")
            lines.append("")

        # Interpretation
        if decay_curve:
            best = max(
                (p for p in decay_curve if p.num_signals >= 5),
                key=lambda p: abs(p.information_coefficient),
                default=None,
            )
            if best:
                if abs(best.information_coefficient) > 0.15:
                    lines.append(
                        f"Signal has **strong** predictive power at {best.horizon_days}d "
                        f"(IC={best.information_coefficient:+.3f})."
                    )
                elif abs(best.information_coefficient) > 0.05:
                    lines.append(
                        f"Signal has **moderate** predictive power at {best.horizon_days}d "
                        f"(IC={best.information_coefficient:+.3f})."
                    )
                else:
                    lines.append("Signal shows **limited** predictive power across all horizons.")

        return "\n".join(lines)


def _rank_ic(predictions: list[float], actuals: list[float]) -> float:
    """Compute Spearman rank IC."""
    n = len(predictions)
    if n < 3:
        return 0.0

    def _rank(vals):
        sorted_idx = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        for rank, idx in enumerate(sorted_idx):
            ranks[idx] = rank + 1
        return ranks

    r_pred = _rank(predictions)
    r_act = _rank(actuals)

    mean_p = sum(r_pred) / n
    mean_a = sum(r_act) / n

    cov = sum((r_pred[i] - mean_p) * (r_act[i] - mean_a) for i in range(n))
    std_p = sum((r - mean_p) ** 2 for r in r_pred) ** 0.5
    std_a = sum((r - mean_a) ** 2 for r in r_act) ** 0.5

    if std_p == 0 or std_a == 0:
        return 0.0
    return cov / (std_p * std_a)
