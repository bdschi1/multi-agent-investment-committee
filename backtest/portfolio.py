"""
Multi-asset portfolio construction — aggregates IC signals across tickers.

Takes the latest signal for each ticker, constructs a weighted portfolio
based on T signals and conviction scores, and computes portfolio-level
risk metrics (gross/net exposure, concentration, sector balance).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

import yfinance as yf

from backtest.database import SignalDatabase
from backtest.models import PortfolioSnapshot, SignalRecord

logger = logging.getLogger(__name__)


class MultiAssetPortfolio:
    """Constructs and tracks a paper portfolio from IC signals."""

    def __init__(self, db: SignalDatabase | None = None):
        self.db = db or SignalDatabase()

    def build_snapshot(
        self,
        max_positions: int = 20,
        min_conviction: float = 5.0,
        weight_method: str = "t_signal",
    ) -> PortfolioSnapshot:
        """
        Build a portfolio snapshot from the latest signal per ticker.

        Args:
            max_positions: Maximum number of positions
            min_conviction: Minimum conviction to include (0-10)
            weight_method: "t_signal" (weight by T), "equal" (equal weight), or "conviction"

        Returns:
            PortfolioSnapshot with weights, exposures, and risk metrics.
        """
        all_tickers = self.db.get_all_tickers()

        # Get latest signal per ticker
        latest_signals: dict[str, SignalRecord] = {}
        for ticker in all_tickers:
            sigs = self.db.get_signals(ticker=ticker, limit=1)
            if sigs:
                latest_signals[ticker] = sigs[0]

        # Filter by conviction threshold and non-zero direction
        eligible = {
            t: s for t, s in latest_signals.items()
            if s.conviction >= min_conviction
            and s.position_direction != 0
        }

        if not eligible:
            return PortfolioSnapshot(
                snapshot_date=datetime.now(UTC),
            )

        # Sort by abs(t_signal) descending, take top N
        sorted_tickers = sorted(
            eligible.keys(),
            key=lambda t: abs(eligible[t].t_signal),
            reverse=True,
        )[:max_positions]

        # Compute weights
        selected = {t: eligible[t] for t in sorted_tickers}
        weights = self._compute_weights(selected, weight_method)

        # Compute t_signals dict
        t_signals = {t: s.t_signal for t, s in selected.items()}

        # Exposure metrics
        long_weights = {t: w for t, w in weights.items() if w > 0}
        short_weights = {t: w for t, w in weights.items() if w < 0}
        gross_exposure = sum(abs(w) for w in weights.values())
        net_exposure = sum(weights.values())

        return PortfolioSnapshot(
            snapshot_date=datetime.now(UTC),
            tickers=sorted_tickers,
            weights=weights,
            t_signals=t_signals,
            gross_exposure=gross_exposure,
            net_exposure=net_exposure,
            num_longs=len(long_weights),
            num_shorts=len(short_weights),
        )

    def _compute_weights(
        self,
        signals: dict[str, SignalRecord],
        method: str,
    ) -> dict[str, float]:
        """Compute portfolio weights using the specified method."""
        if method == "equal":
            n = len(signals)
            return {
                t: s.position_direction / n
                for t, s in signals.items()
            }

        if method == "conviction":
            total = sum(s.conviction for s in signals.values())
            if total == 0:
                return {t: 0.0 for t in signals}
            return {
                t: s.position_direction * (s.conviction / total)
                for t, s in signals.items()
            }

        # Default: weight by abs(t_signal)
        total_t = sum(abs(s.t_signal) for s in signals.values())
        if total_t == 0:
            n = len(signals)
            return {t: s.position_direction / n for t, s in signals.items()}

        return {
            t: s.t_signal / total_t
            for t, s in signals.items()
        }

    def track_performance(
        self,
        snapshot: PortfolioSnapshot,
        horizon_days: int = 20,
    ) -> PortfolioSnapshot:
        """
        Compute realized portfolio return for a snapshot over a given horizon.

        Fetches actual returns for each ticker and computes weighted P&L.
        Returns updated snapshot with portfolio_return filled in.
        """
        if not snapshot.weights:
            return snapshot

        start_date = snapshot.snapshot_date.date()
        end_date = start_date + timedelta(days=horizon_days + 5)

        portfolio_return = 0.0
        for ticker, weight in snapshot.weights.items():
            try:
                df = yf.download(
                    ticker,
                    start=str(start_date),
                    end=str(end_date),
                    progress=False,
                )
                if df.empty:
                    continue
                if hasattr(df.columns, 'levels') and len(df.columns.levels) > 1:
                    df.columns = df.columns.get_level_values(0)
                prices = df["Close"]
                if len(prices) > horizon_days:
                    ret = (prices.iloc[horizon_days] - prices.iloc[0]) / prices.iloc[0]
                    portfolio_return += weight * float(ret)
            except Exception as e:
                logger.warning(f"Failed to fetch {ticker}: {e}")

        snapshot.portfolio_return = portfolio_return
        return snapshot

    def format_report(self, snapshot: PortfolioSnapshot) -> str:
        """Format portfolio snapshot as a markdown report."""
        lines = [
            "## Multi-Asset Portfolio",
            "",
            f"**Date:** {snapshot.snapshot_date.strftime('%Y-%m-%d %H:%M')}",
            f"**Positions:** {len(snapshot.tickers)} ({snapshot.num_longs} long, {snapshot.num_shorts} short)",
            f"**Gross Exposure:** {snapshot.gross_exposure:.1%}",
            f"**Net Exposure:** {snapshot.net_exposure:+.1%}",
            "",
        ]

        if snapshot.weights:
            lines.extend([
                "| Ticker | Weight | T Signal | Direction |",
                "|--------|--------|----------|-----------|",
            ])

            for ticker in snapshot.tickers:
                w = snapshot.weights.get(ticker, 0)
                t = snapshot.t_signals.get(ticker, 0)
                direction = "LONG" if w > 0 else "SHORT"
                lines.append(f"| {ticker} | {w:+.1%} | {t:+.3f} | {direction} |")

        if snapshot.portfolio_return != 0:
            lines.append("")
            lines.append(f"**Portfolio Return:** {snapshot.portfolio_return:+.2%}")

        return "\n".join(lines)
