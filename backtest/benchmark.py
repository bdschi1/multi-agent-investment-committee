"""
Benchmark comparison — evaluates IC signals vs naive strategies.

Compares the IC's directional signals against:
    1. SPY buy-and-hold
    2. Equal-weight buy-and-hold on all tickers
    3. Momentum (buy winners, short losers based on trailing returns)

Provides excess return and risk-adjusted comparisons.
"""

from __future__ import annotations

import logging
import statistics
from datetime import timedelta
from typing import Optional

import yfinance as yf

from backtest.database import SignalDatabase
from backtest.models import SignalRecord, BenchmarkComparison

logger = logging.getLogger(__name__)


class BenchmarkAnalyzer:
    """Compares IC signal performance against benchmark strategies."""

    def __init__(self, db: SignalDatabase | None = None):
        self.db = db or SignalDatabase()

    def run_comparison(
        self,
        ticker: str | None = None,
        horizon: str = "return_20d",
    ) -> list[BenchmarkComparison]:
        """
        Compare IC signals against benchmark strategies.

        Returns list of BenchmarkComparison results.
        """
        signals = self.db.get_signals(ticker=ticker, limit=10000)
        evaluated = [
            s for s in signals
            if getattr(s, horizon, None) is not None
            and s.position_direction != 0
        ]

        if not evaluated:
            return []

        # IC strategy P&L
        ic_pnls = [
            s.position_direction * getattr(s, horizon)
            for s in evaluated
        ]
        ic_metrics = _compute_strategy_metrics("IC Signal", ic_pnls)

        # Benchmark 1: Always long (buy-and-hold equivalent)
        long_pnls = [getattr(s, horizon) for s in evaluated]
        long_metrics = _compute_strategy_metrics("Always Long", long_pnls)

        # Benchmark 2: SPY over same dates
        spy_pnls = _compute_spy_returns(evaluated, horizon)
        spy_metrics = _compute_strategy_metrics("SPY Buy & Hold", spy_pnls) if spy_pnls else None

        # Benchmark 3: Momentum (long top-half T signals, short bottom-half)
        sorted_sigs = sorted(evaluated, key=lambda s: s.t_signal)
        mid = len(sorted_sigs) // 2
        mom_pnls = []
        for i, s in enumerate(sorted_sigs):
            ret = getattr(s, horizon)
            if i >= mid:  # top half → long
                mom_pnls.append(ret)
            else:  # bottom half → short
                mom_pnls.append(-ret)
        momentum_metrics = _compute_strategy_metrics("Momentum (T-Signal Ranked)", mom_pnls)

        results = [ic_metrics, long_metrics, momentum_metrics]
        if spy_metrics:
            results.append(spy_metrics)

        # Add excess return vs IC for each benchmark
        ic_total = ic_metrics.total_return
        for r in results:
            r.excess_return_vs_ic = r.total_return - ic_total

        return results

    def format_report(self, comparisons: list[BenchmarkComparison]) -> str:
        """Format benchmark comparison as a markdown report."""
        if not comparisons:
            return "## Benchmark Comparison\n\nNo signals with realized returns available."

        lines = [
            "## Benchmark Comparison",
            "",
            "| Strategy | Total Return | Sharpe | Max DD | Win Rate | Excess vs IC |",
            "|----------|-------------|--------|--------|----------|--------------|",
        ]

        for c in comparisons:
            bold = "**" if c.strategy_name == "IC Signal" else ""
            lines.append(
                f"| {bold}{c.strategy_name}{bold} | "
                f"{c.total_return:+.2%} | "
                f"{c.sharpe_ratio:.2f} | "
                f"{c.max_drawdown:.1%} | "
                f"{c.win_rate:.0%} | "
                f"{c.excess_return_vs_ic:+.2%} |"
            )

        lines.append("")

        # Find IC vs SPY spread
        ic = next((c for c in comparisons if c.strategy_name == "IC Signal"), None)
        spy = next((c for c in comparisons if "SPY" in c.strategy_name), None)
        if ic and spy:
            spread = ic.total_return - spy.total_return
            if spread > 0:
                lines.append(f"IC signals **outperformed** SPY by {spread:+.2%}.")
            else:
                lines.append(f"IC signals **underperformed** SPY by {spread:+.2%}.")

        return "\n".join(lines)


def _compute_strategy_metrics(name: str, pnls: list[float]) -> BenchmarkComparison:
    """Compute performance metrics for a strategy."""
    if not pnls:
        return BenchmarkComparison(strategy_name=name)

    total_return = sum(pnls)
    win_rate = sum(1 for p in pnls if p > 0) / len(pnls)

    # Sharpe
    if len(pnls) > 1:
        mean_pnl = statistics.mean(pnls)
        std_pnl = statistics.stdev(pnls)
        sharpe = mean_pnl / std_pnl if std_pnl > 0 else 0
    else:
        sharpe = 0

    # Max drawdown
    cumulative = 0
    peak = 0
    max_dd = 0
    for p in pnls:
        cumulative += p
        if cumulative > peak:
            peak = cumulative
        dd = (peak - cumulative) / max(peak, 0.001)
        if dd > max_dd:
            max_dd = dd

    return BenchmarkComparison(
        strategy_name=name,
        total_return=total_return,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        win_rate=win_rate,
    )


def _compute_spy_returns(
    signals: list[SignalRecord],
    horizon: str,
) -> list[float]:
    """Compute SPY returns at each signal date for apples-to-apples comparison."""
    horizon_days_map = {
        "return_1d": 1, "return_5d": 5, "return_10d": 10,
        "return_20d": 20, "return_60d": 60,
    }
    horizon_days = horizon_days_map.get(horizon, 20)

    if not signals:
        return []

    earliest = min(s.signal_date for s in signals)
    latest = max(s.signal_date for s in signals)
    end_date = latest.date() + timedelta(days=horizon_days + 10)

    try:
        df = yf.download("SPY", start=str(earliest.date()), end=str(end_date), progress=False)
        if df.empty:
            return []
        if hasattr(df.columns, 'levels') and len(df.columns.levels) > 1:
            df.columns = df.columns.get_level_values(0)
    except Exception:
        return []

    spy_returns = []
    for s in signals:
        sig_date = s.signal_date.date()
        # Find nearest trading day on or after signal date
        matching = df.index[df.index.date >= sig_date]
        if len(matching) == 0:
            spy_returns.append(0.0)
            continue

        start_idx = df.index.get_loc(matching[0])
        end_idx = start_idx + horizon_days

        if end_idx >= len(df):
            spy_returns.append(0.0)
            continue

        start_price = df["Close"].iloc[start_idx]
        end_price = df["Close"].iloc[end_idx]
        spy_returns.append(float((end_price - start_price) / start_price))

    return spy_returns
