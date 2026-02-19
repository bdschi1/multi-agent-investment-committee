"""
Historical backtest engine — runs stored IC signals against actual market returns.

Computes realized P&L for each signal by fetching forward-looking prices from
the signal date and comparing predicted direction vs actual return.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import yfinance as yf

from backtest.database import SignalDatabase
from backtest.models import SignalRecord, BacktestResult, PortfolioSnapshot

logger = logging.getLogger(__name__)

# Forward horizons to evaluate (trading days)
_HORIZONS = {
    "return_1d": 1,
    "return_5d": 5,
    "return_10d": 10,
    "return_20d": 20,
    "return_60d": 60,
}


def _fetch_forward_returns(
    ticker: str,
    signal_date: datetime,
    horizons: dict[str, int] | None = None,
) -> dict[str, float | None]:
    """
    Fetch realized returns for a ticker at each forward horizon from signal_date.

    Returns dict of {horizon_name: return_pct} or None if data unavailable.
    """
    horizons = horizons or _HORIZONS
    max_days = max(horizons.values()) + 10  # extra buffer for weekends/holidays

    start = signal_date.date()
    end = start + timedelta(days=max_days + 5)

    try:
        df = yf.download(ticker, start=str(start), end=str(end), progress=False)
        if df.empty or len(df) < 2:
            return {k: None for k in horizons}
    except Exception as e:
        logger.warning(f"Failed to fetch prices for {ticker}: {e}")
        return {k: None for k in horizons}

    # Handle MultiIndex columns from yfinance
    if hasattr(df.columns, 'levels') and len(df.columns.levels) > 1:
        df.columns = df.columns.get_level_values(0)

    prices = df["Close"].values
    base_price = prices[0]

    results = {}
    for name, days in horizons.items():
        if days < len(prices):
            results[name] = float((prices[days] - base_price) / base_price)
        else:
            results[name] = None

    return results


class BacktestRunner:
    """Runs historical backtests on stored IC signals."""

    def __init__(self, db: SignalDatabase | None = None):
        self.db = db or SignalDatabase()

    def fill_returns(self, limit: int = 500) -> int:
        """
        Fill in realized returns for signals that don't have them yet.

        Returns count of signals updated.
        """
        signals = self.db.get_signals(limit=limit)
        updated = 0

        for sig in signals:
            if sig.return_1d is not None:
                continue  # already filled

            returns = _fetch_forward_returns(sig.ticker, sig.signal_date)

            # Also get price at signal if missing
            update_data = {}
            for k, v in returns.items():
                if v is not None:
                    update_data[k] = v

            if sig.price_at_signal is None:
                try:
                    df = yf.download(
                        sig.ticker,
                        start=str(sig.signal_date.date()),
                        end=str(sig.signal_date.date() + timedelta(days=3)),
                        progress=False,
                    )
                    if not df.empty:
                        if hasattr(df.columns, 'levels') and len(df.columns.levels) > 1:
                            df.columns = df.columns.get_level_values(0)
                        update_data["price_at_signal"] = float(df["Close"].iloc[0])
                except Exception:
                    pass

            if update_data and sig.id is not None:
                self.db.update_returns(sig.id, **update_data)
                updated += 1

        return updated

    def run_backtest(
        self,
        ticker: str | None = None,
        provider: str | None = None,
        horizon: str = "return_20d",
    ) -> BacktestResult:
        """
        Run a backtest on stored signals and compute performance metrics.

        Args:
            ticker: Filter to a specific ticker (None = all tickers)
            provider: Filter to a specific provider (None = all)
            horizon: Which return horizon to use for P&L (default: 20d)
        """
        signals = self.db.get_signals(ticker=ticker, limit=10000)
        if provider:
            signals = [s for s in signals if s.provider == provider]

        # Filter to signals that have realized returns
        evaluated = [s for s in signals if getattr(s, horizon, None) is not None]

        if not evaluated:
            return BacktestResult(
                run_date=datetime.now(timezone.utc),
                tickers=list({s.ticker for s in signals}),
                provider=provider or "all",
                num_signals=0,
            )

        # Compute metrics
        returns = [getattr(s, horizon) for s in evaluated]
        directions = [s.position_direction for s in evaluated]
        convictions = [s.conviction for s in evaluated]
        t_signals = [s.t_signal for s in evaluated]

        # P&L: direction * return (long signals profit when returns are positive)
        pnls = []
        for sig in evaluated:
            ret = getattr(sig, horizon)
            direction = sig.position_direction or (1 if sig.t_signal > 0 else (-1 if sig.t_signal < 0 else 0))
            pnls.append(direction * ret if direction != 0 else 0.0)

        # Direction accuracy: did we predict the right sign?
        correct = sum(
            1 for sig in evaluated
            if getattr(sig, horizon, 0) is not None
            and sig.position_direction != 0
            and (sig.position_direction > 0) == (getattr(sig, horizon, 0) > 0)
        )
        total_directional = sum(1 for sig in evaluated if sig.position_direction != 0)

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        total_return = sum(pnls)
        win_rate = len(wins) / len(pnls) if pnls else 0
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf')
        direction_accuracy = correct / total_directional if total_directional > 0 else 0

        # Conviction calibration
        right_convictions = [
            sig.conviction for sig in evaluated
            if sig.position_direction != 0
            and getattr(sig, horizon, 0) is not None
            and (sig.position_direction > 0) == (getattr(sig, horizon, 0) > 0)
        ]
        wrong_convictions = [
            sig.conviction for sig in evaluated
            if sig.position_direction != 0
            and getattr(sig, horizon, 0) is not None
            and (sig.position_direction > 0) != (getattr(sig, horizon, 0) > 0)
        ]

        # Sharpe / Sortino (simple, not annualized — signals are irregular)
        import statistics
        if len(pnls) > 1:
            mean_pnl = statistics.mean(pnls)
            std_pnl = statistics.stdev(pnls)
            sharpe = mean_pnl / std_pnl if std_pnl > 0 else 0
            downside = [p for p in pnls if p < 0]
            downside_std = statistics.stdev(downside) if len(downside) > 1 else std_pnl
            sortino = mean_pnl / downside_std if downside_std > 0 else 0
        else:
            sharpe = sortino = 0

        # Max drawdown (cumulative P&L curve)
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

        # Information coefficients at each horizon
        ic_values = {}
        for h_name in ["return_1d", "return_5d", "return_20d", "return_60d"]:
            pairs = [
                (sig.t_signal, getattr(sig, h_name))
                for sig in evaluated
                if getattr(sig, h_name, None) is not None
            ]
            if len(pairs) >= 5:
                ic_values[h_name] = _rank_ic(
                    [p[0] for p in pairs],
                    [p[1] for p in pairs],
                )

        # SPY benchmark return over same period
        spy_return = _benchmark_return(evaluated, horizon)

        tickers_list = sorted(set(s.ticker for s in evaluated))

        return BacktestResult(
            run_date=datetime.now(timezone.utc),
            start_date=min(s.signal_date for s in evaluated).isoformat(),
            end_date=max(s.signal_date for s in evaluated).isoformat(),
            tickers=tickers_list,
            provider=provider or "all",
            num_signals=len(evaluated),
            total_return=total_return,
            annualized_return=0.0,  # irregular signals — not annualizable
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            direction_accuracy=direction_accuracy,
            avg_conviction_when_right=(
                statistics.mean(right_convictions) if right_convictions else 0
            ),
            avg_conviction_when_wrong=(
                statistics.mean(wrong_convictions) if wrong_convictions else 0
            ),
            spy_return=spy_return,
            ic_1d=ic_values.get("return_1d"),
            ic_5d=ic_values.get("return_5d"),
            ic_20d=ic_values.get("return_20d"),
            ic_60d=ic_values.get("return_60d"),
        )


def _rank_ic(predictions: list[float], actuals: list[float]) -> float:
    """Compute rank information coefficient (Spearman correlation)."""
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
    std_p = (sum((r - mean_p) ** 2 for r in r_pred)) ** 0.5
    std_a = (sum((r - mean_a) ** 2 for r in r_act)) ** 0.5

    if std_p == 0 or std_a == 0:
        return 0.0
    return cov / (std_p * std_a)


def _benchmark_return(
    signals: list[SignalRecord],
    horizon: str,
) -> float:
    """Compute equal-weight buy-and-hold SPY return over signal dates."""
    if not signals:
        return 0.0

    earliest = min(s.signal_date for s in signals)
    latest = max(s.signal_date for s in signals)

    # Add buffer for the horizon lookforward
    horizon_days = _HORIZONS.get(horizon, 20)
    end_date = latest.date() + timedelta(days=horizon_days + 10)

    try:
        df = yf.download("SPY", start=str(earliest.date()), end=str(end_date), progress=False)
        if df.empty:
            return 0.0
        if hasattr(df.columns, 'levels') and len(df.columns.levels) > 1:
            df.columns = df.columns.get_level_values(0)
        prices = df["Close"]
        return float((prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0])
    except Exception:
        return 0.0
