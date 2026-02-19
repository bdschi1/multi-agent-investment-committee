"""
SQLite persistence for IC signals, portfolio snapshots, and backtest results.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from backtest.models import SignalRecord, PortfolioSnapshot, BacktestResult

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path("store/signals.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    signal_date TEXT NOT NULL,
    provider TEXT DEFAULT '',
    model_name TEXT DEFAULT '',
    recommendation TEXT DEFAULT '',
    t_signal REAL DEFAULT 0.0,
    conviction REAL DEFAULT 5.0,
    position_direction INTEGER DEFAULT 0,
    raw_confidence REAL DEFAULT 0.5,
    bull_conviction REAL DEFAULT 5.0,
    bear_conviction REAL DEFAULT 5.0,
    macro_favorability REAL DEFAULT 5.0,
    bl_optimal_weight REAL,
    bl_sharpe REAL,
    bl_sortino REAL,
    sharpe_heuristic REAL,
    sortino_heuristic REAL,
    price_at_signal REAL,
    return_1d REAL,
    return_5d REAL,
    return_10d REAL,
    return_20d REAL,
    return_60d REAL,
    duration_s REAL DEFAULT 0.0,
    total_tokens INTEGER DEFAULT 0,
    bull_influence REAL,
    bear_influence REAL,
    macro_influence REAL,
    debate_shift REAL
);

CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_date TEXT NOT NULL,
    tickers TEXT DEFAULT '[]',
    weights TEXT DEFAULT '{}',
    t_signals TEXT DEFAULT '{}',
    gross_exposure REAL DEFAULT 0.0,
    net_exposure REAL DEFAULT 0.0,
    portfolio_return REAL DEFAULT 0.0,
    cumulative_return REAL DEFAULT 0.0,
    drawdown REAL DEFAULT 0.0,
    portfolio_sharpe REAL,
    num_longs INTEGER DEFAULT 0,
    num_shorts INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS backtest_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date TEXT NOT NULL,
    start_date TEXT DEFAULT '',
    end_date TEXT DEFAULT '',
    tickers TEXT DEFAULT '[]',
    provider TEXT DEFAULT '',
    num_signals INTEGER DEFAULT 0,
    total_return REAL DEFAULT 0.0,
    annualized_return REAL DEFAULT 0.0,
    sharpe_ratio REAL DEFAULT 0.0,
    sortino_ratio REAL DEFAULT 0.0,
    max_drawdown REAL DEFAULT 0.0,
    win_rate REAL DEFAULT 0.0,
    avg_win REAL DEFAULT 0.0,
    avg_loss REAL DEFAULT 0.0,
    profit_factor REAL DEFAULT 0.0,
    direction_accuracy REAL DEFAULT 0.0,
    avg_conviction_when_right REAL DEFAULT 0.0,
    avg_conviction_when_wrong REAL DEFAULT 0.0,
    spy_return REAL DEFAULT 0.0,
    momentum_return REAL DEFAULT 0.0,
    excess_return_vs_spy REAL DEFAULT 0.0,
    ic_1d REAL,
    ic_5d REAL,
    ic_20d REAL,
    ic_60d REAL,
    optimal_holding_period INTEGER
);

CREATE INDEX IF NOT EXISTS idx_signals_ticker ON signals(ticker);
CREATE INDEX IF NOT EXISTS idx_signals_date ON signals(signal_date);
CREATE INDEX IF NOT EXISTS idx_portfolio_date ON portfolio_snapshots(snapshot_date);
"""


class SignalDatabase:
    """SQLite-backed storage for IC signals and analytics."""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else _DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self) -> None:
        conn = self._get_conn()
        conn.executescript(_SCHEMA)
        conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── Signals ──────────────────────────────────────────────────

    def store_signal(self, signal: SignalRecord) -> int:
        """Store a signal and return its ID."""
        conn = self._get_conn()
        cursor = conn.execute(
            """INSERT INTO signals (
                ticker, signal_date, provider, model_name,
                recommendation, t_signal, conviction, position_direction,
                raw_confidence, bull_conviction, bear_conviction,
                macro_favorability, bl_optimal_weight, bl_sharpe, bl_sortino,
                sharpe_heuristic, sortino_heuristic, price_at_signal,
                return_1d, return_5d, return_10d, return_20d, return_60d,
                duration_s, total_tokens,
                bull_influence, bear_influence, macro_influence, debate_shift
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                signal.ticker, signal.signal_date.isoformat(),
                signal.provider, signal.model_name,
                signal.recommendation, signal.t_signal, signal.conviction,
                signal.position_direction, signal.raw_confidence,
                signal.bull_conviction, signal.bear_conviction,
                signal.macro_favorability, signal.bl_optimal_weight,
                signal.bl_sharpe, signal.bl_sortino,
                signal.sharpe_heuristic, signal.sortino_heuristic,
                signal.price_at_signal,
                signal.return_1d, signal.return_5d, signal.return_10d,
                signal.return_20d, signal.return_60d,
                signal.duration_s, signal.total_tokens,
                signal.bull_influence, signal.bear_influence,
                signal.macro_influence, signal.debate_shift,
            ),
        )
        conn.commit()
        return cursor.lastrowid

    def get_signals(
        self,
        ticker: str | None = None,
        limit: int = 100,
    ) -> list[SignalRecord]:
        """Retrieve signals, optionally filtered by ticker."""
        conn = self._get_conn()
        if ticker:
            rows = conn.execute(
                "SELECT * FROM signals WHERE ticker = ? ORDER BY signal_date DESC LIMIT ?",
                (ticker, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM signals ORDER BY signal_date DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_signal(r) for r in rows]

    def get_signal_by_id(self, signal_id: int) -> Optional[SignalRecord]:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM signals WHERE id = ?", (signal_id,)).fetchone()
        return self._row_to_signal(row) if row else None

    def update_returns(self, signal_id: int, **returns: float) -> None:
        """Update realized returns for a signal."""
        valid_cols = {"return_1d", "return_5d", "return_10d", "return_20d", "return_60d", "price_at_signal"}
        updates = {k: v for k, v in returns.items() if k in valid_cols}
        if not updates:
            return
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        conn = self._get_conn()
        conn.execute(
            f"UPDATE signals SET {set_clause} WHERE id = ?",
            (*updates.values(), signal_id),
        )
        conn.commit()

    def count_signals(self) -> int:
        conn = self._get_conn()
        return conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]

    def get_all_tickers(self) -> list[str]:
        conn = self._get_conn()
        rows = conn.execute("SELECT DISTINCT ticker FROM signals ORDER BY ticker").fetchall()
        return [r[0] for r in rows]

    # ── Portfolio Snapshots ──────────────────────────────────────

    def store_snapshot(self, snapshot: PortfolioSnapshot) -> int:
        conn = self._get_conn()
        cursor = conn.execute(
            """INSERT INTO portfolio_snapshots (
                snapshot_date, tickers, weights, t_signals,
                gross_exposure, net_exposure, portfolio_return,
                cumulative_return, drawdown, portfolio_sharpe,
                num_longs, num_shorts
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                snapshot.snapshot_date.isoformat(),
                json.dumps(snapshot.tickers),
                json.dumps(snapshot.weights),
                json.dumps(snapshot.t_signals),
                snapshot.gross_exposure, snapshot.net_exposure,
                snapshot.portfolio_return, snapshot.cumulative_return,
                snapshot.drawdown, snapshot.portfolio_sharpe,
                snapshot.num_longs, snapshot.num_shorts,
            ),
        )
        conn.commit()
        return cursor.lastrowid

    def get_snapshots(self, limit: int = 100) -> list[PortfolioSnapshot]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM portfolio_snapshots ORDER BY snapshot_date DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_snapshot(r) for r in rows]

    # ── Backtest Results ─────────────────────────────────────────

    def store_backtest(self, result: BacktestResult) -> int:
        conn = self._get_conn()
        cursor = conn.execute(
            """INSERT INTO backtest_results (
                run_date, start_date, end_date, tickers, provider, num_signals,
                total_return, annualized_return, sharpe_ratio, sortino_ratio,
                max_drawdown, win_rate, avg_win, avg_loss, profit_factor,
                direction_accuracy, avg_conviction_when_right, avg_conviction_when_wrong,
                spy_return, momentum_return, excess_return_vs_spy,
                ic_1d, ic_5d, ic_20d, ic_60d, optimal_holding_period
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                result.run_date.isoformat(), result.start_date, result.end_date,
                json.dumps(result.tickers), result.provider, result.num_signals,
                result.total_return, result.annualized_return,
                result.sharpe_ratio, result.sortino_ratio,
                result.max_drawdown, result.win_rate, result.avg_win,
                result.avg_loss, result.profit_factor,
                result.direction_accuracy, result.avg_conviction_when_right,
                result.avg_conviction_when_wrong,
                result.spy_return, result.momentum_return,
                result.excess_return_vs_spy,
                result.ic_1d, result.ic_5d, result.ic_20d, result.ic_60d,
                result.optimal_holding_period,
            ),
        )
        conn.commit()
        return cursor.lastrowid

    # ── Row converters ───────────────────────────────────────────

    @staticmethod
    def _row_to_signal(row: sqlite3.Row) -> SignalRecord:
        return SignalRecord(
            id=row["id"],
            ticker=row["ticker"],
            signal_date=datetime.fromisoformat(row["signal_date"]),
            provider=row["provider"] or "",
            model_name=row["model_name"] or "",
            recommendation=row["recommendation"] or "",
            t_signal=row["t_signal"] or 0.0,
            conviction=row["conviction"] or 5.0,
            position_direction=row["position_direction"] or 0,
            raw_confidence=row["raw_confidence"] or 0.5,
            bull_conviction=row["bull_conviction"] or 5.0,
            bear_conviction=row["bear_conviction"] or 5.0,
            macro_favorability=row["macro_favorability"] or 5.0,
            bl_optimal_weight=row["bl_optimal_weight"],
            bl_sharpe=row["bl_sharpe"],
            bl_sortino=row["bl_sortino"],
            sharpe_heuristic=row["sharpe_heuristic"],
            sortino_heuristic=row["sortino_heuristic"],
            price_at_signal=row["price_at_signal"],
            return_1d=row["return_1d"],
            return_5d=row["return_5d"],
            return_10d=row["return_10d"],
            return_20d=row["return_20d"],
            return_60d=row["return_60d"],
            duration_s=row["duration_s"] or 0.0,
            total_tokens=row["total_tokens"] or 0,
            bull_influence=row["bull_influence"],
            bear_influence=row["bear_influence"],
            macro_influence=row["macro_influence"],
            debate_shift=row["debate_shift"],
        )

    @staticmethod
    def _row_to_snapshot(row: sqlite3.Row) -> PortfolioSnapshot:
        return PortfolioSnapshot(
            id=row["id"],
            snapshot_date=datetime.fromisoformat(row["snapshot_date"]),
            tickers=json.loads(row["tickers"] or "[]"),
            weights=json.loads(row["weights"] or "{}"),
            t_signals=json.loads(row["t_signals"] or "{}"),
            gross_exposure=row["gross_exposure"] or 0.0,
            net_exposure=row["net_exposure"] or 0.0,
            portfolio_return=row["portfolio_return"] or 0.0,
            cumulative_return=row["cumulative_return"] or 0.0,
            drawdown=row["drawdown"] or 0.0,
            portfolio_sharpe=row["portfolio_sharpe"],
            num_longs=row["num_longs"] or 0,
            num_shorts=row["num_shorts"] or 0,
        )
