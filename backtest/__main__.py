"""
CLI entry point for backtest analytics.

Usage:
    python -m backtest stats              # Show signal/portfolio stats
    python -m backtest fill-returns       # Fill realized returns for stored signals
    python -m backtest run [--ticker X]   # Run backtest
    python -m backtest calibration        # Calibration analysis
    python -m backtest decay              # Alpha decay curve
    python -m backtest benchmark          # Benchmark comparison
    python -m backtest portfolio          # Build portfolio snapshot
    python -m backtest explain            # Agent attribution analysis
    python -m backtest report             # Full analytics report
"""

from __future__ import annotations

import argparse

from backtest import (
    BacktestRunner,
    CalibrationAnalyzer,
    MultiAssetPortfolio,
    SignalDatabase,
)
from backtest.alpha_decay import AlphaDecayAnalyzer
from backtest.benchmark import BenchmarkAnalyzer
from backtest.explainability import ExplainabilityAnalyzer


def main():
    parser = argparse.ArgumentParser(description="IC Backtest Analytics CLI")
    sub = parser.add_subparsers(dest="command")

    # Stats
    sub.add_parser("stats", help="Show signal and portfolio statistics")

    # Fill returns
    fill = sub.add_parser("fill-returns", help="Fill realized returns for stored signals")
    fill.add_argument("--limit", type=int, default=500)

    # Run backtest
    bt = sub.add_parser("run", help="Run historical backtest")
    bt.add_argument("--ticker", type=str, default=None)
    bt.add_argument("--provider", type=str, default=None)
    bt.add_argument("--horizon", type=str, default="return_20d")

    # Calibration
    cal = sub.add_parser("calibration", help="Conviction-return calibration analysis")
    cal.add_argument("--ticker", type=str, default=None)
    cal.add_argument("--horizon", type=str, default="return_20d")

    # Alpha decay
    decay = sub.add_parser("decay", help="Alpha decay curve analysis")
    decay.add_argument("--ticker", type=str, default=None)

    # Benchmark comparison
    bench = sub.add_parser("benchmark", help="Compare IC vs benchmarks")
    bench.add_argument("--ticker", type=str, default=None)
    bench.add_argument("--horizon", type=str, default="return_20d")

    # Portfolio
    port = sub.add_parser("portfolio", help="Build portfolio snapshot")
    port.add_argument("--max-positions", type=int, default=20)
    port.add_argument("--min-conviction", type=float, default=5.0)

    # Explainability
    sub.add_parser("explain", help="Agent attribution analysis")

    # Reflections
    ref = sub.add_parser("reflect", help="Generate post-trade reflections")
    ref.add_argument("--horizon", type=str, default="return_20d")

    # Full report
    sub.add_parser("report", help="Full analytics report")

    args = parser.parse_args()
    db = SignalDatabase()

    if args.command == "stats":
        _cmd_stats(db)
    elif args.command == "fill-returns":
        _cmd_fill_returns(db, args.limit)
    elif args.command == "run":
        _cmd_run_backtest(db, args.ticker, args.provider, args.horizon)
    elif args.command == "calibration":
        _cmd_calibration(db, args.ticker, args.horizon)
    elif args.command == "decay":
        _cmd_decay(db, args.ticker)
    elif args.command == "benchmark":
        _cmd_benchmark(db, args.ticker, args.horizon)
    elif args.command == "portfolio":
        _cmd_portfolio(db, args.max_positions, args.min_conviction)
    elif args.command == "explain":
        _cmd_explain(db)
    elif args.command == "reflect":
        _cmd_reflect(db, args.horizon)
    elif args.command == "report":
        _cmd_full_report(db)
    else:
        parser.print_help()


def _cmd_stats(db: SignalDatabase):
    n_signals = db.count_signals()
    tickers = db.get_all_tickers()
    n_snapshots = len(db.get_snapshots(limit=10000))
    print(f"Signals: {n_signals}")
    print(f"Tickers: {len(tickers)} — {', '.join(tickers[:10])}")
    print(f"Portfolio snapshots: {n_snapshots}")


def _cmd_fill_returns(db: SignalDatabase, limit: int):
    runner = BacktestRunner(db)
    updated = runner.fill_returns(limit=limit)
    print(f"Updated returns for {updated} signals")


def _cmd_run_backtest(db: SignalDatabase, ticker, provider, horizon):
    runner = BacktestRunner(db)
    result = runner.run_backtest(ticker=ticker, provider=provider, horizon=horizon)
    print(f"Backtest: {result.num_signals} signals")
    print(f"  Total return: {result.total_return:+.2%}")
    print(f"  Sharpe: {result.sharpe_ratio:.2f}")
    print(f"  Win rate: {result.win_rate:.0%}")
    print(f"  Direction accuracy: {result.direction_accuracy:.0%}")
    print(f"  Max drawdown: {result.max_drawdown:.1%}")


def _cmd_calibration(db: SignalDatabase, ticker, horizon):
    analyzer = CalibrationAnalyzer(db)
    buckets = analyzer.compute_calibration(horizon=horizon, ticker=ticker)
    corr = analyzer.compute_conviction_return_correlation(horizon=horizon, ticker=ticker)
    print(analyzer.format_report(buckets, corr))


def _cmd_decay(db: SignalDatabase, ticker):
    analyzer = AlphaDecayAnalyzer(db)
    curve = analyzer.compute_decay_curve(ticker=ticker)
    optimal = analyzer.find_optimal_horizon(curve)
    print(analyzer.format_report(curve, optimal))


def _cmd_benchmark(db: SignalDatabase, ticker, horizon):
    analyzer = BenchmarkAnalyzer(db)
    comps = analyzer.run_comparison(ticker=ticker, horizon=horizon)
    print(analyzer.format_report(comps))


def _cmd_portfolio(db: SignalDatabase, max_positions, min_conviction):
    portfolio = MultiAssetPortfolio(db)
    snapshot = portfolio.build_snapshot(
        max_positions=max_positions,
        min_conviction=min_conviction,
    )
    print(portfolio.format_report(snapshot))


def _cmd_reflect(db: SignalDatabase, horizon: str):
    from backtest.reflection import ReflectionEngine
    engine = ReflectionEngine(db)
    count = engine.run_reflections(horizon=horizon)
    print(f"Generated {count} reflections (horizon={horizon})")


def _cmd_explain(db: SignalDatabase):
    analyzer = ExplainabilityAnalyzer(db)
    attributions = analyzer.attribute_all(limit=100)
    stats = analyzer.compute_agent_statistics(attributions)
    print(analyzer.format_report(attributions, stats))


def _cmd_full_report(db: SignalDatabase):
    """Generate a comprehensive analytics report."""
    sections = []
    sections.append("# IC Analytics Report\n")

    # Stats
    n = db.count_signals()
    tickers = db.get_all_tickers()
    sections.append(f"**Signals:** {n} | **Tickers:** {len(tickers)}\n")

    # Backtest
    runner = BacktestRunner(db)
    result = runner.run_backtest()
    if result.num_signals > 0:
        sections.append(f"## Backtest Summary ({result.num_signals} signals)\n")
        sections.append(f"- Total return: {result.total_return:+.2%}")
        sections.append(f"- Sharpe: {result.sharpe_ratio:.2f}")
        sections.append(f"- Win rate: {result.win_rate:.0%}")
        sections.append(f"- Direction accuracy: {result.direction_accuracy:.0%}\n")

    # Calibration
    cal = CalibrationAnalyzer(db)
    buckets = cal.compute_calibration()
    corr = cal.compute_conviction_return_correlation()
    if buckets:
        sections.append(cal.format_report(buckets, corr))
        sections.append("")

    # Alpha decay
    decay = AlphaDecayAnalyzer(db)
    curve = decay.compute_decay_curve()
    optimal = decay.find_optimal_horizon(curve)
    if any(p.num_signals >= 5 for p in curve):
        sections.append(decay.format_report(curve, optimal))
        sections.append("")

    # Benchmark
    bench = BenchmarkAnalyzer(db)
    comps = bench.run_comparison()
    if comps:
        sections.append(bench.format_report(comps))
        sections.append("")

    # Portfolio
    port = MultiAssetPortfolio(db)
    snap = port.build_snapshot()
    if snap.tickers:
        sections.append(port.format_report(snap))
        sections.append("")

    # Explainability
    explain = ExplainabilityAnalyzer(db)
    attrs = explain.attribute_all(limit=50)
    if attrs:
        stats = explain.compute_agent_statistics(attrs)
        sections.append(explain.format_report(attrs, stats))

    print("\n".join(sections))


if __name__ == "__main__":
    main()
