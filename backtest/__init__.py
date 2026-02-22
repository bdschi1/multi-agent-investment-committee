"""
Backtest, calibration, and portfolio analytics for the investment committee.

Provides:
    - Signal persistence (SQLite)
    - Historical backtesting
    - Calibration analysis (conviction -> realized return)
    - Alpha decay curves
    - Benchmark comparison (vs momentum, equal-weight, SPY)
    - Multi-asset portfolio construction
    - Explainability / attribution
"""

from backtest.alpha_decay import AlphaDecayAnalyzer
from backtest.benchmark import BenchmarkAnalyzer
from backtest.calibration import CalibrationAnalyzer
from backtest.database import SignalDatabase
from backtest.explainability import ExplainabilityAnalyzer
from backtest.models import BacktestResult, PortfolioSnapshot, ReflectionRecord, SignalRecord
from backtest.portfolio import MultiAssetPortfolio
from backtest.reflection import ReflectionEngine
from backtest.runner import BacktestRunner

__all__ = [
    "SignalDatabase",
    "SignalRecord",
    "PortfolioSnapshot",
    "BacktestResult",
    "ReflectionRecord",
    "ReflectionEngine",
    "BacktestRunner",
    "CalibrationAnalyzer",
    "AlphaDecayAnalyzer",
    "BenchmarkAnalyzer",
    "MultiAssetPortfolio",
    "ExplainabilityAnalyzer",
]
