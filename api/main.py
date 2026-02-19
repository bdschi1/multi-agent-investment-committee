"""
FastAPI application for the Investment Committee.

Endpoints:
    POST /analyze          — Run full IC pipeline for a ticker
    GET  /signals          — List stored signals
    GET  /signals/{id}     — Get a specific signal
    POST /backtest         — Run backtest on stored signals
    GET  /portfolio        — Build portfolio snapshot from latest signals
    GET  /health           — Health check with stats

Usage:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from api.models import (
    AnalysisRequest,
    AnalysisResponse,
    BacktestResponse,
    HealthResponse,
    PortfolioResponse,
    SignalResponse,
)
from backtest import (
    SignalDatabase,
    SignalRecord,
    BacktestRunner,
    MultiAssetPortfolio,
)
from config.settings import settings, LLMProvider

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Multi-Agent Investment Committee API",
    description="Programmatic access to multi-agent IC analysis, signals, and analytics.",
    version="3.8.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared database instance
_db = SignalDatabase()

# Provider name → LLMProvider enum
_PROVIDER_MAP = {
    "anthropic": LLMProvider.ANTHROPIC,
    "google": LLMProvider.GOOGLE,
    "openai": LLMProvider.OPENAI,
    "deepseek": LLMProvider.DEEPSEEK,
    "huggingface": LLMProvider.HUGGINGFACE,
    "ollama": LLMProvider.OLLAMA,
}


@app.get("/health", response_model=HealthResponse)
def health():
    """Health check with signal stats."""
    return HealthResponse(
        status="ok",
        version="3.8.0",
        num_signals=_db.count_signals(),
        num_tickers=len(_db.get_all_tickers()),
    )


@app.post("/analyze", response_model=AnalysisResponse)
def analyze(req: AnalysisRequest):
    """
    Run the full IC pipeline for a ticker.

    Returns the committee memo, all agent outputs, and stores the signal.
    """
    from app import create_model
    from orchestrator.committee import InvestmentCommittee
    from tools.data_aggregator import DataAggregator

    ticker = req.ticker.strip().upper()
    if not ticker:
        raise HTTPException(400, "Ticker is required")

    provider_enum = _PROVIDER_MAP.get(req.provider.lower())
    if not provider_enum:
        raise HTTPException(400, f"Unknown provider: {req.provider}")

    try:
        t0 = time.time()

        # Override debate rounds
        original_rounds = settings.max_debate_rounds
        settings.max_debate_rounds = req.debate_rounds

        model = create_model(provider_enum, model_name=req.model_name)
        committee = InvestmentCommittee(model=model)
        context = DataAggregator.gather_context(ticker, req.user_context)

        result = committee.run(ticker, context)

        settings.max_debate_rounds = original_rounds

        duration_s = time.time() - t0

        # Store signal in database
        signal_id = None
        if result.committee_memo:
            memo = result.committee_memo
            signal = SignalRecord(
                ticker=ticker,
                signal_date=datetime.now(timezone.utc),
                provider=req.provider,
                model_name=req.model_name or "",
                recommendation=memo.recommendation,
                t_signal=memo.t_signal,
                conviction=memo.conviction,
                position_direction=memo.position_direction,
                raw_confidence=memo.raw_confidence,
                bull_conviction=result.bull_case.conviction_score if result.bull_case else 5.0,
                bear_conviction=result.bear_case.bearish_conviction if result.bear_case else 5.0,
                macro_favorability=result.macro_view.macro_favorability if result.macro_view else 5.0,
                duration_s=duration_s,
                total_tokens=result.total_tokens,
            )

            # Add BL optimizer results if available
            if result.optimization_result and hasattr(result.optimization_result, 'success'):
                opt = result.optimization_result
                if opt.success:
                    signal.bl_optimal_weight = opt.optimal_weight
                    signal.bl_sharpe = opt.computed_sharpe
                    signal.bl_sortino = opt.computed_sortino

            signal_id = _db.store_signal(signal)

        return AnalysisResponse(
            success=True,
            signal=SignalResponse(
                ticker=ticker,
                recommendation=result.committee_memo.recommendation if result.committee_memo else "ERROR",
                conviction=result.committee_memo.conviction if result.committee_memo else 0,
                t_signal=result.committee_memo.t_signal if result.committee_memo else 0,
                position_direction=result.committee_memo.position_direction if result.committee_memo else 0,
                bull_conviction=result.bull_case.conviction_score if result.bull_case else 5.0,
                bear_conviction=result.bear_case.bearish_conviction if result.bear_case else 5.0,
                macro_favorability=result.macro_view.macro_favorability if result.macro_view else 5.0,
                thesis_summary=result.committee_memo.thesis_summary if result.committee_memo else "",
                duration_s=duration_s,
                provider=req.provider,
                model_name=req.model_name or "",
            ),
            signal_id=signal_id,
            bull_case=result.bull_case.model_dump() if result.bull_case else None,
            bear_case=result.bear_case.model_dump() if result.bear_case else None,
            macro_view=result.macro_view.model_dump() if result.macro_view else None,
            committee_memo=result.committee_memo.model_dump() if result.committee_memo else None,
            optimization_result=(
                result.optimization_result.model_dump()
                if result.optimization_result and hasattr(result.optimization_result, 'model_dump')
                else None
            ),
            duration_s=duration_s,
        )

    except Exception as e:
        logger.exception(f"Analysis failed for {ticker}")
        return AnalysisResponse(success=False, error=str(e))


@app.get("/signals")
def list_signals(
    ticker: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=10000),
):
    """List stored signals, optionally filtered by ticker."""
    signals = _db.get_signals(ticker=ticker, limit=limit)
    return [
        {
            "id": s.id,
            "ticker": s.ticker,
            "signal_date": s.signal_date.isoformat(),
            "provider": s.provider,
            "recommendation": s.recommendation,
            "t_signal": s.t_signal,
            "conviction": s.conviction,
            "position_direction": s.position_direction,
            "return_1d": s.return_1d,
            "return_5d": s.return_5d,
            "return_20d": s.return_20d,
        }
        for s in signals
    ]


@app.get("/signals/{signal_id}")
def get_signal(signal_id: int):
    """Get a specific signal by ID."""
    signal = _db.get_signal_by_id(signal_id)
    if not signal:
        raise HTTPException(404, f"Signal {signal_id} not found")
    return signal.model_dump()


@app.post("/backtest", response_model=BacktestResponse)
def run_backtest(
    ticker: Optional[str] = Query(None),
    provider: Optional[str] = Query(None),
    horizon: str = Query("return_20d"),
):
    """Run backtest on stored signals."""
    runner = BacktestRunner(_db)
    result = runner.run_backtest(ticker=ticker, provider=provider, horizon=horizon)
    return BacktestResponse(
        num_signals=result.num_signals,
        total_return=result.total_return,
        sharpe_ratio=result.sharpe_ratio,
        sortino_ratio=result.sortino_ratio,
        win_rate=result.win_rate,
        direction_accuracy=result.direction_accuracy,
        max_drawdown=result.max_drawdown,
        spy_return=result.spy_return,
        excess_return_vs_spy=result.excess_return_vs_spy,
    )


@app.get("/portfolio", response_model=PortfolioResponse)
def get_portfolio(
    max_positions: int = Query(20, ge=1, le=100),
    min_conviction: float = Query(5.0, ge=0, le=10),
):
    """Build a portfolio snapshot from latest signals."""
    portfolio = MultiAssetPortfolio(_db)
    snapshot = portfolio.build_snapshot(
        max_positions=max_positions,
        min_conviction=min_conviction,
    )
    return PortfolioResponse(
        tickers=snapshot.tickers,
        weights=snapshot.weights,
        t_signals=snapshot.t_signals,
        gross_exposure=snapshot.gross_exposure,
        net_exposure=snapshot.net_exposure,
        num_longs=snapshot.num_longs,
        num_shorts=snapshot.num_shorts,
    )


@app.post("/fill-returns")
def fill_returns(limit: int = Query(500, ge=1, le=10000)):
    """Fill realized returns for stored signals."""
    runner = BacktestRunner(_db)
    updated = runner.fill_returns(limit=limit)
    return {"updated": updated}
