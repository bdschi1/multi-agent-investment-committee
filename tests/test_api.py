"""
Tests for the FastAPI endpoint.

Tests API models and route structure without requiring live LLM providers.
"""

from __future__ import annotations

from api.models import (
    AnalysisRequest,
    AnalysisResponse,
    BacktestResponse,
    HealthResponse,
    PortfolioResponse,
    SignalResponse,
)


class TestApiModels:
    def test_analysis_request_defaults(self):
        req = AnalysisRequest(ticker="NVDA")
        assert req.provider == "anthropic"
        assert req.debate_rounds == 2
        assert req.user_context == ""

    def test_analysis_request_custom(self):
        req = AnalysisRequest(
            ticker="COST",
            provider="ollama",
            model_name="llama3.1:8b",
            debate_rounds=3,
            user_context="Concerned about tariff impact",
        )
        assert req.ticker == "COST"
        assert req.provider == "ollama"

    def test_signal_response(self):
        resp = SignalResponse(
            ticker="NVDA",
            recommendation="BUY",
            conviction=7.5,
            t_signal=0.72,
            position_direction=1,
            bull_conviction=8.2,
            bear_conviction=5.5,
            macro_favorability=6.5,
        )
        assert resp.ticker == "NVDA"
        assert resp.t_signal == 0.72

    def test_analysis_response_success(self):
        resp = AnalysisResponse(
            success=True,
            signal=SignalResponse(
                ticker="NVDA",
                recommendation="BUY",
                conviction=7.5,
                t_signal=0.72,
                position_direction=1,
                bull_conviction=8.2,
                bear_conviction=5.5,
                macro_favorability=6.5,
            ),
            signal_id=42,
            duration_s=350.0,
        )
        assert resp.success
        assert resp.signal_id == 42

    def test_analysis_response_failure(self):
        resp = AnalysisResponse(success=False, error="API key missing")
        assert not resp.success
        assert "API key" in resp.error

    def test_backtest_response(self):
        resp = BacktestResponse(
            num_signals=50,
            total_return=0.15,
            sharpe_ratio=1.2,
            win_rate=0.65,
        )
        assert resp.num_signals == 50
        assert resp.total_return == 0.15

    def test_portfolio_response(self):
        resp = PortfolioResponse(
            tickers=["NVDA", "META"],
            weights={"NVDA": 0.6, "META": 0.4},
            gross_exposure=1.0,
            net_exposure=1.0,
            num_longs=2,
            num_shorts=0,
        )
        assert len(resp.tickers) == 2
        assert resp.gross_exposure == 1.0

    def test_health_response(self):
        resp = HealthResponse(status="ok", version="3.8.0", num_signals=100)
        assert resp.status == "ok"


class TestApiRoutes:
    """Test that routes can be imported and FastAPI app is constructable."""

    def test_app_import(self):
        """FastAPI app should import without errors."""
        from api.main import app
        assert app is not None
        assert app.title == "Multi-Agent Investment Committee API"

    def test_routes_registered(self):
        from api.main import app
        route_paths = [r.path for r in app.routes]
        assert "/health" in route_paths
        assert "/analyze" in route_paths
        assert "/signals" in route_paths
        assert "/backtest" in route_paths
        assert "/portfolio" in route_paths
