"""
Peer comparison tool using yfinance.

Compares a stock against sector peers on key fundamental metrics.
Enables agents to assess relative valuation, growth, and quality
vs. competitors — a core part of equity analysis.
"""

from __future__ import annotations

import logging
from typing import Any

import yfinance as yf

logger = logging.getLogger(__name__)

# Fallback peer groups by sector (used when yfinance doesn't provide peers)
_SECTOR_PEERS: dict[str, list[str]] = {
    "Technology": ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "CRM", "ADBE", "ORCL"],
    "Healthcare": ["JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "TMO", "ABT"],
    "Financial Services": ["JPM", "BAC", "WFC", "GS", "MS", "BLK", "C", "AXP"],
    "Consumer Cyclical": ["AMZN", "TSLA", "HD", "NKE", "MCD", "SBUX", "TJX", "LOW"],
    "Consumer Defensive": ["PG", "KO", "PEP", "WMT", "COST", "CL", "MDLZ", "GIS"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO"],
    "Industrials": ["CAT", "HON", "UPS", "RTX", "BA", "GE", "DE", "LMT"],
    "Communication Services": ["GOOGL", "META", "DIS", "NFLX", "CMCSA", "T", "VZ", "TMUS"],
    "Utilities": ["NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL"],
    "Real Estate": ["PLD", "AMT", "CCI", "EQIX", "SPG", "O", "WELL", "PSA"],
    "Basic Materials": ["LIN", "APD", "SHW", "FCX", "NEM", "ECL", "DD", "NUE"],
}


class PeerComparisonTool:
    """Compares a stock against sector peers on key metrics."""

    @staticmethod
    def compare_peers(
        ticker: str,
        peers: list[str] | None = None,
        max_peers: int = 5,
    ) -> dict[str, Any]:
        """
        Compare a ticker against sector peers on fundamental metrics.

        Args:
            ticker: Stock ticker symbol
            peers: Optional list of peer tickers. If None, auto-detected from sector.
            max_peers: Maximum number of peers to compare (default 5)

        Returns:
            Dict with target stock metrics, peer metrics, and relative positioning.
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            sector = info.get("sector", "Unknown")

            # Get peer list
            if not peers:
                peers = PeerComparisonTool._find_peers(ticker, sector, max_peers)
            else:
                peers = [p for p in peers if p.upper() != ticker.upper()][:max_peers]

            # Get target metrics
            target_metrics = PeerComparisonTool._extract_metrics(ticker, info)

            # Get peer metrics
            peer_metrics = []
            for peer_ticker in peers:
                try:
                    peer_info = yf.Ticker(peer_ticker).info
                    metrics = PeerComparisonTool._extract_metrics(peer_ticker, peer_info)
                    peer_metrics.append(metrics)
                except Exception as e:
                    logger.warning(f"Failed to fetch peer {peer_ticker}: {e}")
                    continue

            # Compute relative positioning
            relative = PeerComparisonTool._compute_relative(target_metrics, peer_metrics)

            return {
                "ticker": ticker,
                "sector": sector,
                "target": target_metrics,
                "peers": peer_metrics,
                "relative_positioning": relative,
                "peer_count": len(peer_metrics),
            }

        except Exception as e:
            logger.error(f"Peer comparison failed for {ticker}: {e}")
            return {"ticker": ticker, "error": str(e), "peers": []}

    @staticmethod
    def _find_peers(ticker: str, sector: str, max_peers: int) -> list[str]:
        """Find peer tickers from sector mapping, excluding the target."""
        sector_list = _SECTOR_PEERS.get(sector, [])
        return [p for p in sector_list if p.upper() != ticker.upper()][:max_peers]

    @staticmethod
    def _extract_metrics(ticker: str, info: dict) -> dict[str, Any]:
        """Extract comparable metrics from yfinance info dict."""
        return {
            "ticker": ticker,
            "name": info.get("longName", info.get("shortName", ticker)),
            "market_cap": info.get("marketCap"),
            "pe_trailing": info.get("trailingPE"),
            "pe_forward": info.get("forwardPE"),
            "peg_ratio": info.get("pegRatio"),
            "price_to_book": info.get("priceToBook"),
            "ev_to_ebitda": info.get("enterpriseToEbitda"),
            "profit_margin": info.get("profitMargins"),
            "revenue_growth": info.get("revenueGrowth"),
            "roe": info.get("returnOnEquity"),
            "dividend_yield": info.get("dividendYield"),
            "debt_to_equity": info.get("debtToEquity"),
            "recommendation": info.get("recommendationKey"),
        }

    @staticmethod
    def _compute_relative(
        target: dict[str, Any], peers: list[dict[str, Any]]
    ) -> dict[str, str]:
        """Compute where the target sits relative to peers on key metrics."""
        if not peers:
            return {"summary": "No peers available for comparison"}

        relative = {}
        metrics_to_compare = [
            ("pe_trailing", "P/E", "lower_better"),
            ("pe_forward", "Forward P/E", "lower_better"),
            ("revenue_growth", "Revenue Growth", "higher_better"),
            ("profit_margin", "Profit Margin", "higher_better"),
            ("roe", "ROE", "higher_better"),
            ("ev_to_ebitda", "EV/EBITDA", "lower_better"),
        ]

        for metric_key, label, direction in metrics_to_compare:
            target_val = target.get(metric_key)
            if target_val is None:
                continue

            peer_vals = [p.get(metric_key) for p in peers if p.get(metric_key) is not None]
            if not peer_vals:
                continue

            avg_peer = sum(peer_vals) / len(peer_vals)
            if avg_peer == 0:
                continue

            pct_diff = ((target_val - avg_peer) / abs(avg_peer)) * 100

            if direction == "lower_better":
                assessment = "cheaper" if pct_diff < -10 else "pricier" if pct_diff > 10 else "in-line"
            else:
                assessment = "stronger" if pct_diff > 10 else "weaker" if pct_diff < -10 else "in-line"

            relative[label] = f"{assessment} vs peers ({pct_diff:+.1f}%)"

        return relative
