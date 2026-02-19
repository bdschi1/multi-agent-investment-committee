"""Build the asset universe for Black-Litterman optimization."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# GICS sector → SPDR Select Sector ETF
SECTOR_ETF_MAP: dict[str, str] = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Financial Services": "XLF",
    "Financials": "XLF",
    "Consumer Cyclical": "XLY",
    "Consumer Discretionary": "XLY",
    "Consumer Defensive": "XLP",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Communication Services": "XLC",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Basic Materials": "XLB",
    "Materials": "XLB",
}

# Reuse peer groups from tools/peer_comparison.py
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


def build_universe(
    ticker: str,
    sector: str,
    max_peers: int = 5,
    lookback: str = "2y",
) -> tuple[list[str], pd.DataFrame, dict[str, float]]:
    """
    Build the asset universe for BL optimization.

    Universe = [target, peer1..peerN, sector_ETF, SPY]

    Returns:
        (tickers, prices_df, market_caps)
        - tickers: ordered list of ticker symbols
        - prices_df: daily adjusted close prices (DatetimeIndex, columns=tickers)
        - market_caps: {ticker: market_cap_usd}
    """
    import yfinance as yf

    # Build ticker list
    sector_etf = SECTOR_ETF_MAP.get(sector, "XLK")
    peers = _SECTOR_PEERS.get(sector, [])
    peers = [p for p in peers if p.upper() != ticker.upper()][:max_peers]

    universe = [ticker] + peers + [sector_etf, "SPY"]
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for t in universe:
        t_upper = t.upper()
        if t_upper not in seen:
            seen.add(t_upper)
            unique.append(t_upper)
    universe = unique

    logger.info(f"BL universe ({len(universe)} assets): {universe}")

    # Download prices
    data = yf.download(universe, period=lookback, auto_adjust=True, progress=False)

    # Handle multi-level columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data

    # Drop any tickers with insufficient data (< 60% of max rows)
    min_rows = int(len(prices) * 0.6)
    valid_cols = [c for c in prices.columns if prices[c].dropna().shape[0] >= min_rows]
    prices = prices[valid_cols].dropna()

    # Ensure target ticker survived
    if ticker not in prices.columns:
        raise ValueError(f"Target ticker {ticker} has insufficient price data")

    # Update universe to match surviving columns
    universe = list(prices.columns)

    # Market caps
    market_caps = {}
    for t in universe:
        try:
            info = yf.Ticker(t).info
            mc = info.get("marketCap")
            if mc and mc > 0:
                market_caps[t] = float(mc)
            else:
                # ETFs don't have marketCap — use totalAssets or estimate
                total_assets = info.get("totalAssets")
                if total_assets and total_assets > 0:
                    market_caps[t] = float(total_assets)
                else:
                    market_caps[t] = 1e10  # fallback
        except Exception:
            market_caps[t] = 1e10  # fallback

    return universe, prices, market_caps
