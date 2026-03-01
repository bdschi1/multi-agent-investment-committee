"""Bridge to fund-tracker-13f — 13F conviction signals for agent context.

fund-tracker-13f tracks 52 hedge funds across 5 tiers and surfaces
high-conviction moves from SEC 13F-HR filings. This bridge queries
the fund-tracker's SQLite database and cross-fund aggregation engine
to produce per-ticker conviction signals that MAIC agents consume.

Design: sys.path injection + graceful degradation (same pattern as
backtest-lab/bridges/fund_tracker_bridge.py). Returns plain dicts
so agents get structured context without depending on fund-tracker
being installed.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_FUND_TRACKER_PATH = Path(
    "/Users/bdsm4/code/bds_repos/Tier_1/fund-tracker-13f"
)


def _ensure_import() -> None:
    """Add fund-tracker-13f to sys.path if not already present."""
    path_str = str(_FUND_TRACKER_PATH)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def is_available() -> bool:
    """Check if fund-tracker-13f is importable and has data.

    Returns True if the core modules can be imported and the
    SQLite database exists. Does not guarantee data for any
    specific ticker or quarter.
    """
    _ensure_import()
    try:
        from core.models import CrowdedTrade  # noqa: F401
        from data.store import HoldingsStore  # noqa: F401

        db_path = _FUND_TRACKER_PATH / "data_cache" / "fund_tracker.db"
        return db_path.exists()
    except ImportError:
        return False


def get_fund_conviction_signals(ticker: str) -> dict[str, Any]:
    """Get 13F conviction signals for a specific ticker.

    Queries the fund-tracker-13f database for the most recent quarter
    and returns structured signals about hedge fund activity.

    Args:
        ticker: Stock ticker symbol (e.g., "NVDA", "AAPL").

    Returns:
        Dict with keys:
            available: bool — whether data was successfully retrieved
            consensus_buys: list of fund names that initiated or added
            high_conviction_adds: list of funds with significant adds (>50%)
            new_positions: list of funds that opened new positions
            exits: list of funds that exited entirely
            net_sentiment: int (positive = more buying, negative = more selling)
            crowding_risk: bool — True if tracked funds own >5% of float
            float_ownership_pct: float | None — aggregate float ownership
            total_funds_holding: int — count of funds holding the name
            aggregate_value_millions: float — total $ across tracked funds
            summary: str — plain-English summary for agent prompts
    """
    _ensure_import()
    try:
        return _query_signals(ticker.upper())
    except ImportError:
        logger.warning("fund-tracker-13f not available — skipping 13F signals")
        return _unavailable("fund-tracker-13f package not importable.")
    except Exception as e:
        logger.warning(f"fund-tracker-13f query failed for {ticker}: {e}")
        return _unavailable(f"Fund conviction data query failed: {e}")


def _unavailable(reason: str) -> dict[str, Any]:
    """Return a standardized unavailable-data dict."""
    return {
        "available": False,
        "consensus_buys": [],
        "high_conviction_adds": [],
        "new_positions": [],
        "exits": [],
        "net_sentiment": 0,
        "crowding_risk": False,
        "float_ownership_pct": None,
        "total_funds_holding": 0,
        "aggregate_value_millions": 0.0,
        "summary": reason,
    }


def _query_signals(ticker: str) -> dict[str, Any]:
    """Internal: query fund-tracker database for ticker-specific signals.

    Runs a simplified version of the full analysis pipeline:
    1. Open the HoldingsStore
    2. Get the latest quarter with data
    3. Find the CUSIP(s) for this ticker
    4. Query which funds hold the position and how it changed
    5. Build the signal dict
    """
    from data.store import HoldingsStore

    db_path = _FUND_TRACKER_PATH / "data_cache" / "fund_tracker.db"
    if not db_path.exists():
        return _unavailable("fund-tracker-13f database not found.")

    store = HoldingsStore(db_path)

    # Find the latest quarter with data
    quarters = store.get_all_available_quarters()
    if not quarters:
        return _unavailable("No quarter data available in fund-tracker-13f.")

    latest_q = quarters[0]  # Most recent (sorted desc)

    # Find CUSIP(s) for this ticker via the cusip_map table
    cusips = _get_cusips_for_ticker(store, ticker)
    if not cusips:
        return _unavailable(
            f"No 13F filings found for {ticker} — "
            f"ticker may not be in the CUSIP mapping."
        )

    # Get all holdings for the latest quarter
    all_holdings = store.get_all_holdings_for_quarter(latest_q)

    # Find prior quarter for diff analysis
    prior_q = quarters[1] if len(quarters) > 1 else None

    prior_holdings = (
        store.get_all_holdings_for_quarter(prior_q) if prior_q else {}
    )

    # Scan all funds for this ticker's CUSIPs
    funds_holding: list[dict] = []
    funds_initiated: list[str] = []
    funds_added: list[str] = []
    high_conviction_adds: list[str] = []
    funds_exited: list[str] = []
    funds_trimmed: list[str] = []
    total_value_thousands = 0
    total_shares = 0

    # Build a fund name lookup
    fund_lookup: dict[str, str] = {}
    for cik in all_holdings:
        fund = store.get_fund(cik)
        if fund:
            fund_lookup[cik] = fund.name

    # Also check prior quarter for funds that exited
    prior_ciks_with_ticker: set[str] = set()
    if prior_holdings:
        for cik, holdings in prior_holdings.items():
            for h in holdings:
                if h.cusip in cusips and not h.is_option:
                    prior_ciks_with_ticker.add(cik)

    # Analyze current quarter holdings
    current_ciks_with_ticker: set[str] = set()
    for cik, holdings in all_holdings.items():
        for h in holdings:
            if h.cusip in cusips and not h.is_option:
                current_ciks_with_ticker.add(cik)
                fund_name = fund_lookup.get(cik, f"CIK:{cik}")
                total_value_thousands += h.value_thousands
                total_shares += h.shares_or_prn_amt

                # Check how position changed vs prior quarter
                prior_holding = _find_holding_in_fund(
                    prior_holdings.get(cik, []), cusips
                )

                if prior_holding is None:
                    # New position — fund initiated this quarter
                    funds_initiated.append(fund_name)
                    funds_holding.append({
                        "fund": fund_name,
                        "action": "initiated",
                        "value_thousands": h.value_thousands,
                        "shares": h.shares_or_prn_amt,
                    })
                elif h.shares_or_prn_amt > prior_holding.shares_or_prn_amt:
                    # Added to position
                    prior_shares = prior_holding.shares_or_prn_amt
                    if prior_shares > 0:
                        pct_change = (
                            (h.shares_or_prn_amt - prior_shares) / prior_shares
                        )
                    else:
                        pct_change = 1.0
                    funds_added.append(fund_name)
                    if pct_change >= 0.50:
                        high_conviction_adds.append(fund_name)
                    funds_holding.append({
                        "fund": fund_name,
                        "action": "added",
                        "pct_change": round(pct_change * 100, 1),
                        "value_thousands": h.value_thousands,
                        "shares": h.shares_or_prn_amt,
                    })
                elif h.shares_or_prn_amt < prior_holding.shares_or_prn_amt:
                    # Trimmed position
                    funds_trimmed.append(fund_name)
                    funds_holding.append({
                        "fund": fund_name,
                        "action": "trimmed",
                        "value_thousands": h.value_thousands,
                        "shares": h.shares_or_prn_amt,
                    })
                else:
                    # Unchanged
                    funds_holding.append({
                        "fund": fund_name,
                        "action": "unchanged",
                        "value_thousands": h.value_thousands,
                        "shares": h.shares_or_prn_amt,
                    })
                break  # One match per fund is enough

    # Detect exits: funds that held in prior quarter but not in current
    for cik in prior_ciks_with_ticker - current_ciks_with_ticker:
        fund_name = fund_lookup.get(cik)
        if not fund_name:
            fund = store.get_fund(cik)
            fund_name = fund.name if fund else f"CIK:{cik}"
        funds_exited.append(fund_name)

    # Compute net sentiment
    total_buying = len(funds_initiated) + len(funds_added)
    total_selling = len(funds_exited) + len(funds_trimmed)
    net_sentiment = total_buying - total_selling

    # Check crowding risk via sector_map float data
    crowding_risk = False
    float_ownership_pct = None
    try:
        sector_info = store.get_sector_info(ticker)
        if sector_info and sector_info.get("float_shares"):
            float_shares = sector_info["float_shares"]
            if float_shares > 0 and total_shares > 0:
                float_ownership_pct = round(
                    total_shares / float_shares * 100, 2
                )
                crowding_risk = float_ownership_pct >= 5.0
    except Exception:
        pass  # Float data is nice-to-have, not critical

    aggregate_value_millions = round(total_value_thousands / 1000, 2)

    # Build plain-English summary
    summary = _build_summary(
        ticker=ticker,
        quarter=latest_q,
        funds_initiated=funds_initiated,
        funds_added=funds_added,
        high_conviction_adds=high_conviction_adds,
        funds_trimmed=funds_trimmed,
        funds_exited=funds_exited,
        total_funds=len(current_ciks_with_ticker),
        net_sentiment=net_sentiment,
        crowding_risk=crowding_risk,
        float_ownership_pct=float_ownership_pct,
        aggregate_value_millions=aggregate_value_millions,
    )

    return {
        "available": True,
        "quarter": latest_q.isoformat(),
        "consensus_buys": funds_initiated + funds_added,
        "high_conviction_adds": high_conviction_adds,
        "new_positions": funds_initiated,
        "exits": funds_exited,
        "net_sentiment": net_sentiment,
        "crowding_risk": crowding_risk,
        "float_ownership_pct": float_ownership_pct,
        "total_funds_holding": len(current_ciks_with_ticker),
        "aggregate_value_millions": aggregate_value_millions,
        "fund_details": funds_holding,
        "summary": summary,
    }


def _get_cusips_for_ticker(store: Any, ticker: str) -> set[str]:
    """Reverse-lookup: find CUSIP(s) mapped to a ticker.

    Uses the cusip_map table in the fund-tracker database.
    A ticker may map to multiple CUSIPs (e.g., different share classes).
    """
    try:
        conn = store._conn
        rows = conn.execute(
            "SELECT cusip FROM cusip_map WHERE UPPER(ticker) = ?",
            (ticker.upper(),),
        ).fetchall()
        return {r["cusip"] for r in rows}
    except Exception:
        return set()


def _find_holding_in_fund(
    holdings: list, cusips: set[str]
) -> Any | None:
    """Find the first equity holding matching one of the target CUSIPs."""
    for h in holdings:
        if h.cusip in cusips and not h.is_option:
            return h
    return None


def _build_summary(
    ticker: str,
    quarter: Any,
    funds_initiated: list[str],
    funds_added: list[str],
    high_conviction_adds: list[str],
    funds_trimmed: list[str],
    funds_exited: list[str],
    total_funds: int,
    net_sentiment: int,
    crowding_risk: bool,
    float_ownership_pct: float | None,
    aggregate_value_millions: float,
) -> str:
    """Build a plain-English summary for agent prompt injection."""
    parts = []
    parts.append(
        f"13F CONVICTION SIGNALS ({ticker}, Q{_quarter_label(quarter)}):"
    )

    if total_funds == 0:
        parts.append(
            f"  No tracked hedge funds hold {ticker} as of this quarter."
        )
        return "\n".join(parts)

    parts.append(
        f"  {total_funds} tracked fund(s) hold {ticker} "
        f"(~${aggregate_value_millions:.1f}M aggregate)."
    )

    if funds_initiated:
        names = ", ".join(funds_initiated[:4])
        if len(funds_initiated) > 4:
            names += f" +{len(funds_initiated) - 4}"
        parts.append(f"  NEW POSITIONS: {names}")

    if high_conviction_adds:
        names = ", ".join(high_conviction_adds[:4])
        parts.append(f"  HIGH-CONVICTION ADDS (>50% increase): {names}")
    elif funds_added:
        names = ", ".join(funds_added[:4])
        parts.append(f"  ADDED: {names}")

    if funds_exited:
        names = ", ".join(funds_exited[:4])
        parts.append(f"  EXITS: {names}")

    if funds_trimmed:
        names = ", ".join(funds_trimmed[:4])
        parts.append(f"  TRIMMED: {names}")

    # Net sentiment
    if net_sentiment > 0:
        parts.append(
            f"  NET SENTIMENT: +{net_sentiment} "
            f"(more funds buying than selling — likely bullish signal)"
        )
    elif net_sentiment < 0:
        parts.append(
            f"  NET SENTIMENT: {net_sentiment} "
            f"(more funds selling than buying — warrants caution)"
        )
    else:
        parts.append("  NET SENTIMENT: neutral (balanced buying/selling)")

    # Crowding risk
    if crowding_risk and float_ownership_pct is not None:
        parts.append(
            f"  CROWDING RISK: tracked funds own ~{float_ownership_pct:.1f}% "
            f"of float — elevated crowding. Exits by multiple funds "
            f"could amplify selling pressure."
        )
    elif float_ownership_pct is not None:
        parts.append(
            f"  Float ownership by tracked funds: ~{float_ownership_pct:.1f}%"
        )

    return "\n".join(parts)


def _quarter_label(quarter_date: Any) -> str:
    """Convert a date to quarter label like '3 2025'."""
    try:
        month = quarter_date.month
        year = quarter_date.year
        q = (month - 1) // 3 + 1
        return f"{q} {year}"
    except Exception:
        return str(quarter_date)
