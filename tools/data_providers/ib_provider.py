"""Interactive Brokers data provider via ib_insync.

Requires:
    - IB account (paper or live)
    - TWS or IB Gateway running locally
    - ib_insync: ``pip install ib_insync``

Connection ports:
    - 7497: TWS paper trading (default)
    - 7496: TWS live
    - 4002: Gateway paper
    - 4001: Gateway live

All methods return the **same dict schemas** as YahooProvider so the
tool layer works identically regardless of data source.

Usage:
    from tools.data_providers.ib_provider import IBProvider
    provider = IBProvider(port=7497)   # paper trading
    overview = provider.get_company_overview("AAPL")
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd

from tools.data_providers.base import MarketDataProvider
from tools.data_providers.yahoo_provider import _format_large_number, _pct

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional import — ib_insync may not be installed
# ---------------------------------------------------------------------------

try:
    from ib_insync import IB, Stock, util

    _HAS_IB = True
except ImportError:
    IB = None  # type: ignore[assignment, misc]
    Stock = None  # type: ignore[assignment, misc]
    util = None  # type: ignore[assignment]
    _HAS_IB = False


def is_available() -> bool:
    """Return True if ib_insync is importable."""
    return _HAS_IB


# ---------------------------------------------------------------------------
# Connection manager
# ---------------------------------------------------------------------------

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 7497
_DEFAULT_CLIENT_ID = 10


class _IBConnection:
    """Managed IB connection with auto-connect and reconnect."""

    def __init__(
        self,
        host: str = _DEFAULT_HOST,
        port: int = _DEFAULT_PORT,
        client_id: int = _DEFAULT_CLIENT_ID,
        timeout: int = 15,
        readonly: bool = True,
    ):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.timeout = timeout
        self.readonly = readonly
        self._ib: Any = None

        if not _HAS_IB:
            raise ImportError(
                "ib_insync is not installed. Install with: pip install ib_insync"
            )

    def connect(self) -> Any:
        """Return a connected IB instance, reconnecting if needed."""
        if self._ib is not None and self._ib.isConnected():
            return self._ib

        ib = IB()
        try:
            ib.connect(
                host=self.host, port=self.port,
                clientId=self.client_id, timeout=self.timeout,
                readonly=self.readonly,
            )
        except Exception as exc:
            raise ConnectionError(
                f"Cannot connect to IB on {self.host}:{self.port}. "
                f"Ensure TWS/Gateway is running. Error: {exc}"
            ) from exc

        logger.info(
            "IB connected to %s:%d (clientId=%d)",
            self.host, self.port, self.client_id,
        )
        self._ib = ib
        return ib

    def disconnect(self) -> None:
        if self._ib is not None:
            try:
                self._ib.disconnect()
            except Exception:
                pass
            self._ib = None

    def __del__(self) -> None:
        self.disconnect()


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class IBProvider(MarketDataProvider):
    """Fetch market data from Interactive Brokers via ib_insync.

    Requires TWS or IB Gateway running locally.
    Connects in readonly mode by default (no order capability).
    """

    def __init__(
        self,
        host: str = _DEFAULT_HOST,
        port: int = _DEFAULT_PORT,
        client_id: int = _DEFAULT_CLIENT_ID,
        timeout: int = 15,
    ):
        self._conn = _IBConnection(
            host=host, port=port, client_id=client_id,
            timeout=timeout, readonly=True,
        )

    @property
    def name(self) -> str:
        return "Interactive Brokers"

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def _get_bars(self, ticker: str, days: int) -> pd.DataFrame:
        """Fetch daily OHLCV bars and return as pandas DataFrame."""
        try:
            ib = self._conn.connect()
        except (ConnectionError, ImportError) as exc:
            logger.error("IB connection failed: %s", exc)
            return pd.DataFrame()

        contract = Stock(ticker, "SMART", "USD")
        try:
            ib.qualifyContracts(contract)
        except Exception:
            logger.warning("IB cannot qualify contract for %s", ticker)
            return pd.DataFrame()

        duration = _days_to_ib_duration(days)
        end_dt = datetime.now()
        try:
            bars = ib.reqHistoricalData(
                contract, endDateTime=end_dt,
                durationStr=duration, barSizeSetting="1 day",
                whatToShow="ADJUSTED_LAST", useRTH=True,
                formatDate=1, keepUpToDate=False,
            )
        except Exception as exc:
            logger.warning("IB historical data failed for %s: %s", ticker, exc)
            return pd.DataFrame()

        if not bars:
            return pd.DataFrame()

        rows = []
        for bar in bars:
            rows.append({
                "Open": float(bar.open),
                "High": float(bar.high),
                "Low": float(bar.low),
                "Close": float(bar.close),
                "Volume": float(bar.volume),
            })
        ib.sleep(0.5)  # pacing
        return pd.DataFrame(rows)

    def _get_contract_info(self, ticker: str) -> dict[str, Any]:
        """Get contract details and fundamental snapshot."""
        info: dict[str, Any] = {}
        try:
            ib = self._conn.connect()
        except (ConnectionError, ImportError) as exc:
            logger.error("IB connection failed: %s", exc)
            return info

        contract = Stock(ticker, "SMART", "USD")
        try:
            ib.qualifyContracts(contract)
        except Exception:
            return info

        # Contract details → sector / industry
        try:
            details = ib.reqContractDetails(contract)
            if details:
                d = details[0]
                info["sector"] = getattr(d, "category", "") or ""
                info["industry"] = getattr(d, "industry", "") or ""
                info["longName"] = getattr(d, "longName", ticker) or ticker
        except Exception as exc:
            logger.debug("IB contract details failed for %s: %s", ticker, exc)

        # Fundamental XML (requires subscription)
        try:
            xml_str = ib.reqFundamentalData(contract, "ReportSnapshot")
            if xml_str:
                info.update(_parse_fundamentals_xml(xml_str))
        except Exception as exc:
            logger.debug("IB fundamentals unavailable for %s: %s", ticker, exc)

        return info

    # ------------------------------------------------------------------
    # MarketDataProvider interface
    # ------------------------------------------------------------------

    def get_ticker_object(self, ticker: str) -> Any:
        return {"ticker": ticker, "provider": "interactive_brokers"}

    def get_company_overview(self, ticker: str) -> dict[str, Any]:
        info = self._get_contract_info(ticker)
        mktcap = info.get("market_cap", 0)
        return {
            "ticker": ticker,
            "name": info.get("longName", ticker),
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "market_cap": mktcap,
            "market_cap_formatted": _format_large_number(mktcap),
            "currency": "USD",
            "exchange": "SMART",
            "description": "",
            "website": "",
            "employees": None,
            "country": "US",
        }

    def get_price_data(self, ticker: str, period: str = "6mo") -> dict[str, Any]:
        days = _period_to_days(period)
        hist = self._get_bars(ticker, days)

        if hist.empty:
            return {"ticker": ticker, "error": "No price data from IB"}

        current_price = float(hist["Close"].iloc[-1])
        start_price = float(hist["Close"].iloc[0])
        high_52w = float(hist["Close"].max())
        low_52w = float(hist["Close"].min())
        avg_volume = int(hist["Volume"].mean())

        return {
            "ticker": ticker,
            "current_price": round(current_price, 2),
            "period_return_pct": round((current_price / start_price - 1) * 100, 2),
            "high_52w": round(high_52w, 2),
            "low_52w": round(low_52w, 2),
            "pct_from_high": round((current_price / high_52w - 1) * 100, 2),
            "pct_from_low": round((current_price / low_52w - 1) * 100, 2),
            "avg_daily_volume": avg_volume,
            "avg_volume_formatted": _format_large_number(avg_volume),
            "period": period,
            "data_points": len(hist),
        }

    def get_fundamentals(self, ticker: str) -> dict[str, Any]:
        info = self._get_contract_info(ticker)
        return {
            "ticker": ticker,
            "pe_trailing": info.get("pe_ratio"),
            "pe_forward": None,  # IB doesn't provide forward PE in snapshot
            "peg_ratio": None,
            "price_to_book": info.get("price_to_book"),
            "price_to_sales": None,
            "ev_to_ebitda": None,
            "ev_to_revenue": None,
            "profit_margin": _pct(info.get("profit_margin")) if info.get("profit_margin") else None,
            "operating_margin": None,
            "gross_margin": None,
            "roe": _pct(info.get("roe")) if info.get("roe") else None,
            "roa": None,
            "revenue_growth": None,
            "earnings_growth": None,
            "revenue": info.get("revenue"),
            "revenue_formatted": _format_large_number(info.get("revenue")),
            "ebitda": info.get("ebitda"),
            "ebitda_formatted": _format_large_number(info.get("ebitda")),
            "net_income": info.get("net_income"),
            "total_debt": None,
            "total_debt_formatted": None,
            "total_cash": None,
            "total_cash_formatted": None,
            "debt_to_equity": info.get("debt_to_equity"),
            "current_ratio": None,
            "dividend_yield": _pct(info.get("dividend_yield")) if info.get("dividend_yield") else None,
            "payout_ratio": None,
            "target_mean_price": None,
            "target_high_price": None,
            "target_low_price": None,
            "recommendation": None,
            "num_analysts": None,
        }

    def get_info(self, ticker: str) -> dict[str, Any]:
        """Return a yfinance-compatible info dict using IB data."""
        info = self._get_contract_info(ticker)
        return {
            "longName": info.get("longName", ticker),
            "shortName": info.get("longName", ticker),
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "marketCap": info.get("market_cap"),
            "trailingPE": info.get("pe_ratio"),
            "forwardPE": None,
            "pegRatio": None,
            "priceToBook": info.get("price_to_book"),
            "enterpriseToEbitda": None,
            "profitMargins": info.get("profit_margin"),
            "revenueGrowth": None,
            "returnOnEquity": info.get("roe"),
            "dividendYield": info.get("dividend_yield"),
            "debtToEquity": info.get("debt_to_equity"),
            "recommendationKey": None,
        }

    def get_insider_transactions(self, ticker: str) -> Any:
        """IB does not provide insider transaction data."""
        logger.info("Insider transactions not available via IB for %s", ticker)
        return None

    def get_earnings_history(self, ticker: str) -> Any:
        """IB does not provide earnings surprise history."""
        logger.info("Earnings history not available via IB for %s", ticker)
        return None

    def get_quarterly_earnings(self, ticker: str) -> Any:
        return None

    def get_history(self, ticker: str, period: str = "6mo") -> Any:
        days = _period_to_days(period)
        return self._get_bars(ticker, days)

    def disconnect(self) -> None:
        self._conn.disconnect()

    def close(self) -> None:
        self.disconnect()


# ---------------------------------------------------------------------------
# XML parser for IB ReportSnapshot fundamentals
# ---------------------------------------------------------------------------


def _parse_fundamentals_xml(xml_str: str) -> dict[str, Any]:
    """Parse IB's ReportSnapshot XML into flat dict."""
    result: dict[str, Any] = {}
    try:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(xml_str)
        for ratio in root.iter("Ratio"):
            field = ratio.get("FieldName", "")
            val = ratio.text
            if not val:
                continue
            try:
                if field == "MKTCAP":
                    result["market_cap"] = float(val) * 1e6
                elif field == "BETA":
                    result["beta"] = float(val)
                elif field == "YIELD":
                    result["dividend_yield"] = float(val) / 100.0
                elif field == "SHARESOUT":
                    result["shares_outstanding"] = int(float(val) * 1e6)
                elif field == "AVOLUME":
                    result["avg_volume"] = int(float(val))
                elif field == "TTMREV":
                    result["revenue"] = float(val) * 1e6
                elif field == "TTMEBITD":
                    result["ebitda"] = float(val) * 1e6
                elif field == "TTMNIAC":
                    result["net_income"] = float(val) * 1e6
                elif field == "PEEXCLXOR":
                    result["pe_ratio"] = float(val)
                elif field == "PRICE2BK":
                    result["price_to_book"] = float(val)
                elif field == "TTMPROFM":
                    result["profit_margin"] = float(val) / 100.0
                elif field == "TTMROEPCT":
                    result["roe"] = float(val) / 100.0
                elif field == "QTOTD2EQ":
                    result["debt_to_equity"] = float(val)
            except (ValueError, TypeError):
                continue
    except Exception as exc:
        logger.debug("Failed to parse IB fundamentals XML: %s", exc)
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _period_to_days(period: str) -> int:
    mapping = {
        "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365,
        "2y": 730, "5y": 1825, "10y": 3650, "ytd": 180, "max": 3650,
    }
    return mapping.get(period, 180)


def _days_to_ib_duration(days: int) -> str:
    """Convert day count to IB durationStr (e.g. '1 Y', '6 M', '30 D')."""
    if days > 365:
        return f"{min(days // 365, 5)} Y"
    if days > 30:
        return f"{days // 30} M"
    return f"{days} D"
