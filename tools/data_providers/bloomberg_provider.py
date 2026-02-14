"""Bloomberg Professional API data provider.

Requires:
    - Bloomberg Terminal or B-PIPE running locally
    - blpapi Python SDK: ``pip install blpapi``
    - DAPI entitlement on the Terminal

Connection: localhost:8194 (default DAPI port).

All methods return the **same dict schemas** as YahooProvider so the
tool layer works identically regardless of data source.

Usage:
    from tools.data_providers.bloomberg_provider import BloombergProvider
    provider = BloombergProvider()
    overview = provider.get_company_overview("AAPL")
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

import pandas as pd

from tools.data_providers.base import MarketDataProvider
from tools.data_providers.yahoo_provider import _format_large_number, _pct

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional import — blpapi may not be installed
# ---------------------------------------------------------------------------

try:
    import blpapi

    _HAS_BLPAPI = True
except ImportError:
    blpapi = None  # type: ignore[assignment]
    _HAS_BLPAPI = False


def is_available() -> bool:
    """Return True if blpapi is importable."""
    return _HAS_BLPAPI


# ---------------------------------------------------------------------------
# Ticker helpers
# ---------------------------------------------------------------------------


def _to_bbg(ticker: str) -> str:
    """Convert plain ticker to Bloomberg identifier (e.g. AAPL → AAPL US Equity)."""
    if " " in ticker:
        return ticker
    return f"{ticker} US Equity"


def _from_bbg(bbg_ticker: str) -> str:
    """Strip Bloomberg suffix back to plain ticker."""
    return bbg_ticker.split()[0] if bbg_ticker else bbg_ticker


# ---------------------------------------------------------------------------
# Session manager
# ---------------------------------------------------------------------------


class _BloombergSession:
    """Managed Bloomberg API session with auto-connect."""

    def __init__(self, host: str = "localhost", port: int = 8194):
        self.host = host
        self.port = port
        self._session: Any = None

        if not _HAS_BLPAPI:
            msg = (
                "blpapi is not installed. Install with: pip install blpapi "
                "(requires Bloomberg C++ SDK headers)"
            )
            raise ImportError(msg)

    def connect(self) -> Any:
        """Open a blpapi session, reusing if already connected."""
        if self._session is not None:
            return self._session

        opts = blpapi.SessionOptions()
        opts.setServerHost(self.host)
        opts.setServerPort(self.port)

        session = blpapi.Session(opts)
        if not session.start():
            raise ConnectionError(
                f"Failed to start Bloomberg session on {self.host}:{self.port}"
            )
        if not session.openService("//blp/refdata"):
            session.stop()
            raise ConnectionError("Failed to open //blp/refdata service")

        logger.info("Bloomberg session connected to %s:%d", self.host, self.port)
        self._session = session
        return session

    def close(self) -> None:
        if self._session is not None:
            try:
                self._session.stop()
            except Exception:
                pass
            self._session = None

    def __del__(self) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Safe element extraction
# ---------------------------------------------------------------------------


def _safe_float(element: Any, field: str, default: float = 0.0) -> float:
    try:
        if element.hasElement(field):
            return float(element.getElementAsFloat64(field))
    except Exception:
        pass
    return default


def _safe_str(element: Any, field: str, default: str = "") -> str:
    try:
        if element.hasElement(field):
            return str(element.getElementAsString(field))
    except Exception:
        pass
    return default


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class BloombergProvider(MarketDataProvider):
    """Fetch market data from Bloomberg Professional API (DAPI/B-PIPE).

    Requires a running Bloomberg Terminal or B-PIPE connection.
    Falls back gracefully with clear error messages when unavailable.
    """

    def __init__(self, host: str = "localhost", port: int = 8194):
        self._bbg = _BloombergSession(host, port)

    @property
    def name(self) -> str:
        return "Bloomberg"

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def _ref_request(self, ticker: str, fields: list[str]) -> dict[str, Any]:
        """Execute a ReferenceDataRequest and return field→value dict."""
        result: dict[str, Any] = {}
        try:
            session = self._bbg.connect()
        except (ConnectionError, ImportError) as exc:
            logger.error("Bloomberg connection failed: %s", exc)
            return result

        refdata = session.getService("//blp/refdata")
        request = refdata.createRequest("ReferenceDataRequest")
        request.append("securities", _to_bbg(ticker))
        for f in fields:
            request.append("fields", f)

        session.sendRequest(request)
        done = False
        while not done:
            event = session.nextEvent(5000)
            for msg in event:
                if msg.messageType() == blpapi.Name("ReferenceDataResponse"):
                    arr = msg.getElement("securityData")
                    if arr.numValues() == 0:
                        continue
                    sec = arr.getValueAsElement(0)
                    if sec.hasElement("securityError"):
                        continue
                    fd = sec.getElement("fieldData")
                    for f in fields:
                        if fd.hasElement(f):
                            try:
                                result[f] = fd.getElementAsFloat64(f)
                            except Exception:
                                try:
                                    result[f] = fd.getElementAsString(f)
                                except Exception:
                                    pass
            if event.eventType() == blpapi.Event.RESPONSE:
                done = True
        return result

    def _hist_request(
        self, ticker: str, start: date, end: date
    ) -> pd.DataFrame:
        """Fetch daily OHLCV via HistoricalDataRequest, return pandas DataFrame."""
        try:
            session = self._bbg.connect()
        except (ConnectionError, ImportError) as exc:
            logger.error("Bloomberg connection failed: %s", exc)
            return pd.DataFrame()

        refdata = session.getService("//blp/refdata")
        request = refdata.createRequest("HistoricalDataRequest")
        request.append("securities", _to_bbg(ticker))
        for f in ("PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST", "VOLUME"):
            request.append("fields", f)
        request.set("startDate", start.strftime("%Y%m%d"))
        request.set("endDate", end.strftime("%Y%m%d"))
        request.set("periodicitySelection", "DAILY")
        request.set("adjustmentNormal", True)
        request.set("adjustmentAbnormal", True)
        request.set("adjustmentSplit", True)

        session.sendRequest(request)
        rows: list[dict] = []
        done = False
        while not done:
            event = session.nextEvent(5000)
            for msg in event:
                if msg.messageType() == blpapi.Name("HistoricalDataResponse"):
                    sec_data = msg.getElement("securityData")
                    if sec_data.hasElement("securityError"):
                        continue
                    field_data = sec_data.getElement("fieldData")
                    for i in range(field_data.numValues()):
                        bar = field_data.getValueAsElement(i)
                        dt = bar.getElementAsDatetime("date")
                        rows.append({
                            "Open": _safe_float(bar, "PX_OPEN"),
                            "High": _safe_float(bar, "PX_HIGH"),
                            "Low": _safe_float(bar, "PX_LOW"),
                            "Close": _safe_float(bar, "PX_LAST"),
                            "Volume": _safe_float(bar, "VOLUME"),
                        })
            if event.eventType() == blpapi.Event.RESPONSE:
                done = True

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # MarketDataProvider interface
    # ------------------------------------------------------------------

    def get_ticker_object(self, ticker: str) -> Any:
        """Not applicable for Bloomberg — returns a dict proxy."""
        return {"ticker": ticker, "provider": "bloomberg"}

    def get_company_overview(self, ticker: str) -> dict[str, Any]:
        fields = [
            "LONG_COMP_NAME", "GICS_SECTOR_NAME", "GICS_INDUSTRY_NAME",
            "CUR_MKT_CAP", "CRNCY", "EXCH_CODE", "NUM_OF_EMPLOYEES",
            "COUNTRY_FULL_NAME",
        ]
        ref = self._ref_request(ticker, fields)
        mktcap = ref.get("CUR_MKT_CAP", 0)
        if isinstance(mktcap, (int, float)):
            mktcap = mktcap * 1e6  # BBG reports in millions
        return {
            "ticker": ticker,
            "name": ref.get("LONG_COMP_NAME", ticker),
            "sector": ref.get("GICS_SECTOR_NAME", "Unknown"),
            "industry": ref.get("GICS_INDUSTRY_NAME", "Unknown"),
            "market_cap": mktcap,
            "market_cap_formatted": _format_large_number(mktcap),
            "currency": ref.get("CRNCY", "USD"),
            "exchange": ref.get("EXCH_CODE", "Unknown"),
            "description": "",  # Bloomberg doesn't provide a text summary
            "website": "",
            "employees": ref.get("NUM_OF_EMPLOYEES"),
            "country": ref.get("COUNTRY_FULL_NAME", "Unknown"),
        }

    def get_price_data(self, ticker: str, period: str = "6mo") -> dict[str, Any]:
        days = _period_to_days(period)
        end = date.today()
        start = end - timedelta(days=days)
        hist = self._hist_request(ticker, start, end)

        if hist.empty:
            return {"ticker": ticker, "error": "No price data from Bloomberg"}

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
        fields = [
            "PE_RATIO", "BEST_PE_RATIO", "BEST_PEG_RATIO",
            "PX_TO_BOOK_RATIO", "PX_TO_SALES_RATIO",
            "BEST_CUR_EV_TO_EBITDA", "CUR_EV_TO_T12M_REVENUE",
            "PROF_MARGIN", "OPER_MARGIN", "GROSS_MARGIN",
            "RETURN_COM_EQY", "RETURN_ON_ASSET",
            "SALES_GROWTH", "TRAIL_12M_EPS_GROWTH",
            "SALES_REV_TURN", "EBITDA", "NET_INCOME",
            "TOT_DEBT_TO_TOT_ASSET", "BS_TOT_CASH", "CUR_RATIO",
            "EQY_DVD_YLD_IND", "PAYOUT_RATIO_ADJUSTED",
            "BEST_TARGET_PRICE", "BEST_TARGET_HIGH", "BEST_TARGET_LOW",
            "BEST_ANALYST_RATING", "TOT_ANALYST_REC",
        ]
        ref = self._ref_request(ticker, fields)

        def _val(key: str) -> Any:
            return ref.get(key)

        def _pct_val(key: str) -> str | None:
            v = ref.get(key)
            if v is None:
                return None
            return f"{float(v):.1f}%"

        return {
            "ticker": ticker,
            "pe_trailing": _val("PE_RATIO"),
            "pe_forward": _val("BEST_PE_RATIO"),
            "peg_ratio": _val("BEST_PEG_RATIO"),
            "price_to_book": _val("PX_TO_BOOK_RATIO"),
            "price_to_sales": _val("PX_TO_SALES_RATIO"),
            "ev_to_ebitda": _val("BEST_CUR_EV_TO_EBITDA"),
            "ev_to_revenue": _val("CUR_EV_TO_T12M_REVENUE"),
            "profit_margin": _pct_val("PROF_MARGIN"),
            "operating_margin": _pct_val("OPER_MARGIN"),
            "gross_margin": _pct_val("GROSS_MARGIN"),
            "roe": _pct_val("RETURN_COM_EQY"),
            "roa": _pct_val("RETURN_ON_ASSET"),
            "revenue_growth": _pct_val("SALES_GROWTH"),
            "earnings_growth": _pct_val("TRAIL_12M_EPS_GROWTH"),
            "revenue": _val("SALES_REV_TURN"),
            "revenue_formatted": _format_large_number(ref.get("SALES_REV_TURN")),
            "ebitda": _val("EBITDA"),
            "ebitda_formatted": _format_large_number(ref.get("EBITDA")),
            "net_income": _val("NET_INCOME"),
            "total_debt": None,  # derived differently in BBG
            "total_debt_formatted": None,
            "total_cash": _val("BS_TOT_CASH"),
            "total_cash_formatted": _format_large_number(ref.get("BS_TOT_CASH")),
            "debt_to_equity": _val("TOT_DEBT_TO_TOT_ASSET"),
            "current_ratio": _val("CUR_RATIO"),
            "dividend_yield": _pct_val("EQY_DVD_YLD_IND"),
            "payout_ratio": _pct_val("PAYOUT_RATIO_ADJUSTED"),
            "target_mean_price": _val("BEST_TARGET_PRICE"),
            "target_high_price": _val("BEST_TARGET_HIGH"),
            "target_low_price": _val("BEST_TARGET_LOW"),
            "recommendation": _val("BEST_ANALYST_RATING"),
            "num_analysts": _val("TOT_ANALYST_REC"),
        }

    def get_info(self, ticker: str) -> dict[str, Any]:
        """Return a yfinance-compatible info dict using Bloomberg data."""
        fields = [
            "LONG_COMP_NAME", "GICS_SECTOR_NAME", "GICS_INDUSTRY_NAME",
            "CUR_MKT_CAP", "PE_RATIO", "BEST_PE_RATIO", "BEST_PEG_RATIO",
            "PX_TO_BOOK_RATIO", "BEST_CUR_EV_TO_EBITDA",
            "PROF_MARGIN", "SALES_GROWTH", "RETURN_COM_EQY",
            "EQY_DVD_YLD_IND", "TOT_DEBT_TO_TOT_ASSET",
            "BEST_ANALYST_RATING",
        ]
        ref = self._ref_request(ticker, fields)
        mktcap = ref.get("CUR_MKT_CAP", 0)
        if isinstance(mktcap, (int, float)):
            mktcap = mktcap * 1e6

        return {
            "longName": ref.get("LONG_COMP_NAME", ticker),
            "shortName": ref.get("LONG_COMP_NAME", ticker),
            "sector": ref.get("GICS_SECTOR_NAME", "Unknown"),
            "industry": ref.get("GICS_INDUSTRY_NAME", "Unknown"),
            "marketCap": mktcap,
            "trailingPE": ref.get("PE_RATIO"),
            "forwardPE": ref.get("BEST_PE_RATIO"),
            "pegRatio": ref.get("BEST_PEG_RATIO"),
            "priceToBook": ref.get("PX_TO_BOOK_RATIO"),
            "enterpriseToEbitda": ref.get("BEST_CUR_EV_TO_EBITDA"),
            "profitMargins": _to_decimal(ref.get("PROF_MARGIN")),
            "revenueGrowth": _to_decimal(ref.get("SALES_GROWTH")),
            "returnOnEquity": _to_decimal(ref.get("RETURN_COM_EQY")),
            "dividendYield": _to_decimal(ref.get("EQY_DVD_YLD_IND")),
            "debtToEquity": ref.get("TOT_DEBT_TO_TOT_ASSET"),
            "recommendationKey": ref.get("BEST_ANALYST_RATING"),
        }

    def get_insider_transactions(self, ticker: str) -> Any:
        """Bloomberg does not provide insider transactions via DAPI."""
        logger.info("Insider transactions not available via Bloomberg DAPI for %s", ticker)
        return None

    def get_earnings_history(self, ticker: str) -> Any:
        """Bloomberg earnings history requires BQL/BDS — not implemented here."""
        logger.info("Earnings history not available via Bloomberg DAPI for %s", ticker)
        return None

    def get_quarterly_earnings(self, ticker: str) -> Any:
        """Bloomberg quarterly earnings requires BQL/BDS — not implemented here."""
        return None

    def get_history(self, ticker: str, period: str = "6mo") -> Any:
        days = _period_to_days(period)
        end = date.today()
        start = end - timedelta(days=days)
        return self._hist_request(ticker, start, end)

    def close(self) -> None:
        """Shut down the Bloomberg session."""
        self._bbg.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _period_to_days(period: str) -> int:
    """Convert yfinance period string to approximate number of calendar days."""
    mapping = {
        "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365,
        "2y": 730, "5y": 1825, "10y": 3650, "ytd": 180, "max": 3650,
    }
    return mapping.get(period, 180)


def _to_decimal(val: Any) -> float | None:
    """Convert Bloomberg percentage (e.g. 25.3) to decimal (0.253)."""
    if val is None:
        return None
    try:
        return float(val) / 100.0
    except (ValueError, TypeError):
        return None
