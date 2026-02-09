"""
Insider data tool using yfinance.

Retrieves insider transactions (buys, sells) and computes
net insider sentiment — a powerful signal when insiders
are buying or selling their own stock in size.
"""

from __future__ import annotations

import logging
from typing import Any

import yfinance as yf

logger = logging.getLogger(__name__)


class InsiderDataTool:
    """Retrieves insider transaction data for sentiment analysis."""

    @staticmethod
    def get_insider_activity(ticker: str, max_transactions: int = 20) -> dict[str, Any]:
        """
        Get recent insider transactions and compute net sentiment.

        Args:
            ticker: Stock ticker symbol
            max_transactions: Maximum number of transactions to return

        Returns:
            Dict with transactions list, summary stats, and net sentiment.
        """
        try:
            stock = yf.Ticker(ticker)

            # Get insider transactions
            insider_df = stock.insider_transactions
            if insider_df is None or insider_df.empty:
                return {
                    "ticker": ticker,
                    "transactions": [],
                    "summary": {
                        "total_transactions": 0,
                        "buys": 0,
                        "sells": 0,
                        "net_shares": 0,
                    },
                    "net_sentiment": "neutral",
                    "note": "No insider transaction data available",
                }

            # Process transactions
            transactions = []
            total_buys = 0
            total_sells = 0
            buy_value = 0.0
            sell_value = 0.0

            for _, row in insider_df.head(max_transactions).iterrows():
                text = str(row.get("Text", "")).lower()
                shares = row.get("Shares", 0) or 0
                value = row.get("Value", 0) or 0

                # Classify transaction type
                if any(w in text for w in ["purchase", "buy", "acquisition"]):
                    tx_type = "BUY"
                    total_buys += 1
                    buy_value += abs(float(value))
                elif any(w in text for w in ["sale", "sell", "disposition"]):
                    tx_type = "SELL"
                    total_sells += 1
                    sell_value += abs(float(value))
                else:
                    tx_type = "OTHER"

                transactions.append({
                    "insider": str(row.get("Insider", "Unknown")),
                    "relation": str(row.get("Position", row.get("Insider Trading", ""))),
                    "transaction_type": tx_type,
                    "text": str(row.get("Text", "")),
                    "shares": int(shares) if shares else 0,
                    "value": float(value) if value else 0,
                    "date": str(row.get("Start Date", row.get("Date", ""))),
                })

            # Compute net sentiment
            if total_buys > total_sells * 2:
                sentiment = "bullish"
            elif total_sells > total_buys * 2:
                sentiment = "bearish"
            elif total_buys > total_sells:
                sentiment = "slightly_bullish"
            elif total_sells > total_buys:
                sentiment = "slightly_bearish"
            else:
                sentiment = "neutral"

            return {
                "ticker": ticker,
                "transactions": transactions,
                "summary": {
                    "total_transactions": len(transactions),
                    "buys": total_buys,
                    "sells": total_sells,
                    "buy_value": round(buy_value, 2),
                    "sell_value": round(sell_value, 2),
                    "net_value": round(buy_value - sell_value, 2),
                },
                "net_sentiment": sentiment,
            }

        except Exception as e:
            logger.error(f"Failed to get insider data for {ticker}: {e}")
            return {
                "ticker": ticker,
                "transactions": [],
                "summary": {"total_transactions": 0},
                "net_sentiment": "unknown",
                "error": str(e),
            }
