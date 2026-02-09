"""
News retrieval tool using RSS feeds.

Provides recent news headlines and summaries for agents
to incorporate into their analysis. Uses free sources only
(no API keys required for basic functionality).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import feedparser

logger = logging.getLogger(__name__)

# Free RSS feeds that cover financial news
RSS_FEEDS = {
    "google_finance": "https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en",
    "yahoo_finance": "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
}


class NewsRetrievalTool:
    """Retrieves recent financial news via RSS feeds."""

    @staticmethod
    def get_news(ticker: str, max_articles: int = 10) -> list[dict[str, Any]]:
        """
        Fetch recent news articles for a given ticker.

        Args:
            ticker: Stock ticker symbol
            max_articles: Maximum number of articles to return

        Returns:
            List of article dicts with title, source, published, summary
        """
        articles = []

        for feed_name, feed_url in RSS_FEEDS.items():
            try:
                url = feed_url.format(ticker=ticker)
                feed = feedparser.parse(url)

                for entry in feed.entries[:max_articles]:
                    article = {
                        "title": entry.get("title", ""),
                        "source": feed_name,
                        "published": entry.get("published", ""),
                        "link": entry.get("link", ""),
                        "summary": _clean_summary(entry.get("summary", "")),
                    }

                    # Try to parse the date
                    if entry.get("published_parsed"):
                        try:
                            article["published_dt"] = datetime(
                                *entry.published_parsed[:6]
                            ).isoformat()
                        except (TypeError, ValueError):
                            article["published_dt"] = None

                    articles.append(article)

            except Exception as e:
                logger.warning(f"Failed to fetch {feed_name} feed for {ticker}: {e}")
                continue

        # Sort by published date (newest first), deduplicate by title
        seen_titles = set()
        unique_articles = []
        for article in articles:
            title_key = article["title"].lower().strip()
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_articles.append(article)

        return unique_articles[:max_articles]

    @staticmethod
    def format_for_agent(articles: list[dict[str, Any]]) -> list[str]:
        """
        Format articles into concise strings for agent consumption.

        Returns a list of formatted headline strings.
        """
        formatted = []
        for i, article in enumerate(articles, 1):
            headline = article["title"]
            source = article.get("source", "")
            date = article.get("published_dt", article.get("published", ""))
            formatted.append(f"[{i}] {headline} ({source}, {date})")
        return formatted


def _clean_summary(html_summary: str) -> str:
    """Remove HTML tags from RSS summary text."""
    import re
    clean = re.sub(r"<[^>]+>", "", html_summary)
    clean = clean.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    clean = clean.replace("&quot;", '"').replace("&#39;", "'")
    return clean.strip()[:500]  # Cap at 500 chars
