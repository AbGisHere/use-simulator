import logging
from datetime import datetime, timedelta
from typing import Any

import requests

from config import IST, NEWSAPI_KEY

logger = logging.getLogger(__name__)

NEWSAPI_URL = "https://newsapi.org/v2/everything"

MACRO_KEYWORDS = [
    "RBI",
    "Reserve Bank of India",
    "India GDP",
    "Union Budget",
    "SEBI",
    "Nifty",
    "Sensex",
    "Indian economy",
    "repo rate",
    "monetary policy",
]

INDIA_SOURCES = [
    "the-times-of-india",
    "the-hindu",
    "ndtv",
    "india-today",
    "business-standard",
]


def _parse_newsapi_date(date_str: str) -> datetime:
    """Parse NewsAPI ISO datetime string to IST datetime."""
    try:
        # Remove trailing Z and parse
        clean = date_str.rstrip("Z").replace("T", " ")[:19]
        dt = datetime.strptime(clean, "%Y-%m-%d %H:%M:%S")
        # NewsAPI returns UTC
        from zoneinfo import ZoneInfo
        dt_utc = dt.replace(tzinfo=ZoneInfo("UTC"))
        return dt_utc.astimezone(IST)
    except Exception:
        return datetime.now(tz=IST)


def fetch_macro_news(days_back: int = 30, max_articles: int = 100) -> list[dict[str, Any]]:
    """
    Fetch Indian macro news from NewsAPI.

    Returns list of dicts:
        {headline, description, published_at (IST), source, url}
    """
    if not NEWSAPI_KEY:
        logger.warning("NEWSAPI_KEY not configured. Skipping macro news fetch.")
        return []

    from_date = (datetime.now(tz=IST) - timedelta(days=days_back)).strftime("%Y-%m-%d")
    query = " OR ".join(f'"{kw}"' for kw in MACRO_KEYWORDS[:6])  # NewsAPI query limit

    results: list[dict[str, Any]] = []

    try:
        params = {
            "q": query,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": min(max_articles, 100),
            "apiKey": NEWSAPI_KEY,
        }
        resp = requests.get(NEWSAPI_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") != "ok":
            logger.warning("NewsAPI returned status: %s", data.get("status"))
            return results

        for article in data.get("articles", []):
            published_at = _parse_newsapi_date(article.get("publishedAt", ""))
            results.append(
                {
                    "headline": article.get("title", ""),
                    "description": article.get("description", "") or "",
                    "body": article.get("content", "") or "",
                    "published_at": published_at,
                    "source": article.get("source", {}).get("name", "NewsAPI"),
                    "url": article.get("url", ""),
                    "ticker": "MACRO",  # macro news tagged separately
                }
            )

    except requests.exceptions.HTTPError as exc:
        logger.error("NewsAPI HTTP error: %s", exc)
    except Exception as exc:
        logger.error("NewsAPI fetch error: %s", exc)

    logger.info("Fetched %d macro news articles", len(results))
    return results


def fetch_ticker_news_via_newsapi(ticker: str, days_back: int = 30) -> list[dict[str, Any]]:
    """
    Fetch ticker-specific news from NewsAPI as a supplemental source.
    """
    if not NEWSAPI_KEY:
        return []

    symbol = ticker.upper().replace(".NS", "").replace(".BO", "")
    from_date = (datetime.now(tz=IST) - timedelta(days=days_back)).strftime("%Y-%m-%d")

    results: list[dict[str, Any]] = []

    try:
        params = {
            "q": f'"{symbol}" India stock OR NSE',
            "language": "en",
            "sortBy": "relevancy",
            "pageSize": 30,
            "apiKey": NEWSAPI_KEY,
        }
        resp = requests.get(NEWSAPI_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") != "ok":
            return results

        for article in data.get("articles", []):
            published_at = _parse_newsapi_date(article.get("publishedAt", ""))
            results.append(
                {
                    "headline": article.get("title", ""),
                    "description": article.get("description", "") or "",
                    "body": article.get("content", "") or "",
                    "published_at": published_at,
                    "source": article.get("source", {}).get("name", "NewsAPI"),
                    "url": article.get("url", ""),
                    "ticker": symbol,
                }
            )

    except Exception as exc:
        logger.warning("NewsAPI ticker fetch error for %s: %s", symbol, exc)

    return results
