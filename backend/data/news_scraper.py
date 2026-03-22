import logging
import random
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup

from config import IST, SCRAPE_DELAY_MAX, SCRAPE_DELAY_MIN
from data.sector_taxonomy import get_all_query_terms

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-IN,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Cache-Control": "no-cache",
}


def _polite_delay() -> None:
    time.sleep(random.uniform(SCRAPE_DELAY_MIN, SCRAPE_DELAY_MAX))



def _rfc2822_to_ist(date_str: str) -> datetime:
    """Parse RFC 2822 date string (used in RSS) to IST datetime."""
    try:
        dt = parsedate_to_datetime(date_str)
        return dt.astimezone(IST)
    except Exception:
        return datetime.now(tz=IST)


# ── Google News RSS (primary source — free, no auth, India-focused) ──────────

def _fetch_google_news_rss_raw(
    query_term: str,
    ticker_symbol: str,
    match_type: str = "company",
    max_articles: int = 20,
) -> list[dict[str, Any]]:
    """
    Core Google News RSS fetcher. Fetches for any search term and tags results
    with match_type ('company', 'sector', or 'proxy') for downstream weighting.
    """
    query = quote_plus(query_term)
    rss_url = (
        f"https://news.google.com/rss/search?q={query}"
        f"&hl=en-IN&gl=IN&ceid=IN:en"
    )

    results: list[dict[str, Any]] = []

    try:
        _polite_delay()
        resp = requests.get(rss_url, headers=HEADERS, timeout=15)
        resp.raise_for_status()

        root = ET.fromstring(resp.content)
        channel = root.find("channel")
        if channel is None:
            return results

        items = channel.findall("item")
        for item in items[:max_articles]:
            try:
                headline = item.findtext("title", "").strip()
                url = item.findtext("link", "").strip()
                pub_date_str = item.findtext("pubDate", "")
                pub_date = _rfc2822_to_ist(pub_date_str) if pub_date_str else datetime.now(tz=IST)
                source_el = item.find("source")
                source_name = source_el.text if source_el is not None else "Google News"

                if not headline or len(headline) < 15:
                    continue

                body = item.findtext("description", "") or ""
                if body:
                    body = BeautifulSoup(body, "html.parser").get_text(strip=True)[:2000]

                results.append({
                    "ticker": ticker_symbol,
                    "headline": headline,
                    "body": body,
                    "source": source_name,
                    "published_at": pub_date,
                    "url": url,
                    "match_type": match_type,   # 'company' | 'sector' | 'proxy'
                    "query_term": query_term,    # for debugging
                })
            except Exception as exc:
                logger.debug("Google News RSS item parse error: %s", exc)
                continue

    except Exception as exc:
        logger.warning("Google News RSS fetch error for query '%s': %s", query_term, exc)

    return results


def fetch_google_news_rss(ticker: str, max_articles: int = 20) -> list[dict[str, Any]]:
    """
    Fetch company-level news via Google News RSS.
    Uses sector_taxonomy company_terms when available, falls back to symbol search.
    """
    symbol = ticker.upper().replace(".NS", "").replace(".BO", "")
    terms = get_all_query_terms(symbol)
    company_terms = terms.get("company") or [f"{symbol} NSE India stock"]

    results: list[dict[str, Any]] = []
    per_term = max(5, max_articles // len(company_terms))
    for term in company_terms:
        results.extend(_fetch_google_news_rss_raw(term, symbol, match_type="company", max_articles=per_term))

    logger.info("Fetched %d company Google News RSS articles for %s", len(results), symbol)
    return results


def fetch_sector_news(ticker: str, max_per_term: int = 10) -> list[dict[str, Any]]:
    """
    Fetch sector-level and proxy-event news for a ticker using the taxonomy.
    Returns news tagged with match_type='sector' or match_type='proxy'.

    These are weighted at 0.6x and 0.4x respectively in sentiment aggregation,
    so they influence but don't dominate the daily sentiment score.
    """
    symbol = ticker.upper().replace(".NS", "").replace(".BO", "")
    terms = get_all_query_terms(symbol)

    results: list[dict[str, Any]] = []

    for term in terms.get("sector", []):
        fetched = _fetch_google_news_rss_raw(term, symbol, match_type="sector", max_articles=max_per_term)
        results.extend(fetched)

    for term in terms.get("proxy", []):
        fetched = _fetch_google_news_rss_raw(term, symbol, match_type="proxy", max_articles=max_per_term)
        results.extend(fetched)

    logger.info(
        "Fetched %d sector/proxy news articles for %s (%d sector, %d proxy)",
        len(results),
        symbol,
        sum(1 for r in results if r["match_type"] == "sector"),
        sum(1 for r in results if r["match_type"] == "proxy"),
    )
    return results


# ── Economic Times RSS (more reliable than HTML scraping) ────────────────────

def scrape_economic_times(ticker: str, max_articles: int = 15) -> list[dict[str, Any]]:
    """
    Fetch Economic Times news via their topic RSS feed.
    Much more reliable than HTML scraping — ET's RSS is publicly available.
    """
    symbol = ticker.upper().replace(".NS", "").replace(".BO", "")
    results: list[dict[str, Any]] = []

    # ET provides RSS for company topics
    rss_url = f"https://economictimes.indiatimes.com/topic/{quote_plus(symbol)}/rss.cms"

    try:
        _polite_delay()
        resp = requests.get(rss_url, headers=HEADERS, timeout=15)
        resp.raise_for_status()

        root = ET.fromstring(resp.content)
        channel = root.find("channel")
        if channel is None:
            logger.info("Scraped 0 ET articles for %s (no RSS channel found)", symbol)
            return results

        for item in channel.findall("item")[:max_articles]:
            try:
                headline = item.findtext("title", "").strip()
                url = item.findtext("link", "").strip()
                pub_date_str = item.findtext("pubDate", "")
                pub_date = _rfc2822_to_ist(pub_date_str) if pub_date_str else datetime.now(tz=IST)
                description = item.findtext("description", "") or ""
                if description:
                    description = BeautifulSoup(description, "html.parser").get_text(strip=True)[:2000]

                if headline and len(headline) > 10:
                    results.append({
                        "ticker": symbol,
                        "headline": headline,
                        "body": description,
                        "source": "Economic Times",
                        "published_at": pub_date,
                        "url": url,
                    })
            except Exception as exc:
                logger.debug("ET RSS item parse error: %s", exc)
                continue

    except Exception as exc:
        logger.warning("Economic Times RSS error for %s: %s", symbol, exc)

    logger.info("Scraped %d ET articles for %s", len(results), symbol)
    return results


# ── Moneycontrol news tag page ────────────────────────────────────────────────

def scrape_moneycontrol(ticker: str, max_articles: int = 15) -> list[dict[str, Any]]:
    """
    Scrape Moneycontrol news via their tag/topic page.
    Uses the public tag URL which doesn't require authentication.
    """
    symbol = ticker.upper().replace(".NS", "").replace(".BO", "")
    results: list[dict[str, Any]] = []

    # Moneycontrol tag-based news URL (no auth required)
    tag_url = f"https://www.moneycontrol.com/news/tags/{quote_plus(symbol.lower())}.html"

    try:
        _polite_delay()
        resp = requests.get(tag_url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Moneycontrol news list items
        news_items = (
            soup.select("ul.news_list li")
            or soup.select("li.clearfix")
            or soup.select("div.news_box")
            or soup.select("div#cagetory li")
        )

        for item in news_items[:max_articles]:
            try:
                a_tag = item.find("a", href=True)
                if not a_tag:
                    continue
                headline = a_tag.get_text(strip=True)
                url = a_tag["href"]
                if not url.startswith("http"):
                    url = "https://www.moneycontrol.com" + url

                time_el = item.find("span", class_="ago") or item.find("time") or item.find("span", class_="date")
                pub_date = datetime.now(tz=IST)
                if time_el:
                    raw = time_el.get("datetime") or time_el.get_text(strip=True)
                    try:
                        pub_date = _rfc2822_to_ist(raw)
                    except Exception:
                        pass

                if headline and len(headline) > 10:
                    results.append({
                        "ticker": symbol,
                        "headline": headline,
                        "body": "",
                        "source": "Moneycontrol",
                        "published_at": pub_date,
                        "url": url,
                    })
            except Exception as exc:
                logger.debug("Moneycontrol item parse error: %s", exc)
                continue

    except Exception as exc:
        logger.warning("Moneycontrol scrape error for %s: %s", symbol, exc)

    logger.info("Scraped %d Moneycontrol articles for %s", len(results), symbol)
    return results


# ── Combined fetcher ──────────────────────────────────────────────────────────

def fetch_all_news(ticker: str, include_sector_news: bool = True) -> list[dict[str, Any]]:
    """
    Fetch news from all sources for a ticker.

    Sources:
    - Google News RSS (company terms) — primary, match_type='company'
    - Economic Times RSS — supplemental, match_type='company'
    - Moneycontrol tag page — supplemental, match_type='company'
    - Sector/proxy news (via taxonomy) — contextual, match_type='sector'|'proxy'

    The match_type field is used by build_daily_sentiment() to apply
    source-appropriate weights: company=1.0, sector=0.6, proxy=0.4.
    """
    all_news: list[dict[str, Any]] = []

    # Primary: Google News RSS (most reliable)
    all_news.extend(fetch_google_news_rss(ticker))

    # Supplemental: Economic Times RSS (tag with match_type for those that don't have it)
    for item in scrape_economic_times(ticker):
        item.setdefault("match_type", "company")
        all_news.append(item)

    # Supplemental: Moneycontrol
    for item in scrape_moneycontrol(ticker):
        item.setdefault("match_type", "company")
        all_news.append(item)

    # Sector/proxy news: broader thematic coverage from taxonomy
    if include_sector_news:
        all_news.extend(fetch_sector_news(ticker))

    # Deduplicate by normalized headline
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for item in all_news:
        key = item["headline"].lower()[:80]
        if key not in seen:
            seen.add(key)
            unique.append(item)

    company_count = sum(1 for n in unique if n.get("match_type") == "company")
    sector_count = sum(1 for n in unique if n.get("match_type") == "sector")
    proxy_count = sum(1 for n in unique if n.get("match_type") == "proxy")
    logger.info(
        "Total unique news for %s: %d (company=%d, sector=%d, proxy=%d)",
        ticker, len(unique), company_count, sector_count, proxy_count,
    )
    return unique
