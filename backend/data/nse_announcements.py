import logging
import time
from datetime import datetime
from typing import Any

import requests

from config import IST, NSE_API_BASE

logger = logging.getLogger(__name__)

# Session with NSE-required headers (NSE blocks requests without a browser UA)
_session = requests.Session()
_session.headers.update(
    {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/",
        "X-Requested-With": "XMLHttpRequest",
    }
)


def _warm_session() -> None:
    """NSE API requires a cookie from the main page before API calls work."""
    try:
        _session.get("https://www.nseindia.com", timeout=10)
        time.sleep(1)
    except Exception as exc:
        logger.warning("Could not warm NSE session: %s", exc)


def _to_ist(date_str: str) -> datetime | None:
    """Parse NSE date strings (DD-Mon-YYYY or YYYY-MM-DD) to IST datetime."""
    if not date_str:
        return None
    for fmt in ("%d-%b-%Y", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            return dt.replace(tzinfo=IST)
        except ValueError:
            continue
    return None


def fetch_corporate_announcements(ticker: str) -> list[dict[str, Any]]:
    """
    Fetch corporate announcements from NSE India public API.

    Returns list of dicts:
        {type, date (IST datetime), description, ticker}
    """
    # Strip .NS suffix for NSE API
    symbol = ticker.upper().replace(".NS", "").replace(".BO", "")

    _warm_session()

    results: list[dict[str, Any]] = []

    # Corp-info endpoint
    try:
        url = f"{NSE_API_BASE}/corp-info?symbol={symbol}"
        resp = _session.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        # Board meetings
        for item in data.get("boardMeetings", {}).get("data", []):
            dt = _to_ist(item.get("bm_date", ""))
            results.append(
                {
                    "ticker": symbol,
                    "type": "board_meeting",
                    "date": dt,
                    "description": item.get("bm_purpose", "Board Meeting"),
                }
            )

        # Dividends
        for item in data.get("dividends", {}).get("data", []):
            dt = _to_ist(item.get("exDate", ""))
            results.append(
                {
                    "ticker": symbol,
                    "type": "dividend",
                    "date": dt,
                    "description": f"Dividend: {item.get('dividendType', '')} - {item.get('dividendPerShare', '')}",
                }
            )

        # Splits / bonuses
        for item in data.get("splits", {}).get("data", []):
            dt = _to_ist(item.get("exDate", ""))
            results.append(
                {
                    "ticker": symbol,
                    "type": "split",
                    "date": dt,
                    "description": f"Split: {item.get('faceValueNew', '')} -> {item.get('faceValueOld', '')}",
                }
            )

    except requests.exceptions.HTTPError as exc:
        logger.warning("NSE corp-info HTTP error for %s: %s", symbol, exc)
    except Exception as exc:
        logger.warning("NSE corp-info error for %s: %s", symbol, exc)

    # Announcements endpoint
    try:
        url = f"{NSE_API_BASE}/corporate-announcements?index=equities&symbol={symbol}"
        resp = _session.get(url, timeout=15)
        resp.raise_for_status()
        items = resp.json()
        if isinstance(items, list):
            for item in items[:50]:  # limit to recent 50
                dt = _to_ist(item.get("an_dt", ""))
                results.append(
                    {
                        "ticker": symbol,
                        "type": "announcement",
                        "date": dt,
                        "description": item.get("desc", item.get("subject", "Announcement")),
                    }
                )
    except Exception as exc:
        logger.warning("NSE announcements error for %s: %s", symbol, exc)

    # Filter out items with no date
    results = [r for r in results if r["date"] is not None]
    results.sort(key=lambda x: x["date"], reverse=True)

    logger.info("Fetched %d NSE announcements for %s", len(results), symbol)
    return results
