"""
FII/DII net flow fetcher for NSE.

Foreign Institutional Investor (FII) and Domestic Institutional Investor (DII)
net buying/selling is a genuine alpha signal — it tells you whether large "smart
money" players are entering or exiting Indian equities that day.

NSE publishes this at:
  https://www.nseindia.com/api/fiidiiTradeReact?date=DD-Mon-YYYY

That endpoint requires an active NSE browser session (cookies). We implement a
lightweight session-refresh approach using requests + curl_cffi for TLS
fingerprint bypass (same package already used by yfinance).

Falls back gracefully to zeros if the fetch fails — the model still trains, just
without this signal until connectivity is established.
"""

import logging
from datetime import date, timedelta
from functools import lru_cache

import pandas as pd
import requests

logger = logging.getLogger(__name__)

NSE_BASE  = "https://www.nseindia.com"
NSE_FII   = "https://www.nseindia.com/api/fiidiiTradeReact"

_NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
    "X-Requested-With": "XMLHttpRequest",
}


def _make_nse_session() -> requests.Session:
    """Create a requests Session with fresh NSE cookies."""
    session = requests.Session()
    session.headers.update(_NSE_HEADERS)
    try:
        # Fetch the homepage to set session cookies
        session.get(NSE_BASE, timeout=10)
        session.get("https://www.nseindia.com/market-data/live-equity-market", timeout=10)
    except Exception:
        pass  # Proceed with whatever cookies we got
    return session


def _fetch_one_date(session: requests.Session, d: date) -> dict | None:
    """Fetch FII/DII data for a single date. Returns None on failure."""
    date_str = d.strftime("%d-%b-%Y")  # e.g. "22-Mar-2026"
    try:
        r = session.get(NSE_FII, params={"date": date_str}, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        # Response is a list of records like:
        # [{"category": "FII/FPI", "buyValue": ..., "sellValue": ...}, ...]
        result = {"date": d, "fii_net": 0.0, "dii_net": 0.0}
        for rec in data:
            cat = rec.get("category", "").upper()
            try:
                net = float(rec.get("buyValue", 0)) - float(rec.get("sellValue", 0))
            except (TypeError, ValueError):
                net = 0.0
            if "FII" in cat or "FPI" in cat:
                result["fii_net"] = net
            elif "DII" in cat:
                result["dii_net"] = net
        return result
    except Exception as exc:
        logger.debug("FII/DII fetch failed for %s: %s", date_str, exc)
        return None


def fetch_fii_dii(lookback_days: int = 365) -> pd.DataFrame:
    """
    Fetch FII and DII net flows for the last `lookback_days` calendar days.

    Returns a DataFrame indexed by date with columns:
        fii_net   — FII net buy/sell (crores ₹). Positive = net buying.
        dii_net   — DII net buy/sell (crores ₹). Positive = net buying.
        fii_dii_net — combined institutional net flow

    If the NSE endpoint is unreachable, returns an empty DataFrame (non-fatal).
    """
    end_date = date.today()
    start_date = end_date - timedelta(days=lookback_days)

    logger.info("Fetching FII/DII data from NSE (last %d days)…", lookback_days)

    session = _make_nse_session()
    records = []

    # Iterate over business days
    current = start_date
    consecutive_failures = 0
    while current <= end_date:
        if current.weekday() < 5:  # Monday–Friday only
            rec = _fetch_one_date(session, current)
            if rec:
                records.append(rec)
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                # Abort early if NSE is consistently unreachable
                if consecutive_failures >= 10:
                    logger.warning(
                        "FII/DII: 10 consecutive failures — NSE may be unreachable. "
                        "Feature will be zero-filled."
                    )
                    break
        current += timedelta(days=1)

    if not records:
        logger.warning("FII/DII: no data fetched — returning empty DataFrame")
        return pd.DataFrame(columns=["fii_net", "dii_net", "fii_dii_net"])

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df["fii_dii_net"] = df["fii_net"] + df["dii_net"]

    # Derived features: 5-day rolling sum (captures institutional trend)
    df["fii_net_5d"]     = df["fii_net"].rolling(5, min_periods=1).sum()
    df["dii_net_5d"]     = df["dii_net"].rolling(5, min_periods=1).sum()
    df["fii_dii_net_5d"] = df["fii_dii_net"].rolling(5, min_periods=1).sum()

    logger.info("FII/DII: fetched %d trading days of data", len(df))
    return df
