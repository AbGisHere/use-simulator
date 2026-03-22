"""
NSE Delivery Percentage Fetcher.

The NSE Full Bhavcopy (equity market EOD report) contains DELIV_PER —
the percentage of traded volume that resulted in actual delivery
(as opposed to intraday squaring off).

High delivery % = conviction buying/selling by genuine investors.
Low delivery % = speculative intraday activity.

This is a meaningful feature because stocks making large moves on high
delivery % are much more likely to continue the trend.

Data source (publicly accessible, no auth required):
  https://archives.nseindia.com/products/content/sec_bhavdata_full_DDMMYYYY.csv
"""

import io
import logging
import zipfile
from datetime import date, timedelta

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_BHAV_URL = (
    "https://archives.nseindia.com/products/content/sec_bhavdata_full_{ddmmyyyy}.csv"
)

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; NSE-Fetcher/1.0)",
    "Accept": "text/html,application/xhtml+xml,*/*",
}


def _fetch_bhav_for_date(d: date, ticker: str) -> float | None:
    """
    Fetch the delivery percentage for `ticker` on `d`.
    Returns None if the date is a holiday or the fetch fails.
    """
    ddmmyyyy = d.strftime("%d%m%Y")
    url = _BHAV_URL.format(ddmmyyyy=ddmmyyyy)
    try:
        r = requests.get(url, headers=_HEADERS, timeout=15)
        if r.status_code != 200:
            return None

        # The CSV has columns like:
        # SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,LAST,PREVCLOSE,TOTTRDQTY,TOTTRDVAL,
        # TIMESTAMP,TOTALTRADES,ISIN, DELIV_QTY, DELIV_PER
        df = pd.read_csv(io.StringIO(r.text), skipinitialspace=True)
        df.columns = df.columns.str.strip()

        # Filter for EQ series only and the specific ticker
        mask = (df["SYMBOL"].str.strip() == ticker) & (df["SERIES"].str.strip() == "EQ")
        row = df[mask]
        if row.empty:
            return None

        val = row["DELIV_PER"].iloc[0]
        return float(str(val).strip()) if str(val).strip() not in ("", "-", "nan") else None

    except Exception as exc:
        logger.debug("Bhav fetch failed for %s on %s: %s", ticker, d, exc)
        return None


def fetch_delivery_pct(ticker: str, lookback_days: int = 365) -> pd.DataFrame:
    """
    Fetch the daily delivery percentage for `ticker` over the last N calendar days.

    Returns a DataFrame indexed by date with columns:
        delivery_pct     — % of volume that was delivery (0–100)
        delivery_pct_5d  — 5-day rolling average
        delivery_spike   — 1 if today's delivery_pct > 1.5× its 20-day average

    Falls back gracefully to empty DataFrame if NSE is unreachable.
    """
    symbol = ticker.upper().replace(".NS", "").replace(".BO", "")
    end_date = date.today()
    start_date = end_date - timedelta(days=lookback_days)

    logger.info("Fetching delivery_pct for %s (last %d days)...", symbol, lookback_days)

    records = []
    current = start_date
    consecutive_failures = 0

    while current <= end_date:
        if current.weekday() < 5:
            val = _fetch_bhav_for_date(current, symbol)
            if val is not None:
                records.append({"date": current, "delivery_pct": val})
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures >= 15:
                    logger.warning(
                        "Delivery pct: 15 consecutive failures for %s -- "
                        "NSE bhav archive may be unreachable.",
                        symbol,
                    )
                    break
        current += timedelta(days=1)

    if not records:
        logger.warning("Delivery pct: no data for %s -- returning empty DataFrame", symbol)
        return pd.DataFrame(columns=["delivery_pct", "delivery_pct_5d", "delivery_spike"])

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    df["delivery_pct_5d"]  = df["delivery_pct"].rolling(5, min_periods=1).mean()
    ma20 = df["delivery_pct"].rolling(20, min_periods=5).mean()
    df["delivery_spike"] = (df["delivery_pct"] > ma20 * 1.5).astype(float)

    logger.info("Delivery pct: fetched %d days for %s", len(df), symbol)
    return df
