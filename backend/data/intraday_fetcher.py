"""
Intraday bar fetcher — NSE stocks, 15-minute resolution.

yfinance supports intraday data:
  interval="15m"  →  available for up to 60 days back
  interval="1m"   →  last 7 days only

We fetch "15m" bars for the last 5 trading days, which gives ~125 bars
(25 bars/day × 5 days). This is enough for the intraday model to learn
intraday patterns while staying fast to download.

Data is NOT persisted to DB — it's always fetched fresh from Yahoo Finance.
Yahoo Finance intraday data updates every ~15 seconds during market hours.
"""

import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")

# NSE trading window
MARKET_OPEN_H,  MARKET_OPEN_M  = 9, 15
MARKET_CLOSE_H, MARKET_CLOSE_M = 15, 30


def fetch_intraday_bars(
    ticker: str,
    days_back: int = 5,
    interval: str = "15m",
) -> pd.DataFrame:
    """
    Fetch intraday OHLCV bars for an NSE stock.

    Returns a DataFrame with columns:
        timestamp (tz-aware IST), open, high, low, close, volume
    Sorted oldest → newest. Returns empty DataFrame on failure.
    """
    symbol = ticker.upper().replace(".NS", "").replace(".BO", "")
    yf_sym = symbol + ".NS"

    try:
        raw = yf.download(
            yf_sym,
            period=f"{max(days_back, 1)}d",
            interval=interval,
            progress=False,
            auto_adjust=True,
        )
    except Exception as exc:
        logger.warning("Intraday fetch failed for %s: %s", symbol, exc)
        return pd.DataFrame()

    if raw is None or raw.empty:
        logger.warning("Intraday: empty response for %s", symbol)
        return pd.DataFrame()

    # Flatten MultiIndex columns (yfinance ≥ 0.2)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0] for col in raw.columns]

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index.name = "timestamp"

    # Ensure IST timezone
    if df.index.tz is None:
        df.index = df.index.tz_localize("Asia/Kolkata")
    else:
        df.index = df.index.tz_convert("Asia/Kolkata")

    # Keep only bars within NSE trading hours (9:15–15:30)
    df = df[
        (df.index.time >= _t(MARKET_OPEN_H, MARKET_OPEN_M)) &
        (df.index.time <= _t(MARKET_CLOSE_H, MARKET_CLOSE_M))
    ]

    df = df.dropna(subset=["close"]).sort_index()
    logger.debug("Intraday %s: %d bars fetched (%s interval)", symbol, len(df), interval)
    return df


def fetch_today_bars(ticker: str) -> pd.DataFrame:
    """Fetch only today's intraday bars (1d period, 15m interval)."""
    return fetch_intraday_bars(ticker, days_back=1, interval="15m")


def split_intraday_for_replay(df: pd.DataFrame):
    """
    Split a multi-day intraday DataFrame into a list of per-day DataFrames,
    ordered oldest → newest. Used for offline replay training.
    """
    if df.empty:
        return []
    days = []
    for date_val, group in df.groupby(df.index.date):
        days.append(group.copy())
    return days


def _t(h: int, m: int):
    import datetime
    return datetime.time(h, m)
