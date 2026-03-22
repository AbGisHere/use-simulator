import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf
from sqlalchemy.orm import Session

from config import HISTORY_YEARS, IST

logger = logging.getLogger(__name__)


def _ensure_ns_suffix(ticker: str) -> str:
    """Ensure ticker has .NS suffix for NSE stocks."""
    ticker = ticker.strip().upper()
    if not ticker.endswith(".NS") and not ticker.endswith(".BO"):
        ticker = ticker + ".NS"
    return ticker


def fetch_price_data(
    ticker: str,
    years: int = HISTORY_YEARS,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch OHLCV data for an NSE ticker using yfinance.

    Returns a DataFrame with columns:
        Date (IST DatetimeIndex), Open, High, Low, Close, Volume

    Timestamps are converted to IST. Only market-hours data is returned.
    """
    ns_ticker = _ensure_ns_suffix(ticker)
    end_date = datetime.now(tz=IST)
    start_date = end_date - timedelta(days=years * 365 + 30)  # small buffer

    logger.info("Fetching price data for %s from %s to %s", ns_ticker, start_date.date(), end_date.date())

    try:
        yf_ticker = yf.Ticker(ns_ticker)
        df = yf_ticker.history(
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=True,
            actions=True,
        )
    except Exception as exc:
        logger.error("yfinance error for %s: %s", ns_ticker, exc)
        raise RuntimeError(f"Failed to fetch data for {ns_ticker}: {exc}") from exc

    if df.empty:
        raise ValueError(f"No price data returned for {ns_ticker}. Check the ticker symbol.")

    # Normalise columns
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index.name = "Date"

    # Convert index to IST (yfinance returns UTC or tz-aware)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(IST)

    # Keep only valid trading days (drop weekends and zero-volume holidays)
    df = df[df["Volume"] > 0]
    df = df.dropna(subset=["Close"])

    # Sort ascending
    df = df.sort_index()

    logger.info("Fetched %d trading days for %s", len(df), ns_ticker)
    return df


def get_company_name(ticker: str) -> str:
    """Return the long name for a ticker, falling back to the ticker itself."""
    ns_ticker = _ensure_ns_suffix(ticker)
    try:
        info = yf.Ticker(ns_ticker).info
        name = info.get("longName") or info.get("shortName") or ticker
        return name
    except Exception:
        return ticker


def get_latest_price(ticker: str) -> Optional[float]:
    """Return the most recent closing price for a ticker."""
    try:
        df = fetch_price_data(ticker, years=1)
        if not df.empty:
            return float(df["Close"].iloc[-1])
    except Exception as exc:
        logger.warning("Could not fetch latest price for %s: %s", ticker, exc)
    return None
